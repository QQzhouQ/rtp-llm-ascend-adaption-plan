# vllm-ascend Dense 模型 TP 实现方案分析

> 调研对象：`/home/d30033799/vllm-ascend`（vllm-ascend 插件 + vLLM core ~0.20.2）
> 关注点：dense（非 MoE）模型的张量并行如何实现、Ascend 特有部分是什么、能否被 rtp-llm 复用

---

## 0. 一句话总结

> **vllm-ascend 的 dense TP 就是标准 Megatron 式 TP（与 CUDA 上的 vLLM 完全同构），通信后端换成 HCCL（torch.distributed backend="hccl"）。** 所有"高级"优化（融合 GEMM+AllReduce、序列并行 FlashComm、细粒度 TP 等）**全部默认关闭、按需 opt-in**。
> **核心结论：默认路径 = 通用 TP + HCCL 后端，没有任何 NPU 特有的必须项。**

---

## 1. 架构分层：什么来自 vLLM core，什么是 Ascend 特有

vllm-ascend 是 vLLM 的插件，**TP 拓扑、切分数学、collective API 全部来自 vLLM core（与 CUDA 共享）**；vllm-ascend 只做三件事：换后端（HCCL）、换算子实现、加 Ascend 融合 kernel。

| 组件 | 来源 | 位置 |
|------|------|------|
| TP/PP/DP 组拓扑、rank 布局、GroupCoordinator API | vLLM core | `vllm/distributed/parallel_state.py:1486-1665` |
| `"hccl"` 后端字面量 | Ascend | `vllm_ascend/worker/worker.py:739` |
| GroupCoordinator 猴补丁（HCCL PG 创建+pg_options、gloo 侧车、PG 复用注册表、all_to_all） | Ascend | `vllm_ascend/patch/worker/patch_distributed.py:100-233`、`_hccl_pg_registry.py` |
| collective 执行 = torch.distributed on HCCL PG | 通用基类 + Ascend communicator | `base_device_communicator.py:180-309`、`npu_communicator.py` |
| 列/行/QKV 切分数学 + weight loader | vLLM core，Ascend 子类继承 | `vllm_ascend/ops/linear.py:126-449` |
| 层替换机制（PluggableLayer oot swap） | vLLM core + Ascend 注册 | `custom_op.py:47-100`、`vllm_ascend/utils.py:706-828` |
| 融合 GEMM+AR（`npu_mm_all_reduce_base`）、GEMM+RS | Ascend（torch_npu ATB op） | `linear_op.py:392-420, 500-577` |
| 融合 matmul+AR+add+RMSNorm AscendC kernel + inductor pass | Ascend 独有 | `csrc/mc2/matmul_allreduce_add_rmsnorm/*` |
| FlashComm1/2、细粒度 TP（o_proj/lmhead/embed/mlp）、SP pass、权重分片广播 | Ascend 独有 | 见 §4-§5 |

---

## 2. 进程组初始化流程

`NPUWorker.__init__` → `_init_device` → `_init_worker_distributed_environment`：

1. `init_distributed_environment(world_size, rank, ..., "hccl")` —— **backend 就是字符串 `"hccl"`**（`worker.py:735-748`）
   - 内部 `torch.distributed.init_process_group(backend="hccl", ...)`（vLLM core `parallel_state.py:1424`）
2. `ensure_model_parallel_initialized(tp_size, pp_size, pcp_size, dcp_size)` —— 建 TP/PP/DP 组
   - rank 布局：**ExternalDP × DP × PP × PCP × TP**；**TP 组是连续 rank 块**（`all_ranks.view(-1, tp_size).unbind(0)`，`parallel_state.py:1572`）
3. `init_ascend_model_parallel()` —— Ascend 额外组（MoE/EPLB/细粒度 TP）
4. Ascend 猴补丁 `GroupCoordinatorPatch` 替换 vLLM 的 GroupCoordinator：
   - 每个 rank-list 建 **HCCL device group**（`new_group(ranks, backend="hccl", pg_options=...)`）+ **并行 gloo CPU 组**（元数据协调）
   - `pg_options` 带 `hccl_buffer_size`（`vllm_ascend/utils.py:1093-1130`）
   - 相同 HCCL 组经 `HcclPgRegistry` 去重复用

---

## 3. Dense 模型一个 layer 的 TP 数据流（默认配置）

输入经 CPU 共享内存 MQ 广播到每个 TP worker（**输入侧无 device 广播**），所有 rank 执行相同调度：

```
x (full [tokens, hidden], 每个 rank 复制)         # residual
 ├─ input_layernorm(x)                             # 本地
 ├─ qkv_proj: AscendQKVParallelLinear             # 列并行(按 head 切)
 │    y = x @ W_qkv_shard[rank]                    # 本地 GEMM, 无通信
 │    GQA: tp_size >= total_kv_heads 时每 rank 持 1 个 KV head 并复制
 ├─ attention (本地 heads, RoPE + SDPA)            # 本地
 ├─ o_proj: AscendRowParallelLinear               # 行并行
 │    partial = attn_out @ W_o_shard[rank]          # 本地 GEMM
 │    h = tensor_model_parallel_all_reduce(partial) # ← HCCL ALL-REDUCE #1
 │    x = x + h
 ├─ post_attention_layernorm                       # 本地
 ├─ gate_up_proj: AscendMergedColumnParallelLinear # 列并行, 本地 GEMM
 ├─ silu_and_mul                                   # 本地
 └─ down_proj: AscendRowParallelLinear
      partial = act @ W_down_shard[rank]            # 本地 GEMM
      h = tensor_model_parallel_all_reduce(partial) # ← HCCL ALL-REDUCE #2
      x = x + h
```

**每个 dense layer 恰好 2 次 TP all_reduce；模型级再加 1 次 embedding all_reduce + 1 次 lm_head logits all_gather。**

AllReduce 调用链：`AscendRowParallelLinear.forward` → vLLM core `RowParallelLinear.forward` → `get_tp_group().all_reduce` → `torch.ops.vllm.all_reduce`（custom-op trampoline，为了 ACL graph 可捕获）→ `NPUCommunicator` → **`dist.all_reduce` on HCCL PG**（`base_device_communicator.py:180-182`）。

### Embedding / LM head
- `AscendVocabParallelEmbedding`：vocab 维切分，本地 lookup 后 mask+零填+`maybe_pad_and_reduce`（通常= all_reduce）（`ops/vocab_parallel_embedding.py:44-206`）
- `AscendParallelLMHead`：vocab 列并行；`AscendLogitsProcessor` 默认 `_gather_logits` → all_gather（或 gather 到 rank0）

---

## 4. Ascend 特有 TP kernel（全部 opt-in，默认关闭）

| 优化 | 机制 | 开关（默认） |
|------|------|------------|
| **融合 GEMM+AllReduce** | `torch_npu.npu_mm_all_reduce_base`（ATB op），需要 HCCL comm 名字串：`pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)` | `enable_matmul_allreduce=False` |
| **FlashComm1 序列并行** | 激活在两个行并行 GEMM 之间保持序列分片：o_proj/down_proj 尾部 reduce_scatter，qkv/gate_up 头部 all_gather；token>1000 才启用；`npu_mm_reduce_scatter_base` 融合 GEMM+RS | `enable_flashcomm1=False` |
| **融合 matmul+AR+residual+RMSNorm** | AscendC kernel `torch.ops._C_ascend.matmul_allreduce_add_rmsnorm`（AIC kernel 内嵌 HCCL 设备侧调用），torch.compile inductor pass 做 `GEMM→AR→add→rmsnorm` 模式替换；AR 被优化为 RS+AG 两段 | compile range>512 才适用 |
| **FlashComm2**（PD 分离 prefill） | o_proj 变 all_to_all+GEMM+reduce_scatter | 默认关 |
| **细粒度 TP** | `_OTP`（o_proj）/`_LMTP`/`_EMBED_TP`/`_MLP_TP` 子组，切 DP 维 | 默认 0（关） |
| **HCCL 环境调优** | `HCCL_INTRA_PCIE_ENABLE=1`、`HCCL_INTRA_ROCE_ENABLE=0`、`HCCL_OP_EXPANSION_MODE=AIV`、`hccl_buffer_size` PG option | 部署侧 |

### 值得注意的"没有的东西"
- **没有 custom_all_reduce / pynccl 等价物**：`NPUCommunicator.__init__` 显式 `self.ca_comm = None`（`npu_communicator.py:36-38`）；所有重 all_reduce 走 HCCL
- 有 `PyHcclCommunicator`（ctypes 直连 `HcclCommInitRootInfo`/`HcclAllReduce` + 显式 aclrtStream），但**生产 TP 不走它**，只有单测在用
- 310P 有降级补丁（broadcast 改 all_gather+select 等）

---

## 5. 通信与计算重叠

- FlashComm1（§4）：RS/AG 替代 AR，激活保持序列分片，减少通信量
- 权重预取 side stream：`prefetch_stream`/`attention_calculation_stream`
- PP 传输按 TP 分片：每个 PP rank 只发自己的 TP slice，接收端 TP 组内 all_gather（`worker.py:415-451`）
- vLLM 的 dual-batch-overlap 在 Ascend 上显式禁用（`platform.py:1020-1036`）

---

## 6. Worker 拉起与 rank 布局

- 布局：`ExternalDP × DP × PP × PCP × TP`，TP rank 连续（tp=2,pp=4 → TP组 [0,1],[2,3],[4,5],[6,3]... 实为 [0,1],[2,3],[4,5],[6,7]）
- 拉起：默认 `MultiprocExecutor` 每 rank 一个进程；`SchedulerOutput` 经 **CPU 共享内存 MessageQueue** 广播（非 HCCL）；也支持 ray / torchrun external_launcher
- 组内控制数据可用挂在 TP GroupCoordinator 上的 `mq_broadcaster`（shm）

---

## 7. 对 rtp-llm 的可复用性评估

### ❌ 不可直接复用（深耦合 vLLM 运行时）
- GroupCoordinator/猴补丁机制、PluggableLayer oot 层替换、shm MQ broadcaster、inductor pattern pass（依赖 torch.compile）、forward_context 管道 —— 全部绑定 vLLM Python 运行时；C++ 引擎的 rtp-llm 需自建组簿记（Ascend 侧要复刻的语义仅：**连续 TP rank、每组一个 HCCL PG + 一个 gloo 侧车**）

### ✅ 可复用（自包含）
| 项 | 说明 |
|----|------|
| **Megatron 式 TP 语义本身** | 就是"通用 TP"，无 NPU 特有必须项 —— rtp-llm 已有等价实现（见 02 文档） |
| 融合 AscendC kernel `matmul_allreduce_add_rmsnorm` | 独立 aclnn op，吃 HCCL comm 名 + tp_size/rank，C++ 可直接调 |
| HCCL comm 名提取模式 | `pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)` —— torch.distributed PG ↔ 裸/融合 HCCL op 的桥 |
| `PyHcclCommunicator`/`HCCLLibrary` | 裸 `HcclCommInitRootInfo` + `HcclAllReduce` + 显式 stream 的干净模板（C++ 引擎原生调用即此形态） |
| torch_npu 融合 op | `npu_mm_all_reduce_base` / `npu_mm_reduce_scatter_base`（有 comm 名即可在 Python 模型代码用） |
| SP/FlashComm1 数据流 | RS 替代 AR 的算法模式，可移植 |
| HCCL 调优旋钮 | `hccl_buffer_size`、`HCCL_INTRA_PCIE/ROCE_ENABLE` 等 |

---

## 8. 对「用 vllm-ascend 的 TP 还是通用 TP」的回答

**vllm-ascend 默认 dense TP = 通用 Megatron TP + HCCL 后端。** 所以问题不是"二选一"：
- "vllm-ascend 的方案"中可搬的是**后端接入方式**（backend="hccl" 的 torch.distributed）与**可选融合优化**
- "通用 TP"的切分/通信数学 vllm-ascend 也是同一套，且 **rtp-llm 已经有自己的等价实现**（qkv sp_head / o_proj sp_0 / ffn col+row / lm_head sp_0_pad8 + 每 layer 2×all_reduce）

**结论：rtp-llm 走"点亮自有通用 TP + 接 HCCL 后端"路线，vllm-ascend 的融合优化作为后续增量。** 具体差距与实施计划见 `02`、`03` 文档。
