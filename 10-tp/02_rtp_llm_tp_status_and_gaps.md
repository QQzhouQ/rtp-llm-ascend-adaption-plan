# rtp-llm 现有 TP 基础设施现状与 Ascend 缺口分析

> 调研对象：`/home/d30033799/rtp-llm`
> 结论：**rtp-llm 的 TP 架构（切分数学 + 通信时序 + 引擎锁步）已完整存在（为 CUDA 设计），Ascend 上只差 4 个工程阻塞点。**

---

## 0. 一句话总结

> rtp-llm **不是从零开发 TP**：权重切分（QKV 按头/o_proj 行切/FFN 列+行/lm_head vocab 切）、每层 2 次 all_reduce、C++ 引擎锁步（输入广播/logits 聚合/rank0 采样）全部已实现。**Ascend 上被 4 个 CUDA 假设挡住**：进程拉起早退、backend 硬断言 nccl、C++ 通信回调写死 cuda、pinned memory 待验证。

---

## 1. 已完整实现的部分（架构资产）

### 1.1 权重切分（加载期完成，TP-aware）

`ModelDeployWeightInfo` 捕获 `tp_size/tp_rank`（`model_loader/model_weight_info.py:179-190`），`WeightModule._split` 按 `W.gpt_style_tp_strategy` 切分（`weight_module.py:621-641`）：

| 权重 | 切分函数 | 语义 | 位置 |
|------|---------|------|------|
| `attn_qkv_w` | `sp_head` | Q 按头切；**KV 按 `gcd(kv_heads, tp)` 切 + 复制**（GQA 复制） | `utils/model_weight.py:1399, 445-494` |
| `attn_o_w` | `sp_0` | dim-0 切 → 行并行 o_proj | `model_weight.py:1403` |
| `ffn_w1/w3/w13` | `ffn_sp_neg1` | 输出(中间)维切 → 列并行 gate/up | `model_weight.py:1435-1446` |
| `ffn_w2` | `ffn_sp_0` | 输入(中间)维切 → 行并行 down | `model_weight.py:1447` |
| `embedding` | `sp_neg1` | **hidden 维切**（每 rank 全 vocab） | `model_weight.py:1374` |
| `lm_head` | `sp_0_pad8` | vocab 切 + pad 到 `tp*8` | `model_weight.py:1375, 657-684` |

### 1.2 注意力 TP

- `ModelConfig::getAttentionConfigs(tp_size)`：`head_num /= tp`、`kv_head_num /= tp`（或 `/gcd`，GQA 复制）（`cpp/config/ModelConfig.cc:155-179`）
- Ascend prefill/decode FMHA 直接消费已除好的头数（`ascend_prefill.py:112-167`、`ascend_decode.py`）→ **Ascend attention 路径天然 TP-shaped**

### 1.3 每 layer 通信时序（Python 模型内，与 vllm-ascend 默认路径同构）

```
Qwen3DecoderLayer.forward (model_desc/qwen3.py:57-77):
  LN → qkv(local) → FMHA(local) → o_proj → ALL_REDUCE(TP)   ← causal_attention.py:95-96
  +res → LN → gate_up(local) → silu → down → ALL_REDUCE(TP) ← dense_mlp.py:87-88
层首: embedding hidden 分片 → ALL_GATHER(TP)                 ← base/ascend/embedding.py:23-34
```

**每 layer 恰好 2 次 TP all_reduce + 层首 1 次 embedding all_gather —— 与 vllm-ascend 默认 dense TP 完全同构（Megatron 式）。**

### 1.4 C++ 引擎锁步

- 每个 rank 独立引擎循环线程；`NormalEngine.cc:454-456` 注释明确 **所有 TP rank 必须同时进入 process() 否则 tpSyncModelInputs 死锁**
- `tpSyncModelInputs`（`cpp/models/ModelTypes.cc:7-332`）：广播打包 shape-hints + 打包 CPU/GPU buffer（`execBroadcast`，默认 TP 模式）；已有 `USING_ASCEND → torch::kPrivateUse1` 分支（105-109, 185-189, 288-294）
- logits 聚合在 C++：`PyWrappedModel::forwardPostLayers`（`PyWrappedModel.cc:582-682`）—— lm_head GEMM(vocab 分片) → `tpSyncEmbeddingOrLogits` → `execAllGather` → 全量 logits 在所有 rank
- **采样只在 tp_rank 0**：`NormalExecutor.cc:178-182`（tp_rank>0 forward 后直接返回）
- 控制面 gRPC 广播：`LocalRpcServer` 的 `BroadcastManager`（tp_rank 0 fanout）
- metrics 只在 rank0 上报

### 1.5 通信底层（Python）

`rtp_llm/models_py/distributed/collective_torch.py`：
- `init_distributed_environment(...)` → `torch.distributed.init_process_group(backend, init_method=f"tcp://{ip}:{port}")`（108-120）
- 组构造：TP 组 = 相同 `world_rank // tp_size`（135-208）
- Python 侧 all_reduce/all_gather/broadcast（564-643）
- **C++ 回调桥**：`librtp_compute_ops.register_comm_ops(cpp_broadcast, cpp_allreduce, cpp_allgather)` —— C++ 引擎的数据面 collective 通过回调走 Python torch.distributed（211-384；`ExecOps.cc:491-547, 655-662`）

### 1.6 进程拉起与 rank 发现

- `start_server.py` → `start_backend_server.py`：`multi_rank_start` fork `local_world_size` 个进程（155-185）
- rank 派生：`set_parallelism_config(world_rank)` → local_rank/tp_rank/dp_rank（`server_config_setup.py:184-291`）
- rank 发现：`DistributedServer` TCPStore（master_ip:port-1），支持 `DISTRIBUTE_CONFIG_FILE`/`GANG_CONFIG_STRING` 多节点
- 设备绑定：Python `torch.npu.set_device(local_rank)`（456-466）+ C++ `aclrtSetDevice`（`ExecOps.cc:622-625`）+ `NormalEngine` ctor `DeviceGuard(kPrivateUse1)`（`NormalEngine.cc:66-79`）
- ACCL NIC 亲和已 Ascend-aware（`start_backend_server.py:363-407`）
- `local_world_size` 的 npu.device_count() 逻辑**已写好**（121/143、201-214）但在 NPU 上不可达（见阻塞点1）

---

## 2. Ascend 阻塞点（按依赖序）

| # | 阻塞点 | 位置 | 现象 |
|---|--------|------|------|
| **1** | **进程拉起早退**：`if not torch.cuda.is_available(): return local_rank_start(...)` | `start_backend_server.py:435-436` | NPU-only 主机上 `multi_rank_start` 永不执行，即使 TP_SIZE>1 也单进程 rank0 跑 |
| **2** | **backend 硬断言 nccl**：`assert backend in ["nccl"]` | `collective_torch.py:78`；调用方 `backend_manager.py:58` 传死 "nccl" | 无法用 "hccl" 初始化 |
| **3** | **C++→Python 通信回调写死 cuda**：`cpp_broadcast/cpp_allreduce/cpp_allgather` 用 `_ensure_cuda()`（CPU→`torch.device("cuda")`）与 `torch.cuda.current_device()` | `collective_torch.py:280-384` | `tpSyncModelInputs`（输入广播）与 logits all_gather 第一次 collective 即抛错 |
| **4** | **pinned memory 待验证**：CPU 广播 buffer 用 `pin_memory()`，代码有 TODO "Ascend - check if this is still true for ascend" | `ModelTypes.cc:113-117` | HCCL 下 pinned host buffer 语义需实测 |

次要风险（非阻塞，需验证）：
- `sp_0_pad8` pad 出来的 logits 列（pad 到 tp*8）经 all_gather 后 **sampler 是否 mask 超出 vocab_size 的列**
- `NormalEngine.cc:100-102` warmup 非 CUDA 跳过（正确性中性）
- 无任何 Ascend TP 测试与文档（现有 collective 测试写死 `device=f"cuda:{local_rank}"`，`collective_torch_test.py:69,78`）

---

## 3. 现有端到端 TP 数据流（设计如此，Ascend 上点亮后即生效）

```
HTTP → frontend (world_rank%tp==0 的 rank) → backend rank0 gRPC

每个 rank（独立循环线程）:
NormalEngine::step()
  ├─ [rank0] scheduler_->schedule(); mayAddFakeStream
  └─ NormalExecutor::process()
       ├─ gatherModelInput
       ├─ tpSyncModelInputs ── execBroadcast(TP) ──► Python torch.distributed.broadcast
       ├─ model_->forward (PyWrappedModel → Python Qwen3Model)
       │    Embedding: F.embedding(hidden 分片) → all_gather(TP)
       │    每层 ×N: LN → qkv(local) → FMHA(local) → o_proj → all_reduce(TP)
       │             +res → LN → gate_up → silu → down → all_reduce(TP) → +res
       │    final norm
       │    C++ forwardPostLayers: lm_head GEMM(vocab 分片)
       │      → tpSyncEmbeddingOrLogits: execAllGather(TP) → 全量 logits
       ├─ [tp_rank>0] return
       └─ [rank0] sampler → dispatch → metrics
```

---

## 4. 与 vllm-ascend 的逐项对照

| 维度 | vllm-ascend | rtp-llm | 差距 |
|------|-------------|---------|------|
| QKV 切分（GQA KV 复制） | vLLM core 列并行 | `sp_head` 加载期切分 | ✅ 等价 |
| o_proj 行切 / FFN 列+行 | vLLM core | `sp_0` / `ffn_sp_*` | ✅ 等价 |
| 每 layer 2×AR | vLLM core | causal_attention + dense_mlp | ✅ 等价 |
| embedding | **vocab 并行 + AR** | **hidden 并行 + AG** | ⚠️ 策略不同但都成立（rtp-llm 全 vocab 复制，内存换通信）|
| lm_head/logits | vocab 并行 + AG | `sp_0_pad8` + C++ AG | ✅ 等价 |
| 通信后端 | torch.distributed **"hccl"** | torch.distributed "nccl"（断言） | ❌ **阻塞点2** |
| 进程拉起 | MultiprocExecutor(MQ) | multiprocessing fork | ❌ **阻塞点1**（NPU 早退） |
| C++ 引擎 collective | 无（纯 Python 引擎） | Python 回调（cuda 写死） | ❌ **阻塞点3**（rtp-llm 特有难点，vllm-ascend 无此问题） |
| 融合 GEMM+AR 等 | 有（opt-in） | 无 | 后续增量 |

**关键差异洞察**：vllm-ascend 是纯 Python 引擎，无「C++ 引擎经回调走 Python collective」这一层；rtp-llm 的阻塞点3 是框架结构特有的，但改动集中在 `collective_torch.py` 三个回调函数内。
