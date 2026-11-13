# Qwen3.5-MoE 全特性栈适配：图模式、MTP、Profiling、TP（魏珊）

Qwen3.5-35B-A3B 是 Ascend 适配序列中结构最复杂的模型：40 层中 30 层是 GDN 线性注意力、10 层是 FIA 全注意力，再叠加 MoE 前馈（256 专家 / top-8 + 共享专家）、混合 KV 缓存（分页 K/V + conv state + SSM state 三种物理布局共存）与交错 MRoPE。它几乎踩中了 NPU 推理引擎的每一个技术深水区。

本文记录四个特性在该模型上依次落地的完整过程：**图模式（aclgraph）、MTP 投机解码、Profiling、TP 张量并行**。与前文的 KV Cache 适配章节一起，构成从算子到全栈的完整技术叙事线。

四个特性贯穿同一条主线：**让 NPU 特化代码收敛到最小面积**。图模式的引擎代码收在 `cpp/ascend_graph/` 与 Python `ascend_impl/`，MTP 完全复用主线 speculative 框架只在图捕获侧打补丁，Profiling 只填两处 CUDA-only 缺口，TP 的全部改动不过 5 个文件、约 70 行。最终全栈：**图模式 2.3 倍加速、叠加 MTP 后 2.7 倍（相对 eager），TPOT 从 69.7ms 降到 25.6ms，多模态 MMMU 60.7%，精度全链路无回归**。

---

## 一、图模式适配：ACL Graph 捕获与重放

### 1.1 问题背景

rtp-llm 的图模式（`CudaGraphRunner`）从底层就绑定了 CUDA：`at::cuda::CUDAGraph`、`cudaStreamCaptureModeGlobal`、cuBLAS workspace、图内存池。Ascend 没有直接等价物——`c10_npu::NPUGraph` 有自己的捕获模式体系、内存池语义和流管理方式，而且还有几个 CUDA 世界里不存在的坑。

#### 1.1.1 torch_npu 的捕获模式语义

`NPUGraph::capture_begin` 默认 Global 模式，语义是：捕获体内不允许出现任何非捕获流的操作。而我们的捕获体内——Python 模型的完整 decode forward——恰恰包含跨流操作：FIA workspace 的懒初始化分配、共享更新流上的 `graph_task_update`、`non_blocking` 异步拷贝。Global 模式会直接判捕获非法。

必须切到 `Relaxed` 模式。但 Relaxed 的代价是存在"漏录"风险：某些跨流调用可能被放行而未录进图。兜底方案是每个桶捕获完后立即做 `replayAndSyncCheck`——sanity 重放与 eager 预热步逐值对拍，如果任何必需调用被漏录，输出立刻错、捕获中止（fail-fast 而非静默）。

#### 1.1.2 GDN 线性注意力的 host 元数据死结

这是 GDN 入图的根本障碍。fla-npu 的两个 decode 算子历史上只接受 host int 数组作为元数据：

- `npu_causal_conv1d_update`：`block_map.cpu().tolist()` 算页号、在 Python 里构造 host 索引张量再传下去
- `npu_recurrent_gated_delta_rule`：`actual_seq_lengths` 是 Python list 而非 device tensor

这些 D2H 同步（`.cpu()` 调用）在 ACL 捕获区内是非法的——捕获的图只能录制纯设备侧操作，任何 host↔device 同步都会导致捕获失败或语义未定义。

**关键发现**：调研 fla-npu 源码后发现，新版算子已原生支持 device tensor 元数据通道——`npu_recurrent_gated_delta_rule` 只收 device int32（`actual_seq_lengths`、`ssm_state_indices`），host 通道已标记废弃；`npu_causal_conv1d_update` 的 device tensor 通道为主通道。死结不在算子，而在 rtp-llm 自己 Python 适配层的 host 计算。

#### 1.1.3 捕获缓冲的别名不变式

这是整个图模式适配中最隐蔽的一类 bug，出现过两次真实回归。核心原理：**捕获的图绑定的是捕获期张量的设备地址**。C++ 每次重放前把真实数据写入持久捕获缓冲，图内读到的是同一地址的最新数据——这依赖 Python 侧对该缓冲的引用始终是零拷贝别名（aliasing）。

任何"防御性"的 out-of-place 操作（如 `clamp(min=0)`）都会物化一份新张量：捕获期图绑定的是这份快照的地址，重放时 C++ 更新的是原缓冲，图读到的永远是捕获期的旧值——**静默产出错误 token**。第一次回归发生在 0.6B 上：我加了一个对齐 eager 路径的 `clamp`，短请求测试全过（因为短请求只有一个 KV 块，碰巧与种子表值一致），直到评审 AI 指出才修——改为 in-place `clamp_`。从此回归测试必须包含 >2000 token 的长请求（≥5 个 KV 块），确保块表被真实填充过。

#### 1.1.4 混合 KV 与 MRoPE 的图路径

Qwen3.5 的 40 层按 tag 分成两组：GDN 线性注意力层（30 层）维护 conv state + SSM state 分页池，FIA 全注意力层（10 层）维护标准分页 K/V。引擎侧 `kv_cache_layer_layout` 描述这个拓扑，C++ 组 batch 时传 3-D `[group, batch, blocks]` 块表，Python 按 tag 取对应组。

图捕获需要逐组处理：per-tag 块表从 batch 切片的默认组重建，块表宽度从组页几何推导。MRoPE 交错模式（`combo_position_ids [T,3]`——t/h/w 三轴位置）需按捕获桶切片（`num_tokens * 3` 扁平缓冲），在 NPU 上用纯 torch 逐频率段旋转（interleaved 语义不能用 `npu_mrope`，因其不支持交错模式）。

### 1.2 设计方案

#### 1.2.1 独立 runner 而非侵入 CUDA runner

新建 `cpp/ascend_graph/AscendGraphRunner`，结构上镜像 `CudaGraphRunner` 但底层包裹 `c10_npu::NPUGraph`。通过 `#if USING_ASCEND` 编译分支和 BUILD 的 `select` 接线，CUDA/ROCm 路径零改动。包含 6 个新文件（约 1400 行）：

| 文件 | 职责 |
|---|---|
| `ascend_graph_runner.cc/h` | 核心逻辑：逐桶捕获 + sanity 重放 + prepareInputs 重放数据刷新 |
| `ascend_graph_device_shims.cc/h` | NPU stream/event/memcpy/graph API 隔离层，不侵入 CUDA shim |
| `ascend_graph_utils.h` | `AscendGraphInstance`（NPUGraph 封装）+ `MemHold`（持久捕获缓冲） |
| `BUILD` | `using_ascend` select 接线 |

#### 1.2.2 方案C：单一消费者原则

GDN 不换算子。把 Python 适配层里所有 host 侧计算替换为 device 张量元数据：

- `block_map` 以 device int32 tensor 直传算子，不再 `.cpu().tolist()`
- conv 状态索引在 device 侧计算后传给 `causal_conv1d_update`
- eager 与图共用同一段代码——死结从"换算子"变为"换元数据来源"

这条路线的前提是 fla-npu 算子原生支持 device 元数据（见 1.1.2），布局天然匹配（`[pages, HV, V, K]` 恰等于 rtp 池页内布局），池视图可直传（op def 对 state 有 `IgnoreContiguous`）。

#### 1.2.3 FIA 图任务模式（借鉴 vLLM-Ascend）

FIA v2 的 `npu_fused_infer_attention_score_v2` 有专用的图任务接口：

- **捕获期**：`graph_task_group_begin/end` 包住 FIA v2 调用，捕获的不是一个固定 kernel 而是一个"图任务"，通过 ExternalEvent 在设备侧留 wait 点
- **重放期**：`update_graph_fia` 在专用更新流上以真实 `context_lens` 重发 FIA（`graph_task_update_begin/end`），触发 ExternalEvent 让图任务的注意力计算用新参数执行——全程无 host 同步

类级共享 workspace 在首个（最大桶）捕获时一次性分配，后续各桶复用。这一设计绕开了"图内注意力 context 长度固定"的死限制。

#### 1.2.4 逐桶大→小捕获

5 个 batch 档位（1/8/16/24/32，可配），从大到小捕获：`capture_begin({0,0})` 每图自动创建独立内存池（无共享高水位），大桶先捕是为了启动期先暴露最大分配失败。每桶捕获流程：

```
prepareCaptureInputs（分配种子缓冲）→ warmup forward（eager）→
graphCaptureBegin(Relaxed) → 捕获 forward → 输出 copy 到持久缓冲 →
graphCaptureEnd → replayAndSyncCheck（sanity 重放 vs eager 预热输出）
```

### 1.3 关键正确性修复

评审 AI 六轮审查暴露的问题，每条对应一个真实 bug，全部转化为防御性修复：

| 修复 | 内容 | 触发条件 |
|---|---|---|
| pad 哑行 | `sequence_lengths_plus_1_device` pad 行填 1（CUDA 契约：指向保留块 0 位置 0） | 消除捕获 gather 链的 -1 索引 |
| fail-closed | 捕获体异常时调 `graphCaptureEnd` 再重抛 | 杜绝半捕获 NPUGraph 被析构 |
| 桶键去分叉 | 配置桶 > `max_bs` 时过滤 + 去重 + 空集 CHECK | 消除 C++ 桶键 vs Python FIA 句柄键不一致导致的 sanity 死锁 |
| 权威行数 | `prepare_cuda_graph` 以 C++ 传入的 `graph_bs` 为准 | `(seq!=0)` 计数会把 batch-decode 的真实 0 长度行误判为 padding |
| 别名保护 | 块表 in-place `clamp_`（禁止 out-of-place clamp） | 保留捕获缓冲零拷贝别名 |
| MRoPE 契约 | 交错路径按内核 `get_mrope_position_dim` 逐字重写 | THW 槽位轮转（tidx%3）+ lo/hi 半拆配对，带 CPU 对拍测试 |
| 混合几何守卫 | 构造期 CHECK 各组页几何一致 | 非均匀 hybrid 布局 fail-fast 而非静默截断 |

### 1.4 重放路径的数据流

每次 decode 重放前，C++ `prepareInputs` 执行：

1. 把当前 batch 的真实数据（`input_ids`、块表、`sequence_lengths_plus_1_device`、`combo_position_ids`、per-tag 块表）写入持久捕获缓冲——D2D copy
2. pad 行填哑值（input_ids=0，ctx=1 指向保留块）
3. 调 Python `prepare_cuda_graph` → `update_graph_fia` 在更新流上刷新 FIA context_lens
4. `graphReplay` 重放对应桶的图

### 1.5 结果

| 指标 | 值 |
|---|---|
| 图捕获 | 5/5 桶（含混合双组路径） |
| 单流 TPOT | ~57 → ~16.4 ms/tok（3.5×） |
| 并发 64（2048/512） | eager 69.7 → graph 30.0 ms，吞吐 106.6→245.8 tok/s（2.3×） |
| 精度 | ceval/mmlu/gsm8k/arc 四基准与 eager 最大偏差 3.5pp |
| 上游 | alibaba/rtp-llm#1620（引擎 18 文件）+ #1666（Qwen3.5 增量 3 文件），已合入 |

---

## 二、MTP 投机解码：从 eager 到入图

### 2.1 MTP 的工作原理（以 K=2 为例）

一个完整周期分两个阶段共 11 步（编号与流程图一致，prefill 阶段只发生在首个周期）：

**阶段一 Prefill（①→④）**：

| 步 | 操作 | 执行方式 |
|---|---|---|
| ① | target prefill：输入 prompt（4 token）→ 写 target KV 位置 0-3 → 输出 hidden h0..h3 + 末位 logits | Eager |
| ② | target 采样：从位置 3 的 logits 预测位置 4 → 采出 pending token（追加序列但 KV 未写——领先 KV 一个位置） | — |
| ③ | draft prefill：输入 combo 左移 [今天气不] + target hidden h0..h3 配对 → 写 draft KV 0-3 | Eager |
| ④ | 写 SP buffer（hidden 与 KV 的桥梁） | — |

**阶段二 Decode（⑤→⑪）**：

| 步 | 操作 | 执行方式 |
|---|---|---|
| ⑤ | draft decode 循环 ×(K-1)=1：输入 (上一步草稿, 对应 hidden) → 草稿2 | **ACL Graph** |
| ⑥ | 拼 verify 输入 [pending, 草稿1, 草稿2]（K+1=3 个 token） | — |
| ⑦ | target verify：一次前向 3 token → 写 KV + 输出每 token logits + GDN 逐 token 状态快照 | **ACL Graph** |
| ⑧ | chain speculative sampling：逐 token 判定接受/拒绝 → 确定本轮产出 | 图外（纯 torch） |
| ⑨ | target 采样被接受末位 → bonus token | — |
| ⑩ | draft 重打头：用被接受的真实 token 覆写 draft KV 的草稿位置 | — |
| ⑪ | 下一周期的 pending = bonus（或被接受末位），回到 ⑤ | — |

**不变式**：序列长度 = KV 有效长度 + 1（领先的一个 = pending token，下周期作为 verify 输入首位补写 KV）。

### 2.2 挑战

- **语义层**：rtp-llm 的 chain speculative sampling 是 CUDA kernel，NPU 上 `throw UNIMPLEMENTED`。需纯 torch 重写且逐 token 对拍 CUDA 参考实现。
- **结构层**：target-verify 是 prefill 形态的 forward（一次 K+1 token），而 aclgraph 是 decode-only——verify 步需单独入图。GDN 层在 verify 时 token_count ∈ (1,8]，命中 fla-npu 的多 token 路径。
- **图模式层**：draft re-prop 与 verify 的 MRoPE 三轴位置、`is_target_verify` 分支的 slot 映射、图内块表端点（`cb×ntp` 与桶的乘积可能超 `max_bs`）——每一处都曾在评审中暴露过静默错误。

### 2.3 三框架调研定形态

| 框架 | 形态 | 借鉴点 |
|---|---|---|
| vllm-ascend | V1 单进程双模型 + EAGLE 式 verify | draft 与 target 同 worker、SP buffer 配对 |
| SGLang | 独立 draft worker | 通信开销过大，不采纳 |
| xLLM | verify 入图 | ACL Graph 与 MTP 的组合方式 |

最终选定"target verify 入图 + draft re-prop 入图"为 Ascend 最优形态：draft 与 target 共享捕获缓冲几何，无跨进程通信。

### 2.4 实施分两阶段

**Phase 1（eager MTP）**：4 个根因修复——① `kCUDA` 硬编码清理（27 处）；② `is_cuda()` 在 NPU 恒 false 的判断修正（改 `is_privateuseone()`/device 比较，22 处）；③ rejectionSampling 纯 torch 重写（cumsum/gather/where 组合，无 `nonzero()`/动态 shape/host sync，语义对齐 ROCm kernel：`u·P_draft < P_target` 判定、首个拒绝位扫描、残差逆 CDF 恢复）；④ target verify 的 prefill 图分支接入。产出 15 文件 +711/-75 行，eager MTP 35ms/tok。

**Phase 2（入图）**：verify 图捕获（`is_target_verify=True` 的独立捕获集合）+ draft re-prop 入图。修复图内 MRoPE 三轴位置（`combo_position_ids` 捕获缓冲按 ntp 切片）与 `sequence_lengths_plus_1` 的 ntp 倍数端点。同事后续补齐"MRoPE in MTP verify prefill graphs"（从捕获 `combo_position_ids` 缓冲读三轴位置）与 causal_conv1d 测试对齐。

### 2.5 GDN 多 token 状态推进的陷阱

verify 一次 K+1 token 需要逐 token 写状态快照（回滚基础：接受后从被接受末 token 的状态页续算，被拒页随块覆盖回收）。实现为**逐步 T==1 launch**：每个 token 从前驱写页播种后推进一次 `npu_recurrent_gated_delta_rule`。

选择逐步而非单 launch 的原因：fla-npu 的多 token 单 launch 写错中间页 BF16 快照——同输入对拍 state 偏差 ~1e3（上游算子 bug，绕开而非等待修复）。状态页索引（读页/每 token 写页）在 device 侧计算，可入图。

### 2.6 图外保护点（流程图看不到的）

| 保护点 | 位置 | 原因 |
|---|---|---|
| runner 回放双守卫 | `ascend_graph_runner.cc:prepareInputs` | verify 输入的 sequence_lengths 为空张量（MtpExecutor 置 empty），位置语义全部由 `sequence_lengths_plus_1_device` 承载；空表校验 + ntp 端点 clamp |
| SP hidden 并发聚合 | `CudaOps.cc:multiMergeCopy` | decode 并发 ≥8 时 SP hidden 聚合：基线 host `std::memcpy` 直接解引用 NPU device 指针属非法，改为逐段 `aclrtMemcpyAsync` D2D |
| per-token gather 连续性 | `paged_state.speculative_state_indices` | triton 页迁移 kernel 以裸指针线性读索引，无法消费 strided 视图——写页按 token 独立 gather 成 list[T]，每个切片天然连续 |
| 共享 cache 层零改动 | 不修改 KVCacheManager | layout 全长度问题（getMainModelCacheLayerLayout 返回含 draft 层的全长度）的一次性 memcpy 多写约 8 字节落入对齐 slack，实践无害 |

### 2.7 结果

| 配置 | TPOT avg | 相对 eager | MTP 接受率 |
|---|---|---|---|
| eager | 69.7ms | 1× | — |
| graph | 30.0ms | 2.32× | — |
| **graph+MTP(K=2)** | **25.6ms** | **2.72×** | 51.5% |
| graph+MTP(K=3) | 26.6ms | 2.62× | 56.0% |

K=2 在此接受率下反而优于 K=3：K=3 的接受率增量（51.5→56.0%）不足以覆盖 verify 一次前传 K+1=4 token 的开销增量。开环 Poisson 负载（真实用户延迟）下中位 TTFT < 0.3s（闭环几十秒 TTFT 是饱和排队而非服务缺陷），接受率与负载无关（~51.5% 恒定）。

---

## 三、Profiling：NPU 性能剖析链路

### 3.1 现状与缺口

rtp-llm 的性能剖析框架（`StepWindowProfiler`：窗口控制 + gRPC/per-req 触发 + Kineto 异步落盘）有三处 CUDA-only 断点：

1. `TorchProfile::activities_` 硬编码 `{CPU, CUDA}`——Kineto 无 NPU 活动
2. `cudaProfilerBegin/End` 在 Ascend 是 no-op + TODO（`ExecOps.cc:383-399`）
3. 全代码库零 `torch_npu.profiler` 集成

此外，长期困扰性能归因的是**宿主机 NPU 驱动间歇停摆**：dmesg 出现 `drv_pcie "Message send finish, no resp"` / SVM 错误，表现为 TPOT 在好值与坏值（2-10×劣化）之间无规律跳变。没有剖析链路就无法区分"代码慢"与"环境抖动"。

### 3.2 方案

**取 vllm-ascend 的 `torch_npu.profiler.profile(...)` 调用配方，套进 rtp-llm 已有 C++ 框架**——最小改动、零新依赖。

关键安全前提（已核实）：**引擎循环线程 == 调 `tick()` 的线程 == 调 Python `forward_micro_batch` 的线程**（都在 `NormalEngine::step()` 内，`NormalEngine.cc:471-489`）。因此：
- 在 `TorchProfile::start()/stop()` 里获取 GIL 调 Python torch_npu profiler **无死锁**
- profiler 会在随后的 `forward_micro_batch`（NPU op 发射处）期间处于激活态——与现有 Kineto start/stop 时序完全一致

### 3.3 实施与产出

改动集中在两处：`TorchProfile` 的 Ascend 分支经 pybind 调 `torch_npu.profiler`（activities 覆盖 NPU kernel + CPU 算子 + WithScope），`ExecOps.cc` 填掉 cudaProfiler no-op 缺口。窗口控制、gRPC `StartProfile`、per-req `gen_timeline` 触发链原样复用。

端到端采集 12 万事件成功。该链路直接支撑了图模式多轮优化决策：FIA workspace 一次性查询策略、conv1d 布局消除转置、GDN 优化前后对照（vllm 形态 385ms vs 方案C 28ms 的 trace 佐证）。

---

## 四、TP 张量并行：HCCL 双卡

### 4.1 可复用的架构资产

rtp-llm 已有完整的 Megatron 式 TP 框架，NPU 上不需要新建——只需打通设备管理与通信后端：

| 资产 | 位置 | 说明 |
|---|---|---|
| 权重切分 | `model_weight_info.py` + `model_weight.py` | `sp_head`(QKV按头+GQA复制) / `sp_0`(o_proj行切) / `ffn_sp`(FFN列+行) / `sp_0_pad8`(lm_head vocab切) |
| 注意力切分 | `ModelConfig.cc:155-179` | `getAttentionConfigs(tp_size)`: head_num /= tp, kv_head_num /= tp |
| 每层通信 | `causal_attention.py:95` + `dense_mlp.py:87` | 每 layer 2 次 all_reduce(TP) + 层首 1 次 embedding all_gather |
| 引擎锁步 | `ModelTypes.cc`(tpSyncModelInputs) + `PyWrappedModel.cc`(logits聚合) | C++ 输入广播 / C++ logits allgather / rank0 采样 |
| 回调桥 | `ExecOps.cc:491-547` | `execBroadcast/execAllReduce/execAllGather` → GIL → Python 回调 |

### 4.2 NPU 侧断点与修复（5 文件 ~70 行）

| # | 文件 | 改动 | 作用 |
|---|---|---|---|
| 1 | `start_backend_server.py` | +12/-4 | 按 `get_device_type()` 选拉起路径，解锁 `multi_rank_start` |
| 2 | `backend_manager.py` | +5/-1 | 按设备类型选 hccl/nccl |
| 3 | `collective_torch.py` | +64/-20 | ① 断言放开 hccl；② C++ 通信回调设备无关化（CPU→当前加速器搬运）；③ `_ensure_accel` 助手 |
| 4 | `ModelTypes.cc` | +12/-4 | pin_memory 防御回退（回调先搬 NPU 再 collective，pin 仅为 host 拷贝优化） |
| 5 | `base_model.py` | +8/-1 | `_get_device_str()` 改 `torch.npu.current_device()`（确保权重加载设备与 HCCL communicator 设备一致） |

### 4.3 排障两轮

- **UB 直连配对**：950PR 上非相邻 NPU 卡间无 UB 直连，性能骤降。用 `npu-smi` 拓扑 + 带宽实测确认卡对（4+5），再设 `ASCEND_RT_VISIBLE_DEVICES`。
- **HCCL 初始化陷阱**：`ep_size` 默认 0 会被自动推导为 tp_size 导致拒绝；显式 `--ep_size=1` 解决。

### 4.4 结果

TTFT p99 163s→73s（并发 64）、吞吐 172→321 tok/s（1.87×），精度与单卡一致。启动方式与 GPU 完全一致（`ASCEND_RT_VISIBLE_DEVICES` + 标准 TP 参数）。

---

## 五、特性叠加：全栈终态

### 5.1 全栈数据（Qwen3.5-35B-A3B，Ascend 950PR 单卡，2048/512 负载）

| 配置 | TPOT avg | 吞吐 tok/s | 相对 eager |
|---|---|---|---|
| eager | 69.7ms | 106.6 | 1× |
| + aclgraph | 30.0ms | 245.8 | 2.32× |
| + aclgraph + MTP(K2) | 25.6ms | 289.5 | **2.72×** |

多模态（ViT→MRoPE→注入）在图+MTP 下 MMMU 60.7%（900 题全量），纯文本四基准无回归。TP=2 在 ningweikang 谱系独立验证（1.87×）。

### 5.2 跨特性经验

1. **别名不变式是图模式的宪法**。捕获图绑定捕获期设备地址，任何 out-of-place "防御"都是毒药。回归必须含多 KV 块长请求——短请求会让块表错误"碰巧正确"。
2. **评审是净增益**。AI 评审六轮共暴露 4 个 P1、3 类 P2 真问题（clamp 快照、桶键分叉、行数启发式、acl_sk.h 告警链），全部转化为防御性修复与测试。
3. **先铺地基再起楼**。MRoPE 数学（两个图模式 PR 里带对拍测试交付）成为后来多模态零成本开门的前提；设备串通用化（`device_name()`）提前消掉了 ViT 构建的地雷。
4. **平台门是支持矩阵的显式表达**。`if is_ascend(): return` 式门禁在能力未就绪时是诚实的 fail-fast，能力就绪后删除即完成适配（多模态三门 -20 行即全通）。
5. **环境的错误要显式归档**。宿主 NPU 驱动间歇停摆（dmesg 特征签名）曾三次伪装成代码回归；建立"好值基线 + dmesg 检查"的习惯后避免了大量误判。

### 参考文档

**图模式（aclgraph）**
1. 方案C 总结（AscendC 单一消费者 GDN 入图）：[rtp-llm方案C-AscendC单一消费者GDN入图实施总结.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/qwen35-practice/rtp-llm方案C-AscendC单一消费者GDN入图实施总结.md)
2. SGLang GDN 入图调研（方案A 前身）：[rtp-llm借鉴SGLang的GDN入图实施方案.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/qwen35-practice/rtp-llm借鉴SGLang的GDN入图实施方案.md)
3. vLLM-Ascend GDN 调研：[rtp-llm借鉴vllm-ascend的GDN入图实施方案.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/qwen35-practice/rtp-llm借鉴vllm-ascend的GDN入图实施方案.md)
4. GDN 性能优化探索：[图模式GDN性能优化探索记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/qwen35-practice/图模式GDN性能优化探索记录.md)
5. conv1d 捕获同步问题：[causal_conv1d_graph_capture_sync_issue.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/qwen35-practice/causal_conv1d_graph_capture_sync_issue.md)
6. Qwen3.5 验证总结：[图模式叠加Qwen3.5验证总结.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/qwen35-practice/图模式叠加Qwen3.5验证总结.md)
7. SGLang Decode 阶段 GDN 调研：[SGLang_Decode阶段GDN实现调研.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/qwen35-practice/SGLang_Decode阶段GDN实现调研.md)
8. 0.6B vs 35B 差异：[acl-graph路径-0.6B与35B差异.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/qwen35-practice/acl-graph路径-0.6B与35B差异.md)
9. 图模式适配原始方案：[rtp-llm-aclgraph-adaptation-plan.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/6-graph-mode/rtp-llm-aclgraph-adaptation-plan.md)

**MTP 投机解码**
10. MTP 调研与实施方案（含 Phase1+2 全过程）：[rtp-llm-Ascend-MTP调研与实施方案.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/7-mtp/rtp-llm-Ascend-MTP调研与实施方案.md)
11. Phase1+2 代码改动详细总结：[rtp-llm-Ascend-MTP-Phase1+ Phase 2代码改动详细总结.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/7-mtp/rtp-llm-Ascend-MTP-Phase1+%20Phase%202代码改动详细总结.md)
12. MTP 调用流程图：[mtp_call_flow.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/7-mtp/mtp_call_flow.md)
13. MTP 特性开发记录：[mtp_feature_dev.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/7-mtp/mtp_feature_dev.md)
14. 多 bs 并发 prefill 乱码修复：[多bs并发prefill乱码精度问题修复记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/7-mtp/多bs并发prefill乱码精度问题修复记录.md)
15. 温度 topK topP 950PR 问题：[temperature_topk_topp_950PR问题记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/7-mtp/temperature_topk_topp_950PR问题记录.md)
16. MTP 方案交付说明 PDF（两张流程图 + 11 步全流程 + 图外保护点）：本地 `MTP/Qwen3.5MTP方案交付说明.pdf`

**Profiling**
17. vLLM-Ascend 剖析分析：[01_vllm_ascend_profiling_analysis.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/01_vllm_ascend_profiling_analysis.md)
18. rtp-llm 剖析现状：[02_rtp_llm_current_profiling_state.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/02_rtp_llm_current_profiling_state.md)
19. xLLM 剖析分析：[03_xllm_profiling_analysis.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/03_xllm_profiling_analysis.md)
20. 复用性评估：[04_reuse_assessment.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/04_reuse_assessment.md)
21. rtp-llm 剖析方案（最终推荐）：[05_rtp_llm_profiling_scheme.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/05_rtp_llm_profiling_scheme.md)
22. 落地验证记录：[06_validation_record.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/06_validation_record.md)
23. FIA contiguous 布局分析：[07_fia_contiguous_analysis.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/07_fia_contiguous_analysis.md)

**TP 张量并行**
24. vLLM-Ascend dense TP 分析：[01_vllm_ascend_dense_tp_analysis.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/10-tp/01_vllm_ascend_dense_tp_analysis.md)
25. rtp-llm TP 现状与差距：[02_rtp_llm_tp_status_and_gaps.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/10-tp/02_rtp_llm_tp_status_and_gaps.md)
26. TP 实施方案（最终定稿）：[03_rtp_llm_tp_implementation_plan.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/10-tp/03_rtp_llm_tp_implementation_plan.md)
27. TP 排障记录：[排障记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/10-tp/排障记录.md)

**多模态（附加，图模式的多模态开门）**
28. ViT 调研报告（vLLM-Ascend 考证 + rtp-llm 现状）：[01-调研报告.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/11-multimodal/01-调研报告.md)
29. 多模态实施方案（含全部进度记录）：[02-实施方案.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/11-multimodal/02-实施方案.md)

**端到端链路参考**
30. RTP-LLM-NPU 端到端链路拆解（HTML，服务层→引擎层→模型层→算子层全链路，含代码行号）：本地 `MTP/RTP-LLM-NPU端到端链路拆解-Qwen3.5-35B-v0.1.html`

**通用决策与修复记录**
31. MRoPE npu_mrope 实现备份：[MRoPE_npu_mrope_实现方案备份.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/MRoPE_npu_mrope_实现方案备份.md)
32. do_sample 温度除零修复：[Ascend后处理do_sample温度除零精度问题修复记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/Ascend后处理do_sample温度除零精度问题修复记录.md)
33. getTorchCudaDevice 决策记录：[getTorchCudaDevice改回index=-1的决策记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/getTorchCudaDevice改回index=-1的决策记录.md)
34. dev_0824 合入全记录：[dev_0824合入问题全记录与代码修改清单.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/dev_0824合入问题全记录与代码修改清单.md)

**上游 PR**
35. 图模式引擎（18 文件，已合入）：https://github.com/alibaba/rtp-llm/pull/1620
36. Qwen3.5 全栈：图模式 + MTP + 多模态（3+3 文件）：https://github.com/alibaba/rtp-llm/pull/1666
