# Qwen3.5-MoE 全特性栈适配：图模式、MTP、Profiling、TP（魏珊）

**摘要**：本文是 RTP-LLM 昇腾适配系列的第二篇。前篇以 Qwen3-0.6B 为切入点打通了从 CUDA 到 CANN 的端到端推理，本篇将目标推进到 Qwen3.5-35B-A3B——结构最复杂的模型（30 层 GDN 线性注意力 + 10 层 FIA + MoE + 混合 KV + 交错 MRoPE），并依次落地四个关键特性：图模式（aclgraph，消除 host 侧调度开销）、MTP 投机解码（draft+verify 双模型加速）、Profiling（torch_npu 性能剖析）、TP 张量并行（HCCL 双卡）。四个特性贯穿"NPU 特化代码收敛到最小面积"原则，最终全栈：图模式 2.3 倍、图+MTP 2.7 倍（相对 eager），TPOT 从 69.7ms 降到 25.6ms，多模态 MMMU 60.7%，TP=2 吞吐 99.2 tok/s，精度全链路无回归。文中记录了每轮 AI 评审暴露的真实问题（块表别名回归、桶键分叉死锁等）及其修复，以及从"AI 辅助编码"到"AI 评审驱动工程闭环"的方法演进。

## 从 0.6B 跑通到 35B 全栈

前篇记录了 RTP-LLM 昇腾适配的起步阶段——从 CUDA 到 CANN 的全栈迁移中，团队采用"社区经验 + AI 大模型"双轮驱动，以 Qwen3-0.6B 为切入点打通了端到端推理：编译基础设施、设备抽象、算子接入、KV Cache 与 Attention，一环扣一环。那时的目标是"能跑通"，Qwen3-0.6B 作为结构最简单的验证模型，恰好把整条推理链路从零到一铺了起来。

但"能跑通"和"能上线"之间的距离，远比想象中大。

Qwen3.5-35B-A3B 是接下来要适配的模型，也是 Ascend 适配序列中结构最复杂的一个：40 层中 30 层是 GDN 线性注意力、10 层是 FIA 全注意力，再叠加 MoE 前馈（256 专家 / top-8 + 共享专家）、混合 KV 缓存（分页 K/V + conv state + SSM state 三种物理布局共存）与交错 MRoPE。0.6B 上的每一条经验——FIA 的参数语义、KV Cache 的布局陷阱、设备上下文的封装方式——到了 35B 这里都需要重新审视，因为 GDN 层的状态管理、MoE 的专家路由、混合缓存的按组寻址，在 0.6B 里根本不存在，每一条在 NPU 推理引擎上都是没有现成答案的适配难题。

模型能跑只是第一步。真正让它"能上线"，还需要四个关键特性依次落地：**图模式（aclgraph）消除 host 侧调度开销、MTP 投机解码提升单流吞吐、Profiling 建立性能归因能力、TP 张量并行突破单卡吞吐上限**。本文记录这四个特性在 Qwen3.5-35B 上从设计到验证的完整过程，与前篇的 KV Cache 适配章节一起，构成从"算子跑通"到"全栈可交付"的完整技术叙事线。

四个特性贯穿同一条主线：**让 NPU 特化代码收敛到最小面积**。这条原则从前篇就开始成形——`ascend_impl/` 目录的隔离策略、`#elif USING_ASCEND` 的编译分支、不为 NPU 单独开第二存储通路——在本文的每个特性里都以更成熟的形态出现：图模式的引擎代码收在 `cpp/ascend_graph/` 与 Python `ascend_impl/`，MTP 完全复用主线 speculative 框架只在图捕获侧打补丁，Profiling 只填两处 CUDA-only 缺口，TP 在 alibaba 谱系的全部改动不过 10 个文件、259 行。最终全栈成绩：**图模式 2.3 倍加速、叠加 MTP 后 2.7 倍（相对 eager），TPOT 从 69.7ms 降到 25.6ms，多模态 MMMU 60.7%，TP=2 并发吞吐 99.2 tok/s，精度全链路无回归**。

需要指出的是，这个阶段的工作方法也从前篇的"AI 副驾驶辅助人工适配"，演进到了"AI 评审驱动的工程闭环"——六个评审轮次暴露的每个 P1 级问题（块表别名回归、桶键分叉死锁、行数推断启发式缺陷）都被转化为防御性修复与自动化测试。这种人机协作的深化，从一个侧面反映了适配工作从探索期进入了工程化阶段。

---

## 一、图模式适配：ACL Graph 捕获与重放

### 1.0 图模式原理

在讲适配之前，先解释什么是"图模式"以及它为什么能加速。

#### 为什么需要图模式

NPU 推理引擎的 decode 阶段每生成一个 token，都要执行一次完整的模型前向——对 Qwen3.5 来说就是 40 层（30 次 GDN + 10 次 FIA）+ MoE 路由 + 采样。**eager 模式下每次前向都是"从零调度"**：Python 逐层调用 → 每个 NPU 算子独立 launch → host 侧逐个等待 → 收集结果 → 返回。

这个过程的性能杀手不是计算本身，而是 **host 侧开销**：

- Python 层调度延迟（逐层逐算子的函数调用、张量元数据构造）
- 每个 NPU 算子的 launch 开销（API 调用 → 驱动 → 硬件命令下发）
- host 与 device 之间的隐式同步点（dtype 检查、shape 校验等）

当 batch_size=1（单请求 decode）时，计算本身可能只要 2ms，但 host 侧开销加起来可能 15ms——**大部分时间在"等指令下发"而不是"在算"**。0.6B 的数据证实了这一点：eager 18.6ms/tok → 图模式 6.6ms/tok，节省的 12ms 全是 host 开销。

#### 图模式的核心思想

**把一整次前向录制（capture）成一个"计算图"，之后每次 decode 直接重放（replay）这个图**——跳过全部 Python 调度和算子 launch，一条命令让 NPU 执行整段预录制的操作序列。

这要求解决一个根本矛盾：**图的执行序列是固定的，但每次 decode 的输入数据是变化的**（token ID 不同、KV 位置不同、上下文长度不同）。

解决方案是**固定地址 + 数据刷新**：

```
捕获期（一次性）：
  为每个 batch 档位分配一组持久缓冲区（固定设备地址）
  → 在这些缓冲区上执行一次完整前向，NPU 驱动把整个操作序列录成图
  → 图内所有算子引用的都是这些缓冲区的地址

重放期（每步 decode）：
  把新的输入数据写入同一组缓冲区（D2D 拷贝，地址不变）
  → 一条命令重放图 → NPU 按预录制的序列执行 → 从输出缓冲区读结果
```

因为地址没变，图内录制的所有指针引用依然有效；因为数据变了，计算结果反映的是新输入。

#### 按档位分桶

decode 的 batch_size 是变化的（1 个请求、3 个请求、8 个请求……）。图的形状（矩阵维度）在捕获时就固定了，所以需要**按 batch 档位捕获多个图**。实际 batch 落到 ≥ 它的最小档位（`lower_bound`）：bs=3 → 用 bs=8 的图，多出的 5 行是 padding 行（填哑值：input_ids=0，ctx=1 指向保留块 0）。默认档位 `{1, 8, 16, 24, 32}`，从大到小捕获（先暴露最大分配失败）。

#### ACL Graph 与 CUDA Graph 的对应

| CUDA 概念 | Ascend 等价 | 差异 |
|---|---|---|
| `at::cuda::CUDAGraph` | `c10_npu::NPUGraph` | |
| `cudaStreamBeginCapture` | `NPUGraph::capture_begin(pool, mode)` | 模式语义不同（见下） |
| 共享内存池 `cudaGraphExec` | `capture_begin({0,0})` 自动创建独立池 | 每图各自建池，无共享高水位 |
| `cudaGraphLaunch` | `NPUGraph::replay()` | |
| 不可捕获操作（host sync 等） | 同样不可捕获 | ACL 对跨流限制更严格 |

#### FIA 图任务：让"固定图"处理"变化长度"

注意力是特殊的难题：`context_lens`（每条序列的 KV 长度）每步都在变，但图内录制的 FIA 调用参数是固定的。直接入图会导致注意力永远用捕获期的旧长度。

FIA v2 提供了**图任务接口**（借鉴 vLLM-Ascend 的用法）：

- 捕获期：`graph_task_group_begin/end` 包住 FIA v2 调用——录制的不是一个固定 kernel，而是一个**可更新的图任务**，内部通过 ExternalEvent 在设备侧留 wait 点
- 重放期：`update_graph_fia` 在**专用更新流**上以真实 `context_lens` 重发 FIA（`graph_task_update_begin/end`），ExternalEvent 释放 wait 点，图任务的注意力计算用新参数执行

这样图内的注意力每次重放都能看到正确的 KV 长度，且全程无 host 同步。

#### 别名不变式（最关键的约束）

图绑定的是**捕获期张量的设备地址**。这意味着 Python 侧对捕获缓冲的引用必须是零拷贝别名（aliasing）——C++ 每次重放往持久缓冲写入新数据，图内读到同一地址的最新数据。

**任何 out-of-place 操作（如 `tensor.clamp(min=0)`）都会物化一份新张量**：捕获期图绑定的是这份快照的地址，重放时 C++ 更新的是原缓冲，图读到的永远是捕获期的旧值——静默产出错误 token，且短请求回归测不出来（只有一个 KV 块时，种子表值碰巧与真实块 ID 一致）。必须用 in-place `clamp_()`。这是整个图模式中出现过真实回归的最隐蔽约束。

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

### 2.0 MTP 原理

#### 为什么需要 MTP

decode 的根本瓶颈是**串行性**：每生成一个 token 要跑一次完整的 40 层前向。MTP（Multi-Token Prediction）利用 Qwen3.5 checkpoint 自带的 draft 层（enorm/hnorm/eh_proj + 1 层 MQA decoder，无需单独训练）做**投机解码**：draft 模型很轻（1 层 vs 主模型 40 层），每周期用 draft 快速"猜"出 K 个草稿 token，然后主模型一次前向同时验证 K+1 个 token——用 chain speculative sampling 决定接受多少。理想情况下（草稿全对），一个周期产出 K+1 个 token 但主模型只跑一次前向。

#### 一个完整周期（K=2，以 prompt "今天天气" 为例）

**阶段一 Prefill（仅首周期，全部 Eager）**：

| 步 | 操作 | 说明 |
|---|---|---|
| ① | target prefill：4 token 前向 | 写 target KV 0-3；输出 hidden h0..h3 + 末位 logits |
| ② | target 采样 | 采出 **pending token**「不」——追加进序列但 KV 未写（领先 KV 一个位置） |
| ③ | draft prefill：combo 左移 [天 天 气 不] + hidden 配对 | (词,上文hidden)：(天,h0)(天,h1)(气,h2)(不,h3)——仅末位预测用作草稿 → 草稿1=「错」 |
| ④ | 写 SP buffer | tokens=[不,错]、hidden=d1——draft 与 target 之间的桥梁 |

**阶段二 Decode（每周期重复）**：

| 步 | 操作 | 执行方式 | 说明 |
|---|---|---|---|
| ⑤ | draft decode ×(K-1)=1 | **ACL Graph** | 输入(草稿1=错, d1)，draft 自回归推进 → 草稿2=「很」 |
| ⑥ | 拼 verify 输入 | — | [pending「不」, 草稿1「错」, 草稿2「很」]（K+1=3 token） |
| ⑦ | target verify：3 token 一次前向 | **ACL Graph** | 写 target KV 4-6；输出 3 份 logits + hidden 行 0-2 |
| ⑧ | chain speculative sampling | 图外纯 torch | 逐位判定 `u·P_draft < P_target`；全接受→pos=K=2，末位替换为 target 采样→bonus=「棒」；accept=[错,很,棒] |
| ⑨ | 截断与回滚 | — | combo/hidden 按 accept_len 截取；被拒路径的 KV 靠"下周期直接覆盖"回收，无需显式删除 |
| ⑩ | draft 重打头（ragged） | **Eager** | 输入(accept 全部, verify hidden 前 accept_len 行)；覆写 draft KV 4-6；末位预测→下周期草稿1=「好」 |
| ⑪ | specUpdate | — | 序列=今天天气不错很棒（长10）；SP buffer=[棒,好]→下周期（「棒」成为新 pending） |

**不变式**：序列长度 = KV 有效长度 + 1（领先的一个 = pending token，下周期作为 verify 输入首位补写 KV）。

**截断场景的关键差异**（部分接受）：第 ⑧ 步某位被拒时，该位之后所有草稿的 KV 成残留（红底标记），下周期新 pending 直接覆写回收；hidden 截取从 accept_len=3 行变为 2 行；draft KV 同理被 ⑩ 重打头覆写。

#### hidden 的三条链路

| 环节 | hidden 来源 | 配对语义 |
|---|---|---|
| ③ draft prefill | target prefill 输出 h0..h3 | combo 左移后同下标配对 (天,h0)(天,h1)(气,h2)(不,h3)，仅末位预测用作草稿 |
| ⑤ draft 循环 | draft 自己上步输出 d1/d2 | (草稿i, di) 链式自回归 |
| ⑦→⑩ 重打头 | verify 输出 hidden 前 accept_len 行 | (accept[j], verify_hidden[j]) 天然错位对齐，无需左移；全接受取 3 行、截断取 2 行 |

#### KV：同一 block 池内的双份布局

draft 层以全局层号追加在主模型层之后，主模型与 draft **共享 block 池**、块内按层偏移划分。MtpExecutor 持有两张寻址表（target/draft `kv_cache_layer_to_group`），每次 forward 前切换。target 与 draft 的 KV 池在逻辑上完全独立。

### 2.1 图决策：decode-only 矩阵

| 前向 | 执行方式 | 形状 | 原因 |
|---|---|---|---|
| target verify（⑦） | ACL Graph | 每流 K+1 token | 固定 batch bucket 可捕获 |
| draft decode 循环（⑤） | ACL Graph | 每流 1 token | 与普通 decode 图同构 |
| target prefill（①）/ draft prefill（③） | Eager | 长度可变 | 不固定不可捕获 |
| draft 重打头（⑩） | Eager | 长度=accept_len ∈ [1,K+1] | ragged 长度需 pad 方案，当前 eager 代价已含在 10.3ms 成绩内 |

关键修正：基线对 SP 模型一刀切 `num_tokens_per_bs = K+1`——draft 实例也按 K+1 捕获，与其每流 1 token 的真实输入错配（图不可用）。改为按 `!model_id` 区分：target=K+1、draft=1，draft 图由此可用。

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

### 2.6 FIA verify 图（⑦ 的 full-attn 层）

`ascend_prefill.py` 的 `AscendPrefillImpl` 增加图协议：

- **捕获时**：`graph_task_group_begin/end` 包住 FIA v2 调用并留 ExternalEvent wait 点
- **每次回放前**：`update_graph_fia` 在专用 update stream 上以真实 kvlen 重发 FIA（`graph_task_update_begin/end`）并 record event 释放 wait 点——**图内 kvlen 动态刷新、无 host stall**
- 配套地，图模式下 RoPE positions 与 KV slot_mapping 改为 device 侧由 `sequence_lengths_plus_1` 派生

### 2.7 代码修改总览（对照基线，11 文件 +667/-96）

**C++ 侧（5 文件）**：

| 文件 | 改动 | 对应步骤 |
|---|---|---|
| `cpp/models/PyWrappedModel.h` | 修正 SP 模型的图捕获形状：基线一刀切 `num_tokens_per_bs=K+1`（draft 也按 K+1 捕获，与每流 1 token 错配，图不可用）——改为 `!model_id` 区分：target=K+1、draft=1 | ⑤⑦ 的图/Eager 徽章来源 |
| `cpp/ascend_graph/ascend_graph_runner.cc` | verify 捕获输入（prefill 形态标记、prefix_lengths 捕获）+ 回放双守卫 + layer_to_group min 截断 | ⑦ |
| `cpp/normal_engine/speculative/MtpExecutor.cc` | `#if USING_ASCEND` 跳过 `sp_prefill_draft_model_` 创建（decode-only 语义） | ⑩ 走 eager 的原因 |
| `cpp/normal_engine/speculative/SpeculativeSampler.cc` | generator 设备守卫 | ⑧ |
| `cpp/normal_engine/speculative/MtpBatchStreamProcessor.cc` | ragged 布局注释 + accept_len 上界防御 | ⑨⑩ |

**Python 侧（6 文件，按阶段）**：

| 阶段 | 文件 | 新增/修改函数 |
|---|---|---|
| verify 状态推进（⑦ GDN 层） | `kernels/ascend/paged_state.py` | `speculative_state_indices`（per-token 独立 gather，返回 list[T]，每片天然连续） |
| 同上 | `kernels/ascend/recurrent.py` | `_fused_recurrent_multi_device`（逐步 T==1）+ T>1 分发接线 |
| 同上 | `kernels/ascend/causal_conv1d.py` | `_causal_conv1d_update_tokens_device`（conv 状态逐 token 滚动）+ T>1 分发接线 |
| verify 图（⑦ FIA 层） | `attention/ascend_impl/ascend_prefill.py` | FIA 图捕获 / `prepare_cuda_graph` / `signal_graph_events` / `update_graph_fia` + device 侧 RoPE/slot |
| 采样（⑧） | `bindings/core/CudaSampleOp.cc` | `chainSpeculativeSampling`（Ascend 分支，torch 组合） |
| 并发聚合（⑤ 的 hidden 收集） | `bindings/core/CudaOps.cc` | `multiMergeCopy` D2D（`aclrtMemcpyAsync`） |

注：`state_migration.py`（triton 页迁移 precopy）与基线零差异——strided 索引问题已通过 `speculative_state_indices` 的 per-token gather 在调用侧根治。

### 2.8 步骤与代码对照

| 步 | 环节 | 执行方式 | 代码入口 |
|---|---|---|---|
| ① | target prefill | Eager | `MtpExecutor::prefillStep` → `model_->forward` |
| ② | target 采样（产 pending） | 图外 | `Sampler::forward` |
| ③ | draft prefill（combo 左移 + hidden 配对） | Eager | `updatePrefillPostDraftModelInput` → `draft_model_->forward` |
| ④ | 写 SP buffer | — | `GenerateStream::specUpdate` |
| ⑤ | draft decode 循环 ×(K-1) | **ACL Graph** | `MtpExecutor::draftModelDecode` |
| ⑥ | 拼 verify 输入 [pending, 草稿…] | — | `draftModelDecode` 末尾 |
| ⑦ | target verify（K+1 一次前向） | **ACL Graph** | `model_->forward(is_target_verify=true)` |
| ⑧ | chain speculative sampling | 图外 | `SpeculativeSampler::batchSample` → `CudaSampleOp.cc` |
| ⑨ | 截断与回滚（combo/hidden 按 accept_len 截取） | — | `updateDecodePostDraftModelInput` |
| ⑩ | draft 重打头（ragged 多 token） | Eager | `draft_model_->forward`（fallback 路径） |
| ⑪ | specUpdate 回写 | — | `prepareDecodeSpecUpdateInfo` → `specUpdate` |

### 2.9 图外保护点（流程图看不到的）

| 保护点 | 位置 | 存在原因 |
|---|---|---|
| **runner 回放双守卫** | `ascend_graph_runner.cc:prepareInputs` | ① verify 输入的 sequence_lengths 是空张量（MtpExecutor 置 empty({0})），slice/copy 因 numel=0 崩溃——守卫跳过拷贝；② 基线无条件用 sequence_lengths+1 重算 sequence_lengths_plus_1_d，会覆盖上游已填好的 verify 正确值——守卫 `!is_target_verify` 才重算 |
| **跳过 sp_prefill_draft_model_** | `MtpExecutor.cc` | CUDA 上该实例持有 draft 重打头的 K+1 prefill 图；Ascend decode-only 下该图不存在，`#if USING_ASCEND` 跳过创建。重打头不误入 draft decode 图：其输入为 prefill 形态，被 `canRun` 的 `is_prefill` 检查拒绝 |
| **layer_to_group min 截断** | `ascend_graph_runner.cc` | 构造传的是引擎级全长度 layer_to_groups（target+draft=42 层），回放输入是 executor 切换表（本模型 40 层）——裸 copy_ 直接抛 42 vs 40 尺寸不匹配。按公共宽度 slice 拷贝，截断无损语义 |
| **generator 设备守卫** | `SpeculativeSampler.cc` | NPU 上 GenerateStream 只建 CPU generator，`torch::rand(CPU gen + NPU device)` 抛 device-mismatch——投机解码+random_seed 组合崩引擎。守卫跳过设备不匹配行（接受序列退化为不可复现） |
| **multiMergeCopy D2D** | `CudaOps.cc` | decode 并发 ≥8 时 SP hidden 聚合：基线 host `std::memcpy` 直接解引用 NPU device 指针属非法，改为逐段 `aclrtMemcpyAsync` D2D |
| **per-token gather 连续性** | `paged_state.speculative_state_indices` | triton 页迁移 kernel 以裸指针线性读索引，无法消费 strided 视图——写页按 token 独立 gather 成 list[T]，每片天然连续 |
| **逐步 T==1 的 seq_lengths 复用** | `recurrent.py` | 循环内每 token 复用缓存的 [0,1,1,...] device 张量，整环无 host 同步、可被 ACL Graph 捕获 |
| **共享 cache 层零改动** | 不修改 KVCacheManager | layout 全长度问题（getMainModelCacheLayerLayout 返回含 draft 层的全长度）的一次性 memcpy 多写约 8 字节落入对齐 slack，实践无害——不为 1~2% 显存改共享语义 |

### 2.10 结果

两组数据来自不同测试场景，注意区分：

**高并发闭环**（2048/512 负载，1024 req/round × 8，并发 64）——batch 组合 + KV 管理 + MTP verify 与其他请求交织的真实服务场景：

| 配置 | TPOT avg | 相对 eager | MTP 接受率 |
|---|---|---|---|
| eager | 69.7ms | 1× | — |
| graph | 30.0ms | 2.32× | — |
| **graph+MTP(K=2)** | **25.6ms** | **2.72×** | 51.5% |
| graph+MTP(K=3) | 26.6ms | 2.62× | 56.0% |

K=2 在此接受率下反而优于 K=3：K=3 的接受率增量（51.5→56.0%）不足以覆盖 verify 一次前传 K+1=4 token 的开销增量。开环 Poisson 负载（真实用户延迟）下中位 TTFT < 0.3s（闭环几十秒 TTFT 是饱和排队而非服务缺陷），接受率与负载无关（~51.5% 恒定）。

**单流**（concurrency=1，K=2）——无排队干扰的最佳值，反映 MTP 本身的加速上限（早期基线 d81a9617d 上实测）：

| 配置 | TPOT (ms/token) | 单流 TPS | 相对 eager |
|---|---|---|---|
| eager | 56.7 | 17.6 | 1.0× |
| eager + MTP | 34.8 | 28.7 | 1.63× |
| graph | 15.3 | 65.3 | 3.70× |
| **graph + MTP** | **10.3** | **97.3** | **5.52×** |

### 2.11 已知限制与后续方向

| 项 | 状态 |
|---|---|
| draft 重打头入图 | 未做——ragged 长度需 pad 方案，当前 eager 代价已包含在 10.3ms 成绩内 |
| random_seed 请求 | NPU 上接受判定不可复现（见 generator 守卫），不崩溃 |
| fla-npu 单 launch 多 token | 上游 kernel bug 规避中（逐步 T==1），上游修复后可回退 |
| cache 层 memcpy 微越界 | 与上游一致的技术债，建议上游统一根治 |

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

## 四、TP 张量并行：HCCL 双卡（alibaba 谱系二次移植）

### 4.0 概述与五道门

TP 在 ningweikang 谱系首次跑通（PR #44，5 文件 ~70 行），后来在 alibaba `feature/ascend` 谱系二次移植时遇到了**五道新的门**——因为上游在 #1612/#1620 期间加了一组防御性 fail-fast 和 CUDA 假设，全部需要逐个打开：

| # | 门 | 位置 | 处理 |
|---|---|---|---|
| ① | Python TP fail-fast | `start_backend_server.py:403`："TP>1 not supported on Ascend NPU; run single-card" | 删除（TP 功能即本分支） |
| ② | `backend="nccl"` 硬编码 | `backend_manager.py:59`：虽然上面有 dist_backend 变量的 import 但实际写死 `"nccl"` | 改 `dist_backend`（`get_device_type()` → hccl/nccl） |
| ③ | assert 只允许 nccl | `collective_torch.py:197`：`assert backend in ["nccl"]` | 改 `["nccl", "hccl"]` |
| ④ | `--world_size` 默认 1 | `parallel_group_args.py`：默认 1，不自动从 tp_size 推导 | 用户显式传 `--world_size=2`（帮助文案写"WORLD_SIZE = TP_SIZE * DP_SIZE"但无代码实现） |
| ⑤ | C++ TP fail-fast | `NormalEngine.cc:226`：与 ① 成对的 C++ 侧检查 | 删除（保留 device_input 检查，那是另一特性） |

另有两个运行时陷阱（启动后才暴露）：

| 陷阱 | 现象 | 修复 |
|---|---|---|
| `symm_mem` CUDA 依赖 | `TorchSymmMemCommunicator` 硬要 `torch.cuda`，CPU 版 torch 抛 "Torch not compiled with CUDA enabled" | `_create_process_groups` 中加 `if not _is_npu` 跳过（NPU TP 走 HCCL all-reduce 路径） |
| flashinfer allreduce | CUDA-only 包，NPU 上 init 必崩 | `_init_flashinfer_allreduce` 加 `if _is_npu: return` |

### 4.1 可复用的架构资产

rtp-llm 已有完整的 Megatron 式 TP 框架，NPU 上不需要新建——只需打通设备管理与通信后端：

| 资产 | 位置 | 说明 |
|---|---|---|
| 权重切分 | `model_weight_info.py` + `model_weight.py` | `sp_head`(QKV按头+GQA复制) / `sp_0`(o_proj行切) / `ffn_sp`(FFN列+行) / `sp_0_pad8`(lm_head vocab切) |
| 注意力切分 | `ModelConfig.cc:155-179` | `getAttentionConfigs(tp_size)`: head_num /= tp, kv_head_num /= tp |
| 每层通信 | `causal_attention.py:95` + `dense_mlp.py:87` | 每 layer 2 次 all_reduce(TP) + 层首 1 次 embedding all_gather |
| 引擎锁步 | `ModelTypes.cc`(tpSyncModelInputs) + `PyWrappedModel.cc`(logits聚合) | C++ 输入广播 / C++ logits allgather / rank0 采样 |
| 回调桥 | `ExecOps.cc:491-547` | `execBroadcast/execAllReduce/execAllGather` → GIL → Python 回调 |

### 4.2 NPU 侧核心改动（alibaba 谱系，collective_torch.py 为主）

**设备无关化回调**（`_register_process_groups_to_cpp` 内部）：

```python
# 模块级定义（必须在 _create_process_groups 之前可用——曾因函数作用域嵌套导致
# symm_mem 门读取不到 _is_npu 而漏生效）
def _npu_runtime() -> bool:
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False

_is_npu = _npu_runtime()
def _accel_device_id() -> int:
    return torch.npu.current_device() if _is_npu else torch.cuda.current_device()
def _accel_device(device_id: int) -> torch.device:
    return torch.device("npu" if _is_npu else "cuda", device_id)
def _accel_is_device_tensor(t) -> bool:
    return t.is_npu if _is_npu else t.is_cuda
```

5 处 `torch.cuda.current_device()` / `torch.device("cuda", ...)` 调用全部替换为上述 helper。`_ensure_cuda` 保留函数名（避免无谓重命名 diff），docstring 说明语义为"the accelerator"。

**pin_memory 防御**（`ModelTypes.cc` tpSyncModelInputs）：

```cpp
// NCCL broadcast requires pinned memory for CPU buffers.
// On Ascend (HCCL) the Python comm callbacks move CPU buffers to the
// NPU before the collective, so pinning is a host-copy optimization
// only — fall back to unpinned memory if pinning is unavailable.
try {
    tensor = tensor.pin_memory();
} catch (const std::exception&) {
    // keep unpinned tensor
}
```

### 4.3 排障记录

- **UB 直连配对**（老谱系首次移植时）：950PR 上非相邻 NPU 卡间无 UB 直连，性能骤降。用 `npu-smi` 拓扑 + 带宽实测确认卡对（4+5），再设 `ASCEND_RT_VISIBLE_DEVICES`。
- **HCCL 初始化陷阱**：`ep_size` 默认 0 会被自动推导为 tp_size 导致拒绝；显式 `--ep_size=1` 解决。
- **world_size 崩溃链**（alibaba 谱系二次移植）：`tp_size=2` 但 `world_size` 默认 1 → `init_distributed_environment` 跳过 → C++ 回调未注册 → `synchronizeBlockNum` 的 `execAllGather` 断言崩溃。教训：**启动命令必须显式传 `--world_size`**。
- **图捕获前的 "Waiting for pending NCCL work"**：TP 环境下 `NPUGraph::capture_begin` 会等 HCCL 集合通信的 pending work 排空——正常等待（几秒），不是死锁。

### 4.4 结果

**alibaba 谱系（含 #1666 图+MTP+多模态全栈叠加）：**

| 项 | 结果 | 对照单卡 |
|---|---|---|
| 图捕获（双 rank） | 5/5 桶 × 2 | 5/5 |
| Greedy | 与单卡一致 | — |
| 串行 TPOT | ~17.3 ms/tok | ~16.4（+1ms = all_reduce 开销） |
| **并发 4×40** | **TPOT p50 22.7ms，99.2 tok/s** | 97.3 tok/s |
| p99 尾部 | 22.9ms ≈ p50（极稳） | — |

**老谱系独立验证**（ningweikang，首次移植）：TTFT p99 163s→73s、吞吐 172→321 tok/s（1.87×）。

启动方式与 GPU 完全一致：
```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m rtp_llm.start_server \
    --checkpoint_path=<ckpt> --model_type=qwen35_moe \
    --enable_cuda_graph=1 --kv_cache_mem_mb=8192 \
    --seq_size_per_block=512 --kernel_seq_size_per_block=512 \
    --tp_size=2 --ep_size=1 --world_size=2
```

---

## 五、Profiling：NPU 性能剖析链路（alibaba 谱系移植）

### 5.1 现状与缺口

rtp-llm 的性能剖析框架（`StepWindowProfiler`：窗口控制 + gRPC/per-req 触发 + Kineto 异步落盘）有三处 CUDA-only 断点：

1. `TorchProfile::activities_` 硬编码 `{CPU, CUDA}`——Kineto 无 NPU 活动
2. `cudaProfilerBegin/End` 在 Ascend 是 no-op + TODO
3. 全代码库零 `torch_npu.profiler` 集成

此外，长期困扰性能归因的是**宿主机 NPU 驱动间歇停摆**：dmesg 出现 `drv_pcie "Message send finish, no resp"` / SVM 错误，表现为 TPOT 在好值与坏值（2-10× 劣化）之间无规律跳变。没有剖析链路就无法区分"代码慢"与"环境抖动"。

### 5.2 方案设计

**取 vllm-ascend 的 `torch_npu.profiler.profile(...)` 调用配方，套进 rtp-llm 已有 C++ 框架**——最小改动、零新依赖：

| 组件 | 设计 |
|---|---|
| 触发链 | 原样复用：gRPC `StartProfile` RPC / per-req `gen_timeline` / `--gen_timeline_sync` |
| 窗口控制 | 原样复用 `StepWindowProfiler`（start_step / num_steps / 异步落盘 worker） |
| **NPU 采集** | `TorchProfile` 的 `USING_ASCEND` 分支经 pybind 调 Python `torch_npu.profiler.profile()` |
| 落盘 | torch_npu 自己写 `*_ascend_pt/` 目录（含自动 analyse），`stopAndCollect` 返回 nullptr 跳过 Kineto 的 save worker |

**关键安全前提**（已核实）：**引擎循环线程 == 调 `tick()` 的线程 == 调 Python `forward_micro_batch` 的线程**（都在 `NormalEngine::step()` 内）。因此：
- 在 `TorchProfile::start()/stop()` 里获取 GIL 调 Python torch_npu profiler **无死锁**
- profiler 在随后的 `forward_micro_batch`（NPU op 发射处）期间处于激活态——与现有 Kineto start/stop 时序完全一致

### 5.3 实施

**C++ 侧**（`TorchProfiler.cc/h`，+110 行）：

```cpp
#if USING_ASCEND
// 持有 Python torch_npu.profiler.profile 对象的 opaque handle
struct AscendProfilerImpl { py::object profiler; };
#endif

void TorchProfile::start() {
#if USING_ASCEND
    // GIL → import rtp_llm.utils.ascend_profiler → 构造 profile 对象
    py::gil_scoped_acquire gil;
    py::object cls = py::module_::import("rtp_llm.utils.ascend_profiler")
                         .attr("AscendTorchNpuProfiler");
    ascend_impl_ = std::make_unique<AscendProfilerImpl>(
        cls(output_dir_, prefix_));
    ascend_impl_->profiler.attr("start")();
#endif
    // ... 原 Kineto start ...
}

std::pair<...> TorchProfile::stopAndCollect() {
#if USING_ASCEND
    if (ascend_impl_) {
        // stop() 触发 on_trace_ready → 写 *_ascend_pt/ 目录
        // torch_npu 自己落盘，无 ProfilerResult 需要 enqueue
        return {nullptr, ""};
    }
#endif
    // ... 原 Kineto stop ...
}
#endif
```

**Python 侧**（`ascend_profiler.py` 新增 84 行）：vllm-ascend 同款 profile 配方，activities 覆盖 NPU kernel（`CPU`/`NPU`）+ CPU 算子（`ProfilerActivity.CPU`）+ 引擎 scope（`WithScope`）三层。

**BUILD**：ascend select 添加 `py_utils` 依赖（pybind GIL 操作需要）。

### 5.4 使用方式

```bash
# 启动时设置输出目录（只设目录不启动采集）
python -m rtp_llm.start_server ... --torch_cuda_profiler_dir=/path/to/traces

# 触发（服务运行中，per-request）
curl -s http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen35_moe",
        "messages": [{"role": "user", "content": "东莞有啥特色饮食？"}],
        "max_tokens": 10,
        "extra_configs": {
            "gen_timeline": true,
            "profile_step": 3,
            "profile_trace_name": "test_npu_prof"
        }
    }'
```

trace 落盘到 `<output_dir>/test_npu_prof_wr0__<pid>_<ts>_ascend_pt/`，含：
- `ASCEND_PROFILER_OUTPUT/trace_view.json`（Chrome Tracing 格式，主 timeline）
- `kernel_details.csv` / `operator_details.csv` / `op_statistic.csv` / `step_trace_time.csv`
- `analyse.done` 标记——**自动分析随 stop() 完成**，无需手动 `analyse()`

查看工具：MindStudio Insight（官方推荐）/ `chrome://tracing` / Perfetto。

### 5.5 实测数据（TP=2 环境）

| 检查项 | 实测 |
|---|---|
| trace_view.json | **225,682 个事件**（TP 下 host+device 两侧） |
| NPU kernel | `aclnnMatmulV3`(750) / `aclnnFusedInferAttentionScoreV5`(30) / `aclnnCausalConv1d`(90) / `aclnnRecurrentGatedDeltaRule`(60) |
| **TP 通信算子** | **`HcclAllreduce`(240 次执行)** / `HcclAllGather`(4)——每层两次 all_reduce 完整捕获 |
| 引擎 scope | `executor.tp_sync_input` / `gather_model_input` / `tp_sync_logits` |
| step trace | Computing 87.4ms / Communication 8.0ms / Free 124.0ms——通信开销 ~9% 一目了然 |

该链路直接支撑了图模式多轮优化决策（FIA workspace 一次性查询策略、conv1d 布局消除转置、GDN 方案C 对比等均有 trace 佐证），也是区分"代码慢"与"驱动停摆"的标准工具。

---

## 六、特性叠加：全栈终态

### 6.1 全栈数据（Qwen3.5-35B-A3B，Ascend 950PR 单卡，2048/512 负载）

| 配置 | TPOT avg | 吞吐 tok/s | 相对 eager |
|---|---|---|---|
| eager | 69.7ms | 106.6 | 1× |
| + aclgraph | 30.0ms | 245.8 | 2.32× |
| + aclgraph + MTP(K2) | 25.6ms | 289.5 | **2.72×** |

多模态（ViT→MRoPE→注入）在图+MTP 下 MMMU 60.7%（900 题全量），纯文本四基准无回归。TP=2 在 ningweikang 谱系独立验证（1.87×）。

### 6.2 TP=2 叠加数据（alibaba 谱系，含图+MTP+多模态全栈）

| 项 | TP=2 | 单卡对照 |
|---|---|---|
| 并发 4×40 TPOT p50 | 22.7ms | ~24ms |
| 吞吐 | **99.2 tok/s** | 97.3 tok/s |
| 图捕获 | 5/5 × 2 rank | 5/5 |
| Greedy | 一致 | — |

### 6.3 跨特性经验

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
30. RTP-LLM-NPU 端到端链路拆解（HTML，服务层→引擎层→模型层→算子层全链路，含代码行号）：[RTP-LLM-NPU端到端链路拆解-Qwen3.5-35B-v0.1.html](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/12-full-stack-article/RTP-LLM-NPU%E7%AB%AF%E5%88%B0%E7%AB%AF%E9%93%BE%E8%B7%AF%E6%8B%86%E8%A7%A3-Qwen3.5-35B-v0.1.html)

**通用决策与修复记录**
31. MRoPE npu_mrope 实现备份：[MRoPE_npu_mrope_实现方案备份.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/MRoPE_npu_mrope_实现方案备份.md)
32. do_sample 温度除零修复：[Ascend后处理do_sample温度除零精度问题修复记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/Ascend后处理do_sample温度除零精度问题修复记录.md)
33. getTorchCudaDevice 决策记录：[getTorchCudaDevice改回index=-1的决策记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/getTorchCudaDevice改回index=-1的决策记录.md)
34. dev_0824 合入全记录：[dev_0824合入问题全记录与代码修改清单.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/dev_0824合入问题全记录与代码修改清单.md)

**上游 PR**
35. 图模式引擎（18 文件，已合入）：https://github.com/alibaba/rtp-llm/pull/1620
36. Qwen3.5 全栈：图模式 + MTP + 多模态：https://github.com/alibaba/rtp-llm/pull/1666
37. TP + Profiling（10 文件，与上述零交叠）：https://github.com/alibaba/rtp-llm/pull/1715
