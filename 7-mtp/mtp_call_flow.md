# MTP 调用流程梳理

适用范围：rtp-llm 的 MTP（Multi-Token Prediction）投机解码，以 Qwen3.5 在 Ascend 后端的实现（PR #60）为主线，同时标注 CUDA/ROCm 上的通用路径。文中行号以当前代码库为准。

## 1. 先把概念说清楚

MTP 是一种投机解码：用一个小的 draft 模型（通常是从主模型里拆出来的一层，和主模型共享词表与 hidden size）连猜 K 个 token，再让主模型一次前向验证这 K+1 个 token，按概率接受其中对的部分。每个周期最多产出 K+1 个 token，代价是主模型多算了一些被拒掉的位置。收益来自 draft 轻、猜测命中率高，主模型每周期只跑一次。

几个记号，后文统一用：

- K = `gen_num_per_cycle`，draft 每周期猜测的 token 数（不含起始 token）
- verify 的 T = K+1，主模型一次验证的 token 数（起始 token 也要验）
- accept_len，每条流一个周期实际被接受的 token 数，1 到 K+1 之间

启动配置：

```bash
--sp_type=mtp --sp_model_type=qwen35_moe_mtp \
--sp_checkpoint_path=<主模型ckpt> --gen_num_per_cycle=<K> [--enable_cuda_graph 1]
```

## 2. 启动装配：从参数到 MtpExecutor

这一段回答的问题是：引擎是怎么变成"MTP 引擎"的。

### 2.1 参数与 draft 模型创建

`--sp_type` 等参数注册在 `rtp_llm/server/server_args/speculative_decoding_group_args.py`。Python 侧在 `rtp_llm/model_factory.py` 的 `getAndCreateProposeModel`（约 100 行起）里做归一化：

- `deepseek-v3-mtp`、`mixtbstars-mtp` 强制改回 `SpeculativeType.MTP`；
- `qwen_3_moe-mtp` 会改成 `EAGLE3`——这是同类模型的另一条实现路线，不在本文范围内；
- `qwen35_moe_mtp`（本文主角，注册于 `rtp_llm/models/qwen3_next/qwen3_next_mtp.py:152`）不命中任何改写分支，sp_type 按用户传入的 `mtp` 生效。

MTP 的 draft 权重和主模型放在同一个 checkpoint 里，所以 `--sp_checkpoint_path` 就填主模型路径。注意结构上的分工：Qwen3.5-35B-A3B 主模型是 hybrid 架构（GDN 线性注意力层 + 全注意力层 + MoE），而 MTP 头只有一层，注意力是普通 MQA、不带线性注意力（`_create_config` 里 `hybrid_attention_types` 置为 NONE，`num_layers = 1`）。这决定了第 6 节的麻烦都发生在主模型的 verify 前向里，draft 头反而是常规结构。`ProposeModel` 包装了按 `sp_model_type` 创建的 GPT 模型，`gen_num_per_cycle` 透传给模型配置。

### 2.2 跨越 Python/C++ 边界

`rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc` 的 `prepareMTPEngineInitParams`（25 行起）把 Python 侧的 draft 权重翻译成 C++ 的 `EngineInitParams`：

- 取 draft 模型每层的权重，每层造一个 `num_layers = 1` 的临时 ModelConfig；
- 如果 `gen_num_per_cycle > 1` 而权重只有一份，就把这份权重复制 K 份（42–49 行）——目前只支持单层 MTP 头做多步复用；
- 全部打包进 `ProposeModelEngineInitParams(sp_type, gen_num_per_cycle, mtp_model_params)`。

`ProposeModelEngineInitParams`（`rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h`）是 MTP 和 EAGLE 共用的容器，`draftModel()` 返回第一个 draft 的初始化参数。

### 2.3 KV cache 配额：主模型和 draft 挤在同一块里

`NormalEngine::initCacheManager`（`NormalEngine.cc:307`）发现有 draft 模型时走 `CacheConfigCreator::createSpConfig`，关键逻辑在 `CacheConfigCreator.cc:101` 起：

```
num_mtp_modules = gen_num_per_cycle          # EAGLE 固定为 1
layer_all_num   = score层数 + N × draft层数
block_num       = 可用显存 / (score每块字节 + N × draft每块字节)
```

也就是说每个物理 KV 块内部是"主模型各层 + N 份 draft 层"打包存储的，主模型和 draft 共用同一个块池和同一套块表。每个 MTP module 拿到一个子 `CacheConfig`（`mtp_sub_configs`），其中的 `global_layer_ids` 把 draft 层映射到全局层号，方便统一寻址。

这么做的好处是块的分配、迁移、释放只有一套逻辑，draft 不需要单独的 cache manager；代价是每块要预留 draft 的空间，非 MTP 路径不会走到这里。

### 2.4 MtpExecutor 装配

`NormalEngine::initExecutor`（`NormalEngine.cc:123`）：有 `propose_params_` 就建 `MtpExecutor`，否则建 `NormalExecutor`。MtpExecutor 构造函数（`MtpExecutor.cc:119`）装配出这些角色：

| 角色 | 来源 | 说明 |
|---|---|---|
| `model_`（target） | `params.py_model`，model_id=0 | 主模型 |
| `draft_model_` | `params.py_sp_model`，model_id>0 | MTP 头，目前只用第一个 module 的权重 |
| `sp_prefill_draft_model_` | 同上，开图时多建一份 | CUDA 上 draft 前向的图版本；Ascend 上不建 |
| `speculative_sampler_` | `SpeculativeSampler(K)` | 拒绝采样 |
| `fast_topk_sampler_` | `FastTopKSampler` | draft 的 top-1 快采样（softmax + max） |
| `batch_stream_processor_` | `MtpBatchStreamProcessor` | 输入聚合、输出分发 |
| `sampler_` | `Sampler` | 主模型的常规采样器 |

构造时还从 cache manager 抓两份层布局（`getMainModelCacheLayerLayout` / `getMTPModuleCacheLayerLayout`），做成 `target_kv_cache_layer_to_group` 和 `draft_kv_cache_layer_to_group` 两个 host 张量。之后每次 forward 前把对应的布局和 `kv_block_stride_bytes` 换装进 model_input，让同一份 KV 池在主模型和 draft 眼里呈现不同的层视图。

### 2.5 图模式的决策表

`PyWrappedModel.h:215` 有一张决策表，决定每个模型实例的图按"每流多少 token"捕获：

| 模型 | sp | num_tokens_per_bs |
|---|---|---|
| 主模型 verify 图 | 开 | K+1 |
| 主模型普通 decode 图 | 关 | 1 |
| draft prefill 图（CUDA） | 开 | K+1 |
| draft decode（CUDA） | 开 | 1 |
| draft 任何模式（Ascend） | 开 | 不建图，eager |

Ascend 分支（`PyWrappedModel.h:251` 起）只给 target 建 ACL Graph；draft 前向在 MTP 周期里占比小（实测 <5%），图化的复杂度暂时不匹配收益，所以保持 eager（258–263 行的注释就是这段取舍）。

## 3. 运行时：一个请求的生命周期

引擎主循环在 `NormalEngine::loop` → `step()` → scheduler 调度出一批流 → `executor_->process(streams)`。MtpExecutor::process（`MtpExecutor.cc:787`）是总入口：

```
process
├── prepareStreams        # 按角色分桶 + 初始化 SPOutputBuffer
├── prefillStep           # PREFILL / PDFUSION 角色
├── decodeStep            # DECODE / PDFUSION 角色
└── report metrics
```

### 3.1 prepareStreams：流的状态准备（MtpExecutor.cc:754）

- context stream 进 prefill 桶，其余进 decode 桶；
- decode 流调 `setScoreLen(K+1)`：调度器为每条流预留 K+1 个 token 的 KV 空间（verify 会一次写这么多）；
- 每条流挂一个 `SPOutputBuffer`，它跨周期携带三样东西：上周期末尾的 propose token（`tokens`）、draft 的输出概率（`all_probs`）、draft 输入位置的 hidden states（`hidden_states`）。PD 分离部署时这个 buffer 里的张量会经 gRPC 传输，decodeStep 开头再拷回 NPU（MtpExecutor.cc:540 的 clone_sp_tensors）。

### 3.2 prefillStep：正常出第一个 token（MtpExecutor.cc:318）

prefill 阶段没有可投机的上下文，流程接近普通引擎，只是末尾多做一步 draft：

```
gatherModelInput → target prefill forward → target sample
  → updatePrefillPostDraftModelInput   # combo_tokens 左移一位，末尾填新 token
  → draft forward                      # 吃 (新token, 新hidden)，写 draft KV
  → FastTopKSampler                    # draft logits → 下一个 propose token + 概率
  → dispatchPrefill                    # 新 token 写回流，propose token / probs / hidden 存进 SPOutputBuffer
```

`updatePrefillPostDraftModelInput`（MtpBatchStreamProcessor.cc:228）把主模型最后位置的 hidden states 作为 draft 的输入，MTP 头据此预测下一个 token——这个预测不直接给用户，留给下个周期 verify 用。

### 3.3 decodeStep：投机的主战场（MtpExecutor.cc:509）

一个周期的完整时序（以 K>1 为例；K=1 走 `prepareOneStepSpecDecodeModelInput` 的简化路径，跳过 draft 循环）：

```
① gatherDecodeModelInput            # 聚合各流输入，gatherHiddenStates 拼上周期 hidden
② prepareDecodeDraftModelInput      # 取 SPOutputBuffer.tokens[1] 作为 draft 首个输入
③ draftModelDecode                  # K-1 次 draft 前向循环，逐步出猜测 token
④ target verify forward             # 主模型一次吃 B×(K+1) 个 token
⑤ target sample                     # logits → probs，reshape [B, K+1, V]
⑥ SpeculativeSampler                # 链式拒绝采样，得每流 accept_tokens
⑦ updateDecodePostDraftModelInput   # 按接受长度切 hidden，拼下轮 draft 输入
⑧ draft forward + FastTopKSampler   # 为下一周期准备 propose token
⑨ dispatchDecode                    # 接受的 token 写回流，更新 SPOutputBuffer
```

逐步说明：

**① 输入聚合。** `gatherHiddenStates`（MtpBatchStreamProcessor.cc:460）把各流 SPOutputBuffer 里的 hidden states 拼成一份。流数小于 8 逐段 `copy_`；达到 8 走 `multiMergeCopy` 批量 D2D 合并——Ascend 上这个算子是逐段 `aclrtMemcpyAsync`（PR #60 把它从 host memcpy 换成了设备侧拷贝，因为此时源和目的都在 NPU 上，host memcpy 属于非法访问）。

**③ draft 循环。** `draftModelDecode`（MtpExecutor.cc:833）跑 K-1 轮，每轮：draft forward → top-1 采样 → 收集 token id 和概率 → `updateDecodeDraftModelInput` 换输入、`sequence_lengths+1`。循环结束把 K+1 个 token 拼成候选序列：

```
[target_token, propose_token, draft_1, ..., draft_{K-1}]
```

然后把 model_input 改造成 verify 形态：`input_lengths = K+1`、`prefix_lengths = 原序列长度`、`combo_tokens = 候选序列`、`sequence_lengths` 清空。

**④ verify 前向。** `model_input.is_target_verify = true`（MtpExecutor.cc:614），主模型一次吃下 B×(K+1) 个 token，产出每个位置的 logits 和 hidden states，同时把 K+1 个位置的 KV 全部写进 cache。开图时这一步走 ACL Graph 的 replay（第 5 节展开）。attention 掩码用 sparse_mode=3（表示 kv 长度大于 2048 也正确的前缀+当前模式）。

**⑥ 链式拒绝采样。** `SpeculativeSampler::batchSample`（SpeculativeSampler.cc:33）把 draft 概率、draft token、均匀随机数、target 概率和三组输出缓冲交给 `execChainSpeculativeSampling`：

- 判据：对每个位置 i，`u_i * p_i < q_i` 则接受（u 是均匀随机数，p 是 draft 概率，q 是 target 概率；严格小于）
- 第一个不满足的位置截断链条，`accept_len = pos + 1`
- 拒绝位置从残差分布 `max(q - p, 0)` 逆 CDF 重采样一个 token，之后的位全部填 -1
- 回传 host 后逐流裁剪，最后一格强制替换成主模型自己的采样结果——即使全接受，最后一位也来自 target，保证分布无偏

NPU 实现是纯 torch 算子组合（`CudaSampleOp.cc:487` 的 USING_ASCEND 分支），语义与 CUDA/ROCm kernel 逐条对齐，配套单测 `test_chain_speculative_sampling.py` 用 CPU 参考实现做逐 case 对照。`uniform_samples` 按流支持 random_seed 复现，NPU 上 generator 设备不匹配时跳过（PR #60 加的守卫，SpeculativeSampler.cc:74）。

**⑦ 按接受长度收尾。** `updateDecodePostDraftModelInput`（MtpBatchStreamProcessor.cc:258）把各流接受的 token 拼成下轮输入，hidden states 按 `accept_len` 切片，`input_lengths[i] = accept_len`。verify 写进 cache 的 KV 里，只有接受长度之前的部分继续有效，多余部分随流的长度更新由调度器释放回收。

**⑧ 为下周期播种。** draft 再跑一次前向（CUDA 上有 `sp_prefill_draft_model_` 的图版本），top-1 得到下一周期的 propose token，连同概率、hidden 存回 SPOutputBuffer。这样下个周期第②步拿来就能用。

### 3.4 一个数值例子

K=2，某流上轮末尾的 propose token 是 t0：

```
draft 循环: t0 → 猜出 d1 → 猜出 d2
候选序列:   [t0, d1, d2]          # 3 = K+1 个
target verify: 得到三个位置的 q 分布
采样:        t0 概率上必然接受（它是上轮 target 采样/重采样的结果）
             d1 若 u*p < q 接受；d2 依赖 d1 被接受
结果:        accept_len ∈ {1,2,3}，产出 accept_len 个 token
```

## 4. KV cache 的双视角

第 2.3 节说过主模型和 draft 共用一个块池。运行时的切换发生在每次 forward 前（MtpExecutor 里能反复看到这三行）：

```cpp
const auto& mtp_cache_cfg           = cache_manager_->getMTPModuleCacheConfig(0);
model_input.kv_block_stride_bytes   = mtp_cache_cfg.kv_block_stride_bytes;
model_input.kv_scale_stride_bytes   = mtp_cache_cfg.kv_scale_stride_bytes;
model_input.kv_cache_layer_to_group = draft_kv_cache_layer_to_group;  // 或 target 版本
```

同一个物理块，target 前向时按主模型层布局寻址，draft 前向时按 draft 层布局寻址。块表（block table）只有一张，两层模型各自解释它。PD 分离场景下，DecodeRpcServer.cc:689 起有一段按 mtp_cache_cfg 把全局块表还原成各模型部分的逻辑，用于跨进程迁移。

## 5. Ascend 特有：target-verify 的 ACL Graph（PR #60 Phase B）

**捕获。** `AscendGraphRunner` 按 batch bucket 捕获 verify 图，每流 K+1 个 token（`initCaptureAttentionInputs`，ascend_graph_runner.cc:213，`input_lengths` 直接填 `num_tokens_per_bs`）。捕获用的块表全部播种为 1——页 0 是非法/padding 槽，Qwen3.5 的线性注意力算子在捕获热身时会拒绝非正状态页（237–248 行的注释）。

**分派。** `canRun`（ascend_graph_runner.cc:172）看到 `is_target_verify_` 且输入确实是 verify 时走图，否则回落 eager。prefill 一律不走图。

**回放。** 每次回放前 `prepareInputs`（620 行起）做三件事：

1. 清零捕获的块表缓冲，防止上一次的块 id 污染本批（635 行起）；
2. 把真实的 input_ids、块表、`sequence_lengths_plus_1` 等拷进捕获缓冲；
3. 调 Python 侧 `prepare_cuda_graph` → `update_graph_fia_verify`（ascend_prefill.py:283），用 torch_npu 官方的 `_GraphDispatchMode` 更新每个捕获的 FIA 注意力任务的 `actual_seq_kvlen`。

FIA 任务捕获时带着 ExternalEvent 等待点，回放前必须先走一次 update 流释放事件，否则死锁——ascend_prefill.py 里那段 247–256 行的注释记录了为什么用官方 dispatch 机制而不是手搓的：手搓的回放结果不对，官方的是 bit-exact。配套测试 `test_fia_graph_replay.py` 把"图回放 vs eager 必须逐位一致"固化成回归。

## 6. Ascend 特有：GDN 线性注意力的状态管理（Phase A）

Qwen3.5 主模型的 GDN 层不带传统 KV，状态分页存在 pool 里，每页一行的 ssm 段 + conv 段（draft 头是 MQA，没有这个问题）。verify 前向发生在主模型上，一次吃 K+1 个 token，GDN 的状态管理要跟着支持多 token：

- **页索引**：`decode_state_indices_multi`（paged_state.py:54）在 device 侧算出 verify 的读页和 T 个写页列索引，纯 gather/clamp，可被图捕获；
- **跨页播种**：`seed_state_segment` → triton 的 `migrate_state_rows`，只拷贝跨 `seq_size_per_block` 边界的行，`src == dst` 的行整体跳过；没有 triton 环境时回退语义等价的 `index_copy_`；
- **逐步 launch**：`_fused_recurrent_multi_device`（recurrent.py:102）用 T 次 T==1 的 launch 代替一次多 token launch。原因是 fla-npu 的多 token 单 launch 会写坏中间页的 BF16 状态快照（同输入下与逐步参考 diff 约 1e3），正确性代价约 10% tpot，上游修复后可以退回单 launch；
- **conv 状态**：causal_conv1d.py 同样逐 token 滚动。

## 7. 性能参考

Qwen3.5-35B-A3B，单卡 Ascend 950PR，BF16，verify 图 + draft eager，单流长输出（PR #60 实测）：

| 配置 | TPOT | 单流吞吐 |
|---|---|---|
| MTP K=1（verify T=2） | ~11.8 ms/token | ~85 tokens/s |
| MTP K=2（verify T=3） | ~10.2 ms/token | ~95 tokens/s |
| MTP K=3（verify T=4） | ~10.0 ms/token | ~99 tokens/s |

K=2 相对 K=1 延迟降 13%、吞吐升 12%；K=3 与 K=2 基本持平——draft 保持 eager 和 recurrent 逐步 launch 是两个主要摊薄因素。参考量级：同规模模型在 vllm-ascend 的 MTP acceptance 约 3.14 @ K=3。

## 8. 阅读索引

| 主题 | 文件 |
|---|---|
| 执行器主流程 | `rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc` |
| 输入构造与输出分发 | `rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.cc` |
| 拒绝采样 | `rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.cc` |
| 采样算子（三平台） | `rtp_llm/models_py/bindings/core/CudaSampleOp.cc` |
| KV 配额与布局 | `rtp_llm/cpp/cache/CacheConfigCreator.cc`、`KVCacheManager.cc` |
| 图决策表 | `rtp_llm/cpp/models/PyWrappedModel.h` |
| Ascend 图捕获与回放 | `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc` |
| verify 的 FIA 更新 | `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py` |
| GDN 分页状态 | `rtp_llm/models_py/kernels/ascend/paged_state.py`、`recurrent.py`、`causal_conv1d.py` |
| 单测 | `rtp_llm/cpp/normal_engine/speculative/test/`、`rtp_llm/models_py/bindings/ascend/ops/tests/` |
