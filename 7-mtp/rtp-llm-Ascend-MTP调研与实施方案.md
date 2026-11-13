# rtp-llm Ascend MTP（多 Token 预测 / 投机解码）调研与实施方案

> 时间：2026-09-24 ~ 2026-09-28
> 基线分支：`/home/d30033799/weishan/rtp-llm` `feature/qwen35-graph-mode`（Qwen3.5-35B-A3B 图模式已跑通）
> 调研对象：`/home/d30033799/vllm-ascend`（vLLM v0.20.x V1 架构）、`/home/d30033799/sglang`（v0.5.6.post2）、`/home/d30033799/xllm`
> 目标模型：Qwen3.5-35B-A3B（30 层 GDN 线性注意力 + 10 层 FIA + MoE），checkpoint 自带完整 `mtp.*` 权重（785 keys）
>
> ## 实施状态（2026-09-28 终验通过）
> **Phase 1（MTP eager）+ Phase 2（MTP 图模式）全部完成**：
> - Phase 1：4 个根因修复 → eager MTP 35ms/token（2.0× vs eager 基线）
> - Phase 2：verify 图入图 + 2 个图根因修复 → **图模式 MTP 11-12ms/token（6× vs eager 基线、1.4× vs 纯图模式）**
> - 接受率 2.63（k=2），正确性终验全通过（greedy 中文恢复真实模型行为、700t 无退化、多请求稳定）
> - 最终代码：15 文件 +711/-75 行纯功能代码（调试代码已全部清理）
> - 逐文件改动明细见 `rtp-llm-Ascend-MTP-Phase1-代码改动详细总结.md`

---

## 一、背景与目标

1. Qwen3.5 图模式（方案C：AscendC 单一消费者 GDN 入图）已端到端验证，decode 28~31ms/token。
2. 下一步开发 MTP（Multi-Token Prediction，即 Qwen3-Next/Qwen3.5 的 nextn 投机解码）：draft 模型一次提出 k 个 token，target 一次 forward 验证 k+1 个 token，链式拒绝采样决定接受长度，期望显著提升单请求吞吐。
3. 方案C 已预留伏笔：GDN 算子对 `1 < token_count ≤ 8` 保留了 host 元数据路径（`kernels/ascend/recurrent.py` 注释明确 "multi-token speculative decode only runs eager；target-verify never runs inside the decode graph"）。

---

## 二、三框架 MTP 实现调研

### 2.1 vllm-ascend（重点借鉴对象）

**架构**：vLLM V1 单进程双模型——draft 与 target 在同一 worker 内加载，无独立 draft worker。

| 层次 | 关键文件 | 要点 |
|---|---|---|
| 方法分发 | `vllm_ascend/spec_decode/__init__.py:33` | `method in ("eagle","eagle3","mtp")` 统一路由 `AscendEagleProposer`（MTP 是 EAGLE 的链式特例） |
| Proposer | `spec_decode/llm_base_proposer.py`（1979 行） | 权重共享（MTP 无自有 embedding/lm_head 时直接复用 target）；draft TP=1 独立进程组；每 draft step 独立的持久 attn 元数据 buffer 组（`slot_mapping_group/seq_lens_group/query_start_loc_group`）供 ACL Graph 固定地址回放 |
| Verify 采样 | `sample/rejection_sampler.py:33` + `ops/triton/reject_sample.py` | **Triton-Ascend 拒绝采样核 + 全向量化 PyTorch fallback 双实现**；NPU 约束：无 `tl_rand64`/fp64，philox 仅 int32，用 `tl.rand`(fp32) 替代 |
| 模型映射 | `patch/platform/patch_speculative_config.py` | `qwen3_5/qwen3_5_moe → qwen3_5_mtp`，MTP 层 = "enorm/hnorm 双 RMSNorm + fc(2H→H) + 1 个 **full_attention** 层 + norm + lm_head"（draft 无 GDN） |
| GDN verify | `ops/gdn.py:365-446` + `patch/worker/patch_gdn_attn.py` | target verify 时 GDN 层用 `npu_causal_conv1d_custom(run_mode=1, num_accepted_tokens_opt=...)` + recurrent 算子的 `spec_state_indices`（每 spec 请求 num_spec+1 个状态检查点槽）+ `num_accepted_tokens` 决定状态回退 |
| ACL Graph | `worker/v2/spec_decode/eagle/aclgraph.py` 等 | draft 的 k 步合并成一张图（merged draft graph）；verify 图 capture size 必须是 `n*(k+1)` 的倍数；`graph_task_group` + `ExternalEvent` 机制 replay 前在独立 stream 更新 attention 参数 |
| 算子约束 | `docs/.../Multi_Token_Prediction.md:100-114` | `npu_fused_infer_attention_score` TND 单次 decode ≤16 token → **k ≤ 15**；FULL 图模式 capture_sizes 必须为 (k+1) 整数倍 |

**数据流**：Scheduler 调度 `1+k` token/请求 → target verify forward（logits 仅在 `logits_indices` 处算）→ `AscendRejectionSampler`（greedy=逐位比对 argmax；random=`P_t/P_d ≥ U` 链式接受 + 残差 `relu(P_t−P_d)` 重采样，k≥3 用 block-verify）→ `prepare_next_token_ids_padded` 取每请求最右有效 token 作下轮 draft 输入 → draft 逐 token 自回归 k 步。

### 2.2 sglang

**架构**：`EAGLEWorker(TpModelWorker)` 封装 target worker，Scheduler 的 `model_worker` 直接替换为 draft worker，由它内部回调 target。支持树（topk>1）与链（topk=1）两种 propose。

| 层次 | 关键文件 | 要点 |
|---|---|---|
| Worker | `srt/speculative/eagle_worker.py`（1093 行） | draft → verify → draft-extend 三段式；V2（overlap）版本 draft 计划与 verify 执行重叠 |
| Verify 输入 | `eagle_info.py:103` | `bs × draft_token_num` 个 token（含 root）打平成一批，`forward_mode=TARGET_VERIFY` |
| **GDN verify（多 token recurrent）** | `layers/attention/hybrid_linear_attn_backend.py:624-733` | cu_seqlens 按 `draft_token_num` 步进；conv 用**树感知 kernel**（token 的卷积输入取其父链 token）；recurrent 用 `disable_state_update=True` + `intermediate_states_buffer`（每步状态落盘）+ `retrieve_parent_token`（子 token 从父 token 的中间状态出发）；verify 后 `update_mamba_state_after_mtp_verify` 把状态回滚到被接受路径的终点步 |
| 中间状态池 | `mem_cache/memory_pool.py:144-229` | `MambaPool.SpeculativeState` 额外分配 `intermediate_ssm [L, size+1, k, HV, K, V]` 与 `intermediate_conv_window`，mamba 池容量按 `(1+k)` 放大 |
| NPU 适配 | `hardware_backend/npu/attention/ascend_backend.py:856-966` | `forward_mtp`：`npu_fused_infer_attention_score` + 预生成 2048 下三角 mask + `sparse_mode=3`，`actual_seq_lengths = arange(k, k+q_len, k)`（**链式语义**）；非 CUDA 平台**强制 greedy 验证**（`TREE_SPEC_KERNEL_AVAILABLE = _is_cuda`） |
| NPU Graph | `hardware_backend/npu/graph_runner/eagle_draft_npu_graph_runner.py` | monkey-patch `torch.cuda.*→torch.npu.*`；replay 用 `graph.update(cpu_update_input=[actual_seq_lengths_kv])` 动态改 seq_len |
| draft KV | `eagle_worker.py:112-151` | 共享 allocator、独立物理池，**同尺寸统一 slot 索引空间**（`max_total_num_tokens` 抬大 `max_reqs*(steps*topk + draft_tokens) + 100`） |

### 2.3 xllm

**架构**：C++ 实现。`MTPWorkerImpl : SpeculativeWorkerImpl` 同时持有 target 与 draft 两个 `LLMWorkerImpl`；引擎级 `SpeculativeEngine` = target LLMEngine（`num_decoding_tokens = S+1`）+ 独立 draft LLMEngine。

| 层次 | 关键文件 | 要点 |
|---|---|---|
| Worker | `core/runtime/mtp_worker_impl.cpp`（2412 行） | propose（S-1 步 draft 递推）→ validate（target 一次 forward `[B,S+1]`）→ accepted 状态写 `EmbeddingCache`（draft 侧 target-hidden 缓存，按 block id 索引） |
| **拒绝采样（Ascend 可直接借鉴的形态）** | `framework/sampling/rejection_sampler.cpp` | **纯 torch 组合算子**：greedy=`(target_ids==draft_ids)` + `build_accepted_mask` 首个不匹配处截断；random=`accepted = uniform < p_t/p_d`，拒绝位从 `normalize((p_t−p_d).clamp_min(0))` 重采样 |
| Qwen3.5 GDN verify | `layers/npu_torch/qwen3_gated_delta_net_base.cpp:686-838` | conv1d `kSpecVerify` 分支带 `num_accepted_tokens`（只推进被接受个数）；recurrent 用 `checkpoint_indices = base + step_offsets` 为每个 spec 位置准备一份 ssm 检查点槽 |
| verify batch 组织 | `mtp_worker_impl.cpp:1862-2010` | hybrid 路径组成 `CHUNKED_PREFILL`（`q_seq_lens=S+1`、`kv_seq_lens=kv+S`）——规避 FlashInfer batch-decode 的 bonus token read-before-write 竞争；full-attn 层 verify 走 expanded-decode 注意力 |
| draft 权重 | `tools/export_mtp.py` + `models/llm/qwen3_5_mtp.h` | Qwen3.5 draft 自带完整 embedding+lm_head；loader 强制 `layer_types=["full_attention"]`（draft 无 GDN） |
| ACL Graph | `core/runtime/acl_graph_persistent_param.cpp` | `update_expanded_spec_decode_attention` 在图更新阶段写 `expanded_kv_seq_lens`（pad 行填 1）与块表；conv1d 图任务 `kSpecVerify` 分支重放时按 accepted 个数推进状态 |

### 2.4 三框架对比与借鉴结论

| 维度 | vllm-ascend | sglang | xllm |
|---|---|---|---|
| draft 结构 | enorm/hnorm+fc+1 full-attn 层（复用 target embed/lm_head） | 同左 | 同左（Qwen3.5 自带 embed/lm_head） |
| draft 有无 GDN | **无**（纯 full-attn） | **无** | **无** |
| target verify 的 GDN | conv `num_accepted_tokens` + recurrent 检查点槽 + 状态回滚 | 中间状态缓存 + 父链寻址 + verify 后回滚 | 同 vllm 形态（检查点槽 + accepted 推进） |
| 拒绝采样核 | Triton-Ascend + **PyTorch fallback** | CUDA 核 + 非 CUDA 强制 greedy + `sgl_kernel_npu.verify_tree_greedy` | **纯 torch 组合算子** |
| verify 的 FIA | TND + `actual_seq_lengths=k*(i+1)` + sparse_mode=3 | `forward_mtp` 同左 + 预生成下三角 mask | expanded-decode / CHUNKED_PREFILL |
| 图模式 | verify/draft 三套图参数池 + graph_task 更新 | 三张图（draft 多步/verify/draft-extend）按 bs 档 | ACL 图容量按 S+1 扩容 + 持久 accepted buffer |
| 树 speculative | 仅链（top-1） | 链+树 | 链 |

**借鉴结论**：
1. **draft 无 GDN、verify 的 GDN 状态管理是 target 侧问题**——三框架一致，rtp-llm 的 host 元数据多 token 路径（已存在）首版够用；
2. **拒绝采样核不需要 AscendC**——xllm 的纯 torch 组合、vllm-ascend 的 PyTorch fallback 都证明了向量化 torch 算子足以支撑 verify 采样（eager 场景每步一次、B×(k+1)×V 规模），后续再按 profiling 决定是否 triton 化；
3. verify 的全注意力层三框架都用 **TND varlen（prefill 形态）+ 因果 mask** 处理——rtp-llm 的 `AscendPrefillAttnOp` 已支持 `prefix_lengths`，可直接复用；
4. 图模式叠加是所有框架都花了大力气的部分（检查点槽、num_accepted_tokens、参数池），rtp-llm 应放第二阶段。

---

## 三、rtp-llm 现状盘点

### 3.1 已有资产（基本不用动）

| 模块 | 位置 | 状态 |
|---|---|---|
| draft-verify 主循环 | `cpp/normal_engine/speculative/MtpExecutor.cc`（954 行，含流程图注释） | 完整：`prefillStep`（target prefill→sample→draft prefill→FastTopK）、`decodeStep`（draft 递推 n-1 步→target verify→target sample→链式拒绝采样→draft prefill 补状态→dispatch） |
| batch 聚合/分发 | `speculative/MtpBatchStreamProcessor.cc` | 完整：`prepareDecodeDraftModelInput`、`updateMultiStepDraftSamplerOutput`（拼 `[B,k,V]` draft probs / `[B,k]` draft ids）、`updateDecodePostDraftModelInput`（按 accept_len 切 hidden 供下轮 draft prefill） |
| 目标/草稿 KV 布局 | `cpp/cache/CacheConfigCreator.cc:70-160` `createSpConfig` | target 混合组 + MTP 模块独立子配置（draft 是纯 full-attn MQA 单层，无 linear stride 约束问题） |
| draft 权重抽取 | `cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc:24-102` | `prepareMTPEngineInitParams` 从同一 checkpoint 的 mtp 层权重构造 draft `EngineInitParams`；`gen_num_per_cycle>1` 且只有 1 层时复制权重 |
| Python draft 模型 | `models/qwen3_next/qwen3_next_mtp.py` + `model_desc/qwen3_next_mtp.py` | `qwen35_moe_mtp` 已注册：`_create_config` 强制 `hybrid_attention_types=[NONE]`（MQA）、`num_layers=1`、`is_mtp=True`；权重映射 `mtp.pre_fc_norm_embedding/pre_fc_norm_hidden/fc/norm + 层内 MQA/FFN`，`model_prefix="model.language_model."` 与 checkpoint 匹配 |
| checkpoint 权重 | `/mnt/docker/weights/Qwen3.5-35B-A3B` | **785 个 `mtp.*` keys 已确认存在**：`mtp.fc/pre_fc_norm_*/norm` + `mtp.layers.0.*`（带 q_norm/k_norm 的 MQA + shared_expert + 256 experts + gate + shared_expert_gate） |
| GDN 多 token 算子 | `kernels/ascend/recurrent.py:240-293`、`causal_conv1d.py` | `1 < token_count ≤ 8` host 元数据路径（`_resolve_state_pages` + 连续写页 + 单次 launch），**eager-only**（D2H 合法） |
| verify 的 GDN 模型层 | `model_desc/qwen3_next.py:454-472,747` | `_get_bs_from_attenion_input` 用 `prefix_lengths` 把 token 数折成 `[B, k+1]`；`is_prefill and not is_target_verify` 才走 prefill_gdn，verify 走 decode_gdn 多 token 路径 |
| verify 的 FIA 层 | `ascend_impl/ascend_prefill.py:138-179` | `AscendPrefillAttnOp.prepare` 已算 `seq_lens_kv = prefix_lengths + input_lengths`；TND + block_table + sparse_mode=3，与 vllm/sglang 的 verify 形态一致（**未在 verify 形状下实测**） |
| Ascend 图 runner 骨架 | `cpp/ascend_graph/ascend_graph_runner.cc:175-181,229` | `canRun` 已有 `is_target_verify_` 分支；capture 时 `max_kv_blocks += sp_steps_` 已预留 |
| 多 token 采样 | `models_py/bindings/core/CudaSampleOp.cc:303-485` | Ascend 版 `sampleGreedy` 是 torch 组合式，天然支持 `[B*(k+1), V]` 多行采样 |
| FastTopK | `SpeculativeSampler.cc:8-22` | softmax+max/topk，纯 torch，平台无关 |

### 3.2 唯一硬断点

**`CudaSampleOp.cc:487-489`（`#elif USING_ASCEND` 分支）**：

```cpp
void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);   // ← Ascend fail-fast
}
```

触发链：`MtpExecutor::decodeStep`（MtpExecutor.cc:652-671，真实流必经）→ `SpeculativeSampler::forward → batchSample`（SpeculativeSampler.cc:87）→ `execChainSpeculativeSampling` → 抛异常。draft 递推、target verify forward、target 多 token 采样在此之前都已跑完——**只缺最后一步拒绝采样核**。

### 3.3 隐患清单（首版需处理/绕过）

| # | 问题 | 位置 | 影响 | 处置 |
|---|---|---|---|---|
| H1 | **图模式与 MTP 互斥未处理**：sp 开启 + `enable_cuda_graph=true` 时，target-verify 图 capture 会走 GDN 多 token host 路径（`.cpu()` D2H）→ capture 崩溃（C1 同款）；且 `canRun` 的 `is_target_verify_` 分支**不看 enable_graph_** | `ascend_graph_runner.cc:175-181`；`PyWrappedModel.h:271-289` | 首版必须 sp + eager | 在 Ascend 分支跳过 target-verify 图的创建/capture，或 canRun 对 verify 一律 false（第一阶段） |
| H2 | **Ascend 图参数未按 CUDA 决策表区分 draft/target**：CUDA 有 `(!model_id \|\| is_prefill_cuda_graph_mode)` 守卫，Ascend 分支只要 sp≠NONE 就把 `num_tokens_per_bs` 设成 `gen+1`——draft decode 图（应为 1 token/bs）会被错误捕获 | `PyWrappedModel.h:271-289` vs 226-240 | 第二阶段修 | 补 model_id 守卫，对齐 CUDA 决策表 |
| H3 | verify 的 `AscendPrefillAttnOp` 未在 `prefix_lengths + input_lengths=k+1` 形状下实测；2048×2048 静态 mask 与长 kv 的组合待验证 | `ascend_prefill.py:114-179` | verify 全注意力层正确性 | 首版 e2e 重点观察；必要时对齐 sglang 的 mask 生成方式 |
| H4 | GDN 多 token 上限 8（`ValueError`），即 **verify 一次 forward 的 k+1 ≤ 8 → `gen_num_per_cycle ≤ 7`** | `kernels/ascend/recurrent.py:226-229` | 启动参数约束 | 文档化；建议首版 k=2~3（vllm Qwen3-Next MTP 也用 k=3） |
| H5 | draft 权重加载细节：checkpoint 里 mtp 层 expert 是未合并的 `down/gate/up_proj`（target 层是合并的 `gate_up_proj`），且 mtp 层带 `q_norm/k_norm`——需确认 loader 的 fuse/映射逻辑覆盖 | `qwen3_next_mtp.py` + weight loader | draft 能否正确加载 | 首版启动时逐项核对权重 shape/缺余 |
| H6 | 非 CUDA 平台跳过引擎 warmup（`NormalEngine.cc` `skip warm up on non-CUDA`），MTP 的 fake-stream 路径未预热 | `NormalEngine.cc:100-102` | 首个真实请求可能触发懒初始化慢/失败 | 首版观察；必要时手动预热 |
| H7 | `check_sp_supported` 拒绝 batched input / `num_return_sequences>1` / beam search / `return_all_probs` 等请求形态 | `server/backend_rpc_server_visitor.py:263-289` | 使用限制（上游既有） | 首版接受 |
| H8 | `SpeculativeSampler.cc:75` 对设了 random_seed 的流逐流 `torch::rand(gen)`——NPU generator 兼容性待验证 | `SpeculativeSampler.cc:68-79` | 可复现请求 | 首版观察 |
| H9 | PD 分离（P/D/融合）下 MTP 的 hidden 传输（`mtp_hidden_states_key` proto）在 Ascend 未验证 | `model_rpc/*RpcServer.cc` | 首版不支持 PD | 明确排除 |

---

## 四、chainSpeculativeSampling 语义分析与 Ascend 实现方案（核心）

### 4.1 调用契约（从 `SpeculativeSampler.cc:33-134` 与 ROCm HIP 核 `cpp/rocm/speculative_sampling/sampling.cu:285-409` 提取）

输入（全部已在 NPU 上）：

| 张量 | 形状 | 含义 |
|---|---|---|
| `draft_probs_d` | `[B, k, V]` float | draft 分布（MTP 用 FastTopK 的 softmax 输出；首列无 draft 位置不占位） |
| `draft_token_ids_d` | `[B, k]` int | 待验证 draft token |
| `uniform_samples_d` | `[B, k+1]` float | 均匀随机数（种子流已覆写） |
| `target_probs_d` | `[B, k+1, V]` float | target verify logits 的 softmax（已过温度/topk/topp） |

输出：

| 张量 | 形状 | 语义 |
|---|---|---|
| `output_token_ids_d` | `[B, k+1]` int32 | `output[b, 0..pos-1]` = 接受的 draft token；`output[b, pos]` = 残差重采样 token；其后 `-1` |
| `output_emitted_token_num_d` | `[B]` int32 | `+= pos + 1`（张量已 zero 初始化，+= 等价于 =） |
| `output_accepted_token_num_d` | `[B]` int32 | **HIP 核从未写过，占位** |

算法（每行独立）：

```
1. pos = k
   for i in 0..k-1:                       # 链式接受
       q = target_probs[b, i, draft_ids[b,i]]
       p = draft_probs[b, i, draft_ids[b,i]]
       u = uniform_samples[b, i]
       if u * p < q: output[b,i] = draft_ids[b,i]     # 接受
       else: pos = i; break                            # 首个拒绝处截断
2. emitted[b] = pos + 1
3. if pos == k: return                                 # 全接受（bonus 位 [b,k] 不写，
                                                       # 由 host 用 target 采样覆写）
4. 残差重采样：r = relu(target_probs[b, pos, :] - draft_probs[b, pos, :])
   u2 = uniform_samples[b, min(pos+1, k)] * sum(r)
   sampled_id = 第一个 cumsum(r) > u2 的下标（逆 CDF）
   output[b, pos] = sampled_id；output[b, pos+1..k] = -1
```

host 侧后处理（`SpeculativeSampler.cc:96-133`，无需改动）会把每流 `accept_tokens[accept_len-1]` **覆写为 target sampler 自己在该位置的采样**——即残差重采样结果实际被丢弃，但保持核语义完整仍应实现（forceSpAccept/perf 路径及未来语义变化依赖它）。

### 4.2 实现选项对比

| 方案 | 做法 | 优 | 劣 |
|---|---|---|---|
| a. **纯 torch 组合算子（推荐首版）** | 见 4.3；xllm `rejection_sampler.cpp` 同形态、vllm-ascend PyTorch fallback 同思路 | 零新依赖、平台无关、可单测、eager verify 场景性能足够 | 每步多次小核 launch（估计 <1ms 级） |
| b. Triton-Ascend 核 | vllm-ascend `ops/triton/reject_sample.py` 同款 | 单核完成、性能最优档 | 需适配 NPU 约束（无 fp64/rand64）；维护成本 |
| c. AscendC 自定义核 | 上游 CUDA 核直译 | 性能最优 | 开发/维护成本最高 |

verify 是 eager、每 decode step 一次、张量规模 `B×(k+1)×V`（V≈151K）——方案 a 的主要开销是 vocab 维 gather/cumsum，只对 `[B, V]`（pos 行）操作即可，开销可控。**首版选 a，第二阶段按 profiling 决定是否升级 b/c。**

### 4.3 方案 a 设计（向量化 torch）

```python
# 伪代码（B=batch, k=propose_step, V=vocab）
draft_ids  = draft_token_ids                       # [B, k]   int
idx        = arange(B)[:, None], arange(k)[None]   # 广播索引
q_sel      = target_probs[:, :k].gather(-1, draft_ids.unsqueeze(-1)).squeeze(-1)  # [B, k]
p_sel      = draft_probs.gather(-1, draft_ids.unsqueeze(-1)).squeeze(-1)          # [B, k]
accept     = uniform_samples[:, :k] * p_sel < q_sel                                # [B, k] bool
accept_pre = accept.cumprod(dim=1)                 # 链式语义：首个 False 后全 False
pos        = accept_pre.sum(dim=1)                 # [B] 接受数（= HIP 核的 pos）

out        = full([B, k+1], -1, int32)
out[:, :k] = where(accept_pre, draft_ids, -1)

# 残差重采样（仅 pos < k 的行；全接受行跳过）
if (rejected := (pos < k)).any():
    rows  = nonzero(rejected)                       # host 侧一次性同步（eager 合法）
    pos_r = pos[rows]
    t_row = target_probs[rows, pos_r]               # [n, V]  高级索引
    d_row = draft_probs[rows, pos_r]                # [n, V]
    resid = (t_row - d_row).clamp_min(0)            # relu(q - p)
    cdf   = resid.cumsum(dim=-1)
    u2    = uniform_samples[rows, (pos_r + 1).clamp_max(k)] * cdf[:, -1:]
    sampled = (cdf <= u2).sum(dim=-1).clamp_max(V - 1)   # 逆 CDF：首个 cdf>u2 的下标
    out[rows, pos_r] = sampled

output_token_ids.copy_(out)
output_emitted_token_num += (pos + 1).to(int32)
```

落位：`CudaSampleOp.cc` 的 `USING_ASCEND` 分支 `chainSpeculativeSampling`，用 ATen API 直写（与该文件 sampleGreedy 的 torch 组合风格一致），不需要新增 pybind。

正确性要点：
1. `cumprod` 精确复刻"首个拒绝即截断"的链式语义（与 HIP 核逐 token break 等价）；
2. 逆 CDF 用 `(cdf <= u2).sum(-1)` 而非 `searchsorted`（NPU 兼容性更稳，语义 = 第一个 `cdf > u2` 的下标，与核内 `atomicMin(sampled_id)` 一致）；
3. 边界 `u2` 极接近 `sum(r)` 时 `clamp_max(V-1)` 对应核内 `sampled_id = d-1` 初始化兜底；
4. 单测对拍参照：CUDA 语义可用 ROCm HIP 核 host 版做 golden（或直接构造手工用例：全接受/首位拒绝/中间拒绝/k=1 快速路径/随机种子流）。

### 4.4 与 vllm-ascend 的语义差异说明

vllm-ascend 的 rejection sampler 是"target-only 变体"（树/链通用，draft 分布在核内近似）；rtp-llm 的 HIP 核是经典链式 `u*p<q` + 残差重采样（draft_probs 实际可用）。**实施时以 rtp-llm 自身 HIP 核语义为准**（host 侧契约已锁定），不引入 vllm 语义。

---

## 五、rtp-llm MTP 实施方案（分阶段 + 文件级改动细则）

### Phase 0：准备（0.5 天）

1. 启动参数组合：`--sp_type=mtp --sp_model_type=qwen35_moe_mtp --gen_num_per_cycle=2`（MTP 权重在同一 checkpoint 内，无需 `--sp_checkpoint_path`）；`--seq_size_per_block=1024 --kernel_seq_size_per_block=512` 沿用；首版 **不带 `--enable_cuda_graph`**。
2. 权重加载核对（H5）：日志确认 785 个 `mtp.*` keys 全部命中、expert gate/up 正确 fuse、无缺余告警。
   - 已确认的 checkpoint 键形态：`mtp.fc.weight`、`mtp.pre_fc_norm_{embedding,hidden}.weight`、`mtp.norm.weight`、`mtp.layers.0.self_attn.{q,k,v,o}_proj + q_norm/k_norm`、`mtp.layers.0.mlp.{shared_expert.{down,gate,up}_proj, gate, shared_expert_gate} + experts.N.{down,gate,up}_proj`（未合并形态，target 层是合并的 `gate_up_proj`，loader 的 fuse 逻辑需覆盖）。

### Phase 1：eager MTP 端到端跑通（核心，1~2 天）

#### 改动一（唯一功能断点）：`CudaSampleOp.cc` Ascend 分支实现 `chainSpeculativeSampling`

**位置**：`rtp_llm/models_py/bindings/core/CudaSampleOp.cc:487-489`，替换 `throw OpException(ERROR_UNIMPLEMENTED)`。

**调用链（已核实）**：`MtpExecutor::decodeStep`（MtpExecutor.cc:670）→ `SpeculativeSampler::batchSample`（SpeculativeSampler.cc:87 `execChainSpeculativeSampling`）→ `ExecOps.cc:487` → 本函数。入参张量已在 NPU、float32/int32、形状契约见 4.1（`draft_probs [B,k,V]`、`draft_token_ids [B,k]` int32、`uniform_samples [B,k+1]`、`target_probs [B,k+1,V]`、三个输出张量已 zero 分配）。

**实现（ATen 直写，与同文件 `sampleGreedy` Ascend 版的 torch 组合风格一致，无新 pybind/无新依赖）**：

```cpp
void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    // Shapes (see SpeculativeSampler::batchSample):
    //   draft_probs     [B, k, V]   float32 (NPU)
    //   draft_token_ids [B, k]      int32   (NPU)
    //   uniform_samples [B, k + 1]  float32 (NPU)
    //   target_probs    [B, k + 1, V] float32 (NPU)
    // Outputs (pre-zeroed by caller):
    //   output_token_ids_d          [B, k + 1] int32
    //   output_emitted_token_num_d  [B]        int32  (kernel semantics: += pos + 1)
    //   output_accepted_token_num_d [B]        int32  (never written, same as HIP kernel)
    auto device = params.draft_probs_d.device();

    const auto  draft_ids_l = params.draft_token_ids_d.to(torch::kLong);  // gather index
    const auto  sel         = draft_ids_l.unsqueeze(-1);                  // [B, k, 1]

    // ---- 1. chain acceptance: accept while u * p < q ----
    auto q_sel = params.target_probs_d.slice(1, 0, -1).gather(-1, sel).squeeze(-1);  // [B, k]
    auto p_sel = params.draft_probs_d.gather(-1, sel).squeeze(-1);                   // [B, k]
    // float cumprod: first rejection truncates the prefix (chain semantics
    // of the HIP kernel's per-token break)
    auto accept        = (params.uniform_samples_d.slice(1, 0, -1) * p_sel) < q_sel;  // [B, k] bool
    auto accept_prefix = accept.to(torch::kFloat).cumprod(1).to(torch::kBool);        // [B, k]
    auto pos           = accept_prefix.sum(1);                                        // [B] int64

    // ---- 2. output tokens: accepted prefix, -1 elsewhere ----
    auto neg_one   = torch::full_like(draft_ids_l, -1);
    auto out_k     = torch::where(accept_prefix, draft_ids_l, neg_one);               // [B, k]
    auto output    = torch::full({out_k.size(0), out_k.size(1) + 1}, -1,
                                 torch::TensorOptions().dtype(torch::kInt32).device(device));
    output.slice(1, 0, -1).copy_(out_k);

    // ---- 3. residual resample at the first rejected position ----
    //   r = relu(target_probs[b, pos] - draft_probs[b, pos]);
    //   inverse-CDF sample with u[b, min(pos+1, k)] * sum(r)
    auto rejected_rows = (pos < out_k.size(1)).nonzero().squeeze(-1);   // host sync, eager-legal
    if (rejected_rows.numel() > 0) {
        auto pos_r    = pos.index({rejected_rows});                     // [n]
        auto row_idx  = rejected_rows.to(torch::kLong).unsqueeze(1);    // [n, 1]
        auto step_idx = pos_r.unsqueeze(1);                             // [n, 1]
        auto t_row    = params.target_probs_d.index({row_idx, step_idx}).squeeze(1);  // [n, V]
        auto d_row    = params.draft_probs_d.index({row_idx, step_idx}).squeeze(1);   // [n, V]
        auto resid    = (t_row - d_row).clamp_min(0);
        auto cdf      = resid.cumsum(-1);                                             // [n, V]
        auto u_col    = params.uniform_samples_d.index(
                            {rejected_rows, (pos_r + 1).clamp_max(out_k.size(1))});   // [n]
        auto u2       = u_col.unsqueeze(1) * cdf.slice(1, -1);                        // [n, 1] 广播
        auto sampled  = (cdf <= u2).sum(-1).clamp_max(t_row.size(1) - 1);  // first cdf > u2
        output.index_put_({rejected_rows, pos_r}, sampled.to(torch::kInt32));
    }

    params.output_token_ids_d.copy_(output);
    params.output_emitted_token_num_d += (pos + 1).to(torch::kInt32);
}
```

（注：`slice(1, 0, -1)` 即取第 0..k-1 步；`cdf.slice(1, -1)` 取每行末元素 `[n,1]` 与 `u_col` 广播相乘；实施时以编译通过为准微调 ATen 惯用式。）

**语义对齐说明**：
- `cumprod` 前缀精确复刻 HIP 核逐 token `break`（首个拒绝后全 False）；
- 逆 CDF 用 `(cdf <= u2).sum(-1)`（= 第一个 `cdf > u2` 的下标）对齐核内 `atomicMin(sampled_id)`；`clamp_max(V-1)` 对应核内 `sampled_id = d-1` 的边界兜底（u 极接近 1 时）；
- `output_emitted += (pos+1)` 对齐核内 `+= pos + 1`（张量已 zero，等价赋值）；
- `output_accepted_token_num_d` 不写——HIP 核同样从未写过（占位通道）；
- 全接受行（pos==k）：`output[b,k]` 保持 -1，与 HIP 核不写该位一致；host 侧（SpeculativeSampler.cc:127-128）会用 target 采样覆写 accept_tokens 最后一位，语义闭环。

**性能预估**：vocab 维操作仅 `gather [B,k,V→B,k]` ×2 + 拒绝行的 `[n,V]` cumsum ×1，B=1~32、k=2~7、V≈151K 时 <1ms，eager verify 场景可忽略。

#### 改动二：图模式互斥（H1，防启动即崩）

**问题**：`enable_cuda_graph=true` + sp 开启时，`PyWrappedModel` 构造函数尾部无条件 `graph_runner_->initCapture()`（PyWrappedModel.h:312）——target-verify 图（`is_target_verify=true, num_tokens_per_bs=k+1`）capture 时走 GDN 多 token host 路径（`.cpu()` D2H）→ **ACL capture 崩溃（C1 同款）**；draft 图则因 H2 用错误 token 数捕获。

**修改**（`PyWrappedModel.h` 的 `#elif USING_ASCEND` 分支，约 251-289 行）：sp 开启时跳过图 runner 创建：

```cpp
#elif USING_ASCEND
        if (is_prefill_cuda_graph_mode) {
            ... // 原样
        } else if (params.sp_config.type != SP_TYPE_NONE) {
            // Phase-1: MTP on Ascend runs eager. The target-verify graph would
            // capture the GDN multi-token host-metadata path (D2H syncs) and
            // crash ACL capture; the draft decode graph token count is not yet
            // guarded by model_id (see H2). Revisit in Phase 2.
            RTP_LLM_LOG_WARNING(
                "Ascend ACL Graph is disabled while speculative decoding is on; "
                "falling back to eager forward for both target-verify and draft.");
        } else {
            ... // 原样创建 graph_runner_
        }
```

尾部 `if (graph_runner_ != nullptr) { ... initCapture(); }` 已有判空（PyWrappedModel.h:307-315），`graph_runner_` 保持 nullptr 即自动全走 eager（`PyWrappedModel::forward` 的 `enable_cuda_graph_ && graph_runner_->canRun(...)` 判空短路）。

**效果**：Phase 1 验证期允许带 `--enable_cuda_graph=true` 启动也不崩（显式降级 + warning），不带该参数更无影响。

#### 改动三：单元测试

1. **新增 `rtp_llm/test/spec/test_chain_speculative_sampling.py`**（纯 torch，CPU 可跑）：
   - 把 4.1 算法写成 python 参考实现 `reference_chain_sampling(draft_probs, draft_ids, uniforms, target_probs)`；
   - 用例矩阵：全接受 / 首位拒绝 / 中间拒绝 / k=1 / 概率边界（`u*p == q` 取 `<` 严格比较）/ 残差分布退化（`relu(q-p)` 全 0 时 sampled 兜底 V-1）/ B>1 混合接受长度；
   - C++ 实现与参考对拍：在 NPU 环境加载构建产物直接调 binding，或在 e2e 前 CPU 冒烟（ATen 逻辑一致性）。
2. **BUILD 注册**：对齐 `rtp_llm/models_py/kernels/ascend/test/` 现有模式。

#### Phase 1 端到端验证（详见第七节）

greedy 与无 MTP 基线逐 token 一致性、接受率统计、tokens/s 对比、1024 跨块长生成、首位全拒退化路径。观察项：H3（verify 的 FIA prefill 形状）、H6（无 warmup 首请求懒初始化）、H8（种子流 NPU generator）。

### Phase 2：MTP 与图模式叠加（3~5 天，视 Phase 1 profiling 决定优先级）

#### 2.1 修 H2：Ascend 图参数对齐 CUDA 决策表（PyWrappedModel.h:271-289）

```cpp
// 对齐 PyWrappedModel.h:226-240 的 CUDA 决策表：draft(model_id!=0) decode 图按 1 token/bs
if (params.sp_config.type != SP_TYPE_NONE && params.sp_config.gen_num_per_cycle > 0
        && !params.model_id) {          // ← 补 model_id 守卫（仅 target-verify）
    graph_params.sp_steps          = params.sp_config.gen_num_per_cycle;
    graph_params.num_tokens_per_bs = params.sp_config.gen_num_per_cycle + 1;
} else {
    graph_params.num_tokens_per_bs = 1;
}
```

#### 2.2 draft decode 图（1 token/bs，入图障碍小）

draft = 纯 full-attn MQA 单层 + MoE，**无 GDN**（无 C1 类 host 元数据问题）。工作项：
1. 放开改动二中对 draft（`model_id != 0`）的禁用，仅保留 target-verify 禁用；
2. draft 的 `kv_cache_layer_to_group`/混合组块表在 `AscendGraphRunner` capture 的适配（draft cache config 只有 1 个 full 组）；
3. replay 校验：draft 递推每步 batch=streams、1 token/流，图档位 bs 对齐。

#### 2.3 draft prefill 图（可选，收益视 accept 长度分布）

verify 后的 draft prefill 输入长度 = accept_len（1~k+1 变长）：按档位 padding 捕获，或维持 eager（首版）。

#### 2.4 target-verify 图（最难，建议单独立项）

GDN 多 token 路径 device 元数据化，两条可选技术路线（第二~三节已调研）：
1. **vllm-ascend 路线**：fla-npu 算子原生支持 `num_accepted_tokens` + 每 spec 请求 `num_spec+1` 个状态检查点槽（`spec_state_indices`），verify 时算子一次处理 k+1 token 并按 accepted 数推进/回滚状态——依赖 fla-npu 算子版本确认（本仓 `flash-linear-attention-npu` 需核对这两个通道是否可用）；
2. **sglang 路线**：`disable_state_update + intermediate_states_buffer`（每步状态落盘）+ verify 后 `update_mamba_state_after_mtp_verify` 回滚到被接受路径终点；
3. FIA 层复用现有 TND prefill 形态（`actual_seq_qlen` 步进 k+1），本身无 capture 障碍；
4. 相关联动：`AscendGraphRunner` capture 的 `prefix_lengths` 通道、`canRun` 对 verify 的 `enable_graph_` 检查补齐、capture_sizes 与 (k+1) 取整约束（vllm 同款）。

### Phase 3：优化

1. 拒绝采样 triton 化（4.2 方案 b，vllm-ascend `reject_sample.py` 同款，注意无 fp64/rand64 约束）。
2. accept 长度自适应（`sp_min/max_token_match` 语义动态调 k）。
3. verify 每步 D2H 审计：GDN host 路径 30 层×2 次 + `spec_prefix_lengths.cpu()`（MtpExecutor.cc:852）等，评估 device 化收益（与 Phase 2.4 合并考虑）。

---

## 六、风险与开放问题

| 风险 | 等级 | 缓解 |
|---|---|---|
| verify 的 FIA prefill 路径（H3）在 `prefix_lengths` 形状下行为未知 | 中 | Phase 1 首个 e2e 必测；备选是给 `AscendPrefillAttnOp` 加 verify 专用 mask/`actual_seq_lengths` 分支（对齐 sglang `forward_mtp`） |
| GDN 多 token host 路径性能：verify 每步 30 层×2 次 D2H + host 页解析 | 中 | eager 合法但慢；首版接受（vllm 图模式前也是这个形态）；Phase 3 device 化 |
| draft（35B 级 MoE 单层）递推 k-1 步的耗时占比 | 中 | profiling 后决定是否给 draft 入图（Phase 2.1） |
| accept_len=0 的退化路径（首位即拒、`accept_tokens` 仅 1 个 bonus token） | 低 | HIP 核天然覆盖（pos=0 → emitted=1，残差采样后 host 覆写）；单测覆盖 |
| `uniform_samples` 的 NPU generator 种子流 | 低 | H8 观察；不行则退化为全局 generator |
| PD 分离 / TP>1 / 多并发（bs>1）叠加 | 低（首版排除） | 单卡单流验证后再放开 |

---

## 七、验证方案

### 7.1 单测（`rtp_llm/test/spec/test_chain_speculative_sampling.py`，CPU 可跑，见 5. Phase 1 改动三）
1. 手工 golden：构造 [B=4, k=2] 用例覆盖全接受 / 首位拒绝 / 中间拒绝 / 概率边界（u·p == q），对照 4.1 算法逐元素断言；
2. 与 ROCm HIP 核语义对拍：同输入跑 host 参考实现（把 HIP 核逻辑译成 python），断言 output/emitted 一致；
3. k=1 快速路径、随机种子流覆写路径。

### 7.2 端到端（Phase 1 验收标准）
```bash
python3 -m rtp_llm.start_server \
    --checkpoint_path=/mnt/docker/weights/Qwen3.5-35B-A3B \
    --model_type=qwen35_moe --start_port=9000 \
    --kv_cache_mem_mb=8192 \
    --seq_size_per_block=1024 --kernel_seq_size_per_block=512 \
    --sp_type=mtp --sp_model_type=qwen35_moe_mtp --gen_num_per_cycle=2
```
1. **正确性**：greedy（temperature=0）输出与无 MTP 基线**逐 token 一致**（链式拒绝采样对 greedy 的不变性：draft=argmax 时 u*p<q 恒成立或拒绝后 host 用 target 采样覆写——首 token 之后语义应等价，若分叉说明实现有 bug）；默认采样输出质量正常；
2. **接受率统计**：日志/metrics 的 `speculative_num_accepted_tokens_total / total_propose_token_num`（k=2 时期望接受长度 ≥1.5，参考 vllm Qwen3-Next k=3 golden [0.85, 0.46, 0.19]）；
3. **性能**：对比无 MTP eager 基线（~70ms/token）的 ms/token 与有效 tokens/s（MTP 的收益指标是 tokens/s 而非 ms/token）；
4. **长生成**：1024 token 跨块（覆盖 verify 的 GDN 多 token 状态页写 + 下轮 draft prefill 的状态回退）；
5. **退化路径**：首位全拒请求（高温随机 prompt）不崩、输出正常。

### 7.3 Phase 2 验收
draft 图 replay 与 eager 逐位一致；verify 图（若做）与 eager 输出一致；三图（若全做）叠加后吞吐进一步提升。

---

## 附：关键文件速查

| 职责 | 文件 |
|---|---|
| 唯一断点（本方案核心落点） | `rtp_llm/models_py/bindings/core/CudaSampleOp.cc:487` |
| 采样参数结构与 host 契约 | `models_py/bindings/core/OpData.h:360`；`cpp/normal_engine/speculative/SpeculativeSampler.cc:33-134` |
| 语义参考核 | `cpp/rocm/speculative_sampling/sampling.cu:285-409`（ChainSpeculativeSampling HIP 核） |
| draft-verify 主循环 | `cpp/normal_engine/speculative/MtpExecutor.cc`（decodeStep:509 / draftModelDecode:833） |
| batch 组织 | `cpp/normal_engine/speculative/MtpBatchStreamProcessor.cc` |
| 图参数决策表 | `cpp/models/PyWrappedModel.h:216-289`（CUDA 表 vs Ascend 分支，H2） |
| 图 runner verify 分支 | `cpp/ascend_graph/ascend_graph_runner.cc:175-181,213,388` |
| verify 的 GDN 模型层 | `models_py/model_desc/qwen3_next.py:335-472,747` |
| verify 的 FIA 层 | `models_py/modules/factory/attention/ascend_impl/ascend_prefill.py:138-179` |
| GDN 多 token 算子 | `models_py/kernels/ascend/recurrent.py:240-293`；`causal_conv1d.py` |
| draft 模型与权重 | `models/qwen3_next/qwen3_next_mtp.py`；`models_py/model_desc/qwen3_next_mtp.py`；`utils/model_weight.py:1142-1152` |
| sp 启动参数 | `server/server_args/speculative_decoding_group_args.py` |
| vllm-ascend 参照 | `vllm_ascend/spec_decode/llm_base_proposer.py`；`sample/rejection_sampler.py`；`ops/gdn.py`；`patch/worker/patch_gdn_attn.py` |
| sglang 参照 | `python/sglang/srt/speculative/eagle_worker.py`；`layers/attention/hybrid_linear_attn_backend.py:624-733,968-1001`；`hardware_backend/npu/attention/ascend_backend.py:856-966` |
| xllm 参照 | `xllm/core/runtime/mtp_worker_impl.cpp`；`core/framework/sampling/rejection_sampler.cpp`；`core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` |

---

## 八、实施记录（2026-09-24）

> **逐文件改动明细（含每个根因的代码位置、改动内容、原因）见配套文档**：`rtp-llm-Ascend-MTP-Phase1-代码改动详细总结.md`（同目录）。本节为摘要。

### 8.1 已落地的代码改动（全部编译通过 + 单测）

| # | 文件 | 改动 | 验证 |
|---|---|---|---|
| 1 | `models_py/bindings/core/CudaSampleOp.cc` | Ascend `chainSpeculativeSampling` 纯 torch 实现（§4.3） | 单测 10/10（CPU 8 + NPU 2 含 V=1519 fuzz） |
| 2 | `cpp/models/PyWrappedModel.h/.cc` | sp 开启时跳过 Ascend 图 runner 创建 + `graph_runner_` 空指针守卫 | e2e |
| 3 | `models_py/bindings/ascend/ops/tests/SamplerPybind.cc` + `test_chain_speculative_sampling.py` | 测试绑定 + 双层单测 | 10/10 |
| 4 | `models_py/bindings/OpDefs.h/.cc`、`cpp/models/PyWrappedModel.cc`、`models_py/model_desc/block_map.py`、`models_py/model_desc/qwen3_next.py`、`cpp/ascend_graph/ascend_graph_runner.cc` | **GDN 状态寻址物理表化**：透出 `kv_cache_block_id_{host,device}_by_group`，每层选组切换，8 处 GDN 调用改传物理表，图 capture/replay 配套 | e2e（跨请求污染消除） |
| 5 | `models_py/kernels/ascend/recurrent.py` | **T>1 改为 T 次 T==1 逐步 launch**（绕过 fla-npu multi-token 快照 bug） | 逐元素 bit-exact 单测 PASS；plan_c 4/4 无损 |
| 6 | `MtpExecutor.cc`、`GenerateStream.cc`、`LinearKVCacheGroup.cc`、`recurrent.py`、`causal_conv1d.py` | MTP 调试日志（`RTP_LLM_MTP_DEBUG=1` / detail log 门控） | 用于根因定位 |

### 8.2 根因一：GDN 状态页号误用 kernel 块 id

- **现象**：长 MTP 生成后**跨请求全局污染**（新请求乱码持续）
- **根因**：GDN 状态寻址（conv/recurrent/prefill 全链）用 `kv_cache_kernel_block_id`（值 = 物理页号×bpk + half）当状态页号——页号落在 2 倍编号空间，多 token 的连续 `+token_idx` 写脏未分配页，MTP 多轮循环翻动空闲池后脏页被新请求复用
- **修复**：物理表化（8.1 #4）。`LinearKVCacheGroup::needBlocksNum` 的"位置块 + reserve_step-1 额外块"预留契约正好匹配"物理表 + 1024 粒度 + token_idx 连续 entry"，印证这是原始设计意图
- **注**：T==1 图模式此前"正常"是因为低页号下偶数页巧合 + 无并发

### 8.3 根因二：fla-npu recurrent 多 token 单 launch 的中间页快照错误（上游 bug）

- **现象**：修复根因一后仍有**请求内累积劣化**（几百 token 后输出退化、accept 跌至全 1）
- **定位手段**：accept_lens 序列（detail log）+ 独立单测 `/tmp/opencode/test_recurrent_snapshot2.py`（multi vs stepwise 逐页对拍）
- **根因**：`npu_recurrent_gated_delta_rule` 多 token 单次 launch 时 per-token BF16 快照**中间页写错**（page[t=0]、page[t=T-1] 正确，中间页是垃圾——kernel `recurrent_gated_delta_rule.h` 的 `CopyOutState` 单缓冲 pending 路径 flush 时序错误，flush 在 Compute 之后用当前 token 状态写上一 token 位置）。输出本身正确（FP32 链无恙），只有快照页坏
- **与症状吻合**：RTP 契约下 accept=1 读 page[0]✓、**accept=2 swap 进 page[1]（垃圾）→ 劣化**、accept=3 读 page[2]✓——accept 序列中 `...2 1 1 1...` 的退化起点完美对齐
- **修复**：rtp-llm 侧绕过（8.1 #5）——T 次 T==1 逐步 launch（线上 bit-exact 验证过的语义），开销 3×5.5us/层
- **上游建议**：fla-npu `recurrent_gated_delta_rule.h` 的 pending flush 应移到 `Compute()` 之前（或修双缓冲时序）

### 8.4 根因四（最终根因）：prefill/verify 的 FIA 读视图与池布局不符（已修复）

- **定位手段**：四层递进——①纯 prefill 探针（max_tokens=1）被改变 → 排除 verify/decode 路径；②权重/mask/rope 哨兵校验和零漂移 → 排除持久张量被写脏；③逐层 hidden 校验和 → GDN 层 0-2 bit 一致、**层 3（首个 FIA 层）起漂移**；④离线写读一致性测试 → **straddle 写 V 与 split 读 V 落点不符（V 落在 split 的 k_read[3]）**
- **根因**：混合 KV 池 FULL 层的真实布局是 **kernel 交错**（每 kernel 块 [K512|V512]，C++ `getLayerCache` 的 5D reshape 即正确解释；decode 路径 `AscendDecodeAttnOp._split_kv` 早已用 straddle 视图并验证）。但 **`AscendPrefillAttnOp` 仍用旧的 `split_kv_kernel_blocks`**（按物理 [K1024|V1024] 布局重建视图 + 物化拷贝）→ bpk=2 下 **V 被映射到 K 的后半 chunk**，读 stale：
  - 新池 stale=零 → FIA 的 V=0 → FIA 层从首个请求起隐性失效（证据：修复前基线输出英文 "Here's a thinking process"，修复后恢复中文"嗯，用户问…"——与图模式验证文档记录逐字一致）
  - 请求跨 512 token 后 K[512:1023] 写入真值 → stale 变真垃圾 → **后续所有 prefill 累积污染**（长请求后探针 T1 漂移、多请求递减劣化、.item() 同步偶发掩盖）
- **修复**（`ascend_prefill.py`）：FIA 读改用 straddle 零拷贝视图（`base[:,0/1].reshape(...)`，与已验证 decode 路径完全一致），离线 K/V 写读一致性验证通过
- **修复后全部验证通过**（2026-09-24 终验）：
  - 基线探针恢复真实模型行为（中文，与图模式验证一致）
  - 同 prompt 700t × 3：1807/1918/1796ch 全部无退化（修复前 2007→990→164→99 递减）
  - 1024 跨块长生成：3132ch 无退化
  - 长请求后探针 **T1' == T1 bit 级一致**（全局污染彻底消除）
  - **接受率 2.57**（k=2 含 bonus，1200 轮平均，理论加速 2.57×）
  - **性能 35ms/token**（vs 无 MTP eager ~70ms → **2.0× 端到端加速**）

### 8.5 启动与验证命令（最终版）

```bash
cd /home/d30033799/weishan/rtp-llm
export ASCEND_CUSTOM_OPP_PATH=$PWD/bazel-bin/3rdparty/aclnn_custom_ops/opp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/bazel-bin/3rdparty/aclnn_custom_ops/lib"   # libcust_opapi.so（top-p 采样必需）
python3 -m rtp_llm.start_server \
    --checkpoint_path=/mnt/docker/weights/Qwen3.5-35B-A3B \
    --model_type=qwen35_moe --start_port=9000 \
    --kv_cache_mem_mb=8192 \
    --seq_size_per_block=1024 --kernel_seq_size_per_block=512 \
    --sp_type=mtp --sp_model_type=qwen35_moe_mtp --gen_num_per_cycle=2 \
    --sp_checkpoint_path=/mnt/docker/weights/Qwen3.5-35B-A3B \
    --enable_detail_log=true    # 可选：engine.log 里看 accept_lens
```

注意：
- 首版 **sp 与图模式互斥**（代码已自动降级 + warning，无需特殊参数）
- `gen_num_per_cycle ≤ 7`（GDN 算子单次 8 token 上限），推荐 2~3
- 单测：`python3 rtp_llm/models_py/bindings/ascend/ops/tests/test_chain_speculative_sampling.py`（需 LD_LIBRARY_PATH 含 torch/torch_npu lib）
- **Phase 2 已完成**：图模式 + MTP 可同时启用（`--enable_cuda_graph=true` + sp 参数），详见 §九

---

## 九、Phase 1 Profiling 与 Phase 2 方案决策

### 9.1 Profiling 方法（复用已有 `06_validation_record.md` 方案）

- **触发方式**：OpenAI 接口 `extra_configs.gen_timeline`（`StepWindowProfiler` → `torch_npu.profiler`）
- **采集开关**：启动带 `--torch_cuda_profiler_dir=/home/d30033799/prof`（不设 `RTP_LLM_MTP_DEBUG`——校验和同步会污染时序）
- **请求**：`max_tokens=64, profile_step=16, profile_trace_name=mtp_phase1_eager`
- MtpExecutor 内置 `RTP_LLM_PROFILE_SCOPE`（`executor.mtp.decode_step` / `executor.mtp.draft_model_decode` / `executor.mtp.decode_step(target_model_verify)` / `executor.mtp.decode_step(rejection_sampling)` 等）——trace 直接给出 MTP 循环分解

### 9.2 MTP 循环分解（15 轮 profiled，bs=1，k=2）

| 阶段 | ms/轮 | 占比 |
|---|---|---|
| **target_model_verify** | **99.2** | **91.0%** |
| rejection_sampling | 3.4 | 3.1% |
| draft_model_decode（1 步） | 2.6 | 2.4% |
| draft_model_forward（draft prefill） | 2.3 | 2.1% |
| dispatch + 其它 | ~1.6 | ~1.4% |

### 9.3 设备算子分解（16 步 411ms）

| 类别 | 算子 | 占比 |
|---|---|---|
| 模型本职计算 | MatMulV3 + GroupedMatmul（投影 + MoE） | 42% |
| **框架/launch 开销** | **Slice + Mul + Cast + Transpose + ViewCopy** | **~29%** |
| GDN 核心 | CausalConv1d + RecurrentGatedDeltaRule | **仅 4.4%** |

### 9.4 Phase 2 方案决策（数据驱动）

**原计划首选"draft decode 图"被数据否定**：
- draft decode 仅占每轮 **2.4%**（2.6ms/轮）——图化收益 <2%，不值得做
- **target verify 占 91%**（99ms/轮），且其中 ~29% 是图模式可消除的框架开销（与无 MTP 图模式 70→30ms 的收益机制相同）

**修正后目标：target-verify 入图**，预期收益分析：

| 项 | 推导 |
|---|---|
| verify 图内耗时估算 | 设备计算 ~20ms（Phase 1 profiling 实测）+ 图 replay 开销 ~2ms ≈ 22ms |
| 每轮总耗时 | verify 22 + draft 4.9 + 采样 3.4 + 引擎 ~1.5 ≈ 32ms |
| **预期 ms/token** | **32 / 2.57 ≈ 12.4ms** |
| **vs 纯图基线 16ms** | **1.3× 提升** |

**技术路线选择**（三子项，各有已验证参照）：

| 子项 | 方案 | 参照 |
|---|---|---|
| GDN T>1 可捕获化 | 逐步 T==1 device launch（device gather 页解析 + device seed + 缓存 asl 常量） | 方案C 的 T==1 device 路径（已图验证） |
| conv T>1 可捕获化 | 同款逐步 device 索引 + `.contiguous()` 切片 | 同上 |
| FIA verify 可捕获化 | positions/slot 全 device 重算 + torch_npu 官方 `_GraphDispatchMode`（graph.update 底层，内置 IFA v2 handler） | sglang 的 `graph.update(cpu_update_input)` 模式 |

### 9.5 FIA 图方案的探索过程（关键决策记录）

Phase 2 中 FIA 图捕获经历了三次方案迭代：

1. **直接捕获 TND + device actual_seq** → 崩溃（op host 侧读张量值 → stream sync → capture 非法）
2. **手卷 ExternalEvent + graph_task_group/update**（镜像 decode 已验证路径）→ 捕获成功但 **replay 计算错误**（A/B diff 30-56）
3. **torch_npu 官方 `_GraphDispatchMode`**（离线验证逐位正确）→ **成功**

**教训**：手卷机制与官方实现在 Event 时序/record 管理/参数更新等细节上存在难以排查的微妙差异。官方 dispatch mode（`auto_dispatch_capture` / `graph.update` 底层）是唯一可靠路径。

### 9.6 图 replay 错误的定位过程（根因五/六）

图 replay 输出乱码后，通过四层递进探针定位（探针代码已在清理阶段全部移除，此处仅记录方法论）：

1. **A/B 探针**（`RTP_LLM_VERIFY_AB`）：replay 后同输入跑 eager，对比 hidden diff——确认图计算错误（diff 30-56）
2. **逐层 hidden 钩子**（capture 时保存 `hidden.sum()` device 引用，replay 后读取 = 图实际输出）→ **L0（首个 GDN 层）从最开始就 100% 分叉**，embedding 正确
3. **图内输入值钩子**（capture 时保存 `seq_lens[:4].sum()` 和 `block_table[0,:4].sum()` 引用）→ **seq_lens 正确但块表全 0**
4. **prepareInputs 日志** → `capture_empty=1`：**MemHold 构造函数漏拷贝物理组表字段**

**根因五**：verify 输入的 `sequence_lengths` 为空（MtpExecutor 清空）→ `sequence_lengths_plus_1_d` 刷新被跳过 → 保持捕获封顶值 → GDN 页号全错。修复：从 `prefix_lengths + 1` 计算。

**根因六**：`AscendGraphMemHold` 逐字段拷贝捕获输入时**漏了 Phase 1 新增的物理 by_group 表** → 图实例的 GDN 块表 buffer 为空 → prepareInputs 刷新被跳过 → 图内 GDN 页号全 0。修复：MemHold 补两行拷贝。

### 9.7 最终性能总结

| 配置 | ms/token | 加速比（vs eager 基线） |
|---|---|---|
| 无 MTP eager | ~70 | 1.0× |
| 无 MTP 图模式（PR #54） | ~16 | 4.4× |
| MTP eager（Phase 1） | ~35 | 2.0× |
| **MTP 图模式（Phase 2 终态）** | **~11-12** | **~6×** |

Phase 2 从 16ms 纯图基线再获 **1.4× 提升**；MTP 在图基线上的收益从 eager 的 2.0× 缩到 1.4×，因为图模式已消除了 MTP 在 eager 下摊薄的主要开销（框架/launch），MTP 剩余收益来自**内存绑定的稠密权重读取摊薄**（3 token GEMM 权重只读一遍）+ MoE 专家计算按 token 数线性增长但权重复用。
