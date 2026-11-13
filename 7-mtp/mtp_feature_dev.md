## MTP特性开发：Qwen3.5 投机解码（魏珊）

MTP（Multi-Token Prediction）投机解码是 NPU 推理链路上决定解码吞吐上限的特性：用 checkpoint 里自带的单层 MTP 头连续猜测 K 个 token，主模型一次前向验证 K+1 个，按概率接受。rtp-llm 的 MTP 框架代码（C++ `MtpExecutor` / `MtpBatchStreamProcessor` / `SpeculativeSampler` / `CacheConfigCreator`）围绕 CUDA/ROCm 的手写 kernel 和 CUDA Graph 设计，迁移到 Ascend NPU 后面临以下主要挑战：

- **拒绝采样算子缺失**：verify 后的接受判定依赖 `chainSpeculativeSampling`（Leviathan et al. 2023 链式拒绝采样），CUDA/ROCm 上是手写 kernel，NPU 无对应实现；配套的 `multiMergeCopy`（多流 hidden states 合并）在 NPU 上走 host memcpy，对设备指针做 host 访问直接崩溃。
- **GDN paged state 多 token verify**：Qwen3.5-35B-A3B 主模型是 hybrid 架构（GDN 线性注意力 + MoE），GDN 层的状态按页分池管理，现有单 token decode 契约（读一页写一页）满足不了 verify 一次读写 T=K+1 个 token；MTP 头本身是单层 MQA，反而是常规结构。
- **fla-npu 多 token 单 launch 缺陷**：上游 fla-npu 的多 token 单次 launch 会写坏中间页的 BF16 状态快照（同输入下与逐步参考 diff 约 1e3），错误输出"大致对"，极难从现象上分辨。
- **verify 前向 launch 开销**：eager 模式下每步算子下发开销吃掉投机收益，需要 ACL Graph；但图捕获要求路径内零 D2H 同步、块表缓冲每次回放前防污染、FIA 任务的 kvlen 需图内刷新——每个约束都与 eager 路径冲突。
- **共享代码触点**：`CacheConfigCreator` / `KVCacheManager` / `SpeculativeSampler` / `PyWrappedModel` 都是非 MTP 路径也要走的代码，NPU 适配不能在这里留下长期分叉。

宏观目标：

让 Qwen3.5-35B-A3B 在 Ascend 上跑通 MTP 的正确性与合理加速（启动配置 `--sp_type=mtp --sp_model_type=qwen35_moe_mtp --gen_num_per_cycle=<K>`），全部改动收在 Ascend 分支（`USING_ASCEND` / `kernels/ascend/` / `ascend_impl/`）或带守卫的 MTP 限定路径内，CUDA/ROCm 编译零改动，非 MTP 的 Ascend 推理不受影响。

整体策略：

- **两期推进，正确性先行**：Phase A 用 eager 模式把语义做对（采样算子、GDN 多 token 状态、引擎接线），问题全部用单测和 A/B 对照钉死；Phase B 再把 target-verify 前向搬进 ACL Graph，图机制问题与语义问题不混在一起。
- **采样算子用 torch 组合而非 AscendC 重写**：`u*p<q` 接受判据用 cumprod 前缀积等价序列 kernel 的 break-on-first-rejection，拒绝位残差 `max(q-p,0)` 逆 CDF 重采样用 cumsum 实现；先写 CPU 参考实现，三平台逐 case 对齐。
- **target-only 入图**：只给主模型 verify 前向建 ACL Graph（每流 K+1 token，按 batch bucket 捕获），draft 前向保持 eager——Phase A profiling 显示 draft 占 MTP 周期 <5%，图化复杂度暂不匹配收益。
- **状态索引全 device 化**：GDN 页索引、跨页播种全部用 gather/clamp 在 device 侧计算，无 host 同步，为 Phase B 图捕获铺路。
- **共享代码改动全部带守卫**：MTP 限定 / 判空 / 设备检查，NTP 回归确认不受影响。

具体实施内容：

**1. Phase A：采样与合并算子（CudaSampleOp.cc / CudaOps.cc USING_ASCEND 分支）**

`chainSpeculativeSampling` NPU 实现为纯 torch 算子组合，语义与 CUDA/ROCm kernel 逐条对齐：`emitted = pos + 1`、拒绝位残差逆 CDF 重采样、-1 padding、greedy 快路径条件与 CUDA 分支一致。`uniform_samples` 按流支持 random_seed 复现，NPU 上 `GenerateStream` 创建的是 CPU generator（无 NPU generator），`torch::rand(CPU gen + NPU options)` 抛设备不匹配异常——加 generator 设备守卫跳过。`multiMergeCopy` 从 host `std::memcpy` 改为逐段 `aclrtMemcpyAsync` D2D（`gatherHiddenStates` 在 decode 并发 ≥8 时传入的是 NPU buffer，host memcpy 属于非法访问）。

**2. Phase A：GDN paged state 多 token verify（paged_state.py / state_migration.py / causal_conv1d.py / recurrent.py）**

- **页索引**：`decode_state_indices_multi` 在 device 侧算出 verify 的读页与 T 个写页列索引（`read = block_map[b, max(len-2,0)//page]`，`write[t] = block_map[b, (len-1)//page + t]`），与单 token decode 路径同一契约。
- **跨页播种**：`seed_state_segment` 委托 triton 页行迁移 precopy，只拷跨 `seq_size_per_block` 边界的行，`src==dst` 行自拷贝 no-op；无 triton 环境回退语义等价的 `index_copy_`。性能上踩过两版：全行 embedding gather + cat 重建约 0.1 ms/层，列切片变体反而慢 40–60%（`aclnnInplaceCopy_Slice` 退化成逐元素拷贝），最终按页行块级拷贝。
- **recurrent 逐步 launch**：`_fused_recurrent_multi_device` 用 T 次 T==1 launch 替代一次多 token launch，规避 fla-npu 写坏中间页快照的缺陷，索引全 device 化、无 host 同步，整环可被 ACL Graph 捕获。正确性代价约 10% tpot，上游修复后可回退单 launch。conv/ssm 状态同样逐 token 滚动。

这里有一个值得记录的教训：改逐步之前曾测得 K=2 只有 9.3 ms（约 107 tps）的"好成绩"，该数值依赖 fla-npu 单 launch 写出的错误中间页状态，不作为有效结论。性能数据必须在正确性之后谈。

**3. Phase B：target-verify ACL Graph（PyWrappedModel.h / ascend_graph_runner.cc / ascend_prefill.py）**

- **决策表对齐 CUDA**：`num_tokens_per_bs = K+1` 按 `model_id` 判定（target verify = K+1，普通 decode = 1），Ascend 上 draft 前向不建图、保持 eager（`PyWrappedModel.h` 258–263 行注释记录该取舍）。
- **按 batch bucket 捕获**：verify 图每流 K+1 token；捕获块表播种为页 1——页 0 是 pad/invalid 槽，Qwen3.5 线性注意力算子在捕获热身时拒绝非正状态页。
- **FIA 图内 kvlen 刷新**：verify 的 sparse_mode=3 掩码接通（kvlen>2048 语义已验证）；FIA 任务捕获时带 ExternalEvent 等待点，回放前经 `prepare_cuda_graph` → `update_graph_fia` 用 torch_npu 官方 `_GraphDispatchMode` 刷新 `actual_seq_kvlen`。第一版手搓的任务更新回放结果不对，官方机制离线验证 bit-exact。
- **回放防污染**：每次回放前清零捕获的块表缓冲——块表稀疏写，上批块 id 残留在尾部列里，本批更短时不被覆盖，表现为"长上下文偶发胡言乱语"。

**4. 引擎与 cache 接线（MtpExecutor.cc / MtpBatchStreamProcessor.cc / CacheConfigCreator.cc / KVCacheManager.cc）**

- Ascend draft 前向（eager）与 verify 输入构造：draft 循环 K-1 轮逐步出猜测 token，候选序列 `[target_token, propose_token, draft_1..draft_{K-1}]` 共 K+1 个，verify 前向一次吃下 B×(K+1)。
- `num_mtp_modules = min(gen_num_per_cycle, draft 层数)` 作为 KV 配额上限，主模型与 draft 层的 KV 打包进同一物理块、共用同一块表，draft 前向前换装 `kv_block_stride_bytes` / `layer_to_group` 呈现 draft 视角；verify 后 KV 截断到接受长度，多余部分由调度器回收。
- 对应更新 3 个 cache 单测。

**5. 验证与性能**

按"每个坑一个回归"组织测试：`test_chain_speculative_sampling`（CPU 参考 vs 真实算子，edge case 矩阵 + 随机 fuzz）、`test_recurrent_stepwise_vs_single`（多 token 逐步入口 vs 逐 token 参考，给上游修复后的回退做仲裁）、`test_fia_graph_replay`（图回放 vs eager bit-exact + 过期元数据阴性对照）、`test_prefill_kv_pairing`（并发 prefill KV 写读配对守卫）、`test_linear_attention`（varlen 必须走 TND 布局）。e2e 覆盖 >2048 needle 检索（single + bs=4）、混合 batch 交替回放（bs 2..8 × 4 轮无漂移）、2048 长输出累积无腐蚀、8 并发 prefill 与长生成交叠；greedy 一致性上 MTP vs NTP 输出长前缀逐位一致。

MTP 性能对照（Qwen3.5-35B-A3B，单卡 Ascend 950PR，BF16，verify 图 + draft eager，单流长输出，实测）：

| 配置 | TPOT | 单流吞吐 |
|------|------|---------|
| MTP K=1（verify T=2） | ~11.8 ms/token | ~85 tokens/s |
| MTP K=2（verify T=3） | ~10.2 ms/token | ~95 tokens/s |
| MTP K=3（verify T=4） | ~10.0 ms/token | ~99 tokens/s |

K=2 相对 K=1 每 token 延迟 -13%、吞吐 +12%；K=3 与 K=2 基本持平，主因是 draft 保持 eager 与 recurrent 逐步 launch（约 10% tpot 代价）两个摊薄因素。参考量级：同规模模型在 vllm-ascend 的 MTP acceptance ≈ 3.14 @ K=3。

已知限制：top_k/top_p 过滤走 AscendC 自定义算子 `aclnnApplyTopKTopPCustom`，运行需 `ASCEND_CUSTOM_OPP_PATH`（opp/）与 `LD_LIBRARY_PATH`（lib/）指向 `3rdparty/aclnn_custom_ops` 构建产物；draft 图化与 recurrent 单 launch 回退是两个后续优化项。

参考文档：

1. PR #60（feat(ascend): Qwen3.5 MTP speculative decoding support Phase A/B）：https://github.com/ningweikang/rtp-llm/pull/60
2. MTP 调用流程梳理（启动装配 / prefill / decode / KV 双视角 / 图捕获全链路）：`mtp_call_flow.md`
3. 链式拒绝采样单测（CPU 参考实现 + NPU 对齐）：`rtp_llm/models_py/bindings/ascend/ops/tests/test_chain_speculative_sampling.py`
4. GDN 多 token 逐步 vs 单 launch 仲裁测试：`test_recurrent_stepwise_vs_single`
5. FIA verify 图回放 bit-exact 回归：`test_fia_graph_replay`
