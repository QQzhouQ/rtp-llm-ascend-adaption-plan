# Qwen3.5-MoE 全特性栈适配：图模式、MTP、Profiling、TP（魏珊）

Qwen3.5-35B-A3B 或许是我们在 Ascend 上遇到过的结构最复杂的模型。它同时拥有 30 层 GDN 线性注意力、10 层 FIA 全注意力、MoE 前馈、混合 KV 缓存和交错 MRoPE——几乎每一个模块都踩在 NPU 推理引擎的技术深水区里。本文记录四个特性在该模型上依次落地的完整过程：图模式、MTP 投机解码、Profiling 和 TP 张量并行。每章遵循同一结构（挑战、目标、策略、实施、结果），与前文的 KV Cache 适配章节一起，构成一条完整的技术叙事线。

四个特性贯穿同一条主线：**让 NPU 特化代码收敛到最小面积**。图模式的引擎代码收在 `cpp/ascend_graph/` 与 Python `ascend_impl/`，MTP 完全复用主线 speculative 框架、只在图捕获侧打补丁，Profiling 只填两处 CUDA-only 的缺口，TP 的全部改动不过 5 个文件、约 70 行。最终的全栈成绩：图模式 2.3 倍加速、叠加 MTP 后 2.7 倍（相对 eager），TPOT 从 69.7ms 降到 25.6ms，多模态 MMMU 60.7%，精度全链路无回归。

---

## 一、图模式适配：ACL Graph 捕获与重放（魏珊）

### 挑战

rtp-llm 的图模式（CudaGraphRunner）从底层就绑定了 CUDA：`at::cuda::CUDAGraph`、`cudaStreamCaptureModeGlobal`、cuBLAS workspace、图内存池，每一层都依赖 CUDA 运行时。Ascend 没有直接等价物，而且还有几个自己的深坑：

- **torch_npu 的捕获模式语义**：`NPUGraph::capture_begin` 默认 Global 模式拒绝捕获期分配，而 FIA 的 workspace 懒初始化恰恰在捕获体内做首次分配——必须切到 `Relaxed` 模式，代价是"漏录"风险需自行兜底。
- **GDN 线性注意力的 host 元数据**：fla-npu 的 `npu_causal_conv1d_update` 与 `npu_recurrent_gated_delta_rule` 历史上只收 host int 数组（`.cpu().tolist()` 算页号、构造 host 索引），这些 D2H 同步在捕获区内非法，是 GDN 入图的根本死结。
- **捕获缓冲的别名不变式**：捕获的图绑定的是捕获期张量的设备地址。任何"防御性"的 out-of-place 操作（如 `clamp`）都会物化一份快照副本，重放时图读的还是旧地址——静默产出错误 token，且短请求回归测不出来。
- **多模型结构差异**：Qwen3.5 的混合 KV（GDN 线性组 + FIA 全注意力组按 tag 分组）、交错 MRoPE（`combo_position_ids [T,3]`）、MoE 回退策略，每一项都要在图捕获路径找到对应位置。

### 目标

让 decode 阶段完整跑进图里：5 个 batch 档位全部捕获、重放语义正确、greedy 输出与 eager 逐字一致，单流解码从约 57ms/tok 降到 16ms/tok 左右。

### 策略

- **新建独立 runner 而非侵入 CUDA runner**：`cpp/ascend_graph/AscendGraphRunner` 镜像 CudaGraphRunner 结构但包裹 `c10_npu::NPUGraph`，通过 `#if USING_ASCEND` 与 BUILD select 接线，CUDA/ROCm 路径零改动。
- **单一消费者原则（方案C）**：GDN 不换算子，把 Python 适配层的 host 计算全部换成 device 张量元数据——eager 与图共用同一段代码，死结从"换算子"变为"换元数据来源"。
- **FIA 图任务模式（借鉴 vLLM-Ascend）**：捕获期 `graph_task_group_begin/end` 包住 FIA v2 调用，重放期 `graph_task_update_begin/end` 动态刷新 context_lens，通过 ExternalEvent 在设备侧排序，避免 host 同步。
- **逐条对拍验证**：每轮修复跑"捕获 5/5 + 短 greedy 逐字 + >2000 tok 长请求（≥5 个 KV 块）+ 并发 pad 桶一致"四件套。

### 实施

**1. 引擎骨架**（`ascend_graph/` 6 个新文件，~1400 行）：`AscendGraphRunner` 负责按档位（大→小，先暴露最大分配）逐桶 `captureOneGraphIns` + `replayAndSyncCheck`（捕获后立即 sanity 重放与 eager 预热对拍，漏录即刻暴露）；`AscendGraphStreamLife` 建专用捕获流；`device_shims` 隔离 NPU stream/event/memcpy API，不动 CUDA shim。

**2. 关键正确性修复**（评审驱动，每条都对应一个真实 bug）：
- pad 行 `sequence_lengths_plus_1_device` 填 1（CUDA 哑行契约：指向保留块 0 位置 0），消除捕获 gather 链的 -1 索引；
- 捕获体异常时 fail-closed 调 `graphCaptureEnd`，杜绝半捕获 NPUGraph 带着未闭合会话被析构；
- `getDecodeBatchSizesToCapture` 过滤超限桶 + 去重 + 空集 CHECK，消除 C++ 桶键与 Python FIA 句柄键分叉导致的 sanity 死锁；
- `prepare_cuda_graph` 以 C++ 传入的 `graph_bs` 为权威行数（废弃 `(seq!=0)` 计数启发式——batch-decode 准入产生的真实 0 长度行会被误判为 padding）；
- 块表 in-place `clamp_` 保持捕获缓冲别名（out-of-place clamp 曾是回归，靠长请求测试抓出）。

**3. 混合 KV 与 MRoPE**：`invokeOnAttnImpl` 把 `signal_graph_events`/`prepare_cuda_graph` 扇出到 `prepare_fmha_impl` 返回的 per-tag 字典；per-tag 块表从 batch 切片默认组重建；`combo_position_ids [T,3]` 按桶切片。MRoPE 交错路径按内核 `get_mrope_position_dim` 契约逐字重写（THW 槽位轮转 + lo/hi 半拆配对），带 CPU 对拍测试锁定。

**4. Embedding/采样链路**：`GraphParams` 增加 Ascend 字段（`tokens_per_block`、hybrid 组拓扑）；`PyWrappedModel` 构造期 `initCapture`；`rpc_engine.py` 的 `cuda:` 硬编码改 `device_name()`。

### 结果

| 指标 | 值 |
|---|---|
| 图捕获 | 5/5 桶（含混合双组路径） |
| 单流 TPOT | ~57 → ~16.4 ms/tok（3.5×） |
| 并发 64（2048/512 负载） | eager 69.7 → graph 30.0 ms，吞吐 106.6→245.8 tok/s（2.3×） |
| 精度 | ceval/mmlu/gsm8k/arc 四基准与 eager 最大偏差 3.5pp |
| 上游 | alibaba/rtp-llm#1620（图引擎 18 文件）+#1666（Qwen3.5 3 文件增量）已合入 |

---

## 二、MTP 投机解码：从 eager 到入图（魏珊）

### 挑战

MTP 的 draft-propose → target-verify → 拒绝采样链路在 NPU 上有三层叠加的困难：

- **语义层**：rtp-llm 的拒绝采样是 CUDA kernel（`rejectionSampling`），NPU 无对应实现；`accepted = pos+1`（含 bonus）的语义需纯 torch 重写并逐 token 对拍。
- **结构层**：Qwen3.5 的 target-verify 是 **prefill 形态的 forward**（一次验证 k+1 个 token），而 aclgraph 是 decode-only——verify 步必须走 eager 或单独入图；GDN 层在 verify 时 token_count ∈ (1,8]，命中 fla-npu 的多 token 路径。
- **图模式层**：draft re-prop 与 verify 的位置编码（MRoPE 三轴）、`is_target_verify` 分支的 slot 映射、图内块表端点（`cb*ntp` 与桶的乘积可能超 `max_bs`）——每一处都在评审中暴露过静默错误。

### 目标

MTP 在图模式 decode 上叠加：draft/verify 全链路正确、接受率 ≥2.4（k=2）、图+MTP 相对 eager ≥2.5×。

### 策略

- **三框架调研定形态**：对比 vllm-ascend（V1 单进程双模型 + EAGLE 式 verify）、SGLang、xLLM 的 MTP 实现，选定"target verify 入图 + draft re-prop 入图"为 Ascend 最优形态（draft 与 target 共享捕获缓冲几何）。
- **eager 先行、图模式跟进**：Phase 1 先修 4 个根因跑通 eager MTP；Phase 2 再把 verify/re-prop 搬进 ACL graph，隔离变量。
- **拒绝采样纯 torch 化**：`chainSpeculativeSampling` 的 NPU 移植逐分支对拍 CUDA 参考（含 bonus 语义），fla-npu recurrent 算子 `num_accepted_tokens` 通道天然支持变长接受。

### 实施

**Phase 1（eager MTP）**：4 个根因修复——`kCUDA` 硬编码清理（27 处）、`is_cuda()` 在 NPU 恒 false 的判断修正（改 `is_privateuseone()`/`device` 比较）、rejectionSampling 纯 torch 重写、target verify 的 prefill 图分支接入。产出 15 文件 +711/-75 行，eager MTP 35ms/tok。

**Phase 2（入图）**：verify 图捕获（`is_target_verify=True` 的独立捕获集合）+ draft re-prop 入图；修复图内 MRoPE 三轴位置（`combo_position_ids` 捕获缓冲）与 `sequence_lengths_plus_1` 的 ntp 倍数端点。后续同事在 fork 侧补齐"MRoPE in MTP verify prefill graphs"与 causal_conv1d 测试对齐（fork PR #1/#2），最终形成 **graph_mtp_k2 = 25.6ms TPOT @并发64**。

### 结果

| 配置 | TPOT avg | 相对 eager | MTP 接受率 |
|---|---|---|---|
| eager | 69.7ms | 1× | — |
| graph | 30.0ms | 2.32× | — |
| **graph+MTP(K=2)** | **25.6ms** | **2.72×** | 51.5% |
| graph+MTP(K=3) | 26.6ms | 2.62× | 56.0% |

K=2 在此接受率下反而优于 K=3（verify 开销增量超过接受收益）。开环 Poisson 负载下中位 TTFT < 0.3s，接受率与负载无关。精度四基准 + MMMU 全量 900 题（60.7%）无回归。

---

## 三、Profiling：NPU 性能剖析链路（魏珊）

### 挑战

rtp-llm 的性能剖析全链路（`StepWindowProfiler` 窗口控制 + gRPC/per-req 触发 + Kineto 落盘）有三处 CUDA-only 断点：`activities_` 硬编码 `{CPU, CUDA}`、`cudaProfilerBegin/End` 在 Ascend 是 no-op、全代码库无 `torch_npu.profiler` 集成。此外长期困扰性能归因的是**宿主机 NPU 驱动间歇停摆**（dmesg `drv_pcie`/SVM 错误，表现为 TPOT 在好/坏值间跳变），没有剖析链路就无法区分"代码慢"与"环境抖动"。

### 目标

复用现有触发链与窗口框架，接入 torch_npu profiler 实现完整 NPU 剖析（kernel 级 + host 算子 + 引擎 scope），并建立 vllm-ascend / xLLM 的基线对照。

### 策略

**取 vllm-ascend 的 `torch_npu.profiler.profile(...)` 调用配方，套进 rtp-llm 已有 C++ 框架**——最小改动、零新依赖。关键安全前提（已核实）：引擎循环线程 == tick() 线程 == Python forward 线程，GIL 介入无死锁。

### 实施

- `TorchProfile` Ascend 分支经 pybind 调用 `torch_npu.profiler`（activities 覆盖 NPU kernel + CPU 算子 + WithScope），`ExecOps.cc` 填掉 cudaProfiler no-op 缺口；
- 窗口控制（start_step/num_steps）、gRPC `StartProfile`、per-req `gen_timeline` 触发链原样复用；
- 剖析驱动的问题定位方法论沉淀：kernel 时间占比 → host 侧 gap → dmesg 驱动错误三步归因，直接支撑了图模式多轮优化（FIA workspace 一次性查询、conv1d 布局消除转置等决策都有 trace 证据）。

### 结果

端到端采集 12 万事件成功；FIA contiguous 布局分析、GDN 优化前后对照（385ms→28ms 的 vllm 形态 vs 方案C 对比）等关键决策均由该链路产出 trace 佐证。

---

## 四、TP 张量并行：HCCL 双卡（魏珊）

### 挑战

rtp-llm 的 TP 框架（Megatron 式权重切分 + 每层 all_reduce + 引擎锁步）在 NPU 上断点分散且隐蔽：Python 侧 `multi_rank_start` 拉起路径按 CUDA 假设分支、C++ 通信回调把张量钉死在 CPU、`collective_torch` 断言只放 NCCL、HCCL communicator 设备与权重加载设备可能不一致。另外 950PR 环境的 NPU 卡间 UB 直连配对（非相邻卡性能骤降）是环境侧暗坑。

### 目标

Qwen3.5-35B TP=2 HCCL 双卡：吞吐近似线性扩展、精度与单卡一致、改动面最小。

### 策略

**不新建 TP 框架，激活已有资产**：rtp-llm 的权重切分（`sp_head`/`sp_0`/`ffn_sp`）、注意力切分（`getAttentionConfigs`）、每层通信（`all_reduce` + embedding `all_gather`）、引擎锁步（`tpSyncModelInputs` + logits 聚合）全部现成——NPU 侧只需打通设备管理与通信后端。

### 实施

5 个文件约 70 行：
1. `start_backend_server.py`：按 `get_device_type()` 选拉起路径，解锁 `multi_rank_start`；
2. `backend_manager.py`：按设备类型选 hccl/nccl；
3. `collective_torch.py`：断言放开 hccl；C++ 回调设备无关化（CPU→当前加速器搬运）+ `_ensure_accel` 助手；
4. `ModelTypes.cc`：pin_memory 防御回退（回调先搬 NPU 再 collective）；
5. `base_model.py`：`_get_device_str()` 改 `torch.npu.current_device()`，权重设备与 HCCL communicator 对齐。

排障两轮：UB 直连配对确认（`npu-smi` 拓扑 + 带宽实测）、HCCL 初始化超时与 `ep_size` 默认推导陷阱（默认 0 自动推成 tp_size 被拒，显式 `--ep_size=1`）。

### 结果

TTFT p99 163s→73s（并发 64）、吞吐 172→321 tok/s（1.87×），精度与单卡一致。启动方式与 GPU 完全一致（`ASCEND_RT_VISIBLE_DEVICES` + 标准 TP 参数）。

---

## 五、特性叠加：全栈终态与经验

### 全栈数据（Qwen3.5-35B-A3B，Ascend 950PR 单卡，2048/512 负载）

| 配置 | TPOT avg | 吞吐 tok/s | 相对 eager |
|---|---|---|---|
| eager | 69.7ms | 106.6 | 1× |
| + aclgraph | 30.0ms | 245.8 | 2.32× |
| + aclgraph + MTP(K2) | 25.6ms | 289.5 | **2.72×** |

多模态（ViT→MRoPE→注入）在图+MTP 下 MMMU 60.7%（900 题全量），纯文本四基准无回归。TP=2 在 ningweikang 谱系独立验证（1.87×）。

### 跨特性经验

1. **别名不变式是图模式的宪法**：捕获图绑定捕获期设备地址，任何 out-of-place "防御"都是毒药。回归必须含多 KV 块长请求——短请求会让块表错误"碰巧正确"。
2. **评审是净增益**：AI 评审六轮共暴露 4 个 P1、3 类 P2 真问题（clamp 快照、桶键分叉、行数启发式、acl_sk.h 告警链），全部转化为防御性修复与测试。
3. **先铺地基再起楼**：MRoPE 数学（两个图模式 PR 里带对拍测试交付）成为后来多模态零成本开门的前提；设备串通用化（`device_name()`）提前消掉了 ViT 构建的地雷。
4. **平台门是支持矩阵的显式表达**：`if is_ascend(): return` 式门禁在能力未就绪时是诚实的 fail-fast，能力就绪后删除即完成适配（多模态三门 -20 行即全通）。
5. **环境的错误要显式归档**：宿主 NPU 驱动间歇停摆（dmesg 特征签名）曾三次伪装成代码回归；建立"好值基线 + dmesg 检查"的习惯后避免了大量误判。

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

**Profiling**
16. vLLM-Ascend 剖析分析：[01_vllm_ascend_profiling_analysis.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/01_vllm_ascend_profiling_analysis.md)
17. rtp-llm 剖析现状：[02_rtp_llm_current_profiling_state.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/02_rtp_llm_current_profiling_state.md)
18. xLLM 剖析分析：[03_xllm_profiling_analysis.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/03_xllm_profiling_analysis.md)
19. 复用性评估：[04_reuse_assessment.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/04_reuse_assessment.md)
20. rtp-llm 剖析方案（最终推荐）：[05_rtp_llm_profiling_scheme.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/05_rtp_llm_profiling_scheme.md)
21. 落地验证记录：[06_validation_record.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/06_validation_record.md)
22. FIA contiguous 布局分析：[07_fia_contiguous_analysis.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/9-profiling/07_fia_contiguous_analysis.md)

**TP 张量并行**
23. vLLM-Ascend dense TP 分析：[01_vllm_ascend_dense_tp_analysis.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/10-tp/01_vllm_ascend_dense_tp_analysis.md)
24. rtp-llm TP 现状与差距：[02_rtp_llm_tp_status_and_gaps.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/10-tp/02_rtp_llm_tp_status_and_gaps.md)
25. TP 实施方案（最终定稿）：[03_rtp_llm_tp_implementation_plan.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/10-tp/03_rtp_llm_tp_implementation_plan.md)
26. TP 排障记录：[排障记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/10-tp/排障记录.md)

**多模态（附加，图模式的多模态开门）**
27. ViT 调研报告（vLLM-Ascend 考证 + rtp-llm 现状）：[01-调研报告.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/11-multimodal/01-调研报告.md)
28. 多模态实施方案（含全部进度记录）：[02-实施方案.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/11-multimodal/02-实施方案.md)

**通用决策与修复记录**
29. MRoPE npu_mrope 实现备份：[MRoPE_npu_mrope_实现方案备份.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/MRoPE_npu_mrope_实现方案备份.md)
30. do_sample 温度除零修复：[Ascend后处理do_sample温度除零精度问题修复记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/Ascend后处理do_sample温度除零精度问题修复记录.md)
31. getTorchCudaDevice 决策记录：[getTorchCudaDevice改回index=-1的决策记录.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/getTorchCudaDevice改回index=-1的决策记录.md)
32. dev_0824 合入全记录：[dev_0824合入问题全记录与代码修改清单.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/8-notes/dev_0824合入问题全记录与代码修改清单.md)

**上游 PR**
33. 图模式引擎（18 文件，已合入）：https://github.com/alibaba/rtp-llm/pull/1620
34. Qwen3.5 全栈：图模式 + MTP + 多模态（3+3 文件）：https://github.com/alibaba/rtp-llm/pull/1666
