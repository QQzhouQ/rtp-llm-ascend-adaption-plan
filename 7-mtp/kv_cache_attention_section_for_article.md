## KV cache适配&Attention算子接入（魏珊）

KV Cache 适配与 Attention 算子接入是 NPU 推理链路中改动量最大、问题最密集的模块，直接决定模型能否在 Ascend NPU 上完成完整的 prefill → decode → 多轮对话推理。rtp-llm 原有的 KV Cache 管理（C++ `BlockPool` / `MemoryLayoutStrategy` / `KVCacheManager`）和 Attention 调度（Python `ascend_prefill.py` / `ascend_decode.py`）围绕 GPU FlashInfer 的合并 KV 布局和 paged attention 接口设计，迁移到 Ascend NPU 后面临以下主要挑战：

- **K/V Cache 拆分要求**：rtp-llm 原生将 K/V 合并在一个 Tensor 中（dim=1 区分：`[blocks, 2, kv_heads, seq, head_dim]`），而昇腾 torch_npu 的算子（`_npu_reshape_and_cache`、`npu_fused_infer_attention_score`）要求 `key_cache` 和 `value_cache` 作为两个独立的 Tensor 传入。早期 FIA 算子只接受连续内存，从合并 buffer slice 出来的非连续 view 无法处理，迫使适配初期在 C++ 层物理拆分出 `k_cache_buffer_` + `v_cache_buffer_` 两块独立内存（separate_kv_cache 机制）。这条第二通路横跨 `BlockPool`、`MemoryLayoutStrategy`、`CacheConfig`、`OpDefs`、`ConfigInit` 等全部 cache 核心模块，改动面大且与 GPU/AMD 路径分叉。
- **存储布局读写不一致**：写入算子（`_npu_reshape_and_cache`）期望 BNDH `[blocks, seq, kv_heads, dim]`，读取算子（`npu_fused_infer_attention_score`）期望 BNSD `[blocks, kv_heads, seq, dim]`。GPU 路径上 FlashInfer 读写统一使用 HND 布局，不存在此问题。
- **算子 format 约束**：PyTorch 标准 `torch::from_blob` 在 NPU 上创建的张量 format 字段未初始化（垃圾值），下游算子检查 ACL format 时崩溃或报 `unknown format type`。
- **FIA 参数语义差异**：`actual_seq_lengths` 以元素个数作为 batch 值、Q 使用 cumsum 而 KV 使用原始长度、`context_lens` 是长度还是位置索引、`sparse_mode` 对 atten_mask 的要求——每个参数都与 GPU FlashInfer 不同，需逐个验证。
- **paged attention 算子缺陷**：`_npu_paged_attention` 在特定 block_table 配置下触发 aicore 异常（CANN 已知问题，vllm-ascend issue #3263），且 `_npu_paged_attention_v2` 在当前 CANN 版本不存在，需 fallback 到 FIA 路径。
- **采样后处理缺失**：NPU 上无专用采样 kernel，`sampleGreedy` 需从 CPU fallback 逐步迁移到 NPU 原生实现。

宏观目标：

打通 Ascend NPU 上 KV Cache 的完整生命周期管理——从 buffer 分配、RoPE 变换、cache 写入、attention 读取到采样后处理——使 Qwen3 等标准 MHA 模型能在 NPU 上完成端到端推理，且推理精度与 GPU 对齐。

整体策略：

- **C++ 层物理拆分 K/V buffer**：将合并 buffer 拆成 `k_cache_buffer_` + `v_cache_buffer_` 两块独立连续内存（separate_kv_cache 机制），满足 torch_npu 算子对独立连续 tensor 的要求。
- **Prefill / Decode 统一走 FIA**：放弃存在 aicore 缺陷的 `_npu_paged_attention`，prefill 和 decode 统一使用 `npu_fused_infer_attention_score`（FIA）+ TND layout + block_table，与 vllm-ascend 默认路径对齐。
- **采样后处理 NPU 化**：引入 `aclnnApplyTopKTopPCustom` 自定义 AscendC 算子做 top-k/top-p 过滤，将采样主路径从 CPU 搬到 NPU。
- **NPU 专属逻辑收敛**：NPU 分支放在 `ascend_impl/` 下少数 Python 文件和 C++ bindings 的 `#elif USING_ASCEND` 中，尽量不侵入 GPU/ROCm 路径。

具体实施内容：

**1. C++ 基础通路打通**

在 KV Cache 适配之前，需先解决三个基础阻塞问题，否则模型无法启动：

- **fusedCopy SIGSEGV**：`FusedCopyOp.cc` 缺少 `USING_ASCEND` 分支，Ascend 编译时走 `#else` 的 `memcpy` 访问 NPU device 地址，直接段错误。修复方式为新增 `#elif USING_ASCEND` 分支，使用 `aclrtMemcpyAsync(ACL_MEMCPY_DEVICE_TO_DEVICE)`。
- **AttentionConfigs 缺 scale 属性**：`AttentionConfigs` C++ 绑定无 `scale` 字段（仅有 `softmax_extra_scale` 和 `q_scaling`），导致所有 FMHA 实现实例化失败。修复方式为 `getattr(attn_configs, "scale", None) or attn_configs.q_scaling * head_dim ** -0.5`。
- **PyAttentionInputs 缺 kv_cache 字段**：`AscendPrefillImpl.support()` 需访问 `attn_inputs.kv_cache.separate_kv_cache`，但 C++ 结构体无此成员。修复方式为 `OpDefs.h` 新增 `std::optional<KVCache> kv_cache` 字段并暴露 pybind 绑定。

**2. KV Cache 存储布局适配（separate_kv_cache 物理拆分）**

rtp-llm 原生用合并 KV 存储（一整块 buffer，每层 reshape 成 `[blocks, 2, kv_heads, seq, head_dim]`，dim=1 区分 K/V），FlashInfer 和 AITer 直接吃这个布局。NPU 的 torch_npu 算子要求 K/V 以独立 tensor 传入，早期 FIA 只接受连续内存。

适配方案是在 C++ 层将合并 buffer 拆成 `k_cache_buffer_` + `v_cache_buffer_` 两块独立内存（40 文件，+455/-87 行），涉及 `BlockPool`（分离 K/V buffer 分配、`at_npu::native::from_blob` 替代 `torch::from_blob`）、`MemoryLayoutStrategy`（新增 `separate_kv_cache_` 分支，按层逐层创建视图）、`CacheConfig`（新增 `separate_kv_cache`、`k_block_stride_bytes`、`v_block_stride_bytes` 字段）、`OpDefs`（`getLayerCache()` 新增 `if (separate_kv_cache)` BNDH reshape 路径）、`ConfigInit`（Pickle 序列化新增 `separate_kv_cache` 项）等全部 cache 核心模块。写入走 `_npu_reshape_and_cache`（BNDH 布局），读取走 FIA + `permute(0,2,1,3).contiguous()` 转 BNSD。

**3. Attention 算子接入**

Prefill 路径（`AscendPrefillImpl`）组合 RoPE → KVCacheWrite → FIA。QKV 经融合线性层产出后 split，key/value 做 `.contiguous()`（`torch.split` 返回 strided view，非连续），写入 KV cache，再用 FIA（TND + block_table）做 attention 计算。Decode 路径（`AscendDecodeImpl`）同样走 FIA，放弃 `_npu_paged_attention`。`context_lens = sequence_lengths + 1`（`sequence_lengths` 是 0-indexed 位置索引，不 +1 会漏读当前 decode 步写入的 token）。

关键参数适配（每个均踩过坑）：

| 参数 | 正确值 | 常见错误 |
|------|--------|---------|
| `actual_seq_lengths`（Q） | cumsum 前缀和 | 前导 0 导致 batch 翻倍 |
| `actual_seq_lengths_kv`（KV） | 原始长度，不 cumsum | prefill/decode 均误用过 cumsum，多并发时 KV 长度超范围 |
| `context_lens`（decode） | `sequence_lengths + 1` | 不 +1 导致 off-by-one，输出乱码但不崩溃 |
| `sparse_mode=3` | 必须传 `atten_mask` | CANN 在 sparse_mode≠0 时要求 atten_mask 不为 null |
| `block_table` | 2D、NPU device 上、kernel 块粒度 | C++ 传来 3D / 在 CPU 上 / 用了物理块粒度 |

**4. C++ 侧 format 修复**

`MemoryLayoutStrategy::processKVTensor()` 使用 `torch::from_blob` 创建每层 tensor，在 NPU 上 format 字段为垃圾值（如 1752432660），导致下游算子报 `unknown format type` 或 SIGSEGV。修复方式为用 `narrow().view(dtype).reshape()` 替代 `from_blob`，继承 `torch::empty` 分配时正确的 `ACL_FORMAT_ND`；V cache 使用 `at_npu::native::from_blob`（torch_npu 实现，正确设置 format）替代。此问题为 NPU 适配 torch 时的通用注意点。

**5. 采样后处理 NPU 化**

`sampleGreedy` 从 CPU 逐 batch 循环改为 NPU 原生实现：temperature 走 `logits.div_()`，top-k/top-p 走 `aclnnApplyTopKTopPCustom` 自定义 AscendC 算子，multinomial 走 PyTorch（torch_npu 映射到 NPU）。bindings 层新增 `ascend/ops/`（NPUBridge、op_api_common.h、ascend_apply_top_k_top_p_custom.h），通过 `EXEC_NPU_CMD` 宏动态加载算子 .so。具体介入细节见算子接入（士泽）章节。

参考文档：

1. 40 文件适配改动总结：[rtp-llm-ascend-adaptation-summary.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/4-kv_cache/rtp-llm-ascend-adaptation-summary.md)
2. 逐文件详细代码修改点：[rtp-llm-ascend-adaptation-detail.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/4-kv_cache/rtp-llm-ascend-adaptation-detail.md)
3. KV Cache 格式问题（from_blob format 垃圾值根因分析）：[0525-from_blob-npu-format-garbage-value-fix.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/4-kv_cache/0525-from_blob-npu-format-garbage-value-fix.md)
4. KV Cache 适配方案设计 v0.3：[ascend_kv-cache-adaptation_v0.3.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/4-kv_cache/ascend_kv-cache-adaptation_v0.3.md)
5. Attention 详细设计方案 v0.2：[ascend_attention_detailed_plan_v0.2.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/4-kv_cache/ascend_attention_detailed_plan_v0.2.md)
6. Attention bugfix 汇总：[ascend_attention_bugfix_summary.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/4-kv_cache/ascend_attention_bugfix_summary.md)
7. Decode context_lens off-by-one 精度问题：[ascend_decode_bug_report.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/4-kv_cache/ascend_decode_bug_report.md)
8. vLLM-Ascend KV Cache 管理机制：[kv cache manage-vLLM-Ascend.md](https://github.com/QQzhouQ/rtp-llm-ascend-adaption-plan/blob/main/4-kv_cache/kv%20cache%20manage-vLLM-Ascend.md)
9. upstream PR（applyTopKTopP + NPU sampleGreedy）：https://github.com/alibaba/rtp-llm/commit/1e93ecc130bda678d627493c6b51f226aa7bfaff

---

后续计划中 KV cache 管理重构（魏珊）部分：

### KV cache管理重构（魏珊）

一阶段虽然跑通了 Qwen3-0.6B 端到端推理，但 separate_kv_cache 机制存在明显的结构性问题。该机制在 C++ cache 核心里开了一条 NPU 专属的第二存储通路——`BlockPool` 分配两块独立 buffer、`MemoryLayoutStrategy` 维护两套 per-layer tensor、`ConfigInit` pickle 序列化多一项、读写路径布局不一致需要 Python permute 桥接。这意味着框架后续每一次 cache 管理演进（block pool 扩容、TP 切分、scale 支持、MLA、hybrid attention），都得在 NPU 分支里再实现一遍，跟 GPU/AMD 的合并 KV 路径长期分叉维护。

二阶段计划移除整套 separate_kv_cache 机制，回归与 GPU 一致的合并 KV 布局。前提条件是新版 FIA 算子 .so 支持非连续（strided）访问——从合并 buffer slice 出来的非连续 view 能直接喂给 FIA，不再需要物理拆分。重构方向包括：C++ 层删除 `k_cache_buffer_` / `v_cache_buffer_` / `separate_kv_cache` 全部字段和分支（预计净删约 320 行），Python 层用 `kv_cache_base[:, 0]` 零拷贝 view 替代 permute+contiguous，写入路径评估 `npu_scatter_pa_kv_cache` 算子替代 `_npu_reshape_and_cache`。目标是让 NPU 适配收敛到 `ascend_impl/` 下少数 Python 文件，C++ cache 核心零 NPU 分支，与 GPU/AMD 三端共享同一套合并 KV + 分页 attention 逻辑。

---

## KV cache管理重构：回归合并 KV + FIA v2 + scatter_pa_kv_cache（魏珊）

一阶段跑通 Qwen3-0.6B 端到端推理后，separate_kv_cache 机制的结构性问题暴露得越来越明显。该机制在 C++ cache 核心里开了一条 NPU 专属的第二存储通路，`BlockPool` 分配两块独立 buffer、`MemoryLayoutStrategy` 维护两套 per-layer tensor、`ConfigInit` pickle 序列化多一项、读写路径布局不一致需要 Python permute 桥接。框架后续每一次 cache 管理演进（block pool 扩容、TP 切分、scale 支持、MLA、hybrid attention）都得在 NPU 分支里再实现一遍，跟 GPU/AMD 的合并 KV 路径长期分叉维护。

重构的转折点是新版 FIA 算子 .so 到位。新版 `fused_infer_attention_score` 原生支持 discontinuous（strided）访问：block_table 分页寻址 + 非连续 view 输入都能正确处理。这意味着 K/V 不用再是两块独立的连续物理内存，从合并 buffer 上 slice 出来的非连续 view 也能直接吃。「算子要两个独立连续 tensor」这个前提不成立了，C++ 层的物理拆分也就没了存在意义。重构过程中面临的主要挑战包括：

- **写入算子布局要求**：替代 `_npu_reshape_and_cache` 的 `npu_scatter_pa_kv_cache` 算子要求 BSND 布局（`dim2 == numHead`），而合并 buffer 的自然布局是 BNSD，需要在 C++ 层调整 reshape 顺序或在 Python 层 permute。
- **cache_mode 参数摸索**：`npu_scatter_pa_kv_cache` 的 tiling 路径由 `cache_mode` 参数控制，支持 `None`、`Norm`、`PA_NZ` 三个值，每个值走完全不同的 kernel 路径（NZ 要求 lastDim=16，SISO 要求 BSND）。op-plugin 原始代码硬编码 `PA_NZ`，需逐个试值定位正确的 `Norm` 路径。
- **算子连续性约束**：`npu_scatter_pa_kv_cache` 的 OpDef 未设置 `AutoContiguous`，拒绝非连续输入。合并 buffer 的 strided view 天然非连续，需要 clone + scatter + copy_back 的适配方案，或推动 CANN 侧开启 AutoContiguous。
- **FIA 版本切换**：FIA v1 的 page attention 只认 BNSD 布局（`dim1 == num_kv_heads`），不接受 BSND。切换到 FIA v2 后，v2 的 page attention 支持 3D cache 格式 `(blocknum, blocksize, H)`，合并 buffer 的 BSND view 可零拷贝 reshape 为 3D 传入。

宏观目标：

移除 separate_kv_cache 整套机制，C++ cache 核心回归与 GPU 一致的合并 KV 单 buffer 路径，NPU 适配收敛到 Python `ascend_impl/` 层，C++ 核心零 NPU 分支。

整体策略：

- **C++ 层零 NPU 分支**：删除 `separate_kv_cache` 全部字段和分支，`BlockPool` 回归单一 `cache_aligned_buffer_`，`MemoryLayoutStrategy` 回归整体 reshape 路径。
- **读取路径切换 FIA v2**：使用 `npu_fused_infer_attention_score_v2` + 3D cache 格式，合并 buffer 的 BSND view 通过 `reshape(blocks, seq, -1)` 零拷贝转为 3D，适配 v2 page attention 对 `(blocknum, blocksize, H)` 格式的支持。
- **写入路径切换 scatter 算子**：使用 `npu_scatter_pa_kv_cache`（`cache_mode="Norm"`，SISO 路径），合并 buffer 的 BSND view 直接满足 `dim2 == numHead` 校验。
- **op-plugin 适配**：`ScatterPaKvCacheNpuOpApi.cpp` 的 `cache_mode` 从 `PA_NZ` 改为 `Norm`，走 ND 格式的 SISO kernel 路径。

具体实施内容：

**1. C++ cache 核心：移除 separate_kv_cache 机制（16 文件，净删 320 行）**

将一阶段铺在 cache 链路里的 NPU 特化分支全部摘掉：
- `BlockPool.h/cc`：删除 `k_cache_buffer_`、`v_cache_buffer_`、`separate_kv_cache_`、`global_layer_k/v_tensors_`，`initializeCacheBuffer()` 回归单一 `cache_aligned_buffer_`。
- `MemoryLayoutStrategy.h/cc`：`init()` 签名删除 `separate_kv_cache` 和 `v_cache_base_ptr` 参数；`convertIndexToAddr()` / `createBasicBlockInfo()` 不再构造 k/v/scale 三元组。
- `CacheConfig.h` / `BlockPoolConfig.h`：删除 `separate_kv_cache`、`k_block_stride_bytes`、`v_block_stride_bytes`。
- `BufferTypes.h`：`CacheLayerLayout` 删除 `layers_to_k/v_buffer_ptrs`。
- `OpDefs.h`：`getLayerCache()` 删除 `if (separate_kv_cache)` 分支，MHA 统一 reshape 为 `[kernel_block_num, 2, kernel_seq_size_per_block, num_kv_heads, head_dim]`（BSND 布局）。
- `ConfigInit.cc`：Pickle tuple 删除 `separate_kv_cache` 项（index 9），后续索引前移一位，tuple size 从 45 降回 44。

**2. 读取路径：FIA v2 + 3D cache 格式**

FIA v1 的 page attention 校验 `dim1 == num_kv_heads`（BNSD），合并 buffer 的 BSND 布局 `dim1 = seq` 通不过。FIA v2 的 page attention 在 `input_layout="TND"` 下支持 3D cache 格式 `(blocknum, blocksize, H)`，其中 H = num_kv_heads × head_dim。

合并 buffer 的 BSND view `[blocks, seq, heads, dim]` 可通过 `reshape(blocks, seq, -1)` 零拷贝转为 3D `[blocks, seq, heads*dim]`，适配 v2 要求。prefill 和 decode 的 `forward()` 统一切换到 `npu_fused_infer_attention_score_v2`，参数名相应调整（`actual_seq_lengths` → `actual_seq_qlen`，`num_heads` → `num_query_heads`，`scale` → `softmax_scale`）。

**3. 写入路径：npu_scatter_pa_kv_cache**

替代 `_npu_reshape_and_cache`，使用 `npu_scatter_pa_kv_cache` 算子。合并 buffer 的 BSND view `kv_cache_base[:, 0]` 直接满足 SISO 路径的 `dim2 == numHead` 校验，无需 permute。算子接口接收扁平 `slot_mapping`，内部自动做 block/offset 分解。

适配过程中的关键问题及解决方式：
- **cache_mode 摸索**：op-plugin 原始 `cache_mode="PA_NZ"` 触发 NZ 路径（要求 lastDim=16），经 `PA_BSND`（无效值）→ `"None"` 字符串（无效）→ `nullptr`（仍走 SISO）逐个试值，最终定位 `"Norm"` 走 SISO 的 ND 路径。
- **连续性约束**：OpDef 未设 `AutoContiguous`，strided view 被拒。当前方案为 clone strided view → contiguous BSND → scatter → copy_back，后续推动 CANN 侧在 OpDef 加 `AutoContiguous()` 从根本上消除拷贝。

**4. key/value 非连续性修复**

`torch.split(qkv, [...], dim=-1)` 返回原 QKV 的 strided view，每个 chunk 行步幅是完整 QKV 宽度而非 chunk 宽度。reshape 为 3D 后仍非连续，导致 scatter 算子报非连续输入错误。修复方式为在 split 之后、reshape 之后对 key/value 调用 `.contiguous()`（与 vLLM-Ascend 的 `reshape_and_cache` 调用方式一致），数据量小（当前 step 的 token 数），拷贝代价可忽略。

**5. C++ 侧 format 修复（沿用一阶段方案）**

`processKVTensor()` 继续使用 `narrow().view(dtype).reshape()` 替代 `torch::from_blob`，继承 `torch::empty` 分配时正确的 `ACL_FORMAT_ND`。一阶段在 separate_kv_cache 路径下用 `at_npu::native::from_blob` 修过此问题，重构后回归合并路径碰到同样的 format 垃圾值，沿用相同修复方案。

**重构收益**

| 维度 | 重构前（separate_kv_cache） | 重构后（合并 KV + FIA v2） |
|------|---------------------------|--------------------------|
| C++ 改动 | cache 核心全套 NPU 分支 | 零 NPU 分支（净删 320 行） |
| 读写布局 | 写 BNDH / 读 BNSD，Python permute 桥接 | 统一 BSND，零 permute |
| 多端一致性 | NPU 特例，与 GPU/AMD 分叉 | 三端共享合并 KV + 分页 attention |
| 适配面 | C++ + Python 双层侵入 | 收敛到 ascend_impl/ 少数 Python 文件 |
| 跟随上游 | NPU 分支持续产生合并冲突 | 零成本继承主线演进 |

参考文档：

1. 重构后案例总结（合并 KV + FIA v2 + scatter 全过程）：`case_study_kv_cache_refactor_after.md`
2. FIA v2 算子文档：`op-plugin/docs/context/torch_npu-npu_fused_infer_attention_score_v2.md`
3. scatter_pa_kv_cache 算子文档：`op-plugin/docs/context/torch_npu-npu_scatter_pa_kv_cache.md`
4. 重构前后改动清单（30 文件，-414/+94 行）：`fia-discontinuous-refactor-changes.md`
