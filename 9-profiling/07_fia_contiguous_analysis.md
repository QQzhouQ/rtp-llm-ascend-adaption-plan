# aclnnContiguous_SliceAiCore_Slice 问题分析报告（FIA 算子引入）

> 结论先行：**decode 阶段的 `aclnnContiguous_SliceAiCore_Slice` 由 FIA 算子（`npu_fused_infer_attention_score_v2`）引入**，每次 FIA 调用引入 2 个（k、v 各一），decode 每 step 28 层 × 2 = **56 个**，与生产 trace 观测精确吻合。
>
> 本报告记录了从「初步误判为 MatMulV3 引入」到「单测隔离实锤为 FIA 引入」的完整分析过程、单测方法与证据链。

---

## 1. 背景

在 rtp-llm（Qwen3-0.6B, 910 NPU）的 profiling trace（MindStudio Insight / trace_view.json）中观察到大量 `aclnnContiguous_SliceAiCore_Slice` 设备 kernel：

| 场景 | 数量 |
|------|------|
| prefill step | 84 |
| decode step | 56 / step |

初步分析曾将其归因为 **MatMulV3（线性层）对非连续输入的自动连续化**。用户复核 trace 后提出质疑（怀疑是 FIA 引入），遂设计隔离单测验证。

## 2. 关键机制：三层 contiguous 的区别

分析此问题必须区分三层：

| 层 | 事件名 | 含义 |
|----|--------|------|
| Python 层 | `aten::contiguous` | 显式 `.contiguous()` 调用（仅 prefill 的 `_split_qkv`，56 次）|
| host 派发层 | `aclnnContiguous` | aclnn 框架发现输入非连续后的 host 侧处理 |
| **设备 kernel 层** | **`aclnnContiguous_SliceAiCore_Slice`** | 实际执行的拷贝 kernel（MindStudio 可见，共 196 个）|

设备层的 196 个 = 显式 56（prefill）+ **隐式 140**（全 step 分布，框架自动插入）——隐式部分才是问题主体。

## 3. 生产路径数据流（decode）

```
kv_cache_base [blocks, 2, page, kv_heads, dim]     ← K/V 交错的 KV cache 布局
  └─ kv_base[:, 0].reshape(blocks, page, -1)        ← ascend_decode.py _forward_fia
       ↓ 关键事实：这是 strided VIEW，不是拷贝！
       strides = (262144, 1024, 1)，data_ptr 与 kv_base 同源
  └─ 非连续 k/v 喂给 npu_fused_infer_attention_score_v2
       └─ aclnn 框架内部: 自动插 aclnnContiguous ×2 (k、v 各一)
            └─ 再执行 FusedInferAttentionScore 主 kernel
```

代码位置：`rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`（`_forward_fia`）：

```python
kv_base = kv_cache.kv_cache_base
k_cache = kv_base[:, 0].reshape(kv_base.shape[0], self.page_size, -1)  # strided view
v_cache = kv_base[:, 1].reshape(kv_base.shape[0], self.page_size, -1)  # strided view
...
attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
    query=q, key=k_cache, value=v_cache, ...)
```

## 4. 隔离单测设计（`test_fia_contiguous.py`）

4 个 Case，每 case 独立 profile 20 次迭代，统计设备 kernel 直方图：

| Case | 场景 | 隔离目的 |
|------|------|---------|
| **A** reshape_only | 仅 `kv_base[:, 0/1].reshape(...)`，不调 FIA | 验证视图变换本身是否产生 kernel |
| **B** fia_contig_kv | 预先 `.contiguous()` 的 k/v + FIA | 隔离 FIA 算子本身的行为 |
| **C** production | strided view + FIA（与 `_forward_fia` 完全同款）| 复现生产路径 |
| **D** matmul_control | 非连续输入的 `F.linear` | MatMul 行为对照 |

参数与 Qwen3-0.6B decode 生产一致：`NUM_HEADS=16, KV_HEADS=8, HEAD_DIM=128, PAGE=128, BLOCKS=16, BS=1, CTX=300`。

## 5. 单测结果

| Case | 设备 kernel | Contiguous kernel |
|------|------------|-------------------|
| A 仅 reshape | **0** | **0** |
| B FIA + 连续 k/v | 6/iter（FIA + 单测自身的 InplaceCopy）| **0** |
| **C 生产路径** | 4/iter | **2/iter ⚠️（每次 FIA 调用 k、v 各 1 个）** |
| D MatMul 对照 | 3/iter（MatMulV3 直接用 Slice kernel，无独立 Contiguous）| 0 |

**Case C trace 铁证**——40 个 Contiguous kernel 的同流紧随事件 100% 是 `FusedInferAttentionScore`：

```
aclnnContiguous_SliceAiCore_Slice ×40
  └─ 同流下一个 kernel → FusedInferAttentionScore ×40 (100%)
```

**与生产 trace 数字吻合**：56/decode-step = 2 × 28 层，精确对上。

## 6. 误判复盘：为什么之前错归因给 MatMulV3

之前对生产 trace 做「同流下一个 kernel」归因时，过滤器只匹配 `aclnn*` / `AscendCL@*` 名字。而 FIA 的设备 kernel 名叫 **`FusedInferAttentionScore`**（无 aclnn 前缀），被过滤器排除了——于是 Contiguous 的真实消费者（FIA）被跳过，错误归因到了排在 FIA 之后的 o_proj `MatMulV3`。

> 教训：Ascend 设备 kernel 命名不统一（`aclnnXxx_YyyAiCore_Zzz` / 裸 `FusedInferAttentionScore` / `AscendCL@aclnnXxx`），归因过滤器必须基于排除法（排除 host 派发/元数据类）而非名字匹配法。

## 7. 根因与优化方向

### 根因
KV cache 布局 `[blocks, 2, page, kv_heads, dim]` 将 K/V 交错存放（第 2 维是 2），`kv_base[:, 0/1]` 切出的 view 必然非连续；FIA 的 aclnn 实现要求连续 key/value，框架自动补拷贝。

### 优化方向（按可行性）
| 方案 | 做法 | 效果 |
|------|------|------|
| **A. K/V 分池**（推荐）| C++ 侧把 `kv_cache_base` 拆成两个独立连续 buffer（k_pool、v_pool），`kv_cache.kv_cache_base` 接口改为返回 k/v 两个 view | view 天然连续 → FIA 零拷贝（Case B 已证明可行），每 decode step 省 56 个 kernel；需改 C++ KVCache 内存布局 + `AscendKVCacheWriteOp` 写入目标 |
| B. FIA strided 支持 | 查证 aclnnFIA 是否有支持非连续 key 的 layout/参数 | 零代码改动，但依赖算子能力，需查 CANN 文档 |
| C. 保持现状 | 56 × ~45μs ≈ 2.5ms/step? | 实测 Case C 每个 Contiguous 约 45μs（小拷贝），decode 每 step 额外 ~2.5ms，占比可观，建议优化 |

## 8. 占比随序列长度的变化（补充实验：是否会"快速变小"？）

**问题**：`aclnnContiguous_SliceAiCore_Slice` 的耗时占比是否随序列长度增加而快速变小？

**实验设计**（`test_fia_seq_scaling.py` / `test_fia_pool_scaling.py`）：
- **ctx 扫描**：pool 固定 64 块（K pool 16.8MB），ctx ∈ {128, 256, 512, 1024, 2048}
- **pool 扫描**：ctx 固定 1024，blocks ∈ {16, 64, 256, 1024}

### ctx 扫描结果（pool 固定）

| ctx | Contig(μs/iter) | FIA(μs/iter) | Contig/(C+F) |
|-----|----------------|--------------|--------------|
| 128 | 22.6 | 7.7 | **74.6%** |
| 256 | 22.5 | 7.9 | 74.0% |
| 512 | 22.4 | 8.8 | 71.8% |
| 1024 | 22.4 | 9.9 | 69.4% |
| 2048 | 22.6 | 12.7 | **63.9%** |

> ctx > 2048 无法测：`sparse_mode=3` 强制 mask 必须为 2048×2048（`mask_checker.cpp:480`），生产同款限制。

### pool 扫描结果（ctx=1024 固定）

| blocks | pool K+V (MB) | Contig(μs) | FIA(μs) |
|--------|--------------|-----------|---------|
| 16 | 8.4 | 8.5 | 10.2 |
| 64 | 33.6 | 22.6 | 10.2 |
| 256 | 134.2 | 193.0 | 12.8 |
| 1024 | 536.9 | 753.7 | 13.7 |

### 结论：**不会快速变小，且真正的风险在 pool 侧**

1. **Contig 恒定 vs ctx**：拷贝对象是**整个 pool strided view**（大小=总块数），与实际用了多少上下文无关——ctx 从 128→2048（16×），Contig 稳定在 ~22.5μs。
2. **FIA 随 ctx 缓慢增长**：decode BS=1 是访存受限（只读实际使用的 KV，~0.0027μs/token），128→2048 只增加 5μs。
3. **占比下降很慢**：74.6% → 63.9%（16× ctx 仅降 ~11pp）。按线性外推 FIA(8192)≈29μs → 占比≈44%；FIA(32k)≈93μs → ≈20%。**始终不会消失，绝对浪费恒定。**
4. **拷贝与 pool 大小线性正相关**（8.5→753.7μs / 64× pool），FIA 几乎不变——**生产中 KV pool 越大（`kv_cache_mem_mb` 越大、并发越多），这项浪费越大**。生产配置（2048MB KV / 28 层 ≈ 36.6MB/池）实测每拷贝 ~45μs、每 decode step 浪费 ~2.5ms，与本模型吻合。

> 推论：该问题**不能靠"序列变长"自然稀释**，K/V 分池（方案 A）才是根治手段。

## 9. 附件

| 文件 | 说明 |
|------|------|
| `/home/d30033799/prof/test_fia_contiguous.py` | 隔离单测脚本（同步副本于 `rtp-llm/test_fia_contiguous.py`）|
| `/home/d30033799/prof/fia_contig_test_traces/A_reshape_only/` | Case A trace（证明 reshape 是纯 view，0 kernel）|
| `/home/d30033799/prof/fia_contig_test_traces/B_fia_contig_kv/` | Case B trace（连续 k/v + FIA，0 Contiguous）|
| `/home/d30033799/prof/fia_contig_test_traces/C_production/` | Case C trace（**实锤**：2 Contiguous/FIA 调用）|
| `/home/d30033799/prof/fia_contig_test_traces/D_matmul_control/` | Case D trace（MatMul 对照，无独立 Contiguous）|
| `/home/d30033799/prof/test_fia_seq_scaling.py` + `fia_seq_scaling_traces/` | ctx 扫描实验（128→2048）及 trace |
| `/home/d30033799/prof/test_fia_pool_scaling.py` + `fia_pool_scaling_traces/` | pool 扫描实验（16→1024 块）及 trace |

**复现方法**（NPU 环境）：
```bash
python /home/d30033799/prof/test_fia_contiguous.py
# trace 输出到 /tmp/opencode/fia_contig_test/<case>/<case>_*_ascend_pt/
#   用 MindStudio Insight 打开 ASCEND_PROFILER_OUTPUT/trace_view.json
```

## 10. 相关分析沿革

- 本问题是 profiling 分析的延伸：profiling 工具本身见 `05_rtp_llm_profiling_scheme.md`、调测记录见 `06_validation_record.md`
- eager vs graph 模式的 `aclnnContiguous` 分布差异（84/56/56）最早在 graph 模式对比分析中发现
- prefill 的 84 个中：56 个来自 `_split_qkv` 显式 `.contiguous()`（RoPE 原地写 + scatter 源连续性约束，**必要开销**），另 28 个与 decode 同源（FIA 隐式）
