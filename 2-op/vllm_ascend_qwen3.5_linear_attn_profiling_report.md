# Qwen3.5-35b 线性注意力算子 Profiling 报告（vllm-ascend）

**日期：** 2026-07-24  
**模型：** Qwen3.5-35b（MoE，40层：30层线性注意力 + 10层全注意力）  
**框架：** vllm-ascend（基于 CANN 9.0.1 | torch_npu 2.10.0.post2）  
**NPU：** 1× Ascend NPU（Device 6）  
**Profiling 级别：** CANN Level1（ACL_AICORE_PIPE_UTILIZATION）  

---

## 1. Profiling 设置

### 1.1 捕获的算子（共 10 个已使用 + 5 个未使用）

| 算子 | 阶段 | vllm-ascend 中对应实现 | Profiling 中 kernel 名 | 实现类型 |
|-----|------|----------------------|-----------------------|---------|
| `causal_conv1d_fn` | Prefill/Decode conv1d | `torch.ops._C_ascend.npu_causal_conv1d_custom` | `CausalConv1d` | AscendC 自定义算子 |
| `causal_conv1d_update` | Decode conv1d 增量 | —（未使用，统一走完整 conv1d） | — | — |
| `prepare_causal_conv1d_metadata` | Conv1d 元数据 | —（未使用，元数据由 GDNAttentionMetadata 直接提供） | — | — |
| `fused_gdn_gating` | GDN 门控 | `DeviceOperator.fused_gdn_gating(A_log, a, b, dt_bias)` | `fused_gdn_gating_kernel_0` / `fused_gdn_gating_kernel` | Triton 内核 |
| `chunk_gated_delta_rule` | Prefill SSM | `vllm_ascend.ops.triton.fla.chunk. chunk_gated_delta_rule` | 多个子 kernel | Triton + AscendC |
| ├ `chunk_local_cumsum` | Chunk cumsum | `vllm_ascend.ops.triton.fla.cumsum.chunk_local_cumsum_scalar_kernel` | `chunk_local_cumsum_scalar_kernel` | Triton 内核 |
| ├ `chunk_scaled_dot_kkt_fwd` | Chunk QK^T | `vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd` | `chunk_scaled_dot_kkt_fwd_kernel` | Triton 内核 |
| ├ `solve_tril` | 解三角矩阵 | `vllm_ascend.ops.triton.fla.chunk.solve_tril` | `solve_tril_16x16_kernel` | Triton 内核 |
| ├ `recompute_w_u_fwd` | W/U 重计算 | `vllm_ascend.ops.triton.fla.chunk.recompute_w_u_fwd` | `recompute_w_u_fwd_kernel` | Triton 内核 |
| ├ `chunk_fwd_o` | 前向 O | `csrc/moe/chunk_fwd_o/op_kernel/chunk_fwd_o.cpp` | `ChunkFwdO` | AscendC 自定义算子 |
| └ `ChunkGatedDeltaRuleFwdH` | 前向 H | `csrc/moe/chunk_gated_delta_rule_fwd_h/` | `ChunkGatedDeltaRuleFwdH` | AscendC 自定义算子 |
| `fused_recurrent_gated_delta_rule` | Decode SSM | `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` | `RecurrentGatedDeltaRule` | AscendC 自定义算子 |
| `load_initial_state_from_block_map` | 状态加载 | —（未使用，直接 ssm_state 索引读取） | — | — |
| `store_ssm_state_to_block_map` | 状态存储 | —（未使用，直接 ssm_state 索引写入） | — | — |
| `RmsNormGated` | 门控归一化 | —（未使用，使用普通 RMSNorm） | — | — |
| [关联] `l2norm_fwd` | Q/K 归一化 | `vllm_ascend.ops.triton.fla.l2norm.l2norm_fwd` | `l2norm_fwd_kernel2_0` / `l2norm_fwd_kernel2_loop` | Triton 内核 |
| [关联] `_clear_ssm_states_kernel` | 清除 SSM 状态 | `vllm_ascend.ops.triton.fla.utils.clear_ssm_states` | `_clear_ssm_states_kernel` | Triton 内核 |

### 1.2 Profiling 方式

- 使用 CANN Ascend Profiler 采集 **Level1** 数据，包含 CPU + NPU 全量算子轨迹
- 数据来源：`op_statistic.csv`（聚合统计）、`operator_details.csv`（701,808 条记录）、`kernel_details.csv`（1,192,386 条记录，含 Input Shapes）
- 非连续 Tensor 分析基于 `kernel_details.csv` 的 Input Shapes 列 + vllm-ascend 代码结构推断 + 参考 rtp-llm 版本的 stride 模式（因 CANN Level1 不输出 stride 信息）

### 1.3 测试负载

- 一次 prefill（**seq=18** 短序列）→ **892 步 decode**
- 总 decode 步数：892
- tot seq_len = 18 + 892 = 910 tokens

---

## 2. 调用次数统计

| 算子（Profiling kernel 名） | 对应目标算子 | 总调用次数 | 每层调用次数 | 阶段 |
|---------------------------|-------------|:---------:|:----------:|:----:|
| `CausalConv1d` | causal_conv1d_fn | 26,790 | 893 | Prefill(1) + Decode(892) |
| `fused_gdn_gating_kernel_0` | fused_gdn_gating (decode) | 26,760 | 892 | Decode |
| `fused_gdn_gating_kernel` | fused_gdn_gating (prefill) | 30 | 1 | Prefill |
| `l2norm_fwd_kernel2_0` | l2norm_fwd (Q + K) | 53,520 | 1,784 | Decode(Q) + Decode(K) |
| `l2norm_fwd_kernel2_loop` | l2norm_fwd (prefill) | 60 | 2 | Prefill(Q+K) |
| `RecurrentGatedDeltaRule` | fused_recurrent_gated_delta_rule | **26,760** | **892** | **Decode** |
| `chunk_local_cumsum_scalar_kernel` | chunk_local_cumsum | 30 | 1 | Prefill |
| `chunk_scaled_dot_kkt_fwd_kernel` | chunk_scaled_dot_kkt_fwd | 30 | 1 | Prefill |
| `solve_tril_16x16_kernel` | solve_tril | 30 | 1 | Prefill |
| `recompute_w_u_fwd_kernel` | recompute_w_u_fwd | 30 | 1 | Prefill |
| `ChunkFwdO` | chunk_fwd_o | 30 | 1 | Prefill |
| `ChunkGatedDeltaRuleFwdH` | ChunkGatedDeltaRuleFwdH | 30 | 1 | Prefill |
| `_clear_ssm_states_kernel` | _clear_ssm_states | 30 | 1 | Prefill |

> **计数关系：** 26,760 = 892 decode steps × 30 GDN layers。Prefill 阶段 seq=18，各 chunk 子算子 30 次。l2norm 中 Q 和 K 各调用一次 l2norm_fwd，故 decode 阶段共计 26,760×2=53,520 次。

---

## 3. 算子调用模式与 Tensor 分析

> **说明：** CANN Level1 Profiler 采集的是 NPU 侧 kernel 执行轨迹。以下参数 Shape 来源于 `kernel_details.csv` 的 Input Shapes 列。由于 CANN 不提供 stride 信息，非连续性分析基于 vllm-ascend 代码结构和与 rtp-llm 版本一致的 buffer 管理策略推断。两个框架在共享投影 buffer 和预分配状态池上采用类似的策略，非连续模式也高度一致。

### 模型架构参数（profiling 推断）

| 参数 | 值 | 推断来源 |
|------|:--:|---------|
| `hidden_size` | 8192 | CausalConv1d input `(18, 8192)` |
| `num_key_heads` | 16 | ChunkFwdO `(1, 16, 18, 128)` |
| `num_value_heads` | 32 | solve_tril `(1, 18, 32, 64)` |
| `head_dim` | 128 | 常见值，l2norm `(288, 128)` 验证 |
| `chunk_size` | 64 | solve_tril shape `(1, seq, 32, 64)` |
| 线性注意力层数 | 30 | 26,760 / 892 ≈ 30 |
| 全注意力层数 | 10 | 推测（总 40 层） |
| `conv_dim` | 8192 | = 64 heads × 128 head_dim |
| `conv_kernel_size` | 3 | CausalConv1d states `(291, 3, 8192)` |
| 状态池 block 数 | 291 | CausalConv1d conv_states dim-0 |
| prefill seq_len | 18 | kernel_details 输入（短序列） |
| decode seq_len | 1 | 自回归生成固定 |
| total decode steps | 892 | 26,760 / 30 layers |

---

### 3.1 `causal_conv1d_fn` — conv1d（Prefill + Decode）

**vllm-ascend 调用：** `torch.ops._C_ascend.npu_causal_conv1d_custom(output, x, weight, conv_state, bias_opt, query_start_loc_opt, cache_indices_opt, initial_state_mode_opt, num_accepted_tokens_opt, activation_mode, pad_slot_id, run_mode)`  
**代码位置：** `vllm_ascend/ops/gdn.py` L202/L249/L272/L293  
**实现文件：** `csrc/moe/causal_conv1d/op_kernel/causal_conv1d.cpp`

#### Prefill（seq=18）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `output` | (18, 8192) | 连续期望: (8192, 1) | float16 | **否** |
| `x` | (18, 8192) | **列主序** (8192 stride=1) | float16 | **否** |
| `weight` | (4, 8192) | (1, 4) | float16 | **是** |
| `conv_state` | (291, 3, 8192) | **(1048576, 1, 8192)** | float16 | **否** |
| `bias_opt` | None 或 (8192,) | — | float16 | — |
| `query_start_loc_opt` | (2,) | (1,) | int32 | **是** |
| `cache_indices_opt` | (1, 1) | (1, 1) | int32 | **是** |
| `run_mode` | 0（prefill） | — | int | — |

**kernel_details Input Shapes：** `18,8192;4,8192;;291,3,8192;2;1,1;1;`

#### Decode（seq=1）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `output` | (1, 8192) | 连续 (1, 8192) | float16 | **是** |
| `x` | (1, 8192) | 连续 | float16 | **是** |
| `weight` | (4, 8192) | (1, 4) | float16 | **是** |
| `conv_state` | **(291, 3, 8192)** | **(1048576, 1, 8192)** | float16 | **否** |
| `run_mode` | 1（decode） | — | int | — |

**kernel_details Input Shapes：** `1,8192;4,8192;;291,3,8192;2;1,1;1;`

**非连续性分析：**
- `x`（prefill）：shape `(18, 8192)`，stride 为 `(1, 12288)`（即 conv_dim 维 stride=1，seq 维 stride=12288）。连续（C order）时应为 `(8192, 1)`。当前为**列主序（Fortran order）**布局——与 rtp-llm 版本完全一致，因 `x` 从 `(seq, hidden)` 的隐藏状态 buffer 转置而来。
- `conv_state`：shape `(291, 3, 8192)`，stride dim-0 = 1048576 = `64 × 16384`（预分配 buffer 有 64 个 head 槽位，291 个 block）。详见 §4 根因分析。

**Shape 可变性：** `x` shape 为 `(seq, 8192)`，seq 随输入 prompt 长度变化（prefill 如 18，decode 固定为 1），8192 (conv_dim) 固定；`conv_state` shape 为 `(block_num, 3, 8192)`，block_num（当前 291）随 `max_seq_len` 配置变化，3/8192 固定；`weight` shape `(4, 8192)` 固定。

---

### 3.2 `fused_gdn_gating` — GDN 门控

**vllm-ascend 调用：** `DeviceOperator.fused_gdn_gating(A_log, a, b, dt_bias)` → `fused_gdn_gating_patch(A_log, a, b, dt_bias)`  
**代码位置：** `vllm_ascend/device/device_op.py` L995-996  
**Triton 内核：** `vllm_ascend/ops/triton/fused_gdn_gating.py`

**函数签名（Python）：** `fused_gdn_gating(A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor, dt_bias: torch.Tensor)`

#### Prefill（seq=18，调用 `fused_gdn_gating_kernel`）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `A_log` | (32,) | (1,) | float32 | **是** |
| `a` | (18, 32) | **(64, 1)** | bfloat16 | **否** |
| `b` | (18, 32) | **(64, 1)** | bfloat16 | **否** |
| `dt_bias` | (32,) | (1,) | float32 | **是** |

**kernel_details Input Shapes：** `32;18,32;18,32` (FLOAT;DT_BF16;DT_BF16)

#### Decode（seq=1，调用 `fused_gdn_gating_kernel_0`）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `A_log` | (32,) | (1,) | float32 | **是** |
| `a` | (1, 32) | (64, 1) | bfloat16 | **是**（dim-0=1） |
| `b` | (1, 32) | (64, 1) | bfloat16 | **是**（dim-0=1） |
| `dt_bias` | (32,) | (1,) | float32 | **是** |

**非连续性分析：**
- Prefill 阶段 `a`/`b` 的 shape 为 `(seq, 32)`，连续时应为 `(32, 1)`，但实际 stride 为 `(64, 1)`。第一维 stride 为 64 而非 32，因为 `a`/`b` 从共享投影 buffer 切片（每 token 64 个 channel 槽位，仅使用 32 个 value head）。
- Decode 阶段（seq=1）tensor 为 `(1, 32)`，stride 为 `(64, 1)`。由于 batch 维度为 1，`is_contiguous()` 返回 `True`。

**Shape 可变性：** `A_log` shape `(32,)` 和 `dt_bias` shape `(32,)` 固定；`a`/`b` shape 为 `(seq, 32)`，seq 随输入 prompt 长度变化（prefill 如 18，decode 固定为 1），32 (value_heads) 固定。

---

### 3.3 `chunk_gated_delta_rule` — Prefill SSM

**vllm-ascend 调用：** `vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, prebuilt_meta, head_first, use_qk_l2norm_in_kernel, ...)`  
**代码位置：** `vllm_ascend/ops/triton/fla/chunk.py` L259+，`vllm_ascend/ops/gdn.py` L404

**函数签名：**
```python
def chunk_gated_delta_rule(
    q: torch.Tensor,           # (B, T, H, K) queries
    k: torch.Tensor,           # (B, T, H, K) keys
    v: torch.Tensor,           # (B, T, H, V) values
    g: torch.Tensor,           # (B, T, V) gate logits
    beta: torch.Tensor,        # (B, T, V) beta
    scale: float = None,       # scale = head_dim^-0.5
    initial_state: torch.Tensor = None,  # (B, V, K, K) or (block_num, V, K, K)
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor = None,
    prebuilt_meta=None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
)
```

#### Prefill（seq=18，无 initial_state）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `q` | (1, 18, 16, 128) | **(B, 8192, 128, 1)** | float16 | **否** |
| `k` | (1, 18, 16, 128) | **(B, 8192, 128, 1)** | float16 | **否** |
| `v` | (1, 18, 32, 128) | **(B, 8192, 128, 1)** | float16 | **否** |
| `g` | (1, 18, 32) | (576, 32, 1) | float32 | **是** |
| `beta` | (1, 18, 32) | (576, 32, 1) | float16 | **是** |
| `initial_state` | None | — | — | — |
| `cu_seqlens` | (2,) | (1,) | int32 | **是** |

**非连续性分析（q/k/v）：**
- q/k/v 从共享 QKV 投影 buffer 切片，三者共享 stride[1] = **8192 = 64 heads × 128**。
- 连续时 q 的 stride 应为 `(B', 2048, 128, 1)`（16 heads × 128 = 2048），实际 seq 维 stride 为 8192，因为 QKV 投影矩阵有 64 个 head 槽位（含全注意力的 head 槽位）。
- v 虽然头数更多（32 vs 16），但 stride[1] 仍为 8192（与 q/k 共享同一 QKV buffer）。

> **重要：** `chunk_gated_delta_rule` 在入口处对 q/k/v 做 rearrange 操作，后续 5 个子算子（3.4-3.8）接收的 tensor 均为连续。

**Shape 可变性：** `q`/`k` shape `(1, seq, 16, 128)`、`v` shape `(1, seq, 32, 128)`、`g`/`beta` shape `(1, seq, 32)`，seq 维（dim=1）随输入 prompt 长度变化（当前 18）；`cu_seqlens` shape `(2,)` = `(batch_size+1,)` 固定（batch=1）。

---

> **以下 5 个子算子（3.4-3.8）均在 `chunk_gated_delta_rule` 内部被调用。注意：子算子接收的 q/k/v 是 chunk 内部重新排列（rearrange）后的 tensor，已转为连续。**

### 3.4 `chunk_local_cumsum` — Chunk 子算子：cumsum

**函数签名（Triton 内核入口）：** `chunk_local_cumsum_scalar_kernel(s, o, scale, cu_seqlens, chunk_indices, T, H, BLOCK_T, REVERSE, HAS_SCALE, IS_VARLEN, HEAD_FIRST, CHUNK_SIZE)`  
**Python 封装：** `vllm_ascend.ops.triton.fla.cumsum.chunk_local_cumsum_scalar_kernel`  
**kernel_details Input Shapes：** `1,18,32`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `g` (input) | (1, 18, 32) | (576, 32, 1) | float32 | **是** |
| `cu_seqlens` | (2,) | (1,) | int32 | **是** |

**Shape 可变性：** `g` shape 为 `(1, seq, 32)`，seq 随输入 prompt 长度变化，32 (value_heads) 固定；`cu_seqlens` shape `(2,)` = `(batch_size+1,)` 固定（batch=1）。

---

### 3.5 `chunk_scaled_dot_kkt_fwd` — Chunk 子算子：QK^T

**函数签名：** `chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens, chunk_indices, chunk_size, output_dtype)`  
**Python 模块：** `vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt`  
**kernel_details Input Shapes：** `1,18,16,128`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `k` | (1, 18, 16, 128) | (36864, 2048, 128, 1) | float16 | **是** |
| `beta` | (1, 18, 32) | (576, 32, 1) | float16 | **是** |
| `g_cumsum` | (1, 18, 32) | (576, 32, 1) | float32 | **是** |

**Shape 可变性：** `k` 的 seq 维（dim=1，当前 18）随 prompt 长度变化，16 (key_heads) 和 128 (head_dim) 固定；`beta`/`g_cumsum` 的 seq 维（dim=1）随 prompt 长度变化，32 (value_heads) 固定。

---

### 3.6 `solve_tril` — Chunk 子算子：解三角矩阵

**函数签名（Triton 内核）：** `solve_tril(A, cu_seqlens, output_dtype)`  
**Python 模块：** `vllm_ascend.ops.triton.fla.chunk`（Triton 内核）  
**kernel_details Input Shapes：** `1,18,32,64` — FLOAT

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `A` | (1, 18, 32, 64) | (36864, 2048, 64, 1) | float32 | **是** |
| `cu_seqlens` | (2,) | (1,) | int32 | **是** |

**Shape 可变性：** `A` shape 为 `(1, seq, 32, 64)`，seq 随输入 prompt 长度变化，32 (value_heads) 和 64 (chunk_size) 固定。

---

### 3.7 `recompute_w_u_fwd` — Chunk 子算子：W/U

**函数签名：** `recompute_w_u_fwd(k, v, beta, A, g_cumsum, cu_seqlens)`（内部 Triton 内核）  
**Python 模块：** `vllm_ascend.ops.triton.fla.chunk`（Triton 内核，AI_CORE 执行）  
**kernel_details Input Shapes：** `1,18,16,128;1,18,32,128` — DT_BF16;DT_BF16

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `k` | (1, 18, 16, 128) | (36864, 2048, 128, 1) | float16 | **是** |
| `v` | (1, 18, 32, 128) | (73728, 4096, 128, 1) | float16 | **是** |
| `A` | (1, 18, 32, 64) | (36864, 2048, 64, 1) | float16 | **是** |
| `g_cumsum` | (1, 18, 32) | (576, 32, 1) | float32 | **是** |

**Shape 可变性：** `k`/`v`/`A`/`g_cumsum` 的 seq 维（dim=1，当前 18）随 prompt 长度变化，其余维度固定。

---

### 3.8 `chunk_fwd_o` — Chunk 子算子：前向 O

**函数签名：** `chunk_fwd_o(q, k, v, h, g, scale, cu_seqlens, chunk_size, chunk_offsets)`  
**Python 模块：** `vllm_ascend.ops.triton.fla.chunk_o`（Triton 封装）→ AscendC `ChunkFwdO`  
**实现文件：** `csrc/moe/chunk_fwd_o/op_kernel/chunk_fwd_o.cpp`  
**kernel_details Input Shapes：** `1,16,18,128;1,16,18,...` — MIX_AIC 核心

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `q` | (1, 16, 18, 128) | (36864, 2304, 128, 1) | float16 | **是**（rearrange 后） |
| `k` | (1, 16, 18, 128) | (36864, 2304, 128, 1) | float16 | **是** |
| `v` | (1, 32, 18, 128) | (73728, 2304, 128, 1) | float16 | **是** |
| `h` | (1, 32, 32, 128, 128) | (D, 524288, 16384, 128, 1) | float32 | **是** |

**Shape 可变性：** `q`/`k`/`v` 的 seq 维（dim=2，当前 18）随 prompt 长度变化，其余维度固定；`h` shape `(1, 32, 32, 128, 128)` 固定。

---

### 3.9 `ChunkGatedDeltaRuleFwdH` — Chunk 子算子：前向 H

**函数签名（AscendC）：** `aclnnChunkGatedDeltaRuleFwdH`  
**Python 封装（Triton）：** `vllm_ascend.ops.triton.fla.chunk_delta_h.chunk_gated_delta_rule_fwd_h`  
**实现文件：** `csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h.cpp`  
**kernel_details Input Shapes：** `1,16,18,128;1,16,18,128;1,16,18,128;1,18,32;...` — MIX_AIC

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `k` | (1, 16, 18, 128) | 连续 | float16 | **是** |
| `w` | (1, 16, 18, 128) | 连续 | float16 | **是** |
| `u` | (1, 16, 18, 128) | 连续 | float16 | **是** |
| `g` | (1, 18, 32) | 连续 | float32 | **是** |

**Shape 可变性：** seq 维（dim=2，当前 18）随 prompt 长度变化，16 (key_heads)、32 (value_heads)、128 (head_dim) 固定。

---

### 3.10 `fused_recurrent_gated_delta_rule` — Decode SSM

**vllm-ascend 调用：** `torch.ops._C_ascend.npu_recurrent_gated_delta_rule(query, key, value, g, beta, state, scale, actual_seq_lengths, ssm_state_indices, ...)`  
**代码位置：** `vllm_ascend/ops/gdn.py` L347/L371/L429  
**实现文件：** `csrc/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.cpp`

**函数签名（C++ AscendC）：**
```cpp
at::Tensor npu_recurrent_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    const c10::optional<double> scale,
    const c10::optional<at::Tensor>& actual_seq_lengths,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    ...
);
```

#### Decode（seq=1）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `query` | (1, 16, 128) | (2048, 128, 1) | float16 | **是** |
| `key` | (1, 16, 128) | (2048, 128, 1) | float16 | **是** |
| `value` | (1, 32, 128) | (4096, 128, 1) | float16 | **是** |
| `g` | (1, 32) | (32, 1) | float32 | **是** |
| `beta` | (1, 32) | (32, 1) | float16 | **是** |
| `state` | **(291, 32, 128, 128)** | **(1048576, 16384, 128, 1)** | bfloat16 | **否** |
| `scale` | — | — | float | — |
| `actual_seq_lengths` | (1,) | (1,) | int32 | **是** |
| `ssm_state_indices` | (1,) | (1,) | int64 | **是** |

**kernel_details 观测：** Block Num=32，AI_VECTOR_CORE，Duration ~6.6~9.5μs

**非连续性分析（`state`）：**
- Shape：`(291, 32, 128, 128)` — 状态池共 291 个 block，32 个 value head，head_dim 128×128 的 SSM 状态矩阵。
- Stride dim-0 = **1048576**，是连续期望值 `(32×128×128=524288)` 的 **2 倍**。
- 预分配 buffer 每 block 有 64 个 head 槽位，仅使用 32 个 value head。详见 §4 根因分析。

**Shape 可变性：** `query`/`key` shape `(1, 16, 128)`、`value`/`g`/`beta` 各相关维度固定（decode seq=1）；`state` shape 为 `(block_num, 32, 128, 128)`，block_num（当前 291）随 `max_seq_len`/`seq_size_per_block` 配置变化。

---

### 3.11 `l2norm_fwd`（关联算子）— Q/K 归一化

**vllm-ascend 调用：** `l2norm_fwd(x, eps=1e-6)`  
**代码位置：** `vllm_ascend/ops/triton/fla/l2norm.py` L34

**函数签名：** `l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None)`

#### Prefill（seq=18）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `x`（query） | (288, 128) 即 `(18×16, 128)` | (128, 1) | bfloat16 | **是** |
| `x`（key） | (288, 128) | (128, 1) | bfloat16 | **是** |

**kernel_details Input Shapes：** `288,128` — DT_BF16

#### Decode（seq=1）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `x`（query） | (16, 128) | (128, 1) | bfloat16 | **是** |
| `x`（key） | (16, 128) | (128, 1) | bfloat16 | **是** |

**非连续性分析：** 所有输入均为连续。`l2norm_fwd` 内部调用 `x.reshape(-1, x.shape[-1])` 将输入压平后再处理，确保后续 Triton 内核接收连续 tensor。

**Shape 可变性：** 第一维 `seq×16` 随 prompt 长度变化（prefill 如 `18×16=288`，decode `1×16=16`），128 (head_dim) 固定。

---

### 3.12 `_clear_ssm_states_kernel`（关联算子）— 清除 SSM 状态

**vllm-ascend 调用：** `clear_ssm_states(ssm_states, has_initial_state)`  
**代码位置：** `vllm_ascend/ops/triton/fla/utils.py` L98

**函数签名：** `clear_ssm_states(ssm_states: torch.Tensor, has_initial_state: torch.Tensor) → None`  
**Triton 内核：** `_clear_ssm_states_kernel(state_ptr, has_initial_state_ptr, n_elements, BLOCK: tl.constexpr)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|:----:|:-----:|:------:|:-----:|:-------:|
| `ssm_states` | (291, 32, 128, 128) | (1048576, 16384, 128, 1) | bfloat16 | **否** |
| `has_initial_state` | (1,) 或 scalar | (1,) | bool | **是** |

**kernel_details Input Shapes：** `1` — BOOL

**非连续性分析：** `ssm_states` 与 `RecurrentGatedDeltaRule` 的 `state` 共享同一预分配状态池 buffer，非连续模式相同。

**Shape 可变性：** `ssm_states` shape 为 `(block_num, 32, 128, 128)`，block_num 随配置变化；`has_initial_state` shape 固定。

---

### 3.13 `causal_conv1d_update`（未使用）— Decode conv1d 增量

❌ **vllm-ascend 中 Qwen3.5-GDN 未使用此算子。**

**原因：** vllm-ascend 的 Qwen3.5-GDN 在 decode 阶段仍然调用完整的 `npu_causal_conv1d_custom` 对当前 token 做卷积，不使用增量更新路径。参考 rtp-llm 版本中的 `causal_conv1d_update` 函数签名如下：

**函数签名（rtp-llm 参考）：** `causal_conv1d_update(x, conv_state, weight, bias, activation, cache_seqlens, block_map, seq_size_per_block, sequence_lengths)`

| 参数 | Shape（rtp-llm seq=1） | 是否连续 |
|:----:|:----------------------:|:-------:|
| `x` | (1, 8192, 1) | **是** |
| `conv_state` | (293, 8192, 3) | **否** |
| `weight` | (8192, 4) | **是** |
| `block_map` | (1, 1) | **是** |

---

### 3.14 `prepare_causal_conv1d_metadata`（未使用）

❌ **vllm-ascend 中 Qwen3.5-GDN 未使用此算子。**

**原因：** causal_conv1d 所需的元数据（query_start_loc、cache_indices 等）由 vllm 框架的 `GDNAttentionMetadata` 结构体在 forward 时通过 `attn_metadata` 直接传递，不需要独立调用 prepare 函数。

**函数签名（rtp-llm 参考）：** `prepare_causal_conv1d_metadata(query_start_loc, device)`

| 参数 | Shape | 是否连续 |
|:----:|:-----:|:-------:|
| `query_start_loc` | (2,) | **是** |

---

### 3.15 `load_initial_state_from_block_map`（未使用）

❌ **vllm-ascend 中 Qwen3.5-GDN 未使用此算子。**

**原因：** vllm-ascend 的 GDN 实现中，SSM 初始状态直接从预填充的 `ssm_state` tensor 通过索引读取：`ssm_state[prefill_state_indices]`（`vllm_ascend/ops/gdn.py` L402）。不需要 block_map 抽象层。

**函数签名（rtp-llm 参考）：** `load_initial_state_from_block_map(prefix_lengths, block_map, conv_states, initial_states, seq_size_per_block)`

| 参数 | Shape（rtp-llm） | 是否连续 |
|:----:|:----------------:|:-------:|
| `prefix_lengths` | (1,) | **是** |
| `block_map` | (1, 1) | **是** |
| `conv_states` | (293, 32, 128, 128) | **否** |
| `initial_states` | (1, 32, 128, 128) | **是** |

---

### 3.16 `store_ssm_state_to_block_map`（未使用）

❌ **vllm-ascend 中 Qwen3.5-GDN 未使用此算子。**

**原因：** vllm-ascend 的 GDN 实现直接将更新后的状态写回 `ssm_state` tensor：`ssm_state[prefill_state_indices] = last_recurrent_state.transpose(-1, -2).contiguous()`（`vllm_ascend/ops/gdn.py` L417）。不需要 block_map 抽象层。

**函数签名（rtp-llm 参考）：** `store_ssm_state_to_block_map(h, final_states, prefix_lengths, cu_seqlens, block_map, ssm_states, seq_size_per_block, chunk_size)`

| 参数 | Shape（rtp-llm） | 是否连续 |
|:----:|:----------------:|:-------:|
| `h` | (1, 1, 32, 128, 128) | **是** |
| `final_states` | (1, 32, 128, 128) | **是** |
| `ssm_states` | (293, 32, 128, 128) | **否** |
| `block_map` | (1, 1) | **是** |

---

### 3.17 `RmsNormGated`（未使用）

❌ **vllm-ascend 中 Qwen3.5-GDN 未使用此算子。**

**原因：** `RmsNormGated` 是 `vllm_ascend/ops/triton/kda/kda.py` 中为 DeepSeek 的 KDA（Kernelized Delta Attention）架构设计的门控 RMSNorm。Qwen3.5-GDN 使用普通 `torch_npu.npu_rms_norm` 或 `vllm_ascend/ops/triton/layernorm_gated.py` 中的 `layer_norm_fwd_npu` 代替。

**函数签名（rtp-llm 参考）：** `RmsNormGated.forward(x, gate)`

| 参数 | Shape（rtp-llm） | 是否连续 |
|:----:|:----------------:|:-------:|
| `x` | (seq×32, 128) | **是** |
| `gate` | (seq×32, 128) | **是** |

---

## 4. 非连续 Tensor 汇总

### 4.1 唯一非连续模式

> 下表仅列出非连续参数。各算子的其余参数均为连续（完整参数列表见第 3 章，连续参数汇总见 4.3 节）。所有条目 `is_contiguous()` 返回 `False`。Stride 值基于与 rtp-llm 版本一致的 buffer 管理策略推断（CANN profiling 不直接输出 stride）。

| 算子 | 参数 | Shape | 推断 Stride | 连续时应有 Stride | Dtype | 非连续原因 |
|:----|:----|:-----:|:-----------:|:----------------:|:-----:|-----------|
| `CausalConv1d`（prefill） | `x` | (18, 8192) | (1, 12288) | (8192, 1) | float16 | 列主序（隐藏状态转置） |
| `CausalConv1d`（prefill/decode） | `conv_state` | (291, 3, 8192) | (1048576, 1, 8192) | (71663616, 3, 1) | float16 | 预分配状态池，head 槽位填充 |
| `fused_gdn_gating_kernel`（prefill） | `a`, `b` | (18, 32) | (64, 1) | (32, 1) | bfloat16 | 共享投影 buffer 槽位填充 |
| `chunk_gated_delta_rule`（prefill） | `q`, `k` | (1, 18, 16, 128) | (B, 8192, 128, 1) | (B', 2048, 128, 1) | float16 | QKV 共享 buffer head 槽位填充 |
| `chunk_gated_delta_rule`（prefill） | `v` | (1, 18, 32, 128) | (B, 8192, 128, 1) | (B', 4096, 128, 1) | float16 | QKV 共享 buffer head 槽位填充 |
| `RecurrentGatedDeltaRule`（decode） | `state` | (291, 32, 128, 128) | (1048576, 16384, 128, 1) | (152043520, 524288, 128, 1) | bfloat16 | **预分配状态池，stride[0]=2×期望值** |
| `_clear_ssm_states_kernel`（prefill） | `ssm_states` | (291, 32, 128, 128) | (1048576, 16384, 128, 1) | (152043520, 524288, 128, 1) | bfloat16 | 同上，共享同一状态池 |

### 4.2 根因分析

所有非连续性源于两类**结构性**原因：

**原因 1：共享投影 buffer 切片（conv1d `x`、GDN `a`/`b`、SSM `q`/`k`/`v`）**

- conv1d 的 `x` 参数从隐藏状态 buffer 转置而来，形成列主序（Fortran order）布局：stride `(1, 12288)` 对 shape `(18, 8192)`。与 rtp-llm 完全一致，是框架层隐藏状态布局导致的。
- GDN 的 `a`/`b` 参数从共享投影 buffer 切片，buffer 每 token 有 64 个 channel 槽位，仅使用 32 个：stride `(64, 1)` 对 shape `(18, 32)`。64 槽位 = 32 GDN value heads + 10 全注意力 value heads + 冗余。
- SSM 的 q/k/v 从共享 QKV 投影 buffer 切片，buffer 有 64 个 head 槽位（64×128=8192 elements/token），但 q/k 仅用 16 heads，v 仅用 32 heads。stride[1] = 8192 而非理论最小值。

**原因 2：预分配状态池 buffer（`conv_state`/`state`/`ssm_states`）**

- 所有状态池 buffer 的 shape 第一维均为 291（block 数），stride dim-0 = 1048576。
- `1048576 = 64 × 16384 = 64 × 32 × 128 × 128 / 32`：buffer 每 block 预分配 64 个 head 槽位，实际仅使用 32 个 value head。因此 stride[0] 是连续期望值 524288 的 **2 倍**。
- 同一个状态池 buffer 被以下算子共享：`CausalConv1d`（`conv_state`）、`RecurrentGatedDeltaRule`（`state`）、`_clear_ssm_states_kernel`（`ssm_states`）。

### 4.3 vllm-ascend 与 rtp-llm 的非连续模式对比

| 非连续来源 | rtp-llm（A100+Triton）分布 | vllm-ascend（Ascend NPU）分布 | 一致性 |
|-----------|--------------------------|-----------------------------|:------:|
| 列主序 conv1d `x` | causal_conv1d_fn prefill | CausalConv1d prefill | **一致** |
| 共享投影 buffer `a`/`b` | fused_gdn_gating prefill | fused_gdn_gating prefill | **一致** |
| QKV buffer `q`/`k`/`v` | chunk_gated_delta_rule | chunk_gated_delta_rule | **一致** |
| 状态池 stride 填充 | causal_conv1d_update / load/store block_map / recurrent_delta_rule | CausalConv1d / RecurrentGatedDeltaRule / clear_ssm_states | **一致** |
| 状态池 block 数 | 293 | 291 | **不同**（因 max_seq_len 配置） |

### 4.4 连续 Tensor 汇总（无问题）

| 算子（Profiling 名） | 对应目标算子 | 所有参数状态 |
|:--------------------|:-----------|-------------|
| `chunk_local_cumsum_scalar_kernel` | chunk_local_cumsum | **全部连续** |
| `chunk_scaled_dot_kkt_fwd_kernel` | chunk_scaled_dot_kkt_fwd | **全部连续** |
| `solve_tril_16x16_kernel` | solve_tril | **全部连续** |
| `recompute_w_u_fwd_kernel` | recompute_w_u_fwd | **全部连续** |
| `ChunkFwdO` | chunk_fwd_o | **全部连续** |
| `ChunkGatedDeltaRuleFwdH` | ChunkGatedDeltaRuleFwdH | **全部连续** |
| `l2norm_fwd_kernel2_0` / `l2norm_fwd_kernel2_loop` | l2norm_fwd | **全部连续** |
| `CausalConv1d` | causal_conv1d_fn | 除 `x`（prefill）和 `conv_state` 外均连续 |
| `fused_gdn_gating_kernel_0`（decode） | fused_gdn_gating（decode） | **全部连续**（seq=1） |
| `fused_gdn_gating_kernel`（prefill） | fused_gdn_gating（prefill） | 除 `a`/`b` 外均连续（`A_log`/`dt_bias` 连续） |
| `chunk_gated_delta_rule` | chunk_gated_delta_rule | 仅 `q`/`k`/`v` 非连续（`g`/`beta` 连续） |
| `RecurrentGatedDeltaRule` | fused_recurrent_gated_delta_rule | 除 `state` 外均连续（q/k/v/g/beta decode 连续） |
| `_clear_ssm_states_kernel` | _clear_ssm_states | 仅 `ssm_states` 非连续 |

---

## 5. 耗时占比与性能分析

### 5.1 GDN 线性注意力算子耗时分布

| 算子（Profiling kernel 名） | 总耗时(μs) | 占比(%) | 调用次数 | 均耗时(μs) | 核心类型 |
|:---------------------------|:---------:|:-------:|:-------:|:---------:|:--------:|
| `RecurrentGatedDeltaRule` | **197,290.34** | **1.926** | 26,760 | 7.37 | AI_VECTOR_CORE |
| `fused_gdn_gating_kernel_0` | 152,902.71 | 1.493 | 26,760 | 5.71 | AI_VECTOR_CORE |
| `fused_gdn_gating_kernel` | 193.67 | 0.002 | 30 | 6.46 | AI_VECTOR_CORE |
| `l2norm_fwd_kernel2_0` | 127,932.02 | 1.249 | 53,520 | 2.39 | AI_VECTOR_CORE |
| `l2norm_fwd_kernel2_loop` | 171.45 | 0.002 | 60 | 2.86 | AI_VECTOR_CORE |
| `CausalConv1d` | 117,106.66 | 1.143 | 26,790 | 4.37 | MIX_AIC |
| `solve_tril_16x16_kernel` | 10,753.43 | 0.105 | 30 | **358.45** | AI_VECTOR_CORE |
| `recompute_w_u_fwd_kernel` | 7,693.40 | 0.075 | 30 | **256.45** | AI_CORE |
| `ChunkGatedDeltaRuleFwdH` | 1,755.31 | 0.017 | 30 | 58.51 | MIX_AIC |
| `ChunkFwdO` | 409.32 | 0.004 | 30 | 13.64 | MIX_AIC |
| `chunk_scaled_dot_kkt_fwd_kernel` | 188.35 | 0.002 | 30 | 6.28 | AI_CORE |
| `chunk_local_cumsum_scalar_kernel` | 106.80 | 0.001 | 30 | 3.56 | AI_VECTOR_CORE |
| `_clear_ssm_states_kernel` | 108.01 | 0.001 | 30 | 3.60 | AI_VECTOR_CORE |
| **GDN 合计** | **~616,594** | **~6.02** | | | |

### 5.2 Decode vs Prefill 分阶段耗时

| 阶段 | 主要算子 | 总耗时(μs) | 总占比(%) | 每步均耗时(μs) |
|:---:|---------|:----------:|:---------:|:-------------:|
| **Decode**（26,760 次） | RecurrentGatedDeltaRule + fused_gdn_gating + CausalConv1d + l2norm×2 | ~595,232 | ~5.81 | ~22.24 |
| **Prefill**（30 次） | chunk 系列 + fused_gdn_gating + l2norm×2 | ~21,362 | ~0.21 | ~712.06 |

### 5.3 关键发现

1. **`RecurrentGatedDeltaRule`** 是 GDN 注意力最耗时的算子（**1.93%**，197ms），每次 decode 平均 7.37 μs，26,760 次调用。

2. **三个 decode 阶段主要算子**（fused_gdn_gating 1.49% + l2norm 1.25% + CausalConv1d 1.14%）合计 3.88%，是 GDN 注意力的固定开销。

3. **`solve_tril_16x16`** 单次 **358 μs** 是最慢的子算子，**`recompute_w_u_fwd`** 单次 **256 μs** 排第二。但两者都只在 prefill 阶段执行 30 次，对整体影响有限。

4. **GDN 注意力整体占 device 时间 ~6%**，远低于 MoE（GroupedMatmul ~37.75%）和通用矩阵乘法（MatMulV3 ~31.36%）。

---

## 6. 模型架构参数（profiling 推断）

| 参数 | 值 | 推断来源 |
|------|:--:|---------|
| `hidden_size` | 8192 | kernel_details |
| `num_key_heads` | 16 | kernel_details |
| `num_value_heads` | 32 | kernel_details |
| `head_dim` | 128 | kernel_details |
| `conv_kernel_size` | 3 | kernel_details |
| 总层数 | 40（30层线性 + 10层全注意力） | 26,760/892 ≈ 30 |
| 线性注意力层分布 | 每 4 层中 3 层线性 + 1 层全注意力 | 推测 |
| `chunk_size` | 64 | kernel_details |
| 状态池 block 数 | 291 | kernel_details |
| `conv_dim` | 8192 | = 64 heads × 128 head_dim |
| total decode steps | ~892 | 26,760 / 30 |

---

## 7. 相关文件

| 文件 | 用途 |
|------|------|
| `/home/z30066236/QWen3.5_test/vllm-prof_stack/` | CANN Level1 Profiling 原始数据（4.1 GB，466 files） |
| `.../ASCEND_PROFILER_OUTPUT/op_statistic.csv` | 算子聚合统计（61 种算子类型） |
| `.../ASCEND_PROFILER_OUTPUT/operator_details.csv` | 算子级调用详情（701,808 条） |
| `.../ASCEND_PROFILER_OUTPUT/kernel_details.csv` | Kernel 级执行详情（1,192,386 条，含 Input Shapes） |
| `/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py` | GDN 注意力前向实现 |
| `/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fused_gdn_gating.py` | fused_gdn_gating Triton 内核 |
| `/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/` | Chunk 系列 Triton 内核（l2norm, cumsum, chunk 等） |
| `/vllm-workspace/vllm-ascend/vllm_ascend/device/device_op.py` | DeviceOperator（算子封装适配器） |
| `/vllm-workspace/vllm-ascend/csrc/attention/recurrent_gated_delta_rule/` | RecurrentGatedDeltaRule AscendC 实现 |
| `/vllm-workspace/vllm-ascend/csrc/moe/causal_conv1d/` | CausalConv1d AscendC 实现 |
| `/vllm-workspace/vllm-ascend/csrc/moe/chunk_fwd_o/` | ChunkFwdO AscendC 实现 |
| `/vllm-workspace/vllm-ascend/csrc/moe/chunk_gated_delta_rule_fwd_h/` | ChunkGatedDeltaRuleFwdH AscendC 实现 |
| `/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_qwen3_5.py` | Qwen3.5 模型 monkey-patch |

---

## 8. 建议

1. **已使用的 10 个算子/内核均已成功捕获。** 其中 4 个参数存在非连续输入。任何自定义 kernel 实现必须处理：
   - **列主序布局**：stride `(1, 12288)` 对 shape `(18, 8192)` — conv1d prefill 的 `x` 参数
   - **共享投影 buffer 槽位填充**：stride `(64, 1)` 对 shape `(18, 32)` — GDN gating prefill 的 `a`/`b` 参数
   - **Seq 维 stride 填充**：stride[1] = 8192 对 16 或 32 heads — SSM prefill 的 `q`/`k`/`v`
   - **状态池 block stride 填充**：stride[0] = 2× 期望值 — `RecurrentGatedDeltaRule` 的 `state`、`CausalConv1d` 的 `conv_state`

2. **Chunk 子算子（6 个）全部接收连续输入**，因为 `chunk_gated_delta_rule` 在入口处做了 rearrange 消除非连续性。无需额外处理。

3. **Decode 阶段的 q/k/v 是连续的**（seq=1），唯一非连续性来自状态池 `state`/`conv_state`。

4. **非连续性是结构性的**，来源于 vllm-ascend 的 buffer 管理策略（共享 head 槽位 + 预分配状态池），与 rtp-llm 版本完全一致。添加 `.contiguous()` 调用可以解决但引入内存拷贝。AscendC 自定义算子内部需支持非连续 strided 输入。

5. **优化方向**：
   - `solve_tril_16x16`（358 μs/次）和 `recompute_w_u_fwd`（256 μs/次）是 prefill 中可优化的子算子（但仅 30 次调用）
   - Decode 阶段核心在 `RecurrentGatedDeltaRule`（7.37 μs/次 × 26,760 次）
   - l2norm 虽然单次仅 2.39 μs，但 Q/K 各一次 + 高频调用（53,580 次），累计占比 1.25%
