# Qwen3.5-35B-A3B SSM/线性注意力算子 Profiling 报告

**日期：** 2026-07-24
**模型：** Qwen3.5-35B-A3B（MoE，40层：30层线性注意力 + 10层全注意力）
**框架：** rtp-llm v0.2.0
**GPU：** 1× A100-SXM4-80GB（bf16）
**配置：** `--seq_size_per_block 1024 --max_seq_len 2048 --concurrency_limit 1`

---

## 1. Profiling 设置

### 1.1 拦截的算子（共 14 个）

| 算子 | 阶段 | 源模块 |
|---|---|---|
| `causal_conv1d_fn` | Prefill conv1d | `triton_kernels.causal_conv1d` |
| `causal_conv1d_update` | Decode conv1d | `triton_kernels.causal_conv1d` |
| `prepare_causal_conv1d_metadata` | Conv1d 元数据 | `triton_kernels.causal_conv1d` |
| `fused_gdn_gating` | GDN 门控 | `triton_kernels.fla.gdn_gating` |
| `chunk_gated_delta_rule` | Prefill SSM | `triton_kernels.fla.chunk` |
| `chunk_local_cumsum` | Chunk 子算子：cumsum | `triton_kernels.fla.cumsum` |
| `chunk_scaled_dot_kkt_fwd` | Chunk 子算子：QK^T | `triton_kernels.fla.chunk_scaled_dot_kkt` |
| `solve_tril` | Chunk 子算子：解三角矩阵 | `triton_kernels.fla.solve_tril` |
| `recompute_w_u_fwd` | Chunk 子算子：W/U | `triton_kernels.fla.wy_fast` |
| `chunk_fwd_o` | Chunk 子算子：前向 O | `triton_kernels.fla.chunk_o` |
| `fused_recurrent_gated_delta_rule` | Decode SSM | `triton_kernels.fla.fused_recurrent` |
| `load_initial_state_from_block_map` | 状态加载 | `triton_kernels.fla.block` |
| `store_ssm_state_to_block_map` | 状态存储 | `triton_kernels.fla.block` |
| `RmsNormGated` | RMS 归一化 + 门控 | `triton_kernels.common.layernorm_gated` |

### 1.2 Hook 机制

- **`.pth` 文件**（`site-packages/ssm_profiler.pth`）在 Python 启动时自动 import `ssm_profiler_init.py`，确保 hook 在 `multiprocessing.spawn` 子进程中也能生效。
- **Import hook** 包装 `builtins.__import__`，在 `qwen3_next` 模块完整加载其所有依赖后（检测 `fused_gdn_gating` 属性是否已绑定）触发 patch。
- **Monkey-patch** 分三层：消费模块 `qwen3_next`、消费模块 `fla.chunk`（子算子）、源模块（兜底）。
- **`RmsNormGated`** 为类，通过 hook 其 `forward` 方法实现拦截。

### 1.3 测试负载

- **Prompt 1：** 2047 tokens prefill → 110 步 decode
- **Prompt 2：** 13 tokens prefill → 110 步 decode
- 总 decode 步数：220

---

## 2. 调用次数统计

| 算子 | 总调用次数 | 每层调用次数（30层） |
|---|---|---|
| `causal_conv1d_fn` | 60 | 2（每个 prompt prefill 各一次） |
| `causal_conv1d_update` | 6,600 | 220（总 decode 步数） |
| `prepare_causal_conv1d_metadata` | 2 | 仅 prefill 初始化，每 prompt 一次 |
| `fused_gdn_gating` | 6,660 | 222（2 prefill + 220 decode） |
| `chunk_gated_delta_rule` | 60 | 2（每个 prompt prefill 各一次） |
| `chunk_local_cumsum` | 60 | 2 |
| `chunk_scaled_dot_kkt_fwd` | 60 | 2 |
| `solve_tril` | 60 | 2 |
| `recompute_w_u_fwd` | 60 | 2 |
| `chunk_fwd_o` | 60 | 2 |
| `fused_recurrent_gated_delta_rule` | 6,600 | 220（总 decode 步数） |
| `load_initial_state_from_block_map` | 30 | 1（仅 prefill） |
| `store_ssm_state_to_block_map` | 30 | 1（仅 prefill） |
| `RmsNormGated` | 6,660 | 222（2 prefill + 220 decode） |

> 全部 14 个算子均已成功捕获。

---

## 3. 算子调用模式与 Tensor 分析

> **说明：** Profiling 共发送了两个 prompt，各算子在不同 prompt 下表现出不同的 shape（第一维 = 序列长度）。两个 prompt 分别为：
> - **Prompt 1：** 2047 tokens（长序列）
> - **Prompt 2：** 13 tokens（短序列）
>
> 因此同一算子会出现在两种 seq_len 下的调用记录，但非连续模式（stride）完全一致，因为非连续性来源于共享 buffer 切片，与序列长度无关。

### 3.1 `causal_conv1d_fn` — Prefill conv1d

**函数签名：** `causal_conv1d_fn(x, weight, bias, conv_states, query_start_loc, block_map, seq_size_per_block, prefix_lengths, metadata)`

#### Prompt 1（seq=2047，首次 prefill，无 conv_states）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `x` | (8192, 2047) | **(1, 12288)** | float16 | **否（channel_last_2d）** |
| `weight` | (8192, 4) | (4, 1) | float16 | **是** |
| `bias` | None | — | — | — |
| `conv_states` | None | — | — | — |
| `query_start_loc` | (2,) | (1,) | int32 | **是** |
| `block_map` | None | — | — | — |
| `seq_size_per_block` | 1 | — | int | — |

#### Prompt 2（seq=13，增量 prefill，携带 conv_states）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `x` | (8192, 13) | **(1, 12288)** | float16 | **否（channel_last_2d）** |
| `weight` | (8192, 4) | (4, 1) | float16 | **是** |
| `conv_states` | (293, 8192, 3) | **(1048576, 1, 8192)** | float16 | **否** |

**非连续性分析：**
- `x`：stride 为 `(1, 12288)`，连续（C order / 行主序）时应为 `(8192, 1)`。当前为**列主序（Fortran order）**布局，即 dim-0（conv_dim=8192）stride 为 1，dim-1（seq）stride 为 12288。x 是从 `(seq, hidden)` 的隐藏状态 buffer 转置而来。`channel_last_2d` 是 profiler 对此类非连续布局的细分分类，**本质仍为非连续**——PyTorch 的 `is_contiguous()` 返回 `False`。
- `conv_states`（Prompt 2）：shape `(293, 8192, 3)`，stride `(1048576, 1, 8192)`。293 = 状态池 block 数，8192 = conv_dim，3 = conv_kernel_size。stride[0] = 1048576 = 64 × 16384（预分配 buffer 有 64 个 head 槽位，仅使用部分），非连续。

**Shape 可变性：** `x` shape 为 `(8192, seq)`，seq 随输入 prompt 长度变化（prefill 如 2047/13，decode 固定为 1），8192 (conv_dim) 固定；`conv_states` shape 为 `(block_num, 8192, 3)`，block_num（当前 293）随 `max_seq_len`/`seq_size_per_block` 配置变化，8192/3 固定；`weight` shape `(8192, 4)` 固定。

---

### 3.2 `causal_conv1d_update` — Decode conv1d

**函数签名：** `causal_conv1d_update(x, conv_state, weight, bias, activation, cache_seqlens, block_map, seq_size_per_block, sequence_lengths)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `x` | (1, 8192, 1) | (8192, 1, 8192) | float16 | **是** |
| `conv_state` | **(293, 8192, 3)** | **(1048576, 1, 8192)** | float16 | **否** |
| `weight` | (8192, 4) | (4, 1) | float16 | **是** |
| `block_map` | (1, 1) | (1, 1) | int32 | **是** |
| `seq_size_per_block` | 1024 | — | int | — |
| `sequence_lengths` | (1,) | (1,) | int32 | **是** |

**非连续性分析：**
- `conv_state`：与 prefill 中的 `conv_states` 相同的非连续模式，来自预分配的状态池 buffer。

**Shape 可变性：** `x` shape `(1, 8192, 1)` 固定（decode seq=1）；`conv_state` shape 为 `(block_num, 8192, 3)`，block_num（当前 293）随配置变化，8192/3 固定；`weight` shape `(8192, 4)` 固定；`block_map` shape `(1, 1)` 固定（batch=1）。

---

### 3.3 `prepare_causal_conv1d_metadata`

**函数签名：** `prepare_causal_conv1d_metadata(query_start_loc, device)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `query_start_loc` | (2,) | (1,) | int32 | **是** |
| `device` | — | — | device | — |

仅接收 `query_start_loc`（int32 tensor）和 `device` 参数，**无 tensor 连续性问题**。

**Shape 可变性：** `query_start_loc` shape 为 `(batch_size+1,)`，随 batch_size 变化；`device` 非 tensor。

---

### 3.4 `fused_gdn_gating` — GDN 门控

**函数签名：** `fused_gdn_gating(A_log, a, b, dt_bias)`

#### Prefill（Prompt 1，seq=2047）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `A_log` | (32,) | (1,) | float16 | **是** |
| `a` | (2047, 32) | **(64, 1)** | float16 | **否** |
| `b` | (2047, 32) | **(64, 1)** | float16 | **否** |
| `dt_bias` | (32,) | (1,) | float16 | **是** |

#### Prefill（Prompt 2，seq=13）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `A_log` | (32,) | (1,) | float16 | **是** |
| `a` | (13, 32) | **(64, 1)** | float16 | **否** |
| `b` | (13, 32) | **(64, 1)** | float16 | **否** |
| `dt_bias` | (32,) | (1,) | float16 | **是** |

#### Decode（seq=1）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `A_log` | (32,) | (1,) | float16 | **是** |
| `a` | (1, 32) | (64, 1) | float16 | **是** |
| `b` | (1, 32) | (64, 1) | float16 | **是** |
| `dt_bias` | (32,) | (1,) | float16 | **是** |

**非连续性分析：**
- Prefill 阶段 `a`/`b` 的 stride 为 `(64, 1)`，连续时应为 `(32, 1)`。第一维 stride 为 64 而非 32，因为 `a`/`b` 从共享投影 buffer 切片（64 channel 槽位，仅用 32）。
- Decode 阶段（seq=1）tensor 为 `(1, 32)`，stride 为 `(64, 1)`，由于 batch 维度为 1，`is_contiguous()` 返回 `True`。

**Shape 可变性：** `A_log` shape `(32,)` 和 `dt_bias` shape `(32,)` 固定；`a`/`b` shape 为 `(seq, 32)`，seq 随输入 prompt 长度变化（prefill 如 2047/13，decode 固定为 1），32 (value_heads) 固定。

---

### 3.5 `chunk_gated_delta_rule` — Prefill SSM

**函数签名：** `chunk_gated_delta_rule(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel)`

#### Prompt 1（seq=2047，initial_state=None）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `q` | (1, 2047, 16, 128) | **(16769024, 8192, 128, 1)** | float16 | **否** |
| `k` | (1, 2047, 16, 128) | **(16769024, 8192, 128, 1)** | float16 | **否** |
| `v` | (1, 2047, 32, 128) | **(16769024, 8192, 128, 1)** | float16 | **否** |
| `g` | (1, 2047, 32) | (65504, 32, 1) | float32 | **是** |
| `beta` | (1, 2047, 32) | (65504, 32, 1) | float16 | **是** |
| `initial_state` | None | — | — | — |

#### Prompt 2（seq=13，initial_state 来自 block_map 加载）

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `q` | (1, 13, 16, 128) | **(106496, 8192, 128, 1)** | float16 | **否** |
| `k` | (1, 13, 16, 128) | **(106496, 8192, 128, 1)** | float16 | **否** |
| `v` | (1, 13, 32, 128) | **(106496, 8192, 128, 1)** | float16 | **否** |
| `g` | (1, 13, 32) | (416, 32, 1) | float32 | **是** |
| `beta` | (1, 13, 32) | (416, 32, 1) | float16 | **是** |
| `initial_state` | (1, 32, 128, 128) | (524288, 16384, 128, 1) | bfloat16 | **是** |

**非连续性分析（q/k/v）：**
- 三者共享 stride `(batch_stride, 8192, 128, 1)`，尽管 head 数不同（q/k=16，v=32）。
- 连续时 q 的 stride 应为 `(B', 2048, 128, 1)`，实际 seq 维 stride 为 **8192 = 64 heads × 128**。
- q/k/v 从共享 QKV 投影 buffer 切片，该 buffer 有 64 个 head 槽位。

**Shape 可变性：** `q`/`k`/`v`/`g`/`beta` 的 seq 维（dim-1）随输入 prompt 长度变化（prefill 如 2047/13），其余维度固定：`q`/`k` shape 为 `(1, seq, 16, 128)`、`v` shape 为 `(1, seq, 32, 128)`、`g`/`beta` shape 为 `(1, seq, 32)`；`initial_state` shape `(1, 32, 128, 128)` 固定；`cu_seqlens` shape `(2,)` = `(batch_size+1,)` 固定（batch=1）。

---

### 3.6 `chunk_local_cumsum` — Chunk 子算子：cumsum

> 以下 5 个子算子（3.6-3.10）均在 `chunk_gated_delta_rule` 内部被调用。**注意：子算子接收的 q/k/v 是 chunk 内部重新排列（rearrange）后的 tensor，已转为连续。**

**函数签名：** `chunk_local_cumsum(g, chunk_size, cu_seqlens)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `g` | (1, seq, 32) | (seq×32, 32, 1) | float32 | **是** |
| `cu_seqlens` | (2,) | (1,) | int32 | **是** |

**Shape 可变性：** `g` shape 为 `(1, seq, 32)`，seq 随输入 prompt 长度变化，32 (value_heads) 固定；`cu_seqlens` shape `(2,)` = `(batch_size+1,)` 固定（batch=1）。

---

### 3.7 `chunk_scaled_dot_kkt_fwd` — Chunk 子算子：QK^T

**函数签名：** `chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens, output_dtype)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `k` | (1, seq, 16, 128) | (seq×2048, 2048, 128, 1) | float16 | **是** |
| `beta` | (1, seq, 32) | (seq×32, 32, 1) | float16 | **是** |
| `g_cumsum` | (1, seq, 32) | (seq×32, 32, 1) | float32 | **是** |

**Shape 可变性：** `k`/`beta`/`g_cumsum` 的 seq 维（dim-1）随输入 prompt 长度变化，其余维度固定：`k` shape 为 `(1, seq, 16, 128)`、`beta`/`g_cumsum` shape 为 `(1, seq, 32)`；16 (key_heads)、32 (value_heads)、128 (head_dim) 固定。

---

### 3.8 `solve_tril` — Chunk 子算子：解三角矩阵

**函数签名：** `solve_tril(A, cu_seqlens, output_dtype)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `A` | (1, seq, 32, 64) | (seq×2048, 2048, 64, 1) | float32 | **是** |

**Shape 可变性：** `A` shape 为 `(1, seq, 32, 64)`，seq 随输入 prompt 长度变化，32 (value_heads) 和 64 (chunk_size) 固定。

---

### 3.9 `recompute_w_u_fwd` — Chunk 子算子：W/U

**函数签名：** `recompute_w_u_fwd(k, v, beta, A, g_cumsum, cu_seqlens)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `k` | (1, seq, 16, 128) | (seq×2048, 2048, 128, 1) | float16 | **是** |
| `v` | (1, seq, 32, 128) | (seq×4096, 4096, 128, 1) | float16 | **是** |
| `A` | (1, seq, 32, 64) | (seq×2048, 2048, 64, 1) | float16 | **是** |
| `g_cumsum` | (1, seq, 32) | (seq×32, 32, 1) | float32 | **是** |

**Shape 可变性：** `k`/`v`/`A`/`g_cumsum` 的 seq 维（dim-1）随输入 prompt 长度变化，其余维度固定：`k` shape 为 `(1, seq, 16, 128)`、`v` shape 为 `(1, seq, 32, 128)`、`A` shape 为 `(1, seq, 32, 64)`、`g_cumsum` shape 为 `(1, seq, 32)`。

---

### 3.10 `chunk_fwd_o` — Chunk 子算子：前向 O

**函数签名：** `chunk_fwd_o(q, k, v, h, g, scale, cu_seqlens)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `q` | (1, seq, 16, 128) | (seq×2048, 2048, 128, 1) | float16 | **是** |
| `k` | (1, seq, 16, 128) | (seq×2048, 2048, 128, 1) | float16 | **是** |
| `v` | (1, seq, 32, 128) | (seq×4096, 4096, 128, 1) | float16 | **是** |
| `h` | (1, 32, 32, 128, 128) | (16777216, 524288, 16384, 128, 1) | float32 | **是** |

> **子算子结论：** 所有 5 个 chunk 子算子的所有 tensor 输入均为连续。`chunk_gated_delta_rule` 在入口处对 q/k/v 做了 rearrange 操作，消除了非连续性。

**Shape 可变性（3.10）：** `q`/`k`/`v` 的 seq 维（dim-1）随输入 prompt 长度变化，其余维度固定：`q`/`k` shape 为 `(1, seq, 16, 128)`、`v` shape 为 `(1, seq, 32, 128)`；`h` shape `(1, 32, 32, 128, 128)` 固定。

---

### 3.11 `fused_recurrent_gated_delta_rule` — Decode SSM

**函数签名：** `fused_recurrent_gated_delta_rule(q, k, v, g, beta, scale, initial_state, inplace_final_state, block_map, seq_size_per_block, sequence_lengths, use_qk_l2norm_in_kernel)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `q` | (1, 1, 16, 128) | (8192, 8192, 128, 1) | float16 | **是** |
| `k` | (1, 1, 16, 128) | (8192, 8192, 128, 1) | float16 | **是** |
| `v` | (1, 1, 32, 128) | (8192, 8192, 128, 1) | float16 | **是** |
| `g` | (1, 1, 32) | (32, 32, 1) | float32 | **是** |
| `beta` | (1, 1, 32) | (32, 32, 1) | float16 | **是** |
| `initial_state` | **(293, 32, 128, 128)** | **(1048576, 16384, 128, 1)** | bfloat16 | **否** |
| `block_map` | (1, 1) | (1, 1) | int32 | **是** |
| `seq_size_per_block` | 1024 | — | int | — |
| `sequence_lengths` | (1,) | (1,) | int32 | **是** |

**非连续性分析（`initial_state`）：**
- Shape：`(293, 32, 128, 128)` — 状态池共 293 个 block。
- Stride dim-0 = **1048576**，是连续期望值 524288 的 **2 倍**。
- 预分配 buffer 每 block 有 64 个 head 槽位，仅使用 32 个 value head。

**Shape 可变性：** `q`/`k` shape `(1, 1, 16, 128)`、`v` shape `(1, 1, 32, 128)`、`g`/`beta` shape `(1, 1, 32)` 均固定（decode seq=1）；`initial_state` shape 为 `(block_num, 32, 128, 128)`，block_num（当前 293）随 `max_seq_len`/`seq_size_per_block` 配置变化，32/128/128 固定。

---

### 3.12 `load_initial_state_from_block_map` — 状态加载

**函数签名：** `load_initial_state_from_block_map(prefix_lengths, block_map, conv_states, initial_states, seq_size_per_block)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `prefix_lengths` | (1,) | (1,) | int32 | **是** |
| `block_map` | (1, 1) | (1, 1) | int32 | **是** |
| `conv_states` | **(293, 32, 128, 128)** | **(1048576, 16384, 128, 1)** | bfloat16 | **否** |
| `initial_states` | (1, 32, 128, 128) | (524288, 16384, 128, 1) | bfloat16 | **是** |
| `seq_size_per_block` | 1024 | — | int | — |

**非连续性分析：**
- `conv_states`：与 `fused_recurrent_gated_delta_rule` 的 `initial_state` 相同的非连续模式，来自同一个状态池 buffer。

**Shape 可变性：** `conv_states` shape 为 `(block_num, 32, 128, 128)`，block_num（当前 293）随 `max_seq_len`/`seq_size_per_block` 配置变化，32/128/128 固定；`initial_states` shape `(1, 32, 128, 128)` 固定；`block_map` shape `(1, 1)` 和 `prefix_lengths` shape `(1,)` 随 batch_size 变化（当前 batch=1）。

---

### 3.13 `store_ssm_state_to_block_map` — 状态存储

**函数签名：** `store_ssm_state_to_block_map(h, final_states, prefix_lengths, cu_seqlens, block_map, ssm_states, seq_size_per_block, chunk_size)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `h` | (1, 1, 32, 128, 128) | (524288, 524288, 16384, 128, 1) | float32 | **是** |
| `final_states` | (1, 32, 128, 128) | (524288, 16384, 128, 1) | float32 | **是** |
| `block_map` | (1, 1) | (1, 1) | int32 | **是** |
| `ssm_states` | **(293, 32, 128, 128)** | **(1048576, 16384, 128, 1)** | bfloat16 | **否** |
| `seq_size_per_block` | 1024 | — | int | — |
| `chunk_size` | 64 | — | int | — |

**非连续性分析：**
- `ssm_states`：与上述状态池 buffer 相同的非连续模式。

**Shape 可变性：** `h` shape `(1, 1, 32, 128, 128)` 和 `final_states` shape `(1, 32, 128, 128)` 固定；`ssm_states` shape 为 `(block_num, 32, 128, 128)`，block_num（当前 293）随 `max_seq_len`/`seq_size_per_block` 配置变化，32/128/128 固定。

---

### 3.14 `RmsNormGated` — RMS 归一化 + 门控

**函数签名：** `RmsNormGated.forward(x, gate)`

| 参数 | Shape | Stride | Dtype | 是否连续 |
|---|---|---|---|---|
| `x` | (seq×32, 128) 或 (32, 128) | (128, 1) | float16 | **是** |
| `gate` | (seq×32, 128) 或 (32, 128) | (128, 1) | float16 | **是** |

> Prefill 阶段 `x`/`gate` shape 为 `(65504, 128)`（seq=2047 时）或 `(416, 128)`（seq=13 时）；Decode 阶段为 `(32, 128)`。所有输入均为连续。

**Shape 可变性：** `x`/`gate` shape 为 `(seq×32, 128)`，seq 随输入 prompt 长度变化（prefill 如 2047 时为 `(65504, 128)`，13 时为 `(416, 128)`，decode 固定为 `(32, 128)`），128 (head_dim) 固定。

---

## 4. 非连续 Tensor 汇总

### 4.1 唯一非连续模式

> 下表仅列出非连续参数。各算子的其余参数均为连续（完整参数列表见第 3 章，连续参数汇总见 4.3 节）。表中"布局类型"列是对非连续模式的细分分类，**所有条目均为非连续**（`is_contiguous()` 返回 `False`）。

| 算子 | 参数 | Shape | Stride | 连续时应有 Stride | Dtype | 布局类型 |
|---|---|---|---|---|---|---|
| `causal_conv1d_fn`（prefill） | `x` | (8192, seq) | (1, 12288) | (seq, 1) | float16 | 转置 |
| `causal_conv1d_fn`（prefill P2） | `conv_states` | (293, 8192, 3) | (1048576, 1, 8192) | (24576, 3, 1) | float16 | 状态池 |
| `causal_conv1d_update`（decode） | `conv_state` | (293, 8192, 3) | (1048576, 1, 8192) | (24576, 3, 1) | float16 | 状态池 |
| `fused_gdn_gating`（prefill） | `a`, `b` | (seq, 32) | (64, 1) | (32, 1) | float16 | 切片 |
| `chunk_gated_delta_rule`（prefill） | `q`, `k` | (1, seq, 16, 128) | (B, 8192, 128, 1) | (B', 2048, 128, 1) | float16 | 切片 |
| `chunk_gated_delta_rule`（prefill） | `v` | (1, seq, 32, 128) | (B, 8192, 128, 1) | (B', 4096, 128, 1) | float16 | 切片 |
| `fused_recurrent_gated_delta_rule`（decode） | `initial_state` | (293, 32, 128, 128) | (1048576, 16384, 128, 1) | (524288, 16384, 128, 1) | bfloat16 | 状态池 |
| `load_initial_state_from_block_map` | `conv_states` | (293, 32, 128, 128) | (1048576, 16384, 128, 1) | (524288, 16384, 128, 1) | bfloat16 | 状态池 |
| `store_ssm_state_to_block_map` | `ssm_states` | (293, 32, 128, 128) | (1048576, 16384, 128, 1) | (524288, 16384, 128, 1) | bfloat16 | 状态池 |

### 4.2 根因分析

所有非连续性源于三类**结构性**原因：

**原因 1：Channel-last 转置（conv1d `x`）**
- conv1d 的 `x` 参数从隐藏状态 buffer 转置而来，形成列主序（Fortran order）布局：stride `(1, 12288)` 对 shape `(8192, seq)`，连续时应为 `(seq, 1)`。

**原因 2：共享投影 buffer 切片（GDN `a`/`b`、SSM `q`/`k`/`v`）**
- GDN 的 `a`/`b` 参数从共享投影 buffer 切片，buffer 每 token 有 64 个 channel 槽位，仅使用 32 个：stride `(64, 1)` 对 shape `(seq, 32)`。
- SSM 的 q/k/v 从共享 QKV 投影 buffer 切片，buffer 有 64 个 head 槽位（64×128=8192 elements/token），但 q 仅用 16 heads，v 仅用 32 heads。

**原因 3：预分配状态池 buffer（`conv_state`/`conv_states`/`initial_state`/`ssm_states`）**
- 所有状态池 buffer 的 shape 第一维均为 293（block 数），stride 第一维均为 1048576。
- 1048576 = 64 × 16384 = 64 × 128 × 128：buffer 每 block 预分配 64 个 head 槽位，实际仅使用 32 个 value head。
- 同一个状态池 buffer 被以下算子共享：`causal_conv1d_update`（`conv_state`）、`causal_conv1d_fn` P2（`conv_states`）、`load_initial_state_from_block_map`（`conv_states`）、`store_ssm_state_to_block_map`（`ssm_states`）、`fused_recurrent_gated_delta_rule`（`initial_state`）。

### 4.3 连续 Tensor 汇总（无问题）

| 算子 | 所有参数状态 |
|---|---|
| `prepare_causal_conv1d_metadata` | 无 tensor 输入连续性问题 |
| `chunk_local_cumsum` | **全部连续** |
| `chunk_scaled_dot_kkt_fwd` | **全部连续** |
| `solve_tril` | **全部连续** |
| `recompute_w_u_fwd` | **全部连续** |
| `chunk_fwd_o` | **全部连续** |
| `RmsNormGated` | **全部连续** |
| `causal_conv1d_fn` | 除 `x`、`conv_states` 外均连续 |
| `causal_conv1d_update` | 除 `conv_state` 外均连续 |
| `fused_gdn_gating` | 除 `a`/`b`（prefill 非连续）外均连续（`A_log`/`dt_bias` 始终连续；decode 阶段全部连续） |
| `chunk_gated_delta_rule` | 除 `q`/`k`/`v` 外均连续（`g`/`beta`/`initial_state` 连续） |
| `fused_recurrent_gated_delta_rule` | 除 `initial_state` 外均连续（q/k/v/g/beta 在 decode 时连续） |
| `load_initial_state_from_block_map` | 除 `conv_states` 外均连续 |
| `store_ssm_state_to_block_map` | 除 `ssm_states` 外均连续 |

---

## 5. 调用模式分析

对每个算子，按"是否存在结构性不同的参数组合"（仅 seq 长度变化不算不同模式）分析调用模式。14 个算子中，**5 个存在多模式**，**9 个仅单一模式**。

### 5.1 单一模式算子（9 个）

以下算子所有调用的参数结构完全一致，仅 seq 维度大小随输入变化：

| 算子 | 调用次数 | 说明 |
|---|---|---|
| `causal_conv1d_update` | 6600 | 唯一模式（decode） |
| `prepare_causal_conv1d_metadata` | 2 | 唯一模式 |
| `chunk_local_cumsum` | 60 | 仅 seq 变化 |
| `chunk_scaled_dot_kkt_fwd` | 60 | 仅 seq 变化 |
| `solve_tril` | 60 | 仅 seq 变化 |
| `recompute_w_u_fwd` | 60 | 仅 seq 变化 |
| `fused_recurrent_gated_delta_rule` | 6600 | 唯一模式（decode） |
| `load_initial_state_from_block_map` | 30 | 唯一模式 |
| `store_ssm_state_to_block_map` | 30 | 唯一模式 |

### 5.2 `causal_conv1d_fn` — 2 模式

**模式区分条件：** `conv_states` 是否为 `None`

#### 模式 A：首次 prefill（`conv_states=None`）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `x` | (8192, seq) | (1, 12288) | float16 | **否** | `buf = torch.randn(seq, 12288, dtype=fp16, device='cuda'); x = buf.t()[:8192, :]` |
| `weight` | (8192, 4) | (4, 1) | float16 | 是 | `torch.randn(8192, 4, dtype=fp16, device='cuda')` |
| `bias` | None | — | — | — | — |
| `conv_states` | None | — | — | — | — |
| `query_start_loc` | (2,) | (1,) | int32 | 是 | `torch.tensor([0, seq], dtype=int32, device='cuda')` |
| `block_map` | None | — | — | — | — |
| `seq_size_per_block` | 1 | — | int | — | 标量 |
| `prefix_lengths` | (1,) | (1,) | int32 | 是 | `torch.tensor([0], dtype=int32, device='cuda')` |

#### 模式 B：增量 prefill（`conv_states` 有值）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `x` | (8192, seq) | (1, 12288) | float16 | **否** | 同模式 A |
| `weight` | (8192, 4) | (4, 1) | float16 | 是 | 同模式 A |
| `bias` | None | — | — | — | — |
| `conv_states` | (block_num, 8192, 3) | **(1048576, 1, 8192)** | float16 | **否** | `buf = torch.empty(block_num, 128, 8192, 3, dtype=fp16, device='cuda'); conv_states = buf[:, 0, :, :].permute(0, 2, 1).contiguous().as_strided(...)` 或 `buf.as_strided((block_num, 8192, 3), (1048576, 1, 8192))` |
| `query_start_loc` | (2,) | (1,) | int32 | 是 | 同模式 A |
| `block_map` | (1, 1) | (1, 1) | int32 | 是 | `torch.zeros(1, 1, dtype=int32, device='cuda')` |
| `seq_size_per_block` | 1024 | — | int | — | 标量 |
| `prefix_lengths` | (1,) | (1,) | int32 | 是 | 同模式 A |

**模式间差异：**

| 差异项 | 模式 A | 模式 B |
|---|---|---|
| `conv_states` | `None` | `(block_num, 8192, 3)`，非连续 |
| `block_map` | `None` | `(1, 1)` |
| `seq_size_per_block` | 1 | 1024 |

> **`x` 的非连续性构造：** `x` 的 stride `(1, 12288)` 表示列主序（Fortran order）。源 buffer shape 为 `(seq, 12288)`，`.t()` 转置后切前 8192 行。12288 = 64 × 192（共享投影 buffer 的列数），与 seq 无关，仅取决于 buffer 的列宽。
>
> **`conv_states` 的非连续性构造：** stride `(1048576, 1, 8192)` 中，dim-1（8192 conv_dim）stride 为 1（内存最内层），dim-2（3 kernel positions）stride 为 8192，dim-0（block_num）stride 为 1048576 = 64 × 16384（状态池每 block 预分配 64 个 head 槽位，实际仅用部分）。测试中可用 `torch.empty(block_num, 1048576, dtype=fp16).as_strided((block_num, 8192, 3), (1048576, 1, 8192))` 构造，或用较小的 `block_num`（如 2）降低内存。

### 5.3 `fused_gdn_gating` — 2 模式

**模式区分条件：** `a`/`b` 的 seq 维度（dim-0）是否为 1

#### 模式 A：prefill（seq > 1）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `A_log` | (32,) | (1,) | float16 | 是 | `torch.randn(32, dtype=fp16, device='cuda')` |
| `a` | (seq, 32) | **(64, 1)** | float16 | **否** | `buf = torch.randn(seq, 64, dtype=fp16, device='cuda'); a = buf[:, :32]` |
| `b` | (seq, 32) | **(64, 1)** | float16 | **否** | 同 `a` |
| `dt_bias` | (32,) | (1,) | float16 | 是 | `torch.randn(32, dtype=fp16, device='cuda')` |

#### 模式 B：decode（seq = 1）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `A_log` | (32,) | (1,) | float16 | 是 | 同模式 A |
| `a` | (1, 32) | (64, 1) | float16 | **是** | `buf = torch.randn(1, 64, dtype=fp16, device='cuda'); a = buf[:, :32]`（seq=1 时 stride 差异不影响 `is_contiguous()`） |
| `b` | (1, 32) | (64, 1) | float16 | **是** | 同 `a` |
| `dt_bias` | (32,) | (1,) | float16 | 是 | 同模式 A |

**模式间差异：**

| 差异项 | 模式 A（prefill） | 模式 B（decode） |
|---|---|---|
| `a`/`b` shape | `(seq, 32)`，如 `(2047, 32)` | `(1, 32)` |
| `a`/`b` `is_contiguous()` | `False` | `True` |

> **`a`/`b` 的非连续性构造：** stride `(64, 1)` 中，dim-0 stride 为 64 而非 32，因为 `a`/`b` 从共享投影 buffer（列宽 64）切片取前 32 列。测试中 `buf = torch.randn(seq, 64); a = buf[:, :32]` 即可复现。模式 B 中虽 stride 仍为 `(64, 1)`，但 PyTorch 对 shape `(1, 32)` 判定 `is_contiguous() == True`，因为首维大小为 1 时 stride 被忽略。
>
> **测试覆盖要点：** 自定义 kernel 需同时处理 prefill（非连续，stride[0]=64）和 decode（连续）两种情况。若 kernel 内部依赖 `is_contiguous()` 分支，需验证 stride[0] != shape[1] 时的行为。

### 5.4 `chunk_gated_delta_rule` — 2 模式

**模式区分条件：** `initial_state` 是否为 `None`

#### 模式 A：零状态初始化（`initial_state=None`）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `q` | (1, seq, 16, 128) | **(seq×8192, 8192, 128, 1)** | float16 | **否** | `buf = torch.randn(1, seq, 64, 128, dtype=fp16, device='cuda'); q = buf[:, :, :16, :]` |
| `k` | (1, seq, 16, 128) | **(seq×8192, 8192, 128, 1)** | float16 | **否** | 同 `q` |
| `v` | (1, seq, 32, 128) | **(seq×8192, 8192, 128, 1)** | float16 | **否** | `buf = torch.randn(1, seq, 64, 128, dtype=fp16, device='cuda'); v = buf[:, :, :32, :]` |
| `g` | (1, seq, 32) | (seq×32, 32, 1) | float32 | 是 | `torch.randn(1, seq, 32, dtype=fp32, device='cuda')` |
| `beta` | (1, seq, 32) | (seq×32, 32, 1) | float16 | 是 | `torch.randn(1, seq, 32, dtype=fp16, device='cuda')` |
| `initial_state` | None | — | — | — | — |
| `output_final_state` | — | — | bool | — | `True` |
| `cu_seqlens` | (2,) | (1,) | int32 | 是 | `torch.tensor([0, seq], dtype=int32, device='cuda')` |
| `use_qk_l2norm_in_kernel` | — | — | bool | — | `True` |

#### 模式 B：续算状态（`initial_state` 有值）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `q` | (1, seq, 16, 128) | **(seq×8192, 8192, 128, 1)** | float16 | **否** | 同模式 A |
| `k` | (1, seq, 16, 128) | **(seq×8192, 8192, 128, 1)** | float16 | **否** | 同模式 A |
| `v` | (1, seq, 32, 128) | **(seq×8192, 8192, 128, 1)** | float16 | **否** | 同模式 A |
| `g` | (1, seq, 32) | (seq×32, 32, 1) | float32 | 是 | 同模式 A |
| `beta` | (1, seq, 32) | (seq×32, 32, 1) | float16 | 是 | 同模式 A |
| `initial_state` | (1, 32, 128, 128) | (524288, 16384, 128, 1) | bfloat16 | 是 | `torch.randn(1, 32, 128, 128, dtype=bf16, device='cuda')` |
| `output_final_state` | — | — | bool | — | `True` |
| `cu_seqlens` | (2,) | (1,) | int32 | 是 | 同模式 A |
| `use_qk_l2norm_in_kernel` | — | — | bool | — | `True` |

**模式间差异：**

| 差异项 | 模式 A（零状态） | 模式 B（续算状态） |
|---|---|---|
| `initial_state` | `None` | `(1, 32, 128, 128)`，bfloat16，连续 |

> **`q`/`k`/`v` 的非连续性构造：** stride[1] = 8192 = 64 × 128，而非各自 head 数 × head_dim（q/k 应为 16×128=2048，v 应为 32×128=4096）。这是因为三者共享同一个 QKV 投影 buffer（64 heads × 128 head_dim = 8192 elements/token），通过 head 维度切片得到。测试中 `buf = torch.randn(1, seq, 64, 128); q = buf[:, :, :16, :]` 即可复现 stride。
>
> **测试覆盖要点：** 模式 A 需验证 `initial_state=None` 时的零初始化路径；模式 B 需验证非零 `initial_state` 的续算路径。`initial_state` 的 dtype 为 bfloat16，注意与 q/k/v 的 float16 区分。

### 5.5 `chunk_fwd_o` — 2 模式

**模式区分条件：** `h` 的 dim-1（chunk 数）= ⌈seq / 64⌉

#### 模式 A：多 chunk（seq > chunk_size=64）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `q` | (1, seq, 16, 128) | (seq×2048, 2048, 128, 1) | float16 | 是 | `torch.randn(1, seq, 16, 128, dtype=fp16, device='cuda')` |
| `k` | (1, seq, 16, 128) | (seq×2048, 2048, 128, 1) | float16 | 是 | 同 `q` |
| `v` | (1, seq, 32, 128) | (seq×4096, 4096, 128, 1) | float16 | 是 | `torch.randn(1, seq, 32, 128, dtype=fp16, device='cuda')` |
| `h` | (1, **num_chunks**, 32, 128, 128) | (num_chunks×524288, 524288, 16384, 128, 1) | float32 | 是 | `torch.randn(1, num_chunks, 32, 128, 128, dtype=fp32, device='cuda')` |
| `g` | (1, seq, 32) | (seq×32, 32, 1) | float32 | 是 | `torch.randn(1, seq, 32, dtype=fp32, device='cuda')` |
| `scale` | — | — | float | — | 标量 `0.0884`（= 1/√128 × √(1/8)） |
| `cu_seqlens` | (2,) | (1,) | int32 | 是 | `torch.tensor([0, seq], dtype=int32, device='cuda')` |

#### 模式 B：单 chunk（seq ≤ chunk_size=64）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `q` | (1, seq, 16, 128) | (seq×2048, 2048, 128, 1) | float16 | 是 | 同模式 A |
| `k` | (1, seq, 16, 128) | (seq×2048, 2048, 128, 1) | float16 | 是 | 同模式 A |
| `v` | (1, seq, 32, 128) | (seq×4096, 4096, 128, 1) | float16 | 是 | 同模式 A |
| `h` | (1, **1**, 32, 128, 128) | (524288, 524288, 16384, 128, 1) | float32 | 是 | `torch.randn(1, 1, 32, 128, 128, dtype=fp32, device='cuda')` |
| `g` | (1, seq, 32) | (seq×32, 32, 1) | float32 | 是 | 同模式 A |
| `scale` | — | — | float | — | 同模式 A |
| `cu_seqlens` | (2,) | (1,) | int32 | 是 | 同模式 A |

**模式间差异：**

| 差异项 | 模式 A（多 chunk） | 模式 B（单 chunk） |
|---|---|---|
| seq | > 64（如 2047） | ≤ 64（如 13） |
| `h` dim-1（chunk 数） | ⌈seq/64⌉（如 32） | 1 |
| `h` shape | `(1, 32, 32, 128, 128)` | `(1, 1, 32, 128, 128)` |

> **关键区别：** `h` 的 dim-1 = num_chunks = ⌈seq / chunk_size⌉（chunk_size=64）。这不是简单的 shape 变化，而是算法路径不同：多 chunk 时 kernel 需要逐 chunk 传递 inter-chunk 状态（循环 num_chunks 次），单 chunk 时无跨 chunk 计算。
>
> **测试覆盖要点：** 需覆盖 seq > 64（如 seq=128 → 2 chunks）和 seq ≤ 64（如 seq=13 → 1 chunk）两种路径。注意 `h` 的 dim-1 随 seq 变化，stride[0] 和 stride[1] 在模式 B 中相同（均为 524288），因为 dim-1 大小为 1 时 stride 被忽略。所有输入均为连续，无需特殊 stride 处理。

### 5.6 `RmsNormGated` — 2 模式

**模式区分条件：** `x`/`gate` 的 dim-0 是否等于 32（= value_heads）

#### 模式 A：prefill（seq > 1）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `x` | (seq×32, 128) | (128, 1) | float16 | 是 | `torch.randn(seq×32, 128, dtype=fp16, device='cuda')` |
| `gate` | (seq×32, 128) | (128, 1) | float16 | 是 | 同 `x` |

#### 模式 B：decode（seq = 1）

| 参数 | Shape | Stride | Dtype | 连续 | 构造方法 |
|---|---|---|---|---|---|
| `x` | (32, 128) | (128, 1) | float16 | 是 | `torch.randn(32, 128, dtype=fp16, device='cuda')` |
| `gate` | (32, 128) | (128, 1) | float16 | 是 | 同 `x` |

**模式间差异：**

| 差异项 | 模式 A（prefill） | 模式 B（decode） |
|---|---|---|
| `x`/`gate` shape | `(seq×32, 128)`，如 `(65504, 128)` | `(32, 128)` |
| dim-0 | seq × value_heads | value_heads |

> **关键区别：** dim-0 = seq × 32（value_heads），是将 `(seq, num_value_heads, head_dim)` 展平为 2D。所有输入均为连续，stride 始终为 `(128, 1)`。
>
> **测试覆盖要点：** 此算子两模式仅 dim-0 大小不同，stride/dtype/连续性完全一致。测试中只需变化 dim-0（seq×32 vs 32）。模型参数（weight、eps 等）由 `RmsNormGated` 类实例持有，不随模式变化。

### 5.7 总结

**多模式算子（5 个）：**

| 算子 | 模式区分条件 | 模式间关键差异 | 测试需覆盖 |
|---|---|---|---|
| `causal_conv1d_fn` | `conv_states` 是否为 None | 模式 B 额外传入非连续 `conv_states` (293,8192,3)、`block_map`、`seq_size_per_block=1024` | 两种：`conv_states=None` 和 `conv_states` 有值 |
| `fused_gdn_gating` | `a`/`b` dim-0 是否为 1 | 模式 A 非连续（stride 64≠32），模式 B 连续（seq=1） | 两种：seq>1 非连续 + seq=1 连续 |
| `chunk_gated_delta_rule` | `initial_state` 是否为 None | 模式 B 额外传入 `initial_state` (1,32,128,128) bf16 | 两种：`None` 和有值 |
| `chunk_fwd_o` | seq 是否 > chunk_size(64) | `h` dim-1 = ⌈seq/64⌉，多 chunk vs 单 chunk 算法路径不同 | 两种：seq>64 + seq≤64 |
| `RmsNormGated` | `x` dim-0 是否 > 32 | 仅 dim-0 大小不同（seq×32 vs 32），stride/dtype 一致 | 两种：prefill 大 + decode 小 |

**单一模式算子（9 个）：** `causal_conv1d_update`、`prepare_causal_conv1d_metadata`、`chunk_local_cumsum`、`chunk_scaled_dot_kkt_fwd`、`solve_tril`、`recompute_w_u_fwd`、`fused_recurrent_gated_delta_rule`、`load_initial_state_from_block_map`、`store_ssm_state_to_block_map`。所有调用的参数结构一致，仅 seq 维大小变化。

---

## 6. 模型架构参数

| 参数 | 值 |
|---|---|
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 32 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| 总层数 | 40（30层线性 + 10层全注意力） |
| 线性注意力层分布 | 每 4 层中第 4 层为全注意力（第 3, 7, 11, ..., 39 层） |
| `seq_size_per_block` | 1024 |
| 专家数 | 256（MoE） |
| conv_dim | 8192（= 64 heads × 128 head_dim） |
| 状态池 block 数 | 293 |

---

## 7. 相关文件

| 文件 | 用途 |
|---|---|
| `/home/zym/wzb/ssm_ops_profile.log` | 原始 profiling 日志（约 243K 行，~25000 次算子调用） |
| `/home/zym/wzb/profile_ssm_ops.py` | Monkey-patch hook（14 算子完整版，profiling 模式） |
| `/home/zym/wzb/capture_ssm_ops.py` | Monkey-patch hook（14 算子，capture 模式，紧凑存储 + stride 元数据） |
| `/home/zym/wzb/ssm_profiler_init.py` | 子进程 import hook（支持 `CAPTURE_MODE=1` 环境变量切换） |
| `/home/zym/wzb/ssm_profiling_report.md` | 本报告 |
| `/home/zym/wzb/run_qwen35_moe.sh` | 推理启动脚本（支持指定 GPU） |
| `/home/zym/wzb/run_nsys.sh` | nsys GPU profiling 脚本 |
| `/home/zym/wzb/sample/` | 捕获的 NPU 测试用例（25 个 `.pt` 文件，2.3 GB） |

---

## 8. 建议

1. **全部 14 个算子均已成功捕获。** 其中 7 个算子接收非连续输入。任何自定义 kernel 实现必须处理：
   - **Channel-last 2D 布局**：stride `(1, 12288)` 对 shape `(8192, seq)` — conv1d prefill 的 `x` 参数
   - **列 stride 不匹配**：stride `(64, 1)` 对 shape `(seq, 32)` — GDN gating prefill 的 `a`/`b` 参数
   - **Seq 维 stride 填充**：stride[1] = 8192 对 16 或 32 heads — SSM prefill 的 `q`/`k`/`v`
   - **状态池 block stride 填充**：stride[0] = 2× 期望值 — 所有 decode 阶段的状态参数

2. **chunk 子算子（5 个）全部接收连续输入**，因为 `chunk_gated_delta_rule` 在入口处做了 rearrange 消除非连续性。无需额外处理。

3. **Decode 阶段的 q/k/v 是连续的**（seq=1），唯一非连续性来自状态池 `initial_state`/`conv_state`。

4. **非连续性是结构性的**，来源于 rtp-llm 的 buffer 管理策略。添加 `.contiguous()` 调用可以解决但引入内存拷贝。

---

## 9. 算子数据捕获（NPU 等价性测试用例）

### 9.1 捕获系统概述

捕获系统通过 monkey-patch 拦截全部 14 个算子的函数调用，在**首次出现每种调用模式**时保存输入/输出快照。捕获系统解决了三个关键技术问题：

1. **非连续 stride 丢失问题**：PyTorch `.clone()` 会将「切片型非连续」tensor（如 `buf[:, :32]` stride `(64,1)`）归一化为连续 tensor（stride `(32,1)`），丢失非连续信息。解决方案：tensor 数据以 `.contiguous()` 紧凑保存，原始 stride 保存在 `input_meta` 字段中，NPU 端通过 `torch.empty_strided()` + `.copy_()` 重建。
2. **RmsNormGated 权重丢失**：`self.weight` 是普通属性（非 `nn.Parameter`），不会被 `named_parameters()` 捕获。解决方案：显式遍历 `weight`、`bias`、`eps`、`group_size` 属性。
3. **CausalConv1dMetadata 不可移植**：原始对象依赖 rtp_llm 模块。解决方案：dataclass 序列化为 `{"__dataclass__": ..., "fields": {...}}` 字典格式，全部 `.pt` 文件可用 `torch.load(weights_only=True)` 加载，无需安装 rtp_llm。

### 9.2 捕获的测试用例

共 **25 个 `.pt` 文件**，覆盖全部 14 个算子的所有调用模式：

| 算子 | 模式 | 文件 | 非连续输入 | In-place 修改 |
|---|---|---|---|---|
| `causal_conv1d_fn` | prefill_first_seq2047 | `causal_conv1d_fn/prefill_first_seq2047.pt` | x: stride `(1,12288)` | — |
| `causal_conv1d_fn` | prefill_incr_seq32 | `causal_conv1d_fn/prefill_incr_seq32.pt` | x, conv_states: stride `(1048576,1,8192)` | conv_states ✓ |
| `causal_conv1d_update` | decode | `causal_conv1d_update/decode.pt` | conv_state: stride `(1048576,1,8192)` | conv_state ✓ |
| `prepare_causal_conv1d_metadata` | prefill | `prepare_causal_conv1d_metadata/prefill.pt` | — | — |
| `fused_gdn_gating` | prefill_seq2047 | `fused_gdn_gating/prefill_seq2047.pt` | a, b: stride `(64,1)` | — |
| `fused_gdn_gating` | prefill_seq32 | `fused_gdn_gating/prefill_seq32.pt` | a, b: stride `(64,1)` | — |
| `fused_gdn_gating` | decode_seq1 | `fused_gdn_gating/decode_seq1.pt` | — | — |
| `chunk_gated_delta_rule` | prefill_zero_state_seq2047 | `chunk_gated_delta_rule/prefill_zero_state_seq2047.pt` | q,k,v: stride `(...,8192,128,1)` | — |
| `chunk_gated_delta_rule` | prefill_loaded_state_seq32 | `chunk_gated_delta_rule/prefill_loaded_state_seq32.pt` | q,k,v: stride `(...,8192,128,1)` | — |
| `chunk_local_cumsum` | seq2047 | `chunk_local_cumsum/seq2047.pt` | — | — |
| `chunk_local_cumsum` | seq32 | `chunk_local_cumsum/seq32.pt` | — | — |
| `chunk_scaled_dot_kkt_fwd` | seq2047 | `chunk_scaled_dot_kkt_fwd/seq2047.pt` | — | — |
| `chunk_scaled_dot_kkt_fwd` | seq32 | `chunk_scaled_dot_kkt_fwd/seq32.pt` | — | — |
| `solve_tril` | seq2047 | `solve_tril/seq2047.pt` | — | — |
| `solve_tril` | seq32 | `solve_tril/seq32.pt` | — | — |
| `recompute_w_u_fwd` | seq2047 | `recompute_w_u_fwd/seq2047.pt` | — | — |
| `recompute_w_u_fwd` | seq32 | `recompute_w_u_fwd/seq32.pt` | — | — |
| `chunk_fwd_o` | multi_chunk_seq2047_32chunks | `chunk_fwd_o/multi_chunk_seq2047_32chunks.pt` | — | — |
| `chunk_fwd_o` | single_chunk_seq32_1chunks | `chunk_fwd_o/single_chunk_seq32_1chunks.pt` | — | — |
| `fused_recurrent_gated_delta_rule` | decode | `fused_recurrent_gated_delta_rule/decode.pt` | initial_state: stride `(1048576,16384,128,1)` | initial_state ✓ |
| `load_initial_state_from_block_map` | prefill | `load_initial_state_from_block_map/prefill.pt` | conv_states: stride `(1048576,16384,128,1)` | initial_states ✓ |
| `store_ssm_state_to_block_map` | prefill | `store_ssm_state_to_block_map/prefill.pt` | ssm_states: stride `(1048576,16384,128,1)` | ssm_states ✓ |
| `RmsNormGated` | prefill_seq2047 | `RmsNormGated/prefill_seq2047.pt` | — | — |
| `RmsNormGated` | prefill_seq32 | `RmsNormGated/prefill_seq32.pt` | — | — |
| `RmsNormGated` | decode_seq1 | `RmsNormGated/decode_seq1.pt` | — | — |

### 9.3 `.pt` 文件数据格式

每个 `.pt` 文件是一个字典，包含以下字段：

```python
{
    "op_name": str,           # 算子名称
    "mode": str,              # 调用模式名称
    "param_names": list[str], # 参数名列表（按位置参数顺序）
    "inputs": dict,           # 调用前输入参数（tensor 已 .contiguous()）
    "outputs": tensor|list,   # 算子返回值
    "inplace_outputs": dict,  # 被 in-place 修改的参数（调用后快照）
    "input_meta": dict,       # 每个 tensor 参数的原始 shape/stride/dtype/contiguous
    "model_state": dict,      # 仅 RmsNormGated：weight, eps, group_size
}
```

`input_meta` 格式示例：
```python
{
    "x": {"shape": (8192, 2047), "stride": (1, 12288), "dtype": "torch.float16", "contiguous": False},
    "weight": {"shape": (8192, 4), "stride": (4, 1), "dtype": "torch.float16", "contiguous": True},
}
```

### 9.4 NPU 端非连续 Tensor 重建方法

对于 `input_meta` 中 `contiguous: False` 的参数，需在 NPU 端重建非连续 stride：

```python
import torch

def reconstruct_tensor(contiguous_data, meta):
    """从紧凑数据 + stride 元数据重建非连续 tensor"""
    if meta["contiguous"]:
        return contiguous_data
    dtype = getattr(torch, meta["dtype"].replace("torch.", ""))
    tensor = torch.empty_strided(meta["shape"], meta["stride"], dtype=dtype)
    tensor.copy_(contiguous_data)  # 按 stride 填充数据
    return tensor
```

**重建验证**：重建后的 tensor 与原始 tensor 逐元素相等（`torch.equal` 返回 `True`），stride 完全一致，`is_contiguous()` 返回 `False`。

### 9.5 非连续 stride 模式总结

捕获确认了 **7 个算子接收非连续输入**，涉及 **3 种非连续根因**：

| 根因 | stride 模式 | 影响参数 | 涉及算子 |
|---|---|---|---|
| **共享投影 buffer 切片** | dim-1 stride > dim-1 size | q/k/v: stride[1]=8192, a/b: stride[0]=64 | `chunk_gated_delta_rule`, `fused_gdn_gating` |
| **Channel-last 转置** | stride=(1, large) | x: stride=(1, 12288) | `causal_conv1d_fn` |
| **状态池 buffer 共享** | stride[0]=1048576 (block stride) | conv_state(s), ssm_states, initial_state | `causal_conv1d_fn`, `causal_conv1d_update`, `fused_recurrent_gated_delta_rule`, `load/store_*_block_map` |

### 9.6 In-place 修改检测

5 个算子存在 in-place 修改，均已捕获 pre/post 快照：

| 算子 | 被修改参数 | 说明 |
|---|---|---|
| `causal_conv1d_fn` | `conv_states` | 增量 prefill 时更新卷积状态 |
| `causal_conv1d_update` | `conv_state` | decode 时更新卷积状态 |
| `fused_recurrent_gated_delta_rule` | `initial_state` | decode 时更新 SSM 状态 |
| `load_initial_state_from_block_map` | `initial_states` | 从 block_map 加载状态 |
| `store_ssm_state_to_block_map` | `ssm_states` | 将状态写回 block_map |

### 9.7 数据质量验证结果

- **NaN/Inf 检查**：全部 25 个文件的输入/输出 tensor 无 NaN/Inf ✓
- **空 tensor 检查**：无空 tensor ✓
- **weights_only 加载**：25/25 文件可用 `torch.load(weights_only=True)` 加载 ✓
- **总大小**：2.3 GB（`fused_recurrent_gated_delta_rule/decode.pt` 最大，879 MB，因包含 293-block 完整状态池）
- **数据正确性**：所有 stride 与 profiling 报告第 4 章记录一致 ✓
