# Ascend NPU Decode Attention 修复记录

## 基本信息

- **日期**: 2026-05-28
- **模型**: Qwen3-0.6B
- **硬件**: Ascend NPU
- **修改文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

---

## 问题描述

启动 Qwen3-0.6B 推理服务后，decode 阶段触发 NPU aicore 异常，进程崩溃：

```
EZ9999: Inner Error!
The MPU address access is invalid.
fixp_error0 info: 0xb562a6c
Kernel task happen error, retCode=0x26, [aicore exception.]
```

错误发生在 `ascend_decode.py:132`，调用 `torch_npu._npu_paged_attention` 时。

---

## 根因分析

### 1. `_npu_paged_attention` 存在已知 aicore 缺陷

`torch_npu._npu_paged_attention` 是华为 CANN 提供的专用 paged attention 算子，在某些 block_table 配置下会触发 NPU aicore 地址越界错误（`MPU address access is invalid`）。这是 CANN 层面的已知问题：

- vllm-ascend 也有同样的报错（GitHub issue [#3263](https://github.com/vllm-project/vllm-ascend/issues/3263)），标记为 wontfix
- vllm-ascend 已将 `npu_fused_infer_attention_score`（FIA）作为 decode 的**默认主路径**，`_npu_paged_attention` 仅作为可选加速路径，且有严格的 shape/device 限制

### 2. vllm-ascend 的解决方案

vllm-ascend 提供了两种 decode attention 实现路径：

| 路径 | API | 状态 |
|------|-----|------|
| FIA（默认） | `torch_npu.npu_fused_infer_attention_score` + TND layout + block_table | 稳定，prefill/decode 通用 |
| PA（可选） | `torch_npu._npu_paged_attention` | 有 aicore 风险，仅限特定 shape |

本项目 prefill 阶段已经使用 FIA（`ascend_prefill.py`），decode 阶段应统一采用相同方案。

---

## 解决方案

将 `AscendDecodeAttnOp` 的 decode 实现从 `torch_npu._npu_paged_attention` 替换为 `torch_npu.npu_fused_infer_attention_score`（FIA），采用 TND layout + block_table 方式，与 prefill 路径对齐。

### 关键改动点

1. **KV cache 布局转换**: 从 NHD `[blocks, seq, kv_heads, dim]` permute 到 HND `[blocks, kv_heads, seq, dim]`，与 prefill 一致
2. **actual_seq_lengths 计算**: decode 阶段每个请求只有 1 个 query token，`actual_seq_q` 应基于 `q.shape[0]`（= batch_size）动态计算为 `arange(1, batch_size+1)`，而非使用 `input_lengths` 的 cumsum
3. **actual_seq_lengths_kv 计算**: 直接对 `context_lens`（= prefix_lengths + input_lengths，即 KV 序列总长度）做 cumsum
4. **FIA API 参数**: `input_layout="TND"`, `block_size=page_size`, `sparse_mode=3`（causal mask）

---

## 修改代码

### 修改前（原始 `AscendDecodeAttnOp`）

```python
class AscendDecodeAttnOp:
    """Encapsulate NPU decode paged attention op, reads cache only."""

    def __init__(self, attn_configs, attn_inputs):
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.q_scaling * self.head_dim ** -0.5
        self.page_size = attn_inputs.kv_cache.seq_size_per_block if \
                         attn_inputs.kv_cache else 128
        self.block_table = None
        self.context_lens = None

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_kernel_block_id_host
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
            if self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
        self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths

    def forward(self, q, kv_cache):
        block_table = self.block_table
        context_lens = self.context_lens
        k_cache = kv_cache.k_cache_base.contiguous()
        v_cache = kv_cache.v_cache_base.contiguous()
        out = torch.empty_like(q)
        torch_npu._npu_paged_attention(
            q, k_cache, v_cache,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            block_table,
            context_lens,
            out,
        )
        return out
```

### 修改后（当前 `AscendDecodeAttnOp`）

```python
class AscendDecodeAttnOp:
    """Encapsulate NPU decode attention using npu_fused_infer_attention_score.

    Uses FIA with TND layout + block_table (same as vllm-ascend default path)
    instead of _npu_paged_attention which has known aicore errors on certain
    block_table configurations.
    """

    _causal_mask = None

    @classmethod
    def _get_causal_mask(cls, device):
        if cls._causal_mask is None or cls._causal_mask.device.type != device.type:
            cls._causal_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.int8), diagonal=1
            ).to(device)
        return cls._causal_mask

    def __init__(self, attn_configs, attn_inputs):
        self.num_heads = attn_configs.head_num
        self.num_kv_heads = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.scale = attn_configs.q_scaling * self.head_dim ** -0.5
        self.page_size = attn_inputs.kv_cache.seq_size_per_block if \
                         attn_inputs.kv_cache else 128
        self.block_table = None
        self.context_lens = None

    def set_params(self, params):
        self.params = params

    def prepare(self, attn_inputs):
        self.block_table = attn_inputs.kv_cache_kernel_block_id_host
        if self.block_table is not None:
            self.block_table = self.block_table.clamp(min=0)
            if self.block_table.ndim != 2:
                self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
        self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths

    def forward(self, q, kv_cache):
        k_cache = kv_cache.k_cache_base.permute(0, 2, 1, 3).contiguous()
        v_cache = kv_cache.v_cache_base.permute(0, 2, 1, 3).contiguous()
        block_table = self.block_table
        if block_table is not None and block_table.device.type != q.device.type:
            block_table = block_table.to(q.device)
        context_lens = self.context_lens
        if context_lens is not None and context_lens.device.type != q.device.type:
            context_lens = context_lens.to(q.device)
        batch_size = q.shape[0]
        actual_seq_q = torch.arange(
            1, batch_size + 1, dtype=torch.int32, device=q.device
        )
        actual_seq_kv = torch.cumsum(context_lens.to(torch.int32), dim=0)
        if actual_seq_kv.device.type != q.device.type:
            actual_seq_kv = actual_seq_kv.to(q.device)
        atten_mask = self._get_causal_mask(q.device)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=self.page_size,
            actual_seq_lengths=actual_seq_q,
            actual_seq_lengths_kv=actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        return attn_output
```

---

## 修改要点对比

| 项目 | 修改前 (`_npu_paged_attention`) | 修改后 (`npu_fused_infer_attention_score`) |
|------|------|------|
| KV cache 布局 | NHD: `[blocks, seq, kv_heads, dim]` 直接使用 | HND: `[blocks, kv_heads, seq, dim]` 需 permute |
| Query 序列长度 | 通过 `context_lens` 隐式传递 | 显式通过 `actual_seq_lengths` (cumsum of per-request query token counts) |
| KV 序列长度 | 通过 `context_lens` 隐式传递 | 显式通过 `actual_seq_lengths_kv` (cumsum of total KV lengths) |
| Causal mask | 算子内部处理 | 需外部提供 `atten_mask` + `sparse_mode=3` |
| 输出分配 | 手动 `torch.empty_like(q)` | FIA 内部分配并返回 |
| 稳定性 | 有 aicore 地址越界风险 | 稳定，为 vllm-ascend 默认路径 |

---

## 踩坑记录：`actual_seq_lengths` 计算错误

初次修改时，`actual_seq_q` 使用了 `cumsum(input_lengths)` 计算，导致报错：

```
when query's layout is TND, T(1) should be equal to the last element
of the query's actual sequence lengths(9).
```

**原因**: 在 rtp-llm 框架中，decode 阶段的 `input_lengths` 不是每请求的 query token 数（=1），而是累积序列信息。`q.shape[0]` 才是实际的 query token 总数（decode 时 = batch_size，每个请求贡献 1 个 token）。

**修复**: `actual_seq_q` 改为在 `forward()` 中根据 `q.shape[0]` 动态计算：

```python
batch_size = q.shape[0]
actual_seq_q = torch.arange(1, batch_size + 1, dtype=torch.int32, device=q.device)
```

这确保了 FIA 要求的约束：`T (q.shape[0]) == actual_seq_lengths[-1]`。
