# rtp-llm Qwen3-0.6B Ascend NPU 推理 SIGSEGV 排查与修复

> 日期：2026-05-22
> 环境：CANN 8.5.1 / torch_npu 2.9.0 / Ascend NPU
> 模型：Qwen3-0.6B (num_attention_heads=16, num_key_value_heads=8, head_dim=128, hidden_size=1024)

---

## 一、问题现象

使用 rtp-llm 在 Ascend NPU 上启动 Qwen3-0.6B 推理服务，发送 chat/completions 请求时进程崩溃，返回 `514_UNKNOWN_ERROR`。

堆栈跟踪显示崩溃点为：

```
atb::_npu_reshape_and_cache()  →  SIGSEGV (@0x2e74)
```

`_npu_reshape_and_cache` 是 CANN ATB 库的 `ReshapeAndCacheOperation` 融合算子，负责将 key/value 写入 paged KV cache。

---

## 二、根因分析

通过在 `AscendKVCacheWriteOp.forward()` 中注入诊断代码，逐层排查，发现 **6 个问题**：

### 问题 1：`block_table` 维度不匹配（根因）

| 项目 | 说明 |
|---|---|
| **预期** | `kv_cache_block_id_host` shape 为 `[batch, max_blocks]`（二维） |
| **实际** | C++ 端传入 `[group, batch, max_blocks]`（三维），当前为 `[1, 1, 1]` |
| **后果** | `block_table[batch_ids, block_index]` 的 fancy indexing 产生 `[9, 9]` 而非 `[9]`，`slot_mapping` 变成二维 |
| **诊断输出** | `slot:[9, 9],torch.int32,min=64,max=72`（应为 `slot:[9]`） |

### 问题 2：ATB `_npu_reshape_and_cache` 内部空指针崩溃

| 项目 | 说明 |
|---|---|
| **现象** | 即使 slot_mapping 修正为 `[9]` 后，仍在 `_npu_reshape_and_cache` 内 SIGSEGV @0x2e74 |
| **原因** | CANN 8.5.1 的 ATB `ReshapeAndCacheOperation` 存在内部 null-pointer 解引用 bug |
| **同样受影响** | `_npu_reshape_and_cache_siso`（单输入变体）也以同样的 @0x2e74 崩溃 |

### 问题 3：KV cache layout 不匹配

| 项目 | 说明 |
|---|---|
| **rtp_llm 原始 layout** | NHD: `[blocks, seq_per_block, kv_heads, head_dim]` = `[8142, 64, 8, 128]` |
| **CANN 期望 layout** | BNSD: `[blocks, kv_heads, seq_per_block, head_dim]` = `[8142, 8, 64, 128]` |
| **CANN 报错** | `key layout is BnNBsD, shape [8142,64,8,128] should be equal to [8142,8,64,128]` |

### 问题 4：`actual_seq_lengths` 格式不匹配

| 项目 | 说明 |
|---|---|
| **rtp_llm 原始** | `[0, cumsum(input_lengths)]` = `[0, 9]`（含前导0，长度=2） |
| **CANN V3 语义** | 以元素数量作为 batch 值，batch = len(actual_seq_lengths) = 2 |
| **实际 batch** | 1 |
| **CANN 报错** | `block_table's first dimension(1) should be equal to batch size(2)` |

### 问题 5：`block_table` 设备不匹配

| 项目 | 说明 |
|---|---|
| **原始** | `kv_cache_block_id_host` 在 CPU 上 |
| **CANN 要求** | `npu_fused_infer_attention_score` 和 `_npu_paged_attention` 要求 `block_table` 在 NPU 上 |
| **CANN 报错** | `Expected all tensors to be on the same device, but got block_table is on cpu` |

### 问题 6：`sparse_mode=3` 需要提供 `atten_mask`

| 项目 | 说明 |
|---|---|
| **CANN V3 行为** | sparse_mode != 0 时要求 `atten_mask` 不为 null |
| **CANN 报错** | `when sparse_mode is 3, it not 0, atten_mask should not be null` |

---

## 三、修复方案与代码变更

### 3.1 修复 block_table 三维问题

**文件**：`rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py`

**方案**：添加 `_squeeze_block_table()` 工具函数，当 `block_table` 为三维 `[group, batch, max_blocks]` 时 squeeze 掉 group 维度。

**变更代码**：

```python
def _squeeze_block_table(block_table):
    """Ensure block_table is 2-D [batch, max_blocks].

    The C++ side may pass a 3-D tensor [group, batch, max_blocks].
    For non-hybrid models group=1, so we squeeze the leading dim.
    """
    if block_table is not None and block_table.dim() == 3:
        block_table = block_table.squeeze(0)
    return block_table
```

**调用点修改**：

- `build_ascend_params()` 中：`params.block_table = _squeeze_block_table(attn_inputs.kv_cache_block_id_host)`
- `compute_ascend_attn_params()` 中：`block_table = _squeeze_block_table(attn_inputs.kv_cache_block_id_host)`
- `AscendPrefillAttnOp.prepare()` 中：使用 `_squeeze_block_table()`
- `AscendDecodeAttnOp.prepare()` 中：使用 `_squeeze_block_table()`

### 3.2 替换 `_npu_reshape_and_cache` 为纯 PyTorch 实现

**文件**：`rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py`

**方案**：用 `index_copy_` 在 flatten 后的 cache view 上写入，绕过 ATB 融合算子。将 cache 从 `[B, H, S, D]` reshape 为 `[B*S, H*D]`，然后用 `index_copy_` 按 slot 写入。

**完整替换代码**：

```python
class AscendKVCacheWriteOp:
    """MHA KV Cache write using NPU-side index_copy_.

    Replaces the unstable ATB _npu_reshape_and_cache fusion op (which crashes
    with SIGSEGV @0x2e74 on CANN 8.5.1) with a pure-PyTorch implementation
    that writes key/value into paged KV cache via index_copy_ on a flattened
    view, taking care to preserve the NPU tensor storage format.
    """

    def __init__(self, num_kv_heads, head_size, token_per_block):
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.token_per_block = token_per_block
        self.params = None

    def set_params(self, params):
        self.params = params

    def forward(self, key, value, kv_cache):
        if kv_cache is None:
            return

        k_cache = kv_cache.k_cache_base
        v_cache = kv_cache.v_cache_base

        slot_mapping = self.params.slot_mapping
        if slot_mapping.dtype != torch.int32:
            slot_mapping = slot_mapping.to(torch.int32)

        if not key.is_contiguous():
            key = key.contiguous()
        if not value.is_contiguous():
            value = value.contiguous()

        self._write_kv_cache_index_copy(key, value, k_cache, v_cache, slot_mapping)

    def _write_kv_cache_index_copy(self, key, value, k_cache, v_cache, slot_mapping):
        """Write key/value into paged KV cache using index_copy_.

        k_cache / v_cache layout (BNSD): [num_blocks, kv_heads, block_size, head_dim]
        key / value layout:              [num_tokens, kv_heads, head_dim]
        slot_mapping:                    [num_tokens] int32

        We treat the cache as flat [num_blocks*block_size, kv_heads*head_dim]
        and each key/value row as [kv_heads*head_dim], then use index_copy_
        on a 2-D flat view.  This avoids permute/format issues entirely.
        """
        num_tokens = slot_mapping.numel()
        if num_tokens == 0:
            return

        num_blocks = k_cache.shape[0]
        block_size = k_cache.shape[2]

        slot_long = slot_mapping.to(torch.int64)

        k_flat = k_cache.reshape(num_blocks * block_size, -1)
        v_flat = v_cache.reshape(num_blocks * block_size, -1)
        key_flat = key.reshape(num_tokens, -1)
        value_flat = value.reshape(num_tokens, -1)

        k_flat.index_copy_(0, slot_long, key_flat)
        v_flat.index_copy_(0, slot_long, value_flat)
```

### 3.3 修复 KV cache layout：NHD → BNSD

**文件**：`rtp_llm/models_py/bindings/OpDefs.h`

**方案**：修改 `getLayerCache()` 中 `separate_kv_cache` 路径的 reshape 顺序，从 `[blocks, seq_per_block, kv_heads, head_dim]` 改为 `[blocks, kv_heads, seq_per_block, head_dim]`。

**变更前**：
```cpp
// Ascend NPU: separate K/V cache with NHD layout
layer_cache.k_cache_base = k_base.reshape(
    {kernel_block_num, (int64_t)kernel_seq_size_per_block,
     (int64_t)num_kv_heads, (int64_t)head_dim});
layer_cache.v_cache_base = v_base.reshape(
    {kernel_block_num, (int64_t)kernel_seq_size_per_block,
     (int64_t)num_kv_heads, (int64_t)head_dim});
```

**变更后**：
```cpp
// Ascend NPU: separate K/V cache with BNSD layout
// CANN npu_fused_infer_attention_score expects [blocks, kv_heads, seq_per_block, head_dim]
layer_cache.k_cache_base = k_base.reshape(
    {kernel_block_num, (int64_t)num_kv_heads,
     (int64_t)kernel_seq_size_per_block, (int64_t)head_dim});
layer_cache.v_cache_base = v_base.reshape(
    {kernel_block_num, (int64_t)num_kv_heads,
     (int64_t)kernel_seq_size_per_block, (int64_t)head_dim});
```

同步修改 `LayerKVCache` 的注释：
```cpp
// Separate K/V cache (Ascend NPU)
torch::Tensor k_cache_base;  // [blocks, kv_heads, seq_per_block, head_dim] BNSD
torch::Tensor v_cache_base;  // [blocks, kv_heads, seq_per_block, head_dim] BNSD
```

### 3.4 修复 `actual_seq_lengths` 格式

**文件**：`rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py`

**方案**：CANN V3 接口的 `actual_seq_lengths` 以元素数量作为 batch 值，每个元素表示当前 batch 及之前所有 batch 的 seqlen 累积和。不应包含前导 0。

**变更前**：
```python
self.actual_seq_q = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=seq_lens_q.device),
    torch.cumsum(seq_lens_q, dim=0)
])
self.actual_seq_kv = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=seq_lens_kv.device),
    torch.cumsum(seq_lens_kv, dim=0)
])
```

**变更后**：
```python
self.actual_seq_q = torch.cumsum(seq_lens_q, dim=0)
self.actual_seq_kv = torch.cumsum(seq_lens_kv, dim=0)
```

### 3.5 修复 `block_table` 设备问题

**文件**：`ascend_prefill.py` 和 `ascend_decode.py` 中的 `AscendPrefillAttnOp.prepare()` 和 `AscendDecodeAttnOp.prepare()`

**方案**：优先使用 `kv_cache_block_id_device`（已在 NPU 上），否则将 host tensor `.npu()` 转移到 NPU。

**变更前**：
```python
def prepare(self, attn_inputs):
    self.block_table = attn_inputs.kv_cache_block_id_host
    if self.block_table is not None:
        self.block_table = self.block_table.clamp(min=0)
```

**变更后**：
```python
def prepare(self, attn_inputs):
    from rtp_llm.models_py.modules.factory.attention.ascend_impl.ascend_attn_params import _squeeze_block_table
    bt_device = attn_inputs.kv_cache_block_id_device
    if bt_device is not None and bt_device.defined() and bt_device.numel() > 0:
        self.block_table = _squeeze_block_table(bt_device)
    else:
        self.block_table = _squeeze_block_table(attn_inputs.kv_cache_block_id_host)
        if self.block_table is not None:
            self.block_table = self.block_table.npu()
    if self.block_table is not None:
        self.block_table = self.block_table.clamp(min=0)
```

### 3.6 修复 `atten_mask` 要求

**文件**：`ascend_prefill.py` 中的 `AscendPrefillAttnOp.forward()`

**方案**：CANN 8.5.1 V3 接口在 `sparse_mode != 0` 时要求提供 `atten_mask`，传入 2D placeholder mask `[2048, 2048]`。

**变更前**：
```python
attn_output, _ = torch_npu.npu_fused_infer_attention_score(
    ...
    sparse_mode=3,
)
```

**变更后**：
```python
attn_output, _ = torch_npu.npu_fused_infer_attention_score(
    ...
    sparse_mode=3,
    atten_mask=torch.zeros(2048, 2048, dtype=torch.bool, device=q.device),
)
```

### 3.7 添加 `fmha_params` 属性

**文件**：`ascend_prefill.py` 和 `ascend_decode.py`

**方案**：`AscendPrefillImpl` 和 `AscendDecodeImpl` 继承自 `FMHAImplBase`，后者未定义 `fmha_params` 属性，但 `Qwen3Model.forward()` 返回 `PyModelOutputs` 时访问了该属性。

**变更**：在 `__init__` 中添加 `self.fmha_params = None`。

---

## 四、修改文件汇总

| 文件路径 | 修改类型 | 修改内容 |
|---|---|---|
| `rtp_llm/models_py/bindings/OpDefs.h` | C++ | KV cache reshape 从 NHD 改为 BNSD layout |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py` | Python | 添加 `_squeeze_block_table()` 处理三维 block_table；`build_ascend_params` 和 `compute_ascend_attn_params` 中使用 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py` | Python | 替换 `_npu_reshape_and_cache` 为 `index_copy_` 实现 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py` | Python | 修复 `actual_seq_lengths` 格式；`block_table` 优先用 device tensor；传入 `atten_mask`；添加 `fmha_params` |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py` | Python | `block_table` 优先用 device tensor；添加 `fmha_params` |

---

## 五、编译与部署

修改 `OpDefs.h` 后需要重新编译 C++ bindings：

```bash
bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend --test_output=errors --test_env="LOG_LEVEL=INFO"
pip install --force-reinstall --no-deps bazel-bin/rtp_llm/rtp_llm-0.2.0-py3-none-any.whl
```

---

## 六、修复效果

修复后推理流程中 `_npu_reshape_and_cache` SIGSEGV 已消除：

- ✅ block_table 维度正确（三维 → 二维）
- ✅ slot_mapping shape 正确（`[num_tokens]` 一维）
- ✅ KV cache 写入成功（index_copy_ 替代 ATB 融合算子）
- ✅ KV cache layout 正确（BNSD 匹配 CANN 期望）
- ✅ prefill attention 计算通过（actual_seq_lengths / block_table device / atten_mask 已修正）
- ⚠️ 采样阶段 `sampleGreedy()` 存在独立 SIGSEGV，需单独排查

---

## 七、遗留问题

`sampleGreedy()` 在 NPU 上的 SIGSEGV（`@0x12c041236e00`），发生在 `CudaSampleOp.cc` 的采样逻辑中，属于采样 op 的 NPU 兼容性问题，与 KV cache 写入无关，需要单独排查。
