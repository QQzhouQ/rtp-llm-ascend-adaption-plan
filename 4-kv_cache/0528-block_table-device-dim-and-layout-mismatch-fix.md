# rtp_llm Ascend NPU 适配修复记录 (2025-05-28)

## 环境

- 模型: Qwen3-0.6B
- 硬件: Ascend NPU
- 启动命令:
  ```bash
  python -m rtp_llm.start_server \
    --checkpoint_path=/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
    --model_type=qwen_3 \
    --start_port=9000
  ```

## 编译命令

```bash
bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend --test_output=errors --test_env="LOG_LEVEL=INFO"
```

---

## 问题 1: ReshapeAndCacheOperation setup failed!

### 错误信息

```
RuntimeError: ReshapeAndCacheOperation setup failed!
```

调用栈: `ascend_kv_cache_write_op.py:30` -> `torch_npu._npu_reshape_and_cache`

### 原因

在 `rtp_llm/models_py/bindings/OpDefs.h` 的 `getLayerCache()` 中，Ascend NPU 分离 KV cache 的 reshape 产生了 2D 张量:

```cpp
// 修改前 - 2D [total_slots, kv_heads * head_dim]
layer_cache.k_cache_base = k_base.reshape(
    {kernel_block_num * (int64_t)kernel_seq_size_per_block,
     (int64_t)num_kv_heads * (int64_t)head_dim});
```

但 `torch_npu._npu_reshape_and_cache` 要求 KV cache 为 4D NHD 布局:
`[num_blocks, block_size, num_kv_heads, head_dim]`

### 修改

**文件**: `rtp_llm/models_py/bindings/OpDefs.h` (第 93-103 行)

```cpp
// 修改后 - 4D [blocks, block_size, kv_heads, head_dim]
if (num_kv_heads > 0 && head_dim > 0) {
    layer_cache.k_cache_base = k_base.reshape(
        {kernel_block_num,
         (int64_t)kernel_seq_size_per_block,
         (int64_t)num_kv_heads,
         (int64_t)head_dim});
    layer_cache.v_cache_base = v_base.reshape(
        {kernel_block_num,
         (int64_t)kernel_seq_size_per_block,
         (int64_t)num_kv_heads,
         (int64_t)head_dim});
}
```

> 注意: 需要重新编译 C++ 代码。

---

## 问题 2: block_table 在 CPU 上，其他张量在 NPU 上

### 错误信息

```
RuntimeError: Expected all tensors to be on the same device, but got block_table is on cpu,
different from other tensors on npu:0
```

调用栈: `ascend_prefill.py:135` -> `npu_fused_infer_attention_score`

### 原因

`AscendPrefillAttnOp.prepare()` 中 `block_table` 来自 `attn_inputs.kv_cache_block_id_host`（CPU pinned memory），直接传给 NPU 算子时设备不匹配。`actual_seq_q` 和 `actual_seq_kv` 同样在 CPU 上构造，也需要移到 NPU。

### 修改

**文件**: `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py` (`AscendPrefillAttnOp.forward`)

```python
# 修改前
def forward(self, q, kv_cache):
    k_cache = kv_cache.k_cache_base
    v_cache = kv_cache.v_cache_base
    attn_output, _ = torch_npu.npu_fused_infer_attention_score(
        query=q, key=k_cache, value=v_cache,
        block_table=self.block_table,  # CPU tensor!
        ...
        actual_seq_lengths=self.actual_seq_q,  # CPU tensor!
        actual_seq_lengths_kv=self.actual_seq_kv,  # CPU tensor!
    )

# 修改后
def forward(self, q, kv_cache):
    ...
    block_table = self.block_table
    if block_table is not None and block_table.device.type != q.device.type:
        block_table = block_table.to(q.device)
    actual_seq_q = self.actual_seq_q
    if actual_seq_q is not None and actual_seq_q.device.type != q.device.type:
        actual_seq_q = actual_seq_q.to(q.device)
    actual_seq_kv = self.actual_seq_kv
    if actual_seq_kv is not None and actual_seq_kv.device.type != q.device.type:
        actual_seq_kv = actual_seq_kv.to(q.device)
```

---

## 问题 3: block_table 维度为 3，应为 2

### 错误信息

```
the dim num of block_table is 3, it should be 2
```

### 原因

`kv_cache_kernel_block_id_host` 从 C++ 传来时可能是 3D `[group, batch, max_blocks]`，但 NPU 算子要求 2D `[batch, max_blocks]`。

### 修改

**文件**: `ascend_prefill.py` 和 `ascend_decode.py` 的 `prepare()` 方法

```python
# 修改前
def prepare(self, attn_inputs):
    self.block_table = attn_inputs.kv_cache_kernel_block_id_host
    if self.block_table is not None:
        self.block_table = self.block_table.clamp(min=0)

# 修改后
def prepare(self, attn_inputs):
    self.block_table = attn_inputs.kv_cache_kernel_block_id_host
    if self.block_table is not None:
        self.block_table = self.block_table.clamp(min=0)
        if self.block_table.ndim != 2:
            self.block_table = self.block_table.reshape(-1, self.block_table.shape[-1])
```

---

## 问题 4: sparse_mode=3 时 atten_mask 不能为空

### 错误信息

```
when sparse_mode is 3, it not 0, atten_mask should not be null.
```

### 原因

`npu_fused_infer_attention_score` 的 `sparse_mode=3`（rightDownCausal，标准因果注意力）要求传入 `atten_mask` 参数。根据文档，需要传入一个 2048x2048 的压缩掩码矩阵。

### 修改

**文件**: `ascend_prefill.py` (`AscendPrefillAttnOp`)

```python
class AscendPrefillAttnOp:
    _causal_mask = None

    @classmethod
    def _get_causal_mask(cls, device):
        if cls._causal_mask is None or cls._causal_mask.device.type != device.type:
            cls._causal_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.int8), diagonal=1
            ).to(device)
        return cls._causal_mask

    def forward(self, q, kv_cache):
        ...
        atten_mask = self._get_causal_mask(q.device)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q, key=k_cache, value=v_cache,
            atten_mask=atten_mask,
            ...
            sparse_mode=3,
        )
```

上三角矩阵中 `1` 表示 mask 掉（不计算），`0` 表示计算注意力。使用类变量缓存避免重复创建。

---

## 问题 5: block_table 第一维与 batch size 不匹配

### 错误信息

```
block_table's first dimension(1) should be equal to batch size(2)
```

### 原因

NPU 文档明确说明: **"当 query 的 input_layout 为 TND 时，以 actual_seq_lengths 元素的数量作为 Batch 值"**。

原代码在 `actual_seq_lengths` 前面加了一个 0:

```python
self.actual_seq_q = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=seq_lens_q.device),
    torch.cumsum(seq_lens_q, dim=0)
])
```

对于 `seq_lens=[9]`，产生 `[0, 9]`（2 个元素），NPU 认为 batch=2，但实际 batch=1。

### 修改

**文件**: `ascend_prefill.py` (`AscendPrefillAttnOp.prepare`)

```python
# 修改前
self.actual_seq_q = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=seq_lens_q.device),
    torch.cumsum(seq_lens_q, dim=0)
])
self.actual_seq_kv = torch.cat([
    torch.zeros(1, dtype=torch.int32, device=seq_lens_kv.device),
    torch.cumsum(seq_lens_kv, dim=0)
])

# 修改后 - 不加前导 0，直接使用 cumsum
self.actual_seq_q = torch.cumsum(seq_lens_q, dim=0)
self.actual_seq_kv = torch.cumsum(seq_lens_kv, dim=0)
```

> NPU 的 `actual_seq_lengths` 中每个元素表示"当前 Batch 及之前所有 Batch 的 seqlen 之和"，元素个数即 Batch 数。

---

## 问题 6: 使用 kv_cache_block_id_host 而非 kv_cache_kernel_block_id_host

### 原因

`kv_cache_block_id_host` 是 3D `[group, batch, max_blocks]`（物理块粒度），未被 C++ 端 `setupKVCacheForAttentionInputs` 按组切分。而 `kv_cache_kernel_block_id_host` 已被正确切分为 2D `[batch, max_kernel_blocks]`（kernel 块粒度），与 KV cache 的 kernel block 粒度匹配。

### 修改

**文件**: `ascend_prefill.py` 和 `ascend_decode.py` 的 `prepare()` 方法

```python
# 修改前
def prepare(self, attn_inputs):
    self.block_table = attn_inputs.kv_cache_block_id_host

# 修改后
def prepare(self, attn_inputs):
    self.block_table = attn_inputs.kv_cache_kernel_block_id_host
```

---

## 问题 7: KV cache 布局不匹配 (BNDH vs BNHD)

### 错误信息

```
key layout is BnNBsD, shape [8144, 64, 8, 128] should be equal to [8144, 8, 64, 128]
```

### 原因

两个 NPU 算子对 KV cache 布局要求不同:

| 算子 | 期望布局 | 形状 |
|------|----------|------|
| `_npu_reshape_and_cache` (写入) | BNDH | `[blocks, block_size, kv_heads, head_dim]` |
| `npu_fused_infer_attention_score` (读取) | BNHD | `[blocks, kv_heads, block_size, head_dim]` |

OpDefs.h 中的 reshape 产生 BNDH 布局，写入算子可以正常工作，但读取算子期望 BNHD。

### 修改

**文件**: `ascend_prefill.py` 和 `ascend_decode.py` 的 `forward()` 方法

```python
# ascend_prefill.py - AscendPrefillAttnOp.forward
def forward(self, q, kv_cache):
    # BNDH -> BNHD: [blocks, block_size, kv_heads, head_dim] -> [blocks, kv_heads, block_size, head_dim]
    k_cache = kv_cache.k_cache_base.permute(0, 2, 1, 3).contiguous()
    v_cache = kv_cache.v_cache_base.permute(0, 2, 1, 3).contiguous()
    ...

# ascend_decode.py - AscendDecodeAttnOp.forward
def forward(self, q, kv_cache):
    k_cache = kv_cache.k_cache_base.permute(0, 2, 1, 3).contiguous()
    v_cache = kv_cache.v_cache_base.permute(0, 2, 1, 3).contiguous()
    ...
```

OpDefs.h 保持 BNDH reshape 不变（供 `_npu_reshape_and_cache` 写入使用），在读取时通过 `permute(0, 2, 1, 3)` 转换为 BNHD。

---

## 问题 8: AscendPrefillImpl 缺少 fmha_params 属性

### 错误信息

```
AttributeError: 'AscendPrefillImpl' object has no attribute 'fmha_params'
```

调用栈: `qwen3.py:136` -> `PyModelOutputs(hidden_states, fmha_impl.fmha_params)`

### 原因

`qwen3.py` 的 `forward()` 返回 `PyModelOutputs(hidden_states, fmha_impl.fmha_params)`，要求 fmha_impl 有 `fmha_params` 属性。基类 `FMHAImplBase` 未定义此属性（仅在 `MlaImplBase` 中定义），`AscendPrefillImpl` 和 `AscendDecodeImpl` 也没有初始化它。

### 修改

**文件**: `ascend_prefill.py` (`AscendPrefillImpl.__init__`) 和 `ascend_decode.py` (`AscendDecodeImpl.__init__`)

```python
# ascend_prefill.py
class AscendPrefillImpl(FMHAImplBase):
    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = None  # 新增
        ...

# ascend_decode.py
class AscendDecodeImpl(FMHAImplBase):
    def __init__(self, attn_configs, attn_inputs, parallelism_config):
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.fmha_params = None  # 新增
        ...
```

---

## 问题 9: _npu_paged_attention_v2 算子不存在

### 错误信息

```
AttributeError: module 'torch_npu' has no attribute '_npu_paged_attention_v2'
```

调用栈: `ascend_decode.py:137` -> `torch_npu._npu_paged_attention_v2`

### 原因

当前 CANN/torch_npu 版本中不存在 `_npu_paged_attention_v2` 算子。实际可用的是 `_npu_paged_attention`（无 v2 后缀），位于 ATB 扩展模块中。

通过实际探测，`_npu_paged_attention` 的签名为：

```
atb::_npu_paged_attention(
    Tensor query, Tensor key_cache, Tensor value_cache,
    int num_kv_heads, int num_heads, float scale_value,
    Tensor block_table, Tensor context_lens,
    Tensor(a!) out, *, Tensor? workspace=None
) -> ()
```

与原先使用的 `_npu_paged_attention_v2` 相比主要区别：
- `context_lens` 参数是 **Tensor** 而非 list
- 需要**预分配 `out` 张量**（in-place 写入），不返回结果
- 参数顺序不同

### 修改

**文件**: `ascend_decode.py` (`AscendDecodeAttnOp.forward`)

```python
# 修改前 - 使用不存在的 _npu_paged_attention_v2
output = torch_npu._npu_paged_attention_v2(
    query=q,
    key_cache=k_cache,
    block_table=block_table,
    context_lens=context_lens.tolist() if context_lens is not None else [],
    value_cache=v_cache,
    num_kv_heads=self.num_kv_heads,
    num_heads=self.num_heads,
    scale_value=self.scale,
)

# 修改后 - 使用实际可用的 _npu_paged_attention
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

同时保持 `context_lens` 为 Tensor 类型（不再转为 list），以及 `AscendDecodeImpl.forward()` 中对 `self.fmha_impl.context_lens` 的赋值。

---

## 修改文件清单

| 文件 | 修改类型 |
|------|----------|
| `rtp_llm/models_py/bindings/OpDefs.h` | C++ 修改，需重新编译 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py` | Python 修改 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py` | Python 修改 |

## 数据流总结

```
KV Cache 内存 (C++, flat buffer)
    │
    ▼  OpDefs.h getLayerCache() reshape
k_cache_base: [blocks, block_size, kv_heads, head_dim]  (BNDH)
    │
    ├──▶ _npu_reshape_and_cache (写入) ── 直接使用 BNDH
    │
    └──▶ permute(0,2,1,3) ──▶ [blocks, kv_heads, block_size, head_dim] (BNHD)
         │
         └──▶ npu_fused_infer_attention_score (prefill) / _npu_paged_attention (decode)
```
