# Ascend NPU Attention 实现修复总结

## 涉及文件

- `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py`
- `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

---

## Bug 1: Prefill 阶段 `actual_seq_lengths_kv` 错误使用了 cumsum

### 文件

`ascend_prefill.py` 第 137 行

### 现象

多并发压测时 NPU 算子报错：

```
the key/value's actual sequence lengths(4) is 69, it should be in range [0, KV_S(64)]
```

单并发（concurrency=1）正常，多并发（concurrency>=2）崩溃。

### 原因

`npu_fused_infer_attention_score` 在 `input_layout="TND"` 下对两个参数的语义要求不同：

| 参数 | 语义 | 格式 |
|------|------|------|
| `actual_seq_lengths` (Q) | 每个batch在flattened Q tensor中的**结束位置** | **cumsum**（前缀和） |
| `actual_seq_lengths_kv` (KV) | 每个batch的**单独KV长度** | 原始长度（非cumsum） |

原始代码对两者都使用了 `torch.cumsum()`：

```python
# 修改前（错误）
self.actual_seq_q = torch.cumsum(seq_lens_q, dim=0)     # 正确
self.actual_seq_kv = torch.cumsum(seq_lens_kv, dim=0)    # 错误！
```

单并发时 cumsum 结果与原始长度相同（如 `[69]` = `[69]`），所以不报错。多并发时 cumsum 产生错误值（如 `cumsum([69,69,69,69])` = `[69,138,207,276]`），NPU tiling checker 发现第4个batch的KV长度276超出了KV cache范围，导致崩溃。

### 参考

vllm-ascend (`vllm_ascend/attention/attention_v1.py`) 中 `_get_fia_params()` 方法的处理方式：

```python
# DecodeOnly / PrefillCacheHit / ChunkedPrefill:
actual_seq_lengths_kv = attn_metadata.seq_lens_list    # 单独长度列表
# actual_seq_lengths (Q) 始终使用 cumsum
```

### 修复

```python
# 修改后（正确）
self.actual_seq_q = torch.cumsum(seq_lens_q, dim=0)    # cumsum（前缀和）
self.actual_seq_kv = seq_lens_kv                         # 单独长度（非cumsum）
```

---

## Bug 2: Decode 阶段 `context_lens` 计算时 tensor size 不匹配

### 文件

`ascend_decode.py` 第 139 行（`AscendDecodeAttnOp.prepare()`）

### 现象

多并发 decode 时报错：

```
Failed to instantiate AscendDecodeImpl: The size of tensor a (0) must match the size of tensor b (9) at non-singleton dimension 0
```

随后触发 `can not find mha type`（因为唯一的 decode 实现实例化失败，无可用实现）。

### 原因

`AscendDecodeAttnOp.prepare()` 中的 `context_lens` 计算方式：

```python
# 修改前（错误）
self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths
```

在 decode 阶段，C++ 层 `buildPyAttentionInputs()` 对各字段的填充逻辑为：

| 字段 | Prefill 阶段 | Decode 阶段 |
|------|-------------|-------------|
| `prefix_lengths` | shape=[batch_size]，有值 | shape=[0]，**空tensor** |
| `input_lengths` | shape=[batch_size]，有值 | shape=[batch_size]，有值 |
| `sequence_lengths` | shape=[0]，空tensor | shape=[batch_size]，有值 |

decode 时 `prefix_lengths` 为空 tensor（size=0），`input_lengths` 有值（size=9），两者相加时 size 不匹配：`(0) must match (9)`。

而 `AscendDecodeImpl.forward()` 第 87 行已经正确使用了 `sequence_lengths + 1`：

```python
self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths + 1
```

但 `__init__` 阶段调用的 `prepare()` 中的初始计算逻辑不一致。

### 语义说明

- `sequence_lengths`：当前已生成token的位置索引（0-indexed），即历史KV cache中的token数量减1
- `context_lens`（NPU算子需要的）：KV cache中的**总token数**，即 `sequence_lengths + 1`
- decode阶段每个请求的 `input_length` 固定为1（生成一个新token）

### 修复

```python
# 修改后（正确）
if attn_inputs.sequence_lengths.numel() > 0:
    self.context_lens = attn_inputs.sequence_lengths + 1
elif attn_inputs.prefix_lengths.numel() > 0 and attn_inputs.input_lengths.numel() > 0:
    self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths
else:
    self.context_lens = None
```

优先使用 decode 阶段的 `sequence_lengths + 1`，fallback 到 prefill 阶段的 `prefix_lengths + input_lengths`（仅在两者都有值时使用）。

---

## 修改汇总

| 文件 | 行号 | 修改前 | 修改后 | Bug |
|------|------|--------|--------|-----|
| `ascend_prefill.py` | 137 | `self.actual_seq_kv = torch.cumsum(seq_lens_kv, dim=0)` | `self.actual_seq_kv = seq_lens_kv` | Bug 1: cumsum误用 |
| `ascend_decode.py` | 139 | `self.context_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths` | 优先 `sequence_lengths + 1`，fallback 到 `prefix_lengths + input_lengths` | Bug 2: tensor size不匹配 |
| `ascend_decode.py` | 154 | `actual_seq_kv = torch.cumsum(context_lens.to(torch.int32), dim=0)` | `actual_seq_kv = context_lens.to(torch.int32)` | Bug 1: 同上，cumsum误用 |
