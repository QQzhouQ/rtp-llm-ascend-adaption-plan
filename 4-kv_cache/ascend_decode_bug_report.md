# Ascend NPU Decode Attention Off-by-One Bug Report

## 1. 问题描述

在 Ascend NPU 上运行 rtp-llm 端到端推理（模型：Qwen3-0.6B），decode 阶段输出乱码/重复无意义文本，而 prefill 阶段输出正常。

### 现象

```bash
# 修复前：NPU 输出乱码
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen_3","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
# 输出：好的well，，你好很高兴吗的，怎样，，有什么如何好的好的，安和...

# 修复后：NPU 输出正常
# 输出：Okay, the user just said "hello". I should respond in a friendly way...
```

## 2. 排查过程

### 2.1 Prefill vs Decode 定位

通过在 `ascend_prefill.py` 和 `ascend_decode.py` 的 `forward` 方法中添加 tensor dump 代码，逐层对比 NPU 和 GPU 的中间结果：

| 阶段 | Layer 0 MaxAbsDiff | Layer 27 MaxAbsDiff | 结论 |
|------|-------------------|---------------------|------|
| Prefill query | 0.25 | 0.44 | 正常（fp16/bf16 误差范围） |
| Prefill key | 2.00 | 0.41 | 正常 |
| Prefill attn_output | 0.024 | 1.50 | 正常 |
| Prefill logits top-5 | **GPU 与 NPU 完全一致** | | **Prefill 精度正常** |
| **Decode attn_output** | **2.51** | **14.34** | **严重错误** |

Prefill logits 的 top-5 token id 和概率在 GPU 与 NPU 上完全一致，确认问题仅在 **decode 阶段**。

### 2.2 Decode 参数检查

在 `AscendDecodeAttnOp.forward` 中 dump 所有传入 `npu_fused_infer_attention_score` 的参数：

```json
{
  "context_lens": [9],
  "actual_seq_kv": [9],
  "block_table": [[1]],
  "page_size": 64
}
```

KV cache 验证：
- Prefill 写入 9 个 token 到 block 1 offset 0-8：**正确**
- Decode 写入 1 个 token 到 block 1 offset 9（slot=73）：**正确**
- Block 1 中有 10 个有效 token（offset 0-9）：**确认**

### 2.3 Bug 定位

`context_lens = [9]`，但 KV cache 中实际有 **10 个 token**（9 prefill + 1 decode）。

`npu_fused_infer_attention_score` 使用 `actual_seq_kv = cumsum(context_lens) = [9]` 来确定读取范围，因此 FMHA 只读了 offset 0-8（9 个 prefill token），**完全跳过了 offset 9 的 decode token**。

## 3. 根因分析

### 根因代码

文件：`rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`，第 95 行：

```python
# Bug: sequence_lengths 是 0-indexed 位置索引，不是 KV 长度
self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths
```

### 语义混淆

| 变量 | 含义 | 示例值 | 说明 |
|------|------|--------|------|
| `sequence_lengths` | KV cache 中最后一个 token 的**位置索引**（0-indexed） | `[9]` | 表示已有 token 位于 position 0~9 |
| FMHA 所需 `context_lens` | 每个 batch 的 **KV cache 总长度** | `[10]` | 表示共有 10 个 token |

Prefill 后有 9 个 token（position 0-8），第一次 decode 生成 position 9 的 token。此时 `sequence_lengths = [9]`（最后一个 token 的位置），但 FMHA 需要知道总共有 10 个 token 可读。

### 影响

- FMHA 读到的是 prefill 的 9 个 token，**不包含当前 decode 步骤写入的 KV**
- Attention 计算缺少最新 token 的 KV 信息，导致输出完全错误
- 多步 decode 累积后，误差指数级放大，输出变为乱码

### 与 GPU 实现的差异

GPU 使用 `FusedRopeKVCacheDecodeOp`（融合 RoPE + KV cache 的 C++ kernel），内部自行处理 KV 长度计算，不依赖 `sequence_lengths`。Ascend 实现是纯 Python 拆分实现，需要手动从 `sequence_lengths` 计算 `context_lens`，引入了这个 off-by-one 错误。

## 4. 修复方案

### 修改文件

`rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

### 修改内容

```python
# 修复前（第 95 行）
self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths

# 修复后
self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths + 1
```

`sequence_lengths` 是 0-indexed 索引，`+1` 后得到正确的 KV cache 总长度。

## 5. 验证结果

修复后，`context_lens` 从 `[9]` 变为 `[10]`，`actual_seq_kv` 从 `[9]` 变为 `[10]`，FMHA 正确读取全部 10 个 token。

### 端到端测试

| 测试输入 | 修复前输出 | 修复后输出 |
|----------|-----------|-----------|
| "hello" | 乱码：好的well，，你好很高兴吗的... | 正常：Okay, the user just said "hello". I should respond in a friendly way... |
| "东莞有啥特色饮食？" | 乱码 | 正常：完整的中文回答（694 tokens） |

## 6. 排查工具与方法

本次排查使用了以下方法：

1. **逐层 tensor dump**：在 `ascend_prefill.py`、`ascend_decode.py` 的 `forward` 中添加 `torch.save()` 导出每层的 query/key/value/attn_output
2. **参数 dump**：在 `AscendDecodeAttnOp.forward` 中 dump block_table、context_lens、k_cache、v_cache 等参数及 debug_info.json
3. **KV cache 写入验证**：对比 prefill key 与 KV cache 中对应 block 的数据，确认写入正确
4. **Slot mapping dump**：在 `compute_ascend_attn_params` 中 dump decode 阶段的 positions、slot_mapping 计算过程

关键环境变量：`DUMP_TENSOR_DIR=/path/to/dumps` 控制是否启用 dump。

## 7. 后续建议

1. **添加单元测试**：对 Ascend decode 路径添加 `context_lens` 正确性的单元测试
2. **变量命名优化**：考虑在 `AscendDecodeAttnOp` 中将 `context_lens` 命名为更明确的名称（如 `kv_lengths`），避免与 `sequence_lengths`（位置索引）混淆
3. **添加断言**：在 `AscendDecodeAttnOp.forward` 中添加断言，校验 `context_lens > sequence_lengths`，防止类似问题再次出现
