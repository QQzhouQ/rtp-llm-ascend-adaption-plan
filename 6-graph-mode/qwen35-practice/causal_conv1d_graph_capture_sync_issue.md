# causal_conv1d_update 图捕获同步失败问题分析与解决

## 1. 问题现象

在 Ascend ACL Graph 捕获阶段，`causal_conv1d_update` 调用时报错：

```
ACL stream synchronize failed
```

同样的错误也出现在 `fused_recurrent_gated_delta_rule`（recurrent 层）中。

## 2. 根因分析

### 2.1 为什么需要 CPU 同步

`causal_conv1d_update` 的核心逻辑是：根据 `block_map`（KV cache 页表）和 `sequence_lengths`（序列长度）来确定每个 batch 序列应该读写哪个物理 cache page。

关键代码路径（[causal_conv1d.py](file:///home/b30086708/code/rtp_llm/rtp-llm-npu/rtp_llm/models_py/kernels/ascend/causal_conv1d.py#L330-L333)）：

```python
current_lengths = _as_int_list(sequence_lengths)   # ← 触发 CPU sync
block_rows = _as_int_rows(block_map)                # ← 触发 CPU sync
```

其中 `_as_int_list` 和 `_as_int_rows` 的实现（[causal_conv1d.py:44-58](file:///home/b30086708/code/rtp_llm/rtp-llm-npu/rtp_llm/models_py/kernels/ascend/causal_conv1d.py#L44-L58)）：

```python
def _as_int_list(values) -> list[int]:
    ...
    return [int(value) for value in values.detach().to(dtype=torch.int64).cpu().tolist()]
    #                                                    ^^^^^^^^^^^^
    #                                          .cpu() 触发 aclrtSynchronizeStream

def _as_int_rows(values) -> list[list[int]]:
    ...
    return [[int(value) for value in row]
            for row in values.detach().to(dtype=torch.int64).cpu().tolist()]
    #              同样触发 aclrtSynchronizeStream
```

`block_map` 和 `sequence_lengths` 是 **NPU device tensor**，而 CANN 算子 `npu_causal_conv1d` 的参数 `cache_indices` 需要 **Python list**。将 device tensor 转为 Python list 必须经过 `.cpu().tolist()`，这一步会隐式调用 `aclrtSynchronizeStream` 完成设备到主机的数据传输。

recurrent 层的 `_resolve_state_pages` 存在完全相同的问题（[recurrent.py:56](file:///home/b30086708/code/rtp_llm/rtp-llm-npu/rtp_llm/models_py/kernels/ascend/recurrent.py#L56)）：

```python
mapping = block_map.detach().cpu().tolist()  # ← 触发 CPU sync
```

### 2.2 为什么 CPU 同步在图捕获时失败

ACL Graph 的捕获过程要求 **capture stream 上不能有任何 stream synchronization 操作**。`aclrtSynchronizeStream` 会阻塞当前 stream 直到所有排队的操作完成，这直接破坏了图捕获的上下文，导致捕获失败。

```
图捕获 stream:  [capture begin] → [op1] → [op2] → [.cpu() → aclrtSynchronizeStream!] → ✗ 失败
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                    不允许在捕获期间同步
```

### 2.3 问题本质

这是一个 **D2H（Device-to-Host）数据依赖** 问题：

```
NPU Device Tensor          Python List (Host)         CANN 算子
(block_map, seq_lengths) →  .cpu().tolist()  →  cache_indices=[...]
       ↑                         ↑                      ↑
   图捕获时存在              隐式 stream sync         需要 Python list 参数
```

CANN AscendC 算子 `npu_causal_conv1d` 和 `npu_recurrent_gated_delta_rule` 的设计要求：
1. **`cache_indices` / `ssm_state_indices` 是 Python list**，不是 device tensor
2. 没有 `.out()` 形式（无法原地更新输出）
3. 这意味着每次调用都必须从 device 读取数据到 host，无法避免 CPU sync

## 3. 解决方案演进

### 方案一：图捕获时使用固定索引跳过 CPU sync（causal_conv1d.py — 已注释弃用）

**思路**：图捕获/warmup 阶段 `block_map` 和 `sequence_lengths` 都是占位值，不需要真实读取。用简单的顺序索引 `list(range(batch))` 替代。

```python
# causal_conv1d.py:291-328 (已注释)
in_graph_capture = torch.npu.is_current_stream_capturing()

if in_graph_capture:
    # 用固定索引替代 block_map 读取
    cache_indices = list(range(batch))
    # 跳过所有 .cpu().tolist() 调用
    ...
```

recurrent.py 中保留了相同的思路（[recurrent.py:195-217](file:///home/b30086708/code/rtp_llm/rtp-llm-npu/rtp_llm/models_py/kernels/ascend/recurrent.py#L195-L217)）：

```python
if in_graph_capture:
    state_indices = [batch_idx for batch_idx in range(batch)
                     for _ in range(token_count)]
    # 跳过 _resolve_state_pages() 和 _seed_first_write_pages()
    ...
```

**缺陷**：固定索引是 Python list，作为参数传入 CANN 算子后，**图重放（replay）时无法更新这些值**。Python list 在图捕获时被固化为常量，replay 时永远使用捕获时的占位索引，导致结果错误。

### 方案二：图捕获时走 Triton kernel（qwen3_next.py — 最终方案）

**思路**：在模型层 dispatch，图捕获时切换到 **Triton kernel**，eager 时仍用 **CANN 算子**。

关键代码在 [qwen3_next.py:64-77](file:///home/b30086708/code/rtp_llm/rtp-llm-npu/rtp_llm/models_py/model_desc/qwen3_next.py#L64-L77)：

```python
def causal_conv1d_update(*args, **kwargs):
    if torch.npu.is_current_stream_capturing():
        return triton_causal_conv1d_update(*args, **kwargs)   # 图捕获 → Triton
    return ascend_causal_conv1d_update(*args, **kwargs)       # eager → CANN

def fused_recurrent_gated_delta_rule(*args, **kwargs):
    if torch.npu.is_current_stream_capturing():
        return triton_fused_recurrent_gated_delta_rule(*args, **kwargs)  # 图捕获 → Triton
    return ascend_fused_recurrent_gated_delta_rule(*args, **kwargs)      # eager → CANN
```

**为什么 Triton kernel 没有 CPU sync 问题**：

| 特性 | CANN AscendC 算子 | Triton kernel |
|------|------------------|----------------|
| `cache_indices` / `block_map` 参数类型 | Python list（host） | device tensor |
| 读取 block_map 方式 | `.cpu().tolist()`（D2H + sync） | GPU 端直接读取 tensor 索引 |
| 图捕获兼容性 | 不兼容（需要 CPU sync） | 兼容（全程在 device 上操作） |
| 图重放时更新索引 | 不可能（Python list 被固化为常量） | 自动（tensor 是图的可变输入） |

Triton kernel 接收 `block_map` 作为 **device tensor 参数**，在 GPU/NPU 端直接进行索引操作，整个过程不需要 D2H 传输，因此不会触发 stream synchronization。

## 4. 最终架构

```
                    ┌─────────────────────────────┐
                    │     qwen3_next.py (dispatch) │
                    │  torch.npu.is_current_stream_ │
                    │       capturing() ?           │
                    └──────────┬──────────┬─────────┘
                               │          │
                    图捕获=True │          │ 图捕获=False
                               ▼          ▼
                    ┌──────────────┐  ┌──────────────┐
                    │ Triton kernel │  │ CANN 算子     │
                    │ (causal_conv1d│  │ (npu_causal_ │
                    │  _update)     │  │  conv1d)     │
                    │              │  │              │
                    │ block_map 是  │  │ block_map 需  │
                    │ device tensor │  │ .cpu().tolist│
                    │ → 无 CPU sync │  │ → 有 CPU sync │
                    │ → 图捕获兼容  │  │ → 仅 eager 用 │
                    └──────────────┘  └──────────────┘
```

## 5. 总结

| 维度 | 说明 |
|------|------|
| **问题** | CANN 算子需要 Python list 参数 → 必须从 device tensor 读到 host → 触发 `aclrtSynchronizeStream` → 图捕获失败 |
| **根因** | CANN AscendC 算子设计上依赖 host 端参数（`cache_indices` 等），与 ACL Graph 要求"捕获期间无 stream sync"矛盾 |
| **解决原理** | Triton kernel 全程在 device 上操作 tensor 索引，无需 D2H 传输，天然兼容图捕获 |
| **最终方案** | 模型层 dispatch：图捕获 → Triton kernel（正确性优先），eager → CANN 算子（性能优先） |
| **代价** | 需要安装 triton + triton-ascend；图捕获路径用 Triton 可能比 CANN 略慢 |
