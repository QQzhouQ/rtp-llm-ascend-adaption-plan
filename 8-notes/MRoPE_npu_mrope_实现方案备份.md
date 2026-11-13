# Ascend MRoPE 实现方案备份（npu_mrope）

> **状态**：已实现并验证通过，但已从 PR #46 撤销（决策：MRoPE 随 qwen3.5 多模态接入时再合入）。
> **恢复方式**：`git cherry-pick 78c3d6363`（merge 分支对应 `f78c31234`；撤销提交为 `d579e450a` / `fa4132385`）。
> **验证环境**：本机 Ascend NPU + CANN 9.0 + torch_npu，对拍全部通过。
> 归档日期：2026-09-08

---

## 一、背景

- PR #46（上游 PR #1349 评审修复）的评审评论 ⑥ 问到 "Mrope使用场景"。
- MRoPE（多轴位置编码）用于多模态模型（Qwen2-VL / Qwen2.5-VL / qwen3.5 系列）：位置是三维 (t, h, w)，由视觉 patch 网格计算。
- rtp-llm 上游 `PositionIdsGenerator`（`rtp_llm/cpp/models/position_ids/PositionIdsGenerator.cc:49`）已按 MROPE 风格生成 `[num_tokens, 3]` 的三轴位置——与 vllm-ascend 的 `torch_npu.npu_mrope` 输入天然同构。
- 实现完成后按决策撤销：qwen3.5 接入时直接 cherry-pick 恢复。

## 二、npu_mrope 算子契约（CANN 9.0 实测逆向，本方案最有价值的部分）

官方文档语焉不详的地方，全部通过实验确定（**示踪法**：`cos=0, sin=1` + `x[...,d]=d`，从输出直接读出维度配对关系）：

### 2.1 接口签名
```python
torch_npu.npu_mrope(positions, query, key, cos_sin_cache, head_size,
                    mrope_section=[d1,d2,d3],   # 频率段（维度对单位），和 == rotary_dim/2
                    rotary_mode='half',          # 'half'=NeoX 半分 / 'interleave'=GPT-J 交错
                    cache_mode='default')        # 'default'=[cos|sin] 两段式 cache
```

### 2.2 输入要求
| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| positions | **(3, T)** 或 (T,) | int64（实测 int32 会报 dtype 不支持） | 2D=mrope、1D=普通 rope；rtp-llm 的 `[T,3]` 需 `.t()` 转置 |
| query / key | **(T, heads × head_size)** 2D | fp32/fp16/bf16 | 3D 张量会报 "Expected queryIn to be a vector of size 2"，必须 reshape |
| cos_sin_cache | (max_pos, rotary_dim) | 同 q/k | halves `[cos(前半) \| sin(后半)]`，与 `get_rope_cache_once(interleave=False)` 产物直接兼容 |
| mrope_section | `[d1,d2,d3]` | — | **维度对（pair）单位**，三段之和 == rotary_dim/2；Qwen 系 `[16,24,24]` 对应 rotary_dim=128 |

### 2.3 数学语义（关键，容易搞错）
- **配对方式：全局半分配对 `(d, d + rotary_dim/2)`**，不是"段内配对"（示踪法实测：dim0 与 dim64 配对，head_size=128 时）。
- **section 划分的是频率（维度对），不是维度块**：频率 f ∈ [0, rotary_dim/2)，轴归属 `f < d1 → t 轴`、`f < d1+d2 → h 轴`、其余 → w 轴；每个频率用自己的轴位置查 cos/sin。
- 与 **vLLM 官方 `triton_mrope` 内核逐行比对确认一致**（`vllm/model_executor/layers/rotary_embedding/mrope.py`：非交错分支 `t_mask/h_mask/w_mask` 按频率段划分 + first/second half 全局配对）；HF Qwen2-VL 同语义。
- 约束：fp16/bf16 时 rotary_dim 需为 32 的倍数，fp32 需 16 的倍数（64/128 的常规 head_size 天然满足）。

### 2.4 对拍验证结论
- 纯 torch 参考实现（见第五节）vs `npu_mrope`：**fp32 逐位一致（maxdiff=0）**，fp16 舍入内一致（~1.5e-3），bf16 ~1.5e-2；对拍在 NPU 实机执行。
- fp16 断言容差注意：幅值 ~4 时 fp16 机器精度 ~2e-3，`atol=1e-3` 会偶发失败，用 `atol=1e-2`。

## 三、实现改动（5 个文件，cherry-pick 78c3d6363 可完整恢复）

### 3.1 `ascend_impl/ascend_rope_emb.py`
`AscendRotaryEmbeddingOp` 增加 MRoPE 分支：
```python
import torch_npu
from rtp_llm.ops import AttentionConfigs, RopeStyle, get_rope_cache_once

# __init__：
if self.rope_style == RopeStyle.Mrope:
    self.mrope_section = [int(rc.mrope_dim1), int(rc.mrope_dim2), int(rc.mrope_dim3)]
else:
    self.mrope_section = None

# _apply_rope：
if self.mrope_section is not None:
    positions = rope_params.positions_d          # [T,3]
    if positions.dim() != 2: raise RuntimeError(...)   # combo_position_ids 必须上游生成
    pos_t = positions.t().to(torch.int64).contiguous()      # (3,T)
    q2, k2 = query.reshape(query.shape[0], -1), key.reshape(key.shape[0], -1)
    q_out, k_out = torch_npu.npu_mrope(
        pos_t, q2, k2, self.cos_sin_cache, self.head_size,
        mrope_section=self.mrope_section,
        rotary_mode="half", cache_mode="default")
    query.copy_(q_out.view_as(query)); key.copy_(k_out.view_as(key))
    return
# 非 Mrope 走原有 apply_rope_pos_ids_nhd 纯 torch 路径
```

### 3.2 `ascend_impl/ascend_attn_params.py`
`compute_ascend_attn_params` 把 **RoPE 位置与 KV 槽位解耦**：
```python
# slot_mapping 用 token 序号（prefix + token index），与 RoPE 语义无关（逻辑不变）
# RoPE 位置：
combo_position_ids = getattr(attn_inputs, "combo_position_ids", None)
if combo_position_ids is not None and combo_position_ids.dim() == 2:
    positions = combo_position_ids.to(torch.int64)   # MRoPE [T,3]
else:
    positions = seq_positions                        # 1-D 原逻辑
```

### 3.3 `ascend_impl/ascend_prefill.py` / `ascend_decode.py`
```python
@staticmethod
def support(attn_configs, attn_inputs):
    mrope_ok = (attn_configs.rope_config.style != RopeStyle.Mrope
                or not attn_configs.rope_config.mrope_interleaved)  # 非交错放开
    return ... and mrope_ok and torch.npu.is_available()
```
交错 MRoPE（`mrope_interleaved=True`，Qwen3-VL 系）仍拒绝：需要验证 `cache_mode/rotary_mode='interleave'` 的组合行为后才能放开。

### 3.4 `ascend_impl/test/test_ascend_rope.py`
新增 `TestAscendMRopeParity`（NPU 存在才跑，CPU CI 自动 skip）+ 纯 torch 参考实现 `ref_mrope_half`（见下节）。

## 四、纯 torch 参考实现（测试用，语义基准）

```python
def ref_mrope_half(x, cache, pos, sections):
    """全局半分配对 (d, d+rope_dim/2)；频率 f 按其所在段取轴位置。x:[T,H,D] cache:[max,rd] pos:[T,3]"""
    rope_dim = cache.shape[-1]; half = rope_dim // 2
    t_end = sections[0]; h_end = t_end + sections[1]
    out = x.clone()
    for f in range(half):
        s = 0 if f < t_end else (1 if f < h_end else 2)
        idx = pos[:, s].long()
        cos = cache[idx, f].unsqueeze(1); sin = cache[idx, half + f].unsqueeze(1)
        u, v = x[..., f], x[..., f + half]
        out[..., f] = u * cos - v * sin
        out[..., f + half] = u * sin + v * cos
    return out
```

## 五、qwen3.5 接入时的待办清单

| # | 事项 | 说明 |
|---|---|---|
| 1 | 恢复实现 | `git cherry-pick 78c3d6363`（先确认无冲突，或手工重放第三节代码） |
| 2 | 交错 MRoPE | 验证 `rotary_mode='interleave'`（+ `cache_mode`）与 vLLM `triton_mrope` interleaved 分支的对拍；通过后放开 support() |
| 3 | 模型接线 | 确认 qwen3.5 模型侧走 `apply_mrope_section`（`rtp_llm/models/mrope_utils.py`）且 `combo_position_ids` 为 `[T,3]`（PositionIdsGenerator MROPE 风格） |
| 4 | 端到端验证 | NPU 上多模态模型精度对拍（TP=1 先行，TP>1 视支持情况） |
| 5 | `positions_d` 传递 | decode 路径确认每 token 三轴 next position id 正确生成（`generateNextPositionId` MROPE 分支） |
| 6 | CANN 版本门禁 | npu_mrope 的 interleave 支持与 CANN 版本相关（vllm-ascend 注释提到 8.5.0），接入时在 CI 固定 ≥ 验证过的版本 |

## 六、撤销记录

- PR #46：`d579e450a` Revert "feat(ascend): support non-interleaved MRoPE via torch_npu.npu_mrope"（已推送，support() 恢复拒绝 Mrope）
- merge 分支：`fa4132385`（同步 revert）
- 评论⑥线程已回复撤销说明与后续计划
