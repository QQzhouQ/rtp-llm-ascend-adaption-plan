# rtp-llm 图模式（ACL Graph）叠加 Qwen3.5：GDN Decode 入图实施方案

> 仓库：`/home/d30033799/rtp-llm`（`feature/tp-ascend` 分支）
> 关联文档：《SGLang_Decode阶段GDN实现调研.md》（方案出处）、《图模式叠加Qwen3.5验证总结.md》（全量问题清单）
> 补充：第三节已纳入 vllm-ascend（`/home/d30033799/vllm-ascend`）的 GDN 入图路线作为方案 E 对比——它是 host 元数据 AscendC + graph task update 的混合策略，能解决同一个 D2H capture 崩溃问题，且与 rtp-llm 的 FIA 图机制共用同一套 torch.npu API。
> 性能：图模式与 eager 的差距分析及优化探索（含 **triton-ascend strided store 静默丢弃缺陷** 的发现）见《图模式GDN性能优化探索记录.md》。

---

## 一、要解决的问题

### 现象

图 capture 时崩溃：

```
RuntimeError: ACL stream synchronize failed, error code:107027
  Reason: Stream (stream_id=24) during the capture stage is not supported.
At: rtp_llm/models_py/kernels/ascend/causal_conv1d.py(48): _as_int_list
```

### 根因

Qwen3.5 的 40 层中有 30 层是线性注意力（GDN）。decode 阶段这两类层各有状态要按序列寻址：

- `causal_conv1d_update`：conv_state 滑动窗口状态，需要 `cache_indices`（每序列的状态页号）
- `fused_recurrent_gated_delta_rule`：ssm 递归状态，需要 `ssm_state_indices`

rtp-llm eager 路径用的 **fla_npu AscendC 算子**（`npu_causal_conv1d` / `npu_recurrent_gated_delta_rule`）的元数据是 **host 侧 int 数组**：

```python
# fla_npu/ops/ascendc/_aclnn_ctypes.py（fla_npu 侧）
ctx.int_array(cache_indices_cpu)      # ← aclnn 调用把 device tensor 转成 host int 数组
```

```python
# rtp_llm/models_py/kernels/ascend/recurrent.py（rtp-llm 侧）
mapping = block_map.detach().cpu().tolist()          # ← D2H！
first_lengths = _to_int_list(sequence_lengths)       # ← D2H！
...纯 Python 循环算出 read_pages/write_pages...
state_indices = [page for batch_pages in write_pages for page in batch_pages]
ssm_state_indices = torch.tensor(state_indices, ...)  # ← H2D
```

**capture 期间做 D2H 同步被 ACL 禁止**（`Stream during the capture stage is not supported`）。而且即使 capture 能忍过去，replay 时 Python 不执行、host 参数已固化，页号变化后语义也是错的。

### 约束

rtp-llm 的 `AscendGraphRunner` 是**整图 capture** 架构（一次 forward 录完整模型），没有"某几层不入图"的机制。

---

## 二、实施方案（借鉴 SGLang / sgl-kernel-npu）

### 2.0 核心思路

> SGLang 已证明"GDN decode 直接入图"在 NPU 上可行，前提是**算子元数据 device 化**（kernel 内读 device 内存做间接寻址，无 host 指针）。rtp-llm 借用其**纯 Python/triton 部分**（不依赖 sgl-kernel-npu 的 C++ so），并针对 rtp-llm 的"状态进 KV cache 分页块"架构补一层适配。

**分流策略**：eager 走原 fla_npu AscendC 算子（性能好、已验证），仅 **capture 进行中** 切到 device 元数据实现。replay 不进 Python，重放的就是 capture 录下的 device 算子链。

### 2.1 代码结构（新增约 1060 行）

```
rtp_llm/models_py/kernels/ascend/
├── graph_compat.py            # 134 行：图模式 GDN 实现 + rtp 分页架构适配（新写）
└── sgl/                       # 925 行：自 sgl-kernel-npu 拷贝的纯 Python 部分
    ├── fused_sigmoid_gating_recurrent.py   # triton kernel（decode 核心）
    └── fla_utils.py                        # input_guard 等工具（去掉了 sgl_kernel_npu 依赖）
```

模型层改动（`rtp_llm/models_py/model_desc/qwen3_next.py`，约 40 行）：decode 路径的 `_conv1d` / `_fla` 各加一个图分支。

### 2.2 关键设计点

#### （1）分流判断：capture 状态兜底

正常应该用 `attn_inputs.is_cuda_graph` 判断，但 rtp-llm 的 C++ capture 路径对 `PyModelInputs` 做值拷贝，pybind 动态属性会丢失（详见《图模式叠加Qwen3.5验证总结.md》问题 C5），因此加 capture 状态兜底：

```python
# qwen3_next.py（GDN decode 的 _conv1d 和 _fla 里）
use_graph_path = getattr(attn_inputs, "is_cuda_graph", False) or (
    hasattr(torch, "npu") and torch.npu.is_current_stream_capturing()
)
```

三个阶段的行为：
- **warmup**（图外）：capturing=False → 走 eager AscendC（D2H 合法）
- **capture**（图中）：capturing=True → 走 graph 实现（全 device 算子被录入图）
- **replay**：不进 Python，重放图中录好的 device 算子链

#### （2）状态页索引：图内 device gather 现算

rtp-llm 的 GDN 状态不在独立池，而是**复用 KV cache 分页块**（block_map 索引，每 1024 token 换页），read/write 页语义必须与 fla_npu 的 host 实现（`_resolve_state_pages`）严格一致：

```python
# graph_compat.py
def _decode_state_indices(block_map, sequence_lengths_plus_1, seq_size_per_block):
    """device 侧从 block_map 取 read/write 页索引
    read  page = block_map[b, max(len - 2, 0) // page]
    write page = block_map[b, (len - 1) // page]        （len 即 sequence_lengths_plus_1）"""
    length = sequence_lengths_plus_1.reshape(-1).to(torch.int64)
    read_col  = (length - 2).clamp_min(0) // seq_size_per_block
    write_col = (length - 1).clamp_min(0) // seq_size_per_block
    read_idx  = block_map.gather(1, read_col.view(-1, 1)).squeeze(1).long()
    write_idx = block_map.gather(1, write_col.view(-1, 1)).squeeze(1).long()
    return read_idx, write_idx
```

为什么图内现算而不是图外预算好拷入固定 buffer：rtp-llm 的 C++ `AscendGraphRunner::prepareInputs` 在每次 replay 前已把 `block_map`（by_group capture buffer）和 `sequence_lengths_plus_1_d` 刷新为当前值，**图内 gather 读这些 capture buffer，replay 时自动得到新索引**——语义上等价于 SGLang 的"固定地址 + replay 改内容"，且无需额外图外协作代码。

#### （3）跨块状态搬运（rtp 特有，SGLang 没有）

SGLang 的 GDN 状态每请求一个固定槽，无换页概念；rtp 每 1024 token 换页，跨块时需把旧页状态搬到新页：

```python
def _seed_write_pages(state, read_idx, write_idx):
    """跨块时把 read 页状态搬到 write 页；read==write 时写回原值（幂等）。
    index_select / index_copy_ / where 都是普通 device op，可入图。"""
    src  = state.index_select(0, read_idx)
    dst  = state.index_select(0, write_idx)
    same = (read_idx == write_idx).view(-1, *([1] * (state.dim() - 1)))
    state.index_copy_(0, write_idx, torch.where(same, dst, src))
```

#### （4）conv 更新：纯 torch 实现

对应 sgl-kernel-npu 常规 decode 的纯 torch 路径（gather → 窗口卷积 → scatter）：

```python
def causal_conv1d_update_graph(x, conv_states, weight, block_map, ...):
    read_idx, write_idx = _decode_state_indices(...)
    _seed_write_pages(conv_states, read_idx, write_idx)
    xw = x.unsqueeze(-1).to(conv_states.dtype)
    state = conv_states.index_select(0, write_idx)              # [b, dim, state_len]
    window = torch.cat([state[..., -(width - 1):], xw], dim=-1) # [b, dim, width]
    out = (window.to(weight.dtype) * weight.unsqueeze(0)).sum(dim=-1)
    if activation in ("silu", "swish"):
        out = F.silu(out)
    conv_states.index_copy_(0, write_idx, window[..., -(width - 1):])
    return out.to(x.dtype)
```

#### （5）recurrent 更新：sgl triton kernel + 布局转置适配

sgl kernel 按 `[pages, HV, K, V]` 寻址（指针算术 `o_k*V + o_v`），rtp 的 ssm 池是 `[pages, HV, V, K]`（`get_ssm_state_tensor` 定义）——**差一个末两维转置**（这里踩过 133% 误差的坑）：

> **布局冲突的根源（对照 vllm-ascend 后定位）**：上游 fla / sgl-kernel-npu / vllm-ascend 的 state 约定都是 `[.., HV, K, V]`（K 前 V 后），而 **fla_npu 的 AscendC 算子约定相反（`[.., HV, V, K]`）**——两个 "fla 系" 的布局约定不一致。rtp-llm 的池按 eager 消费者（fla_npu）建，图路径借用的 sgl kernel 按上游约定寻址，转置就是这个**双消费者布局冲突**的适配代价。vllm-ascend 通过"单一消费者（eager/graph 同一自研算子）+ 池按算子约定建"从根上避免了转置（零转置、零 gather、零页搬运——它的状态也不分页，每请求固定槽）。若 fla_npu 未来对齐上游约定、或 rtp-llm 落地方案 C（统一布局+单一消费者），本节整个转置适配层可消失。

```python
def fused_recurrent_gated_delta_rule_graph(alog, dt_bias, q, k, v, a, b, ssm_states, ...):
    read_idx, write_idx = _decode_state_indices(...)
    _seed_write_pages(ssm_states, read_idx, write_idx)

    # 布局适配: rtp 池 [pages,HV,V,K] → kernel 要 [pages,HV,K,V]
    ssm_k_major = ssm_states.transpose(-1, -2).contiguous()
    o = fused_sigmoid_gating_delta_rule_update_npu(
        A_log=alog, a=a.reshape(-1, num_v_heads), dt_bias=dt_bias,
        q=q, k=k, v=v, b=b.reshape(-1, num_v_heads),
        initial_state_source=ssm_k_major,
        initial_state_indices=write_idx.to(torch.int32),   # ← device 索引！
        scale=scale, use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    ssm_states.copy_(ssm_k_major.transpose(-1, -2))        # 写回原布局
    return o.reshape(batch, *o.shape[1:])
```

kernel 内部（sgl 移植，`sgl/fused_sigmoid_gating_recurrent.py`）的关键：
- `idx = tl.load(h0_indices + i_n)`——device 读状态页号；`idx >= 0` 才处理（padding 安全）
- gating 融合在 kernel 里，公式与 eager 的 `fused_gdn_gating` 一致：`g = -exp(A_log)*softplus(a+dt_bias)`、`beta = sigmoid(b)`
- delta rule 单步：`h *= exp(g); v -= (h·k); v *= beta; h += k⊗v; o = h·q`
- **读和写回同一个 idx 槽**（in-place 语义）

两次转置拷贝（每层每步 2×约1MB）都是普通 device op，可入图；对 120ms/step 的整体耗时影响可忽略。

### 2.3 与 rtp-llm 现有机制的配合

```
C++ AscendGraphRunner                     Python 模型层
─────────────────────                     ─────────────────────────────
capture:                                  Qwen3NextModel.forward
  initCaptureAttentionInputs                ├─ full-attn 层(10个): FIA 图分支
    块表 ones 初始化(page 1 合法)            │    （graph_task_group/update 机制，另有修复）
  captureDecode per bs:                     └─ GDN 层(30个):
    prepare_fmha_impl(inputs, true)             ├─ eager(warmup): fla_npu AscendC
    warmup ×2  ──────────────────────────────→  └─ capture: graph_compat.py
    capture ×1（capturing=True）──────────→        （sgl triton kernel + torch conv）
                                                  ↓ 录入图
replay:                                    （Python 不执行）
  prepareInputs:
    copyTensorSlice 刷新 block_map ──┐
    sequence_lengths_plus_1_d 刷新 ──┼─→ 图内 gather 重算 read/write 页
    prepare_cuda_graph(graph_bs)     │    → kernel 读写新页
      update_graph_fia(仅 FIA 用) ───┘
  graph.replay()
```

依赖的其他修复（详见《图模式叠加Qwen3.5验证总结.md》）：
- C2：capture 块表 `ones` 初始化（page 0 是非法槽，GDN 算子会拒绝）
- C4：`copyTensorSlice` 支持 3D 分组表（hybrid 的 `[group,batch,cols]`）
- C5：capturing 兜底（本方案的分流判断）
- C7：bpk（kernel/phys 分裂比）从 `attn_configs` 取真实值——图内 slot 映射的正确性前提
- C8：本节（5）的布局转置

### 2.4 验证结果

| 验证 | 方法 | 结果 |
|---|---|---|
| GDN 单算子 | `fused_recurrent_gated_delta_rule_graph` vs fla_npu `fused_recurrent_gated_delta_rule`（同输入/同状态池/同 block_map，NPU 真机） | 输出 err 0.25%、state err 0.13%（bf16 正常水平；修复布局前 133%） |
| conv 单算子 | `causal_conv1d_update_graph` vs fla_npu `causal_conv1d_update` | 输出 err 0.29%、状态页 err 0 |
| 端到端 | 图模式 vs eager 的 greedy 逐 token | **完全一致**；50 / 1024 token（跨 1024 物理块边界）输出正常，~120ms/token |

---

## 三、方案评估：这是不是最好的方案？有没有改动更小的方案？

### 候选方案对比

| | **A. 当前方案**（图内 device 元数据：sgl triton kernel + torch conv） | **B. GDN 层不入图**（分段图） | **C. 改造 AscendC 算子**（device 元数据 + tiling cache） | **E. vllm-ascend 路线**（host 元数据 AscendC + graph task update） | **D. 图外预算 indices**（host 算好塞固定 buffer） |
|---|---|---|---|---|---|
| 思路 | capture 时切到全 device 算子实现，整图照常录 | 把 30 个 GDN 层切到图外，只录其余部分 | 给 fla_npu 算子加 device tensor 接口（参考 sgl-kernel-npu csrc） | host 元数据算子照用：capture 时 graph_task_group 登记为可更新 task，replay 前在 update stream 上带新 host 参数重新 launch，CANN 重跑 tiling 热更新 task；元数据构建（含 D2H/tolist）全部移出 capture 区 | replay 前图外算好页索引，写入固定 device buffer |
| 能否解决 | ✅ 已验证 | ✅ 理论可行 | ✅ | ✅（vllm-ascend 有 e2e 背书：Qwen3-Next FULL_DECODE_ONLY 图模式） | ❌ **不成立**：AscendC 算子的 aclnn 接口只收 host int 数组，不读 device buffer；必须配 A/C/E 的机制才闭环 |
| 改动量 | **中**（~1060 行新代码，不动 runner/算子上游） | **大**（AscendGraphRunner 架构级重构：分段 capture/replay、中间 tensor 地址管理） | **大**（AscendC kernel 开发 + SoC 绑定构建 + tiling 的 graph cache 机制，周级工作量） | **大**（task update 机制：params/handles/events 三列表管理 + ExternalEvent 跨流顺序 + PAD_SLOT_ID padding 对齐 + 双缓冲 pinned host 池；rtp-llm 的 FIA 已在用同套 graph_task API，技术栈相通） | 小（但单独不成立，见上） |
| 性能 | graph 路径：triton kernel + 2 次转置拷贝（每层每步 ~2MB），实测整体 ~120ms/token，无感 | 每步 30 次 GDN 层图外执行 + 30 次图切换，**开销可能抵消图的收益**；且 SGLang 的 split op 只用于 extend/piecewise，decode 无先例 | **最优**（AscendC 原生、无转置、eager/graph 单一路径） | AscendC 原生性能；replay 前每层一次 task update launch（独立 stream 与 replay 并行可隐藏） | — |
| 正确性风险 | 双实现路径需对拍（已有单测覆盖） | 图边界的状态传递易引入新 bug | 算子级开发，需完整 golden 对拍 | task update 与 replay 的跨流顺序、padding 对齐（`qsl[last]==cuSeqlen` 校验）、双缓冲防覆盖——机制较重，易出隐蔽 bug | — |
| 维护性 | 双路径（eager AscendC / graph triton）+ 引入 sgl 代码拷贝 | runner 复杂度大增 | 单一路径，**长期最好** | eager/graph 走同一算子（无双路径），但 update 机制本身是新维护面 | — |
| 先例背书 | SGLang 生产 decode 同款思路 | 无 decode 先例 | sgl-kernel-npu csrc 有完整参考实现 | **vllm-ascend 生产路线**（PR #8842） | SGLang backend 层类似（但配 device kernel） |

> **方案 E（vllm-ascend）补充说明**：vllm-ascend 是**混合策略**——recurrent 算子做成了 device 元数据 AscendC（`csrc/attention/recurrent_gated_delta_rule/`，tiling 只依赖 shape，图内零特殊处理，思路同方案 C）；conv1d 因 tiling 是 `ValueDepend`（host 读值），走 host IntArrayRef + graph task update 补丁。**历史演进值得注意**：vllm-ascend 在 PR #8842 之前 conv1d 就是 triton device 元数据版（即 rtp-llm 当前的方案 A 路线），后来为解决 **triton 重编译问题**才换成 AscendC + task update。即方案 A 是被 vllm-ascend 验证过的早期路线，方案 E 是其性能演进版。

### 结论与建议

1. **对"验证图模式叠加 Qwen3.5 可行性"这个目标，方案 A 是当时的最优落地选择**：改动可控（不动 runner、不动上游算子）、有 SGLang 生产背书、单测+端到端可闭环，且已经跑通验证。改动更小的方案里：
   - **D（图外预算 indices）单独不成立**——AscendC 接口不读 device 内存，这是死结本身；
   - **B（分段图）表面"改动小"实则是架构重构**，且性能存疑，不建议；
   - 真正改动更小的只有"**GDN decode 完全 fallback eager（放弃这 30 层入图）**"，等于放弃图模式对 GDN 的收益，与任务目标矛盾。

2. **长期工程化的两条终态路线**（按团队能力二选一）：
   - **方案 C**（AscendC 算子 device 元数据化，参考 sgl-kernel-npu csrc：`__gm__ int32_t*` + `GetValue` 读索引；**graph mode tiling cache**：tiling 按参数 hash 一次 H2D 进持久 device buffer `globalTilingBuffer`，且必须用 torch_npu dispatch 而非裸 `aclrtMemcpy`，否则破坏图内流序）。顺便统一 state 布局可消除 A 方案的转置拷贝。收益：eager/graph 单一路径、性能最优、去掉 sgl 代码拷贝依赖。前提：有 AscendC 算子开发能力和 SoC 构建环境，适合立项排期。
   - **方案 E**（vllm-ascend 的 task update 路线）：不改算子接口、host 元数据照用，靠 `torch.npu.graph_task_group/update` 在 replay 前打补丁。rtp-llm 的 FIA 图模式**已经在用同一套 API**（`update_graph_fia`），向 GDN 的 conv1d 推广在技术上顺理成章；代价是要补齐 padding 对齐、双缓冲 pinned 池、ExternalEvent 跨流序这一整套机制（vllm-ascend 的 `vllm_ascend/ops/gdn.py` + `patch/worker/patch_gdn_attn.py` 是完整参照）。
   - **触发时机参考**：如果方案 A 在压测中暴露 **triton 重编译/启动慢**问题（vllm-ascend 换路线的直接动因），优先考虑 E（conv1d 换 AscendC + task update）；如果是**纯性能/单一路径维护**诉求，优先 C。

3. **方案 A 的已知代价**（可接受，但要知情）：
   - 双实现路径的维护成本：eager（fla_npu AscendC）与 graph（sgl triton）语义必须保持对拍，建议把本文 2.4 的两个单测固化进 CI；
   - `sgl/` 目录是 sgl-kernel-npu 的代码拷贝（已去掉 `sgl_kernel_npu` 包依赖），后续 sgl 上游演进不会自动同步；
   - 每层每步 2 次转置拷贝（池越大越贵，实测为图模式与 eager 差距的第一贡献项；页级 gather 版已验证可消除 35ms/token——详见《图模式GDN性能优化探索记录》）。
   - **triton-ascend 3.2.2 缺陷警示**：`tl.store` 写非标准 stride 的 storage 视图（如本项目的 typed_storage_view 池）会静默丢弃——kernel 直写池视图不可行，须走 gather→连续 buffer→scatter，且单测必须用 strided 池模拟真实布局。

---

## 四、文件清单

| 文件 | 行数 | 内容 |
|---|---|---|
| `rtp_llm/models_py/kernels/ascend/graph_compat.py` | 134 | 图模式 GDN 实现：`_decode_state_indices` / `_seed_write_pages` / `causal_conv1d_update_graph` / `fused_recurrent_gated_delta_rule_graph` |
| `rtp_llm/models_py/kernels/ascend/sgl/fused_sigmoid_gating_recurrent.py` | 242 | sgl 的 triton kernel（device 索引 + 融合 gating + in-place 状态更新） |
| `rtp_llm/models_py/kernels/ascend/sgl/fla_utils.py` | 683 | sgl 的工具（`input_guard`、`get_device_properties` 等，已内联去依赖） |
| `rtp_llm/models_py/model_desc/qwen3_next.py`（修改） | ~40 | decode 的 `_conv1d` / `_fla` 增加 `use_graph_path` 分流 |

> 验证期临时加的 hidden dump / 诊断打印等调试代码已全部清理（`git diff` 确认干净），当前代码只保留功能性实现。
