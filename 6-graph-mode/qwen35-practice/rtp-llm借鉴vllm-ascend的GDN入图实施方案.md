# rtp-llm 借鉴 vllm-ascend 的 GDN 图模式实施方案（终态形态验证记录）

> 时间：2026-09-17 ~ 2026-09-18
> 关联：《rtp-llm借鉴SGLang的GDN入图实施方案.md》（sglang方案）、《图模式GDN性能优化探索记录.md》
> 代码快照：`code_snapshots/graph_compat_vllm_ascendc.py`、`code_snapshots/conv_update_kernel_triton.py`
> 状态：**已完整验证（数值 bit-exact、图 capture/replay 通过），但因服务级性能低于 sgl 方案，未作为生产版本保留**——本文档是完整的技术记录，供后续演进（方案C/单一消费者）复用。

---

## 一、方案本质（与 sgl 方案的根本区别）

| | sgl 方案（生产版） | **vllm 方案（本记录）** |
|---|---|---|
| recurrent kernel | sgl triton kernel（`fused_sigmoid_gating_delta_rule_update_npu`，state布局`[HV,K,V]`） | **fla_npu AscendC 原生算子**（与 eager 同一算子，state布局`[HV,V,K]`=rtp池页内布局） |
| 单一消费者 | 否（eager=AscendC / graph=triton 双路径） | **是**（eager/graph 同一 AscendC 算子）——vllm-ascend 形态 |
| 转置 | 需要（kernel布局≠池布局） | **不需要**（布局天然匹配） |
| 数值 | err 0.25%（两套实现的bf16差异） | **bit-exact 0.0**（同算子） |
| 算子可capture | sgl kernel device元数据✓ | **fla_npu AscendC 本身就是 device 元数据**（`ssm_state_indices`/`actual_seq_lengths`收device tensor，eager的D2H只发生在Python适配层`_resolve_state_pages`的`.cpu().tolist()`，不在算子内）——**这是本方案的核心发现** |

**核心洞察**：fla_npu 的 recurrent AscendC 算子天然可被 aclgraph 捕获（单测验证 capture+replay 通过），死结只在 Python 适配层的 host 计算——把它换成 device 计算（`_decode_state_indices` 的 gather）即可。

## 二、实施内容（四层）

### 2.1 device 侧页索引（复用 sgl 方案）
```python
_decode_state_indices(block_map, sequence_lengths_plus_1, page):
    read_col  = (len - 2).clamp_min(0) // page
    write_col = (len - 1).clamp_min(0) // page
    # block_map.gather → read_idx / write_idx（图内自动随 capture buffer 刷新）
```

### 2.2 块级页访问（profiling 驱动的关键优化）

**问题链**（profiling 实锤，三次迭代）：
| 版本 | 实现 | 服务速度 | 根因 |
|---|---|---|---|
| v1 | `ssm_states.index_select(0, idx)` | **1757 ms/token** | strided 池视图上 index_select 退化为元素级拷贝（Slice/ViewCopy 占 81%，单次 2.5-3ms） |
| v2 | `F.embedding(idx, row_view)` + `row_view.index_copy_` | **388 ms/token** | 2D 页行视图上 embedding/2D index_copy 走**块级行拷贝**（0.04ms，**快 88 倍**） |
| v3 | v2 + 段赋值改整行 cat | 385 ms/token | `row[:, off:off+len] = x` 会降级为 `aclnnInplaceCopy_Slice`（跨行 strided 段拷贝，单次 5.9ms）——改 `torch.cat([前段, 新段, 后段])` |

**2D 页行视图推导**（`_paged_row_view`，capture 时执行一次）：
```python
def _paged_row_view(state_view):
    """从 4D strided 池视图反推 2D 页行视图 [pages, page_stride] + 段偏移/长度
    （typed_storage_view 布局：每页 = conv段 + ssm段）"""
    page_stride = state_view.stride(0)
    off = state_view.storage_offset()
    seg_offset = off % page_stride
    base = off - seg_offset
    seg_len = prod(state_view.shape[1:])
    row_view = torch.empty(0, dtype=..., device=...)
    row_view.set_(state_view.untyped_storage(), base,
                  (state_view.shape[0], page_stride), (page_stride, 1))
    return row_view, seg_offset, seg_len
```

### 2.3 recurrent 主链路（AscendC 直调）
```python
# gather（块级）+ seed（跨块 where）+ kernel（与 eager 同算子）+ scatter（块级整行）
row = F.embedding(write_idx, row_view)
read_row = F.embedding(read_idx, row_view)
seg = torch.where(same, row段, read_row段)           # 跨块 seed
row = torch.cat([row[:, :off], seg, row[:, off+len:]], dim=1)  # 防止 InplaceCopy_Slice
pages = row[:, off:off+len].reshape(batch, HV, V, K)  # 段内连续，与 row 共享存储
result = ascendc.npu_recurrent_gated_delta_rule(
    q, k, v, pages, beta, g,                              # g/beta 用 torch 公式（同 eager fused_gdn_gating）
    actual_seq_lengths=cat([zeros(1), ones(batch)]),      # 图内 device 构造（decode 恒 T=1，静态）
    ssm_state_indices=arange(batch))                      # pages 是紧邻小 buffer
row_view.index_copy_(0, write_idx, row)                    # 块级整行 scatter
```

### 2.4 conv（两版都验证过）
- **torch 版**：gather → 窗口卷积（cat/mul/sum/silu）→ scatter——注意段存储序是 `[state_len, dim]`（调用方传 transposed 视图），reshape 后要 transpose 回来
- **triton 版**（`conv_update_kernel.py`，从 sgl-kernel-npu 提取）：gather 出连续行后 kernel 只写连续 buffer（**绕开 triton-ascend strided store 缺陷**），环形写位置 = `cache_seqlens % state_len`（=滑动窗口语义，已对拍验证 err 0.0）

## 三、验证结果

| 验证 | 结果 |
|---|---|
| 数值（vs eager AscendC 全链路） | **bit-exact 0.0**（紧邻池/strided池/conv-first布局/跨块/混合 全场景） |
| 图 capture/replay | 通过（`test_terminal_graph.py`：图内3步状态正常累积） |
| 单层性能（真实 strided 250 页布局） | recurrent 1.22ms + conv 0.75~0.98ms |
| 服务速度 | **385 ms/token** |

## 四、为什么最终没采用（vs sgl 方案的实测对比）

| | sgl 方案（页级gather版） | vllm 方案（块级版） |
|---|---|---|
| **服务速度** | **167 ms/token** | 385 ms/token |
| 单层 recurrent | ~0.6ms（sgl triton kernel 本体 0.036ms + gather/转置开销） | 1.22ms（**AscendC 算子在图内的执行比 sgl triton 慢**——aclnn 算子录制重放的开销） |
| 数值 | 0.25%（两实现 bf16 差异） | bit-exact |
| 单一路径 | 否 | 是 |

**根因**：vllm-ascend 的"单一消费者"形态在 **vllm 的自研算子+标准 tensor 池**上是成立的；rtp-llm 复用 fla_npu 算子时，**池的 strided 视图**（typed_storage_view）迫使增加块级 gather/scatter 适配层，而 **fla_npu AscendC 算子在本机（950PR/CANN 9.2）的图内重放性能低于 sgl triton kernel**——两笔开销叠加后不敌 sgl 方案。

## 五、后续演进价值（何时回到这个方案）

1. **C++ 池改造后**（ssm 独立连续池、布局统一）：gather/scatter/seed 全消失，AscendC 直读直写——那时单一消费者的收益（bit-exact、无双路径维护）才能兑现
2. **fla_npu 算子图内性能问题解决后**（aclnn 重放开销——可能与 capture 模式/auto_dispatch 相关，vllm 用 `torch.npu.graph(auto_dispatch_capture=True)` 正常）
3. **关键复用资产**：`_paged_row_view`/块级 embedding/整行 cat/`InplaceCopy_Slice` 规避——这些技巧对所有 strided 池操作普适（sgl 方案的未来优化也用得上）

## 六、代码位置

- 完整实现快照：`code_snapshots/graph_compat_vllm_ascendc.py`（含块级访问全链路）
- conv triton kernel：`code_snapshots/conv_update_kernel_triton.py`
- git 暂存区曾持有的版本：vllm 块级版（本快照同源）
- 测试脚本（/tmp/opencode，建议固化）：`test_terminal_graph.py`（图内验证）、`test_stride.py`（strided池）、`test_mixed_graph.py`
