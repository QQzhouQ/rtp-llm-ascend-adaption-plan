# 多 BS 并发 Prefill 乱码精度问题 — 修复记录

> 时间：2026-09-28 ~ 2026-09-29
> 分支：`perf/fia-kv-kernel-addressing`（PR#54，`79d7554a7`）
> 模型：Qwen3.5-35B-A3B（bf16，单卡 Ascend950PR）
> 启动参数：`--enable_cuda_graph=true --seq_size_per_block=1024 --kernel_seq_size_per_block=512`
> 前置文档：[pre-existing并发prefill乱码问题调查记录.md](pre-existing并发prefill乱码问题调查记录.md)
> 状态：**已修复，服务端全场景验证通过**

---

## 一、结论摘要

本次乱码由**两个相互独立的 bug 叠加**造成，均与 MTP / 图 replay 无关，属 PR#54 基线的 pre-existing 问题：

| | Bug 1：FIA 层 KV 读写布局错配 | Bug 2：solve_tril varlen 调用姿势错误 |
|---|---|---|
| 位置 | `ascend_impl/ascend_prefill.py` | `kernels/ascend/linear_attention.py` |
| 引入 | PR#54（`79d7554a7`）只统一了 KV 写与 decode 读，漏改 prefill 读 | PR#53 迁移时 varlen 误用 `bsnd` 布局 |
| 机制 | prefill 读仍按旧物理布局重排 K/V，与新写入的 kernel-block 布局错位 | fla_npu `npu_solve_tri` 仅 TND 布局支持 `cu_seqlens`；`bsnd` 下全局 64 对齐切块、忽略序列起点 |
| 触发 | 物理块复用带脏数据（块池与 GDN 组共享）/ prompt 跨 512 token 边界 | 合批 prefill 中**非首序列**（起点非 64 对齐） |
| 表现 | 串行也偶乱、长 prompt（>512）确定性错、ceval 失败 | 仅并发合批乱、乱码确定性重复、续写退化为重复循环 |
| 修复 | prefill 读改为 kernel-block 半区零拷贝视图（与写/decode 读一致） | varlen 时 flatten 成 3-D `[T,H,CS]` + `layout="tnd"` 单次调用 |

修复后离线对拍全部 **bit-exact**，服务端验证：串行 ×5、长 prompt ×3、并发 ×4 ×12 轮、长短混合 ×4 ×6 轮、并发 ×8（64 token 输出）×8 轮——**全部通过，0 错误**。

---

## 二、调查过程

### 2.1 第一阶段：定位 Bug 1（FIA KV 布局错配）

按前置文档的嫌疑清单逐项排查：

1. **确认分组与块表链路**：`HybridConfigCreator.cc` 中 FULL 组排 group 0（FIA），LINEAR 组 group 1（GDN）；`BlockIds::updateKernelSlotAt` 确认 kernel block id = `phys * bpk + sub`；`getLayerCache`（`OpDefs.h`）把原始 `[P, stride]` 存储reshape 为 `[P*bpk, 2, 512, H, D]` —— 物理块内部布局由 Python 侧寻址定义。
2. **推导 PR#54 前后的寻址配对**（纸面证明）：
   - 旧（PR#53）：写 = 物理视图 + phys-flat slot，读 = `split_kv_kernel_blocks` 重排——两者虽"乱"但**自洽**（存储乱、读写配对一致）；
   - 新（PR#54）：写与 decode 读统一到 `base[:, 0/1]` kernel 半区 + kernel-flat slot（**正确布局**），但 **prefill 读仍是旧重排**：
     - 偶数 kernel 块（块内位置 0–511）的 **V** 读到物理块 sub1-K 区；
     - 奇数 kernel 块（位置 512–1023）的 **K** 读到 sub0-V 区。
3. **闭环解释全部现象**：
   - 新块该区域为零 → FIA 层近似失效（输出≈0，残差流保留）→ 串行短请求"碰巧"仍对（GDN 扛住）；
   - `HybridTypeKVCacheAllocator.cc:214` 注释确认 **FULL 组与 LINEAR 组共享同一物理块池** → 并发时新请求复用到曾是 GDN 状态页/长序列的块 → FIA 读到非零残留 → 乱码首 token；是否命中取决于分配/释放顺序，故间歇性；
   - prompt >512 token 跨 kernel 块边界 → 确定性读错 → ceval CI 失败。
4. **NPU 离线验证**（`/tmp/opencode/test_prefill_kv_pairing.py`）：脏块池 + 3 序列合批（含跨 512 边界）场景，旧读法 V 偏差 **458.6**（乱码完整复现），新读法 vs fp32 参考 **8.6e-03**（bf16 精度内）。

### 2.2 第二阶段：服务端首验 → 发现 Bug 2

部署 Bug 1 修复后重启服务（已确认进程 cwd 指向仓库、加载修复版文件）：

- 串行全对 ✓、长 prompt（>512）全对 ✓ → **Bug 1 修复生效**；
- 并发 ×3/×4 **仍乱**，且乱码文本按 prompt 呈确定性重复；乱码请求续写退化为重复循环（`3 3 3...`、`222...`）→ 指向 GDN 递归状态被污染。

### 2.3 第三阶段：离线对拍定位 Bug 2

1. **排除法收窄**：错峰 200ms 全对（不合批→无乱码）、×2 偶发、×3 高频 → **合批是必要条件**；MoE 对 token 独立且串行长 prompt 已验证 → 嫌疑集中于 GDN 多序列 varlen 路径（前置文档嫌疑 3/4）。
2. **GDN kernel 逐一对拍**（多序列一次跑 vs 逐序列独立跑，NPU 实测）：
   - `causal_conv1d_fn`：diff = 0 ✓
   - `chunk_gated_delta_rule`：**seq0 bit-exact 正确，seq1/seq2 输出误差达输出幅度的 ~20 倍、final state 偏差 ~1.4** ✗
3. **逐子 kernel 分叉定位**（cumsum → kkt → solve_tril → w/u → fwd_h → fwd_o）：
   - cumsum/kkt：diff = 0 ✓；给定相同输入时 fwd_h/fwd_o 也 bit-exact ✓
   - **`solve_tril`：首个且唯一的分叉点**（diff = 1.115）
4. **决定性实验**：`T_multi == T_global`（与"整段按全局 64 切块"bit-exact 相同）→ **kernel 忽略 cu_seqlens，把 chunk 定位在 `chunk_idx * 64`**；seq0（起点 0 对齐）恒对，起点 61/119 的序列被跨界污染。

### 2.4 第四阶段：查 fla-npu 源码定性（`/home/d30033799/flash-linear-attention-npu`）

- `solve_tri_tiling.cpp:89`：`isVarlen = (layoutMode == 2 || 3)` —— **只有 TND/NTD 布局支持 varlen**；`bsnd` 走 `numChunks = (T + chunkSize-1)/chunkSize` 全局对齐切块，cu_seqlens 被完全忽略；
- `solve_tri_cube.h:310-317`（TND offset）：`(cu_seqlens[seq] + chunk_in_seq*BT) * H*BT + h*BT` —— 正确支持非对齐起点；
- `op_api/solve_tri.cpp`：host aclIntArray 经 `ConvertToTensor(DT_INT64)` 转 device buffer 供 kernel GM 读取；
- `docs/aclnnSolveTri.md`：明确"变长模式：当 layout 为 tnd 时，cu_seqlens 和 chunk_indices 必须提供"。

**结论：不是 kernel bug，是 rtp-llm 包装层 varlen 误传 `layout="bsnd"` + 4-D `[1,T,H,CS]`，kernel 静默忽略 cu 按全局切块。**

---

## 三、根因详述

### Bug 1：FIA 层 KV 读写布局错配

PR#54 将 KV 写入（`AscendKVCacheWriteOp`）与 decode 读取统一到 kernel-block 半区寻址：

```
写：slot = kb * 512 + offset（或 phys-flat，等价分解）→ base[:, 0] / base[:, 1]
读：base[:, 0].reshape(num_kernel, 512, H*D)（零拷贝视图）+ kernel 块表
```

物理块内部实际布局为按 kernel 子块交错：`[sub0K(512), sub0V(512), sub1K(512), sub1V(512)]`。

但 prefill 读仍走旧的 `split_kv_kernel_blocks`：它把物理块重排为 `[K(1024), V(1024)]` 语义（`phys = base.reshape(P, 2, 1024, H, D)` 再切片），导致：

```
k_cache[2p+1] = sub0V 区（正确应为 sub1K）      v_cache[2p] = sub1K 区（正确应为 sub0V）
```

由于 FULL 组与 LINEAR 组共享块池（`HybridTypeKVCacheAllocator.cc`："All groups share the same block pool"），复用块上残留的 GDN conv/ssm 状态数据或前一个长序列的 K 尾部会被 FIA 当作 V 参与注意力计算。

### Bug 2：solve_tril varlen 布局错误

Qwen3.5 的 GDN prefill 链路：`chunk_local_cumsum → chunk_scaled_dot_kkt → solve_tril → recompute_w_u → fwd_h → fwd_o`。

`solve_tril` 包装层在 varlen（多序列 cu_seqlens）时传 `layout="bsnd"` + 4-D 输入，kernel 因此按 token 轴全局 64 对齐切块并忽略序列起点。合批中：

- seq0 起点 0（对齐）→ 恒正确；
- seq1+ 起点非 64 对齐（如 61、119）→ chunk 跨序列边界读取 → `(I+A)^{-1}` 结果错 → w/u 错 → **36 个 GDN 层全部输出错 + final state 错** → 首 token 乱码、错误状态写入状态页、decode 续写持续乱并陷入重复循环。

这解释了：串行/错峰全对（单序列恒为 seq0）、×2 偶发（合批概率低）、×3+ 高频、乱码确定性重复（分配模式确定则错误确定）。

---

## 四、解决方案

### 4.1 Bug 1 修复（`rtp_llm/models_py/modules/factory/attention/ascend_impl/`）

`ascend_prefill.py` —— prefill 读与写/decode 读统一（与 `ascend_decode.py:_split_kv` 相同）：

```python
def _split_kv(self, kv_cache):
    base = kv_cache.kv_cache_base
    k_cache = base[:, 0].reshape(base.shape[0], base.shape[2], -1)
    v_cache = base[:, 1].reshape(base.shape[0], base.shape[2], -1)
    return k_cache, v_cache, base.shape[2]
```

配套清理：
- `ascend_attn_params.py`：删除语义为旧布局的死代码 `_physical_kv_view` / `split_kv_physical` / `split_kv_kernel_blocks`（防止误用）；
- `ascend_decode.py`、`ascend_kv_cache_write_op.py`：删除死导入。

### 4.2 Bug 2 修复（`rtp_llm/models_py/kernels/ascend/linear_attention.py`）

`solve_tril` varlen 改用 TND 布局单次调用（一次 kernel launch、零额外拷贝）：

```python
if cu is None:
    # 非 varlen：保持原 bsnd 路径
    out = ascendc.npu_solve_tri(x=x, cu_seqlens=None, chunk_indices=None, layout="bsnd")
else:
    # varlen：flatten [1, T, H, CS] -> 3-D TND，kernel 才会按 cu_seqlens 定位 chunk
    out = ascendc.npu_solve_tri(
        x=x.reshape(seqlen, A.shape[2], chunk_size),
        cu_seqlens=cu,
        chunk_indices=_prepare_chunk_indices(cu, chunk_size),
        layout="tnd",
    )
return out.reshape(A.shape).to(output_dtype)
```

> 备注：调查中间版本曾采用"按序列分段调用"的 workaround，查证 fla-npu 源码后升级为上述 TND 正解。

### 4.3 新增回归测试

| 测试 | 覆盖 |
|---|---|
| `ascend_impl/test/test_prefill_kv_pairing.py`（NPU） | 脏块池 + 多序列合批（含跨 512 边界序列）下，真实写 op + prefill FIA 读 vs fp32 参考（修复前同场景 V 偏差 458.6，修复后 8.6e-03） |
| `kernels/ascend/test/test_linear_attention.py::TestChunkVarlenMultiSequenceNPU`（NPU） | 合批 GDN chunk vs 逐序列 **bit-exact**：`[61,58,47]` / `[300,700,40]` / `[17,9]` |
| `test_linear_attention.py` mock 用例 | varlen 走 tnd 布局、非 varlen 保持 bsnd、chunk_indices 编码正确 |

全部 44 个 Ascend 测试通过（含既有 plan_c NPU device path 等套件）。

---

## 五、服务端验证结果（最终）

| 场景 | 轮数 | 结果 |
|---|---|---|
| 串行 ×4 prompts | 5 | 全对 |
| 长 prompt（>512 tok）串行 | 3 | 全对 |
| **并发 ×4**（原失败场景，修复前 28 错） | **12** | **全对** |
| 长短混合并发 ×4 | 6 | 全对 |
| 并发 ×8（max_tokens=64，长输出） | 8 | 全对，无退化重复 |

复现脚本同前置文档第三节（注意校验关键词需兼容 `H₂O` 下标字符、数字答案等写法，避免误报）。

---

## 六、经验与建议

1. **排障利器**：多序列 vs 逐序列的离线对拍（bit-exact 断言）+ 逐子 kernel 分叉定位，比逐层 hidden dump 更快锁定 `solve_tril`；"seq0 恒对、后续序列错"是 varlen 边界类 bug 的典型指纹。
2. **布局约定必须全链路一致**：KV cache 的物理布局由 Python 侧寻址隐式定义，改写路径时必须同步审计所有读路径（本例 prefill 读被遗漏）。
3. **读外部 kernel 文档/源码**：fla-npu 的 `aclnnSolveTri.md` 与 tiling 源码明确写了 varlen 仅支持 tnd——调用侧传错布局时 kernel **静默产出错误结果**，已向 fla-npu 提交 issue（见第七节附录，[#819](https://github.com/flashserve/flash-linear-attention-npu/issues/819)）。
4. **MTP 分支同步**：`feature/mtp-ascend` 上已独立修复 Bug 1（`ascend_prefill.py` 中同样的零拷贝视图，内联写法），语义与本分支一致，rebase 无实质冲突；**Bug 2 的 tnd 修复 MTP 分支尚未包含，rebase 时以本分支版本为准**。
5. 部署注意：`rtp_llm` 在 site-packages 另有 pip 安装副本，且环境中存在多个 checkout（`PYTHONPATH=/home/d30033799/rtp-llm` 指向旧分支）——验证修复是否生效时先确认服务进程实际加载的模块路径。

---

## 七、附录：已提交上游 issue（fla-npu #819，中文版）

> 提交时间：2026-09-29，账号 QQzhouQ
> 链接：https://github.com/flashserve/flash-linear-attention-npu/issues/819
> 主题：Bug 2（`solve_tril` varlen 布局）的上游反馈——`bsnd + cu_seqlens` 组合静默产出错误结果

**标题**：[bug] aclnnSolveTri：layout="bsnd" + cu_seqlens 组合会静默产出错误结果——应在 op_host 参数检查中拒绝

### 组件

`aclnnSolveTri` / `npu_solve_tri`（`fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri`）

### 环境

- CANN 9.2.0，Ascend 950PR
- master 主线（验证于源码与本仓 `solve_tri` 算子一致的仓内副本）

### 摘要

当 `layout="bsnd"` 与 `cu_seqlens`/`chunk_indices` 一起使用时，算子会**静默忽略** varlen 元数据，并在（展平后的）token 轴上按 `chunk_idx * chunk_size` 定位切块。对于后续序列起点未按 chunk 对齐的 varlen 批次（例如 `cu_seqlens=[0, 61, 119, 166]`，`chunk_size=64`），求解会**跨序列边界读取**，并**在没有任何报错的情况下**返回错误结果。

### 根因

- `solve_tri_tiling.cpp`：`isVarlen = (layoutMode == 2 || layoutMode == 3)`——只有 TND/NTD 会使用 `cu_seqlens`。BSND 模式下 tiling 落入静态分支 `numChunks = (T + chunkSize - 1) / chunkSize`（全局对齐切块），`cu_seqlens`/`chunk_indices` 从未被读取。
- `solve_tri_cube.h`（BSND 偏移计算）：`b*T*H*BT + chunk*BT*H*BT + h*BT`——完全不涉及 `cu_seqlens[seq]` 的序列起点。

文档（`docs/aclnnSolveTri.md`）已明确 varlen 模式要求 `layout="tnd"`（"变长模式：当 layout 为 tnd 时，cu_seqlens 和 chunk_indices 必须提供"），但 host 侧参数检查中没有任何地方拒绝非法的 `bsnd + cu_seqlens` 组合。

### 复现步骤

```python
import torch, torch_npu
from fla_npu.ops import ascendc

dev, CS = "npu", 64
seq_lens = [61, 58, 47]           # 起点 61 / 119 未按 64 对齐
cu = [0, 61, 119, 166]
chunk_indices = [0, 0, 1, 0, 2, 0]

A = torch.randn(1, sum(seq_lens), 32, CS, device=dev, dtype=torch.float32)

out_bsnd = ascendc.npu_solve_tri(
    x=A.to(torch.float16), cu_seqlens=cu, chunk_indices=chunk_indices, layout="bsnd")
out_global = ascendc.npu_solve_tri(
    x=A.to(torch.float16), cu_seqlens=[0, sum(seq_lens)],
    chunk_indices=[0, 0], layout="bsnd")

print(torch.equal(out_bsnd, out_global))   # -> True：与整批全局切块的结果 bit 级一致

# 与逐序列单独调用对比：
#   seq0（起点 0，对齐）：完全一致
#   seq1（起点 61）：错误
#   seq2（起点 119）：错误
```

### 影响

传入 4-D `[1, T, H, CS]` 张量 + `layout="bsnd"` **且**带 varlen `cu_seqlens` 的调用方（例如推理框架中 Gated Delta Net 的多序列合并 prefill），对于每个起点未按 chunk 对齐的序列都会得到**静默损坏**的输出。在我们的案例中（rtp-llm，Qwen3.5 GDN prefill），这导致合批中所有非首序列的全部 GDN 层输出和最终状态损坏——在上层表现为并发时生成内容乱码，而串行/单序列请求始终正确。

### 建议

尽早拒绝非法组合，例如在 `aclnnSolveTriGetWorkspaceSize` / op_host 参数检查中：

```cpp
if (cuSeqlens != nullptr || chunkIndices != nullptr) {
    if (layout == "bsnd" || layout == "bnsd") {
        return ACLNN_ERR_PARAM_INVALID;  // 或 OP_LOGE 后快速失败
    }
}
```

或者让 BSND 模式也支持 `cu_seqlens`。无论哪种，当前"静默出错误结果"的行为都是最糟的选项——因为错误输出是 bit 级稳定的，在对齐 shape 的冒烟测试中很容易通过。

（参考：rtp-llm 侧的下游修复已将 varlen 调用切换为受支持的 `tnd` 布局——把 `[1, T, H, CS]` 展平为 `[T, H, CS]`——恢复了逐序列 bit 级精确的结果。）
