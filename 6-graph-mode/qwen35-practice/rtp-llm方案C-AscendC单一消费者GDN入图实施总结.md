# rtp-llm 方案C：AscendC 算子单一消费者 GDN 入图——实施方案与总结

> 时间：2026-09-20 ~ 2026-09-22
> 落位分支：`/home/d30033799/weishan/rtp-llm` `feature/qwen35-graph-mode`（基于 `ningweikang/main` `6a55e8df3`；两提交：`9678d044c` 方案C 主体 + `465f0d07d` PR #52 优势移植，合计 51 文件 +4942/-211，已推送 `QQzhouQ/rtp-llm` 开 PR 至 `ningweikang/rtp-llm:main`）
> 关联文档：《rtp-llm借鉴SGLang的GDN入图实施方案.md》（方案A）、《rtp-llm借鉴vllm-ascend的GDN入图实施方案.md》（vllm形态验证）、《图模式GDN性能优化探索记录.md》、《图模式叠加Qwen3.5验证总结.md》（C1-C8 问题清单）
> 状态：**已完成并端到端验证**（5 档位 capture、greedy 正常、28~31ms/token，较方案A 5.4x+）；方案A 代码（`sgl/`、`graph_compat.py`）已全部移除；已融合 PR #52（sai841）的四项优势并以 co-author 署名（§十）

---

## 一、背景与方案定位

### 1.1 方案A 遗留的三个结构性代价

方案A（借鉴 SGLang）用"双路径"解决 GDN decode 算子入图：eager 走 fla_npu AscendC 算子（host 元数据），capture 切换到 vendored sgl triton kernel（device 元数据）。其代价：

1. **双实现路径维护**：eager（AscendC）与 graph（sgl triton）语义必须永远对拍，两套实现的 bf16 差异（err 0.25%）永远存在；
2. **布局转置开销**：sgl kernel 按 `[pages,HV,K,V]` 寻址，rtp 池页内是 `[pages,HV,V,K]`，每层每步两次转置拷贝（方案A 性能差距的第一贡献项）；
3. **vendored 代码依赖**：`sgl/` 目录 925 行拷贝，上游演进不自动同步。

### 1.2 方案C 的定义

**eager 与图模式共用同一段代码、同一个 AscendC 算子（单一消费者）**：把死结从"换算子"改为"换元数据来源"——rtp-llm 适配层里所有 host 侧计算（`block_map.cpu().tolist()` 算页号、host 构造索引张量）全部替换为 device 算子，使整步 forward 可被 aclgraph 捕获且 replay 语义正确。

---

## 二、关键前置调研发现（决定方案C形态的三个事实）

对 `/home/d30033799/flash-linear-attention-npu`（main 分支，26.7.0.dev0）的调研结论：

### 2.1 fla-npu 的两个 decode 算子已原生支持 device tensor 元数据

| 算子 | 元数据通道 |
|---|---|
| `npu_recurrent_gated_delta_rule` | **只收 device int32**（`actual_seq_lengths`、`ssm_state_indices`、可选 `num_accepted_tokens`），无任何 host int 数组通道 |
| `npu_causal_conv1d_update` | device tensor（int32/bool）为主通道；host int64 数组通道**已标记废弃**（2026-12 移除），与 device 通道互斥 |

也就是说：**原 C1 问题的死结"算子只收 host int 数组"已在上游解决**，死结只剩 rtp-llm 自己 Python 适配层的 host 计算——方案C 不需要开发任何 AscendC kernel。

### 2.2 UPDATE 模式 tiling 无 D2H，capture 合法

- recurrent 算子 tiling 只读 shape/dtype/stride（强制 cuSeqlens/ssmStateIndices 为 DT_INT32），从不读值；
- conv1d 的 tiling 值读取只发生在 **FN(prefill)+CPU 输入**通道（`GetOptionalInputTensor(QUERY_START_LOC_CPU_INDEX)->GetData<int64_t>()`）；UPDATE(decode) 走 `ChooseCanonicalUpdateBaseDimChoice`（仅 shape）；device 输入因无 `ValueDepend` 标注对 tiling 不可见（`HasVisibleMetadataData` 返回 false）→ **无 D2H，capture 合法**。

### 2.3 布局天然匹配 + 池可直传

- AscendC recurrent 算子的 state 布局 `[pages, HV, V, K]` **恰等于 rtp 池页内布局**（`get_ssm_state_tensor` 定义）——方案A 的 C8 转置问题在方案C 里根本不存在；
- eager 路径**本来就把 strided 池视图（typed_storage_view）直接传给算子**（op def 对 state 有 `IgnoreContiguous`），因此图内同样可以**池视图直传 + device 索引**，彻底消除 vllm 方案（385ms/token 版）的 gather→连续 buffer→scatter 适配层；
- conv 算子要求 `(num_cache_lines, state_len, dim)`——正是池的原始 conv 段视图。

### 2.4 padding 行语义（图内 bs < 档位 bs）

replay 时 padding 行 block_map 被 C++ 清零 → 页号 0：
- conv 算子：`null_block_id=0` 使其整行跳过（不写状态，输出未定义——生产上 padding 输出本被丢弃）；
- recurrent 算子：`seqLen<=0` 才跳过，ctx 构造为全 1 时会处理 padding 行并写 page 0（pad 槽）——无害垃圾，与方案A 行为一致，单测 `test_padding_row_is_isolated` 覆盖"有效行不受影响"。

---

## 三、总体设计

```
                     ┌────────────────────────────────────────────┐
                     │   fused_recurrent_gated_delta_rule /       │
                     │   causal_conv1d_update（kernels/ascend）    │
                     │                                            │
                     │  token_count == 1（标准 decode）            │
                     │    → device 元数据路径（方案C，本文）        │
                     │       device gather 页索引                  │
                     │       device 段内跨块 seed                  │
                     │       AscendC 直调（池视图直传）             │
                     │                                            │
                     │  token_count > 1（MTP/投机，仅 eager）       │
                     │    → 原 host 元数据路径（不变）              │
                     └────────────────────────────────────────────┘
三个阶段的执行形态（无任何 Python 分支）：
  warmup（图外）：同一路径 eager 执行，D2H 合法与否无关（device 路径本无 D2H）
  capture（图中）：同一路径被录制；gather 读 block_map/seq_lens 的 capture buffer
  replay：不进 Python；C++ prepareInputs 刷新 buffer 内容 → 图内 gather 自动得到新页号
```

核心语义（与 eager host 路径 `_resolve_state_pages` 严格一致）：
```
read  page = block_map[b, max(len - 2, 0) // page_size]
write page = block_map[b, (len - 1) // page_size]      （len 即 sequence_lengths_plus_1）
跨块（read != write）时先把 read 页的段内容播种到 write 页，再由算子原地更新
```

---

## 四、代码改动详解（vs `ningweikang/main`）

### 4.1 新增 `rtp_llm/models_py/kernels/ascend/paged_state.py`（+110）

方案C 的共享基建，三个纯 device 函数：

```python
def decode_state_indices(block_map, sequence_lengths_plus_1, seq_size_per_block):
    """device 侧从 block_map 取 read/write 页索引（gather，可捕获、replay 自动刷新）"""
    length = sequence_lengths_plus_1.reshape(-1).to(torch.int64)
    read_col  = (length - 2).clamp_min(0) // seq_size_per_block
    write_col = (length - 1).clamp_min(0) // seq_size_per_block
    read_idx  = block_map.gather(1, read_col.view(-1, 1)).squeeze(1)
    write_idx = block_map.gather(1, write_col.view(-1, 1)).squeeze(1)
    return read_idx, write_idx

def paged_row_view(seg_view):
    """从 4D strided 段视图反推 2D 页行视图 [pages, page_stride]（host 视图构造，capture 安全）"""

def seed_state_segment(seg_view, read_idx, write_idx):
    """段内跨块播种：只动本段（ssm 或 conv），不碰兄弟段（最终形态）"""
    from rtp_llm.models_py.kernels.ascend.state_migration import migrate_state_rows
    same = read_idx == write_idx
    # 无需迁移的行做成自拷贝：triton kernel 内部直接跳过
    src = torch.where(same, write_idx, read_idx)
    migrate_state_rows(seg_view, src, write_idx)
```

**设计原因**：
- gather 读的是 C++ `prepareInputs` 每次 replay 刷新的 capture buffer，**语义上等价于 SGLang 的"固定地址 + replay 改内容"，但零图外协作代码**；
- seed 必须段内作用：conv 算子先写 conv 段，若 ssm 的 seed 做整行拷贝会 clobber 刚写的 conv 状态（开发中被单测抓到过）；
- 静态形状设计（图捕获关键）：跨块迁移不用布尔掩码索引（`nonzero()` 输出 shape 依赖值，捕获失败或烤死 shape），用 `torch.where` 把无需迁移的行做成自拷贝，索引 shape 恒为 `(batch,)`。

**seed 实现的三次演进（均有实测依据）**：
1. 初版：整页行 gather（`F.embedding`）→ `torch.where` → `cat` 整行重建 → `index_copy_`（~0.1ms/层，全量搬运）；
2. 段级化实验（**已回退**）：对 `row_view[:, a:b]` 列切片直接段级 gather/scatter，实测**慢 40-60%**（非连续列切片使 index 算子退化为元素级）——且后续 profiling 证明 Slice 热点根本不是 seed；
3. **最终形态（自 PR #52 移植）**：委托 `state_migration.migrate_state_rows` 的 triton kernel——kernel 内部按行判断 `src == dst` 直接跳过，同页行只花一次索引比较；CPU/无 triton/`RTP_LLM_GDN_PRECOPY=0` 时回退等价的无条件 `index_copy_`。

### 4.1b 新增 `rtp_llm/models_py/kernels/ascend/state_migration.py`（+151，自 PR #52 移植）

triton 条件页迁移 kernel（vllm-ascend `precopy.py` 同思路）：
- 签名 `(state_ptr, src_pages, dst_pages, page_stride, row_elems, BLOCK, TILES)`——手动平坦寻址（`dst*page_stride + offset`），支持任意 dim0 stride 的段视图（hybrid 池 padded 视图 OK），实测**无 triton-ascend strided store 静默丢弃问题**（bit-exact 单测验证）；
- 大行多 tile 并行（`_TILE_BYTES=256KB`、`_MAX_TILES=32`），grid 只依赖 batch 与静态 tile 数（图捕获安全）；
- 配套 `test_state_migration.py`（7 用例，CPU 可跑）。

### 4.2 修改 `kernels/ascend/recurrent.py`（新增文件中的方案C部分）

T==1 分流 + device 路径（`fused_recurrent_gated_delta_rule` 内部，模型层无感知）：

```python
_ACTUAL_SEQ_LENGTHS_CACHE: dict[tuple[str, int], torch.Tensor] = {}

def _decode_actual_seq_lengths(device, batch):
    """[0, 1, 1, ...] cuSeqlens 风格元数据。device 侧 fills 构造并缓存：
    torch.tensor(host list) 会 H2D 同步（capture 非法）；缓存保证 capture
    录到稳定地址，replay 不再进 Python。"""

def _fused_recurrent_decode_device(q, k, v, g, beta, scale, initial_state,
                                   block_map, sequence_lengths, seq_size_per_block):
    from ...paged_state import decode_state_indices, seed_state_segment
    ascendc = _get_ascendc_ops()
    batch = q.shape[0]
    read_idx, write_idx = decode_state_indices(block_map, sequence_lengths, int(seq_size_per_block))
    seed_state_segment(initial_state, read_idx, write_idx)
    result = ascendc.npu_recurrent_gated_delta_rule(
        q.reshape(batch, *q.shape[2:]).to(torch.bfloat16),
        k.reshape(batch, *k.shape[2:]).to(torch.bfloat16),
        v.reshape(batch, *v.shape[2:]).to(torch.bfloat16),
        initial_state,                                    # 池视图直传，零转置（§2.3）
        beta=beta.reshape(batch, beta.shape[-1]).to(torch.bfloat16),
        scale=float(scale),
        actual_seq_lengths=_decode_actual_seq_lengths(q.device, batch),
        ssm_state_indices=write_idx.to(torch.int32),     # device 索引！
        g=g.reshape(batch, g.shape[-1]).float(),
    )
    out = result[0] if isinstance(result, (tuple, list)) else result
    return out.reshape(batch, 1, *out.shape[1:]).to(q.dtype), initial_state

# 主入口内（l2norm 之后、_resolve_state_pages 之前）：
if (token_count == 1 and block_map is not None and sequence_lengths is not None
        and block_map.ndim == 2 and block_map.shape[0] == batch):
    return _fused_recurrent_decode_device(...)
# token_count > 1（MTP，≤8）：保留原 host 路径 _resolve_state_pages + host seed
```

**原因**：标准 decode（T=1）占绝对流量，eager 与 capture 共用此路径；MTP/target-verify 不进 decode 图，host 路径合法保留。附带收益：**eager decode 也消除了每步 30 层 × 2 次 D2H 同步**（eager 实测 120 → ~70ms/token）。

### 4.3 修改 `kernels/ascend/causal_conv1d.py`（新增文件中的方案C部分）

契约变化 + T==1 device 路径：

```python
def _causal_conv1d_update_device(x, conv_state, weight, bias, activation,
                                 block_map, sequence_lengths, seq_size_per_block, ...):
    from ...paged_state import decode_state_indices, seed_state_segment
    read_idx, write_idx = decode_state_indices(block_map, sequence_lengths, seq_size_per_block)
    seed_state_segment(conv_state, read_idx, write_idx)
    # 专用 out= 输出 buffer（自 PR #52 移植）：算子的 fallback 路径会把结果写进
    # x 参数，out= 避免穿透调用方 buffer，也无需 clone
    x_work = x.to(conv_state.dtype)
    out = torch.empty_like(x_work)
    npu_causal_conv1d_update(
        x=x_work,
        conv_state=conv_state,                    # 原始池视图 (pages, state_len, dim)
        weight=weight.t().contiguous(),           # 算子要求 (width, dim)
        bias=bias,
        activation=_activation_name(activation),
        conv_state_indices=write_idx.to(torch.int32),   # device 索引！
        null_block_id=0,                          # padding 行（页 0）整行跳过
        out=out,
    )
```

- **契约变化**：`causal_conv1d_update` 的 `conv_state` 从转置视图 `(page,dim,state)` 改为**原始池视图 `(pages,state_len,dim)`**（算子原生布局，模型层直传，少两次转置）。
- **prefill 换新入口（自 PR #52 移植）**：`causal_conv1d_fn` 改用 `npu_causal_conv1d_fn`（device 元数据：`query_start_loc`/`has_initial_state` int32），初始状态 device 侧 gather（`_gather_prefill_states_device` + `_gather_pages_from_block_map`）；废弃老入口的 host gather 删除；**host 多块快照写回保留**（`_scatter_prefill_states_host`，prefill 不入图，eager-only 合法）。
- T>1（MTP）保留 host 逐 token 路径（老入口 `npu_causal_conv1d` + host `cache_indices`）。

### 4.4 修改 `model_desc/qwen3_next.py`（±69）

- import 改为**按设备分流**：Ascend → `kernels/ascend`，非 Ascend 保持 main 原 triton 路径（CUDA 零影响）；
- decode 的 `_conv1d`/`_fla` 调用点**直接写成单一路径**（传 `block_map=attn_inputs.kv_cache_kernel_block_id_device`、`sequence_lengths=attn_inputs.sequence_lengths_plus_1_d`），**没有任何 graph 分支**——方案A 的 `use_graph_path`/`is_current_stream_capturing` 兜底全部不存在（问题 C5 随之消失）。

### 4.5 图模式配套修复（与方案C 正交、必须保留）

| 文件 | 内容 | 对应旧问题 |
|---|---|---|
| `cpp/ascend_graph/ascend_graph_runner.cc`（+37） | capture 块表 `zeros→ones`（6 处，page 1 恒有效）；`copyTensorSlice` 支持 3D 分组块表（取 group 0）；图档位 `current_real_graph_bs` 经 `py::int_` 传 Python | C2 / C4 / C6 |
| `ascend_impl/ascend_decode.py`（+146） | `prepare_cuda_graph(attn_inputs, graph_bs=None)` ctx padding 对齐；`signal_graph_events(graph_bs)`；`_graph_handles/_graph_refs/_graph_events` 按档位 bs 分组；混合 KV（layer_idx 选分组表、kernel/phys 双表、slot_mapping 物理寻址） | C6 |
| `ascend_impl/ascend_attn_params.py`（+90） | `_blocks_per_phys_from_config`（bpk 优先取 `tokens_per_block/kernel_tokens_per_block`，表宽推断仅 fallback）；`_physical_kv_view`/`split_kv_kernel_blocks`（bpk>1 的 K/V 跨界切分） | C7 |
| `ascend_impl/ascend_prefill.py`（+31）、`ascend_kv_cache_write_op.py`（+12） | prefill 侧 bpk；物理视图 scatter 适配 | C7 |

### 4.6 测试

- 新增 `test/test_plan_c_device_path.py`（+342，NPU 真机，4 用例）：
  1. device 路径 vs host 路径 replica（同 strided 池、含跨块行）**bit-exact**（输出+池状态）；
  2. `torch.npu.graph` 捕获整步（conv+recurrent wrapper），replay 3 步刷新 `block_map`/`sequence_lengths` 内容，与 eager 循环**逐位一致**；
  3. padding 行（block_map 置零 → page 0）隔离性；
  4. vs 纯 torch GDN reference（fla_npu ATK golden 语义）误差 <3%。
- `test_causal_conv1d.py` fixture 按新布局契约更新（prefill 用例适配新入口 `npu_causal_conv1d_fn` 的 device 元数据断言）；BUILD 注册全部测试。
- 新增 `test/test_state_migration.py`（+127，自 PR #52 移植，7 用例 CPU 可跑：kernel 结构/回退语义/哨兵行为）。
- **采样修复补强（自 PR #52 移植，`CudaSampleOp.cc`）**：temp≤0 的行**全面**按 greedy 处理（OpenAI 约定 temp=0 = greedy）——并入 greedy-row mask（跳过温度缩放/惩罚）与 multinomial 后的 argmax 覆写，而非只防除零。修正了此前 temp=0 且 do_sample=true 仍会随机采样的语义缺口。

---

## 五、方案A vs 方案C

| | 方案A（sgl triton） | **方案C（单一消费者）** |
|---|---|---|
| eager 路径 | fla_npu AscendC（host 元数据） | **同一实现**（device 元数据） |
| graph 路径 | vendored sgl triton kernel | **同一实现** |
| 数值 | 双实现 bf16 差异 err 0.25% | **同算子同输入 → 与 eager 数学一致**（bit-exact 单测） |
| 布局适配 | 每层每步 2 次转置（C8） | **零转置**（AscendC `[HV,V,K]` = 池布局） |
| 池访问 | gather→连续 buffer→scatter | **池视图直传**（仅剩 seed 的段播种） |
| 代码量 | +1060（含 925 行 vendored） | **+213 / -1201**（净删 988 行） |
| eager 性能 | ~120ms/token（60 次 D2H/步） | **~70ms/token**（D2H 消除） |
| 图模式性能 | 167ms/token | **~30ms/token** |
| 算子本体（图内） | sgl kernel 36us/层 + 适配 | **AscendC 5.5us/层** |

---

## 六、旧问题 C2–C8 在方案C 下的状态

| 问题 | 会出现吗 | 结论 |
|---|---|---|
| C2 page 0 非法 | **不会** | 报错源（host 页号校验）不在 T==1 capture 路径上；device 路径容忍 page 0（conv 跳过/recurrent 写 pad 槽）。ones 修复**保留**（防御性，正确） |
| C3 FIA block_size | 会 | 启动参数约束（与 GDN 方案无关），仍需 `--seq_size_per_block=1024 --kernel_seq_size_per_block=512` |
| C4 3D 块表拷贝 | 会 | engine 混合块表结构不变，修复保留 ✓ |
| C5 is_cuda_graph 丢失 | **不会** | 无分流则无此问题；兜底代码随方案C 删除（正确） |
| C6 上下文错位 | 会 | FIA 层机制（与 GDN 无关），graph_bs 修复保留 ✓ |
| C7 bpk 推断 | 会 | 作用于 FIA K/V 视图与 slot_mapping；GDN device 路径不依赖 bpk（语义继承自已验证的 eager host 路径）。修复保留 ✓ |
| C8 池布局不匹配 | **不会** | sgl kernel 已删，AscendC 布局天然匹配；graph_compat.py 已删 |

---

## 七、验证结果

### 7.1 算子级 / 单元（NPU 真机 Ascend950PR）
- device 路径 vs host 路径：输出与池状态 **bit-exact**（含跨块、strided typed_storage_view 池、padding 行）；
- 图 capture/replay vs eager 多步：**逐位一致**；
- vs torch reference：recurrent 0.35% / conv 0.2%（bf16 正常水平）；
- 仓库 **8 个测试文件全过**（含既有 host AST 回归 + 移植后的 state_migration/新 prefill 用例）——triton kernel 直写 strided 池经 bit-exact 确认**无 triton-ascend 静默丢弃问题**。

### 7.2 端到端（`feature/qwen35-graph-mode` 分支，Qwen3.5-35B-A3B 单卡，含 §十 移植后复验）
- 5 档位（32/24/16/8/1）capture 全部成功，`captured 5 graph instances`；
- greedy 输出正常连贯（temp=0 现为确定性 argmax 路径）；默认采样正常；1024 token 长生成跨 1024 物理块边界正常，无重复退化（覆盖 prefill 新入口 + host 快照写回 + triton seed 跨块迁移）；
- `transformers==4.51.2` 下无需 starcoder 兼容改动（分支已按此排除）。

### 7.3 性能（bs=1，热态）
| 配置 | ms/token |
|---|---|
| 方案A 图模式（旧） | 167 |
| eager 旧基线 | ~120 |
| **方案C eager** | ~70 |
| **方案C 图模式（移植前）** | 30~31 |
| **方案C 图模式（移植后，最终）** | **28~31**（长生成 28，triton 条件迁移小幅收益） |

### 7.4 一致性说明（如实记录）
graph vs eager 的 greedy 输出在前 ~25 token（短）/207 字符（长）一致后分叉。**定位结论：非方案C 问题**——eager 服务器自身重跑同请求 11 字符即分叉、图模式重跑 57 字符即分叉，属请求级运行间非确定性（最可能源于 prefill 或 bf16 GEMM 归约序）。组件级 A/B 全部 bit-exact：FIA（含 graph_task update 重launch、host-list vs device 通道、共享 workspace 三因子分离）、MoE 整链（router + init_routing_v2 + grouped_matmul×2 + swiglu + unpermute）、GDN/conv。测试脚本存档：`/tmp/opencode/test_fia_ab.py`、`test_moe_ab.py`。

---

## 八、profiling 结论与后续优化清单

### 8.1 decode 每步构成（~30ms，span 30.2 / busy 27.3 / idle ~3）
| 项 | ms/step | 归属 |
|---|---|---|
| `aclnnInplaceCopy_Slice`（20×571us） | **11.4** | **FIA 层 KV 写慢路径**（非 GDN！） |
| Matmul/Mm（投影/router/lm-head） | 4.8 | 模型本职 |
| GroupedMatmul（MoE） | 2.7 | 本职 |
| Cast×1024 + Mul×612 等 | ~2.7 | 可精简 |
| IndexSelect/Embedding（seed gather 等） | ~4.4 | 适配层 |
| **GDN recurrent 本体**（5.5us×30 层） | **0.17** | **方案C 算子极快** |
| CausalConv1d 本体（4.6us×30） / FIA 本体（11.4us×10） | 0.14 / 0.11 | 极快 |

### 8.2 优化清单状态
| 优化点 | 状态 |
|---|---|
| GDN seed 条件化（同页行零拷贝） | ✅ **已完成**（§十 移植 `state_migration.py` triton precopy） |
| **FIA 层 KV 写慢路径**（最大热点，11.4ms/step）：慢 Slice kernel shape 为整层 KV cache `(4090,2048,2,256)`、每步 20 次（10 FIA 层×K/V）→ `npu_scatter_pa_kv_cache`。微基准：连续 key/value 17us；非连续（rope 输出转置视图）退化（合成最坏 11.9ms，线上实际 571us）。**修复（一行）**：`ascend_kv_cache_write_op.py` scatter 前 `key/value .contiguous()` → 预期 **-11ms/step → ~19ms/token** | 未实施，留档 |
| 次要：Cast 链精简 + l2norm 融合（-2ms）；prefill 侧同类慢 kernel（30×~590us gather/scatter） | 未实施，留档 |

---

## 九、分支落位与构建

- 分支：`feature/qwen35-graph-mode`（weishan/rtp-llm），基于 `ningweikang/main@6a55e8df3`，已推送到 `QQzhouQ/rtp-llm` 并开 PR 至 `ningweikang/rtp-llm:main`；
- 两个提交：
  - `9678d044c` feat(ascend): Qwen3.5 eager support + aclgraph Plan-C GDN single-consumer path（44 文件 +4447/-102）
  - `465f0d07d` perf(ascend): port PR #52 improvements into the Plan-C path（§十，7 文件 +495/-109；**author = sai841**，共同作者署名）
- **识别排除**：`starcoder*.py`（transformers 4.51.2 不需要 A2 兼容）、`base_model.py`/`collective_torch.py`（TP 联动，main 自有 TP 线）、`ci/`；
- 构建适配（**保留为本地未提交修改**，未进 PR）：main 的 `.bazelrc` 把 ARM `-march=armv8.2-a` 挪进了 `build:ascend`（x86 编不过）→ 删除；补本机 distdir/repository_cache（机器相关路径不宜入库）；
- FIA v2 graph_task 机制与 aclgraph runner 主体 main 已含（PR#25），不在本分支 diff 内；
- 启动命令（已验证）：
  ```bash
  python3 -m rtp_llm.start_server \
      --checkpoint_path=/mnt/docker/weights/Qwen3.5-35B-A3B \
      --model_type=qwen35_moe --start_port=9000 \
      --enable_cuda_graph=true --kv_cache_mem_mb=8192 \
      --seq_size_per_block=1024 --kernel_seq_size_per_block=512
  ```

---

## 十、与 PR #52（sai841）的对比与融合移植

### 10.1 背景
同事的 [PR #52](https://github.com/ningweikang/rtp-llm/pull/52)（`sai841:ascend-graph-qwen35`，41 文件 +3924/-82）与本方案**同源同架构**：同为方案C 路线（同一对 wrapper 的 device-tensor 化、`block_map.gather` 索引、`torch.where` 静态形状、page 0 哨兵、CANN MoE、C4、采样修复），~90% 文件重合（同出 rtp-llm-npu 迁移线）。

### 10.2 对比结论

**对方强项（四项已全部移植，见 10.3）**：triton 条件 precopy、conv `out=` 静态输出、prefill 新入口、采样 temp=0 强制 greedy；另有更大规模的 CPU 可跑测试矩阵（50 用例）与 wrapper 签名不变的设计。

**我方关键正确性优势（对方缺失，保留为我方合入基座的依据）**：
| 缺口 | 后果 |
|---|---|
| C6（graph_bs padding + handles 按档位分组）缺失（`_graph_handles` 仍为 flat list，`prepare_cuda_graph` 用 max_bs） | 多并发（bs 2~31 pad 到档位）时 FIA 读错上下文（"模型看到的输入是别的"）；对方 e2e 仅测 bs=1 未暴露 |
| B1（`models/qwen3_next/qwen3_next.py` config_dtype 解析）缺失（main 无此逻辑，已核实） | 引擎默认 FP16 → 乱码/MoE 策略拒绝 |
| C7（bpk 从配置取）用表宽推断 | capture 同宽表下行为存疑，未经跨块长生成实测 |

**最终决策**：以我方 PR 为基座（正确性修复完整 + 真机验证深），cherry-pick 对方四项优势；对方以 co-author 署名（移植提交 author = sai841）。

### 10.3 移植内容（commit `465f0d07d`，7 文件 +495/-109）

| # | 内容 | 说明 |
|---|---|---|
| 1 | **`state_migration.py`（+151）+ 单测** | triton 条件页迁移 kernel：kernel 内跳过 `src==dst` 行（同页行只花索引比较），手动平坦寻址支持 strided 段视图；`RTP_LLM_GDN_PRECOPY` 开关 + 无条件 `index_copy_` 回退。`seed_state_segment` 改为静态形状 `torch.where` + 委托此 kernel（见 §4.1） |
| 2 | **conv `out=` 静态输出** | device 路径不再 clone x，专用输出 buffer（见 §4.3） |
| 3 | **prefill 换 `npu_causal_conv1d_fn`** | device 元数据 + device 侧初始状态 gather；host 多块快照写回保留（见 §4.3） |
| 4 | **采样修复补强** | temp≤0 全面按 greedy（mask + argmax 覆写，见 §4.6） |

### 10.4 移植验证
- **8/8 测试套件通过**（含 NPU bit-exact / 图 capture-replay / padding 隔离；triton 直写 strided 池 bit-exact，确认无静默丢弃问题）；
- 重建 + e2e 复验：5 档位 capture、greedy（确定性 argmax）/默认采样、1024 跨块长生成无退化；
- 性能 28~31 ms/token（长生成 28，较移植前 30~31 小幅提升）。
