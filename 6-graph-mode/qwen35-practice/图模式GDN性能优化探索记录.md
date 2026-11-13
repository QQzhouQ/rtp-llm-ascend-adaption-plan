# 图模式 GDN 性能优化探索记录（含回退说明）

> 时间：2026-09-17
> 关联：《rtp-llm借鉴SGLang的GDN入图实施方案.md》《图模式叠加Qwen3.5验证总结.md》
> 背景：图模式叠加 Qwen3.5 正确性验证通过后，发现 **图模式 203ms/token vs eager 65ms/token（慢3倍）**，启动性能优化探索。

---

## 一、最终代码状态（当前保留版本）

**ExternalEvent 版（203ms/token）**——update/replay 并行架构已就位，转置优化已按决策回退。

| 文件 | 状态 |
|---|---|
| `sgl/fused_sigmoid_gating_recurrent.py` | **sgl 原版 kernel**（寻址 `[.., HV, K, V]`，`idx*HV*K*V + o_k*V + o_v`） |
| `graph_compat.py` recurrent | **全池转置版**：`ssm_states.transpose(-1,-2).contiguous()` → kernel → `copy_` 回写 |
| `graph_compat.py` conv | torch 组合实现（gather→窗口卷积→scatter） |
| `ascend_decode.py` | **ExternalEvent 机制**（capture 录 wait 点 / update record 放行 / signal_graph_events 供 check replay） |
| `ascend_graph_runner.cc` | `replayAndSyncCheck` 前调 `signal_graph_events(bs)` |

单测基线：GDN err 0.25%（vs AscendC 对拍）。

---

## 二、探索历程与结论全景

### 阶段 0：瓶颈定位实验（跳过 FIA update）

**方法**：临时 env 开关跳过 `update_graph_fia`（10 层×FIA 重 launch 全跳，内容会错只看速度）。

**结果**：**202.9ms/token（完全不变）**——`update_graph_fia`（含曾经的 host sync）**不在关键路径**，被图执行时间完全掩盖。

**推论**：瓶颈在**图内 GDN 执行本身**——eager 的 GDN 是 1 个融合 AscendC 算子/层，图分支是 ~20 个小 op 组合/层 × 30 层。

### 阶段 1：ExternalEvent（update/replay 并行）——保留

把 `update_graph_fia` 的 `us.synchronize()`（host 阻塞）改为 ExternalEvent：

```python
# capture 时（图内录等待点）:
event = torch.npu.ExternalEvent()
event.wait(stream); event.reset(stream)
torch.npu.graph_task_group_begin(stream); ...FIA...; handle = ...end(stream)

# update 时（图外 update stream，record 放行，无 host 同步）:
torch.npu.graph_task_update_begin(us, handle); ...FIA(新ctx)...; end(us)
events[i].record(us)   # 图内 wait 点被满足，replay 可立即发射
```

**踩坑**：capture 后 C++ 的 `replayAndSyncCheck`（sanity replay）没有 update 侧 record → **死锁**。修复：C++ 在 check replay 前调 Python `signal_graph_events(bs)`（图外 record 一次）。**注意不能在图上下文内 record（会被录成图 op，无效）**。

**结论**：当前 203ms 下中性（实验 0 证明 update 不在路径）；**图执行降到 ~65ms 级后有 ~5-10ms/step 流水线收益**（update 逐 task 与图内前几层重叠）。vllm-ascend 同款模式。

### 阶段 2：全池转置消除（页级 gather）——已验证 -35ms，**按决策回退**

**profiling 数据**（真实规模单测，每层每步）：

| 环节 | eager (AscendC) | 图实现 | 差距 |
|---|---|---|---|
| recurrent 整体 | 0.466ms | **9.291ms** | 20 倍 |
| ├ triton kernel 本体 | — | **0.036ms** | kernel 很快 |
| └ **transpose×2（全池）** | — | **9.227ms** | 罪魁祸首 |
| conv | 0.202ms | 1.509ms | 次要 |

**根因**：`ssm_states.transpose(-1,-2).contiguous()` 转置的是**整个池**（实测服务池约 250 页×4MB 级，×2 次搬运/层×30 层）。深层根源是**双消费者布局冲突**：fla_npu AscendC（eager）约定 `[HV,V,K]`，与上游 fla/sgl-kernel-npu/vllm-ascend 的 `[HV,K,V]` **恰好相反**——详见实施方案文档 2.2(5) 的布局根源分析。

**修复**（页级 gather）：只 gather 本 batch 的页（B=1 时 1 页）转置 → kernel（indices=arange）→ scatter 回。单测 9.29→**0.21ms**；端到端 **203→167ms/token（-35ms）**。数值 err 0.25% 不变。

> 实际收益（35ms）小于单测外推（池页数 2048 vs 服务实际 ~250）。

**当前状态：已按决策回退到全池转置版**（回退后单测 err 0.25% 复核通过）。此优化路径已完整验证，需要时可快速重放（改 `graph_compat.py` 一处即可）。

### 阶段 3：kernel 免转置 in-place（发现 triton-ascend 缺陷）——不可行，已回退

**思路**：改 kernel 寻址（`o_k*V + o_v` → `o_k + o_v*K`）直接按 rtp 池的 `[pages,HV,V,K]` 布局寻址，kernel 直读直写原池（零转置零 gather），单测 0.146ms/层。

**翻车**：eager 单测全过（含跨块），但服务（图模式）输出乱码。

**定位过程**（层层深挖）：
1. conv 回退 torch 版仍乱 → 锁定 recurrent
2. strided 池单测复现（out err 138%）——**单测池此前用紧邻 tensor，掩盖了问题**
3. 零池探针：**kernel 执行了（out 非零）但 store 零写入**（池全零）
4. 对照实验实锤——

### ⚠️ 重大发现：triton-ascend 3.2.2 的 strided store 静默丢弃缺陷

```
contig pool   | store 正常写入（state sum=35435）
strided pool  | store 零写入，无任何报错（state sum=0）
```

**`tl.store` 写到非标准 stride 的 storage 视图（`as_strided`/`set_` 构造）时静默丢弃。**而 rtp-llm 的 ssm/conv 池都是 `typed_storage_view` 构造的 strided 视图（页 stride=block_size_bytes，**页内夹着 conv 区**，ssm 只是块内一段）——任何 triton kernel 直写池视图都会踩坑。

**连带结论**：
- **conv 不能 triton 化**（conv 池同样 strided；sgl 生产可用是因为其池是标准 tensor）——conv 保持 torch 实现（`index_copy_` 正确处理 stride）
- **GDN 单测必须用 strided 池模拟真实布局**（紧邻池会掩盖此类缺陷）——测试脚本 `test_stride.py` 已验证此方法论
- **建议向 triton-ascend 社区报此缺陷**（对 NPU triton 生态普适价值）

### 阶段 4：conv triton 化——不可行（同缺陷）

sgl 的 conv kernel 用 stride 参数寻址（页维也取 `conv_state.stride(0)`），但本质仍是 store 到 strided 地址——同样被丢弃。已回退为 torch 组合实现。

---

## 三、性能账本

| 版本 | ms/token | 状态 |
|---|---|---|
| 初始（全池转置+串行update） | 203 | - |
| +ExternalEvent | 203 | **当前版本**（中性；低延迟场景流水线收益） |
| +页级gather转置 | 167 | 已验证（-35ms），**已回退** |
| +kernel免转置in-place | 167.9（数值乱，缺陷） | 不可行，已回退 |
| eager 基线 | 65 | - |

**剩余差距（203 vs 65，~138ms）构成**：
| 项 | 估计 | 说明 |
|---|---|---|
| 全池转置 | ~35ms | 阶段2路径可回收 |
| conv torch 组合 op | ~39ms | 需先解决 strided store 缺陷 |
| FIA update | 0 | 实验0证明不在路径 |
| 未定位（C++ prepareInputs 每步固定开销 / 图重放调度 / MoE 图内执行） | ~60ms | 需 profiler 接入服务 |

---

## 四、复现资产（均在 /tmp/opencode，可固化）

| 脚本 | 用途 |
|---|---|
| `prof_gdn.py` | GDN 各环节耗时分解（eager vs graph，真实规模） |
| `test_gdn_full.py` | GDN 全场景单测（含跨块） |
| `test_stride.py` | **strided 池单测**（模拟真实布局，防缺陷回归） |
| `test_conv_graph.py` | conv 图实现对拍 |

---

## 五、后续建议（按优先级）

1. **报 triton-ascend 缺陷**（strided store 静默丢弃）——解锁 conv triton 化 + in-place kernel（合计 ~70ms 空间）
2. **重放页级 gather 优化**（-35ms，改动一处，已验证）
3. **profile 剩余 60ms**：torch_npu profiler 接入服务，定位 prepareInputs/图调度/MoE
4. **长期**（同实施方案文档第三节）：AscendC device 元数据化（方案C）可一并消除转置/gather 全部适配层
