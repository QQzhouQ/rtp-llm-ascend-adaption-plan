# Ascend ACL Graph 图模式：Qwen3-0.6B（alibaba 上游）与 Qwen3.5-MoE（PR #71）代码差异

> 结论先行：**核心引擎代码完全同一套**（15 个文件逐字节一致），区别只在基线谱系与
> Qwen3.5 专属模型文件的取舍。模型差异（MoE/GDN/MRoPE/混合 KV）全部由运行时
> 数据驱动，不引入模型特定的引擎分支。

## 一、两个版本的基本面

| | Qwen3.5-MoE 版（PR #71） | Qwen3-0.6B 版（alibaba 上游） |
|---|---|---|
| 目标仓库 | `ningweikang/rtp-llm` ← PR #71 | `alibaba/rtp-llm:feature/ascend` |
| 基线 | `remote_qwen3.5` @ #69(eager) + #72(P1 修复) | `feature/ascend` @ e0f2639（PR #1481） |
| 叠加文件数 | 18 | 15 |
| 验证模型 | Qwen3.5-35B-A3B（MoE + GDN + MRoPE + 混合 KV） | Qwen3-0.6B（dense MHA + 标准 RoPE + 单组 KV） |
| 性能 | ~16.3 ms/tok（eager ~57 ms，**3.5×**） | ~6.6 ms/tok（eager ~18.6 ms，**2.8×**） |
| 精度 | ceval 串行 84.6-88.5%（与 eager 一致） | 四件套：C-Eval 43.7%（1346 全量）/ MMLU 43.9%（1140）/ GSM8K 63.3%（300）/ ARC 65.5%（200） |
| 图捕获 | 5/5 bucket | 5/5 bucket |

## 二、文件清单对比

### 完全相同的 15 个引擎文件（两边逐字节一致）

| 文件 | 作用 |
|---|---|
| `rtp_llm/cpp/ascend_graph/BUILD` | 图运行时构建规则 |
| `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc/h` | 核心：分 bucket 捕获 + replayAndSyncCheck |
| `rtp_llm/cpp/ascend_graph/ascend_graph_utils.h` | AscendGraphInstance / MemHold |
| `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.cc/h` | NPU stream/event/memcpy/graph API 隔离层 |
| `rtp_llm/cpp/cuda_graph/cuda_graph_base.h` | GraphParams 新增 Ascend 字段 |
| `rtp_llm/cpp/models/BUILD` | `@//:using_ascend` → ascend_graph 依赖接线 |
| `rtp_llm/cpp/models/PyWrappedModel.h` | 图分支：initCapture / GraphParams 装配 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py` | 位置/slot_mapping 计算（含 device 变体） |
| `.../ascend_decode.py` | FIA v2 + graph_task_group 捕获/重放 |
| `.../ascend_prefill.py` | MRoPE support 放开 + kernel/physical 页语义 |
| `.../ascend_rope_emb.py` | MRoPE 双路径（npu_mrope / 交错纯 torch） |
| `rtp_llm/async_decoder_engine/rpc_engine.py` | `cuda:` 硬编码 → `device_name()` |
| `rtp_llm/models_py/utils/arch.py` | `is_ascend` 再导出 |

### 仅 PR #71 有、alibaba 版剔除的 3 个文件（Qwen3.5/GDN 专属）

| 文件 | 内容 | 剔除原因 |
|---|---|---|
| `rtp_llm/models_py/model_desc/qwen3_next.py` (+58/-36) | GDN conv 状态、layer_idx→KV 组路由、per-head norm、MRoPE 接线 | 模型专属；上游 PR 保持引擎通用性 |
| `rtp_llm/models/qwen3_next/qwen3_next.py` | 多模态门禁 warning 移除、`is_ascend` 引入 | 同上 |
| `rtp_llm/models_py/kernels/ascend/recurrent.py` | GDN 线性注意力 kernel docstring 修正 | Qwen3.5 专属 kernel |

> alibaba 的 `feature/ascend` 基线本就不含 Qwen3.5 模型（#1481 仅支持 Qwen3 0.6B），
> Qwen3.5 模型文件留在 ningweikang 谱系随其模型 PR 落盘。

## 三、同一套代码如何覆盖两类模型（通用性设计）

引擎不感知具体模型，差异全部由运行时条件收敛：

| 差异点 | Qwen3.5-35B 走的路径 | Qwen3-0.6B 走的路径 | 判定条件 |
|---|---|---|---|
| 混合 KV（GDN+MHA 分组） | per-tag inputs 重建、按 tag 扇出 `invokeOnAttnImpl` | 全部旁路，单默认组 | `kv_cache_group_num_ > 1 && !group_tags_.empty()`；`attention_inputs_by_tag` 为空即跳过校验 |
| KV 布局参数来源 | `kv_cache_layer_layout`（多组拓扑） | `getMainModelGroupedCacheLayerLayout()` 同样返回**单组**有效布局，block 尺寸照常装配 | 布局抽象对单组模型统一成立，构造期无特判 |
| MRoPE | `combo_position_ids [T,3]` 按 bucket 切片 + 交错旋转 | 休眠（Base rope 走 C++ cache） | `rope_config.style == Mrope` |
| MoE 策略 | CANN 策略优先、triton 回退兜底 | 无关（dense 模型） | 策略注册表按配置选择 |
| blocks_per_phys（kernel/物理块细分） | 运行时从 cache 配置推导 | 同一推导逻辑（ratio=1） | `_blocks_per_phys_from_config` |
| 图捕获 bucket | 同一套 5 bucket 捕获 + replayAndSyncCheck | 同 | 与模型无关 |

## 四、周边（非图模式代码）差异

| 项 | PR #71 谱系 | alibaba 谱系 |
|---|---|---|
| top-k/top-p 采样 | AscendC 自定义算子（apply_top_k_top_p，双 SoC 编译） | CANN ≥9.0 内置 `aclnnApplyTopKTopP`（无需自编译算子） |
| 依赖获取 | havenask/grpc 需 override 本地目录 | 同样依赖 havenask/grpc（git fetch，网络受限时需预取） |
| torch 装载 | pip requirements | `http_archive`（`torch_cpu_ascend`/`torch_npu_ascend`）+ pip 双通道 |
| 已知编译坑 | acl_sk.h 无旗标警告需 `-Wno-error` | 相同（已加） |

## 五、提交形态

- **PR #71**：单提交叠在 #72 上（合入 #72 后 diff 自动缩减为 18 文件），base `remote_qwen3.5`
- **alibaba PR（待提）**：单提交叠在 e0f2639 上，15 文件，纯引擎级；
  定位是补齐 #1481 Known Limitations 中 "CUDA graph capture: Stubbed, not yet available" 一栏，
  证据 = 0.6B 图捕获 5/5 + greedy 正确 + 2.8× 加速 + ceval 43.7%，
  并附 Qwen3.5-35B（3.5×）作为另一结构极性的旁证。

*生成时间：2026-10-31；PR #1620（alibaba）与 PR #71（ningweikang）均已提交，0.6B 四件套精度已补全*
