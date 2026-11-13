# 复用性评估：vllm-ascend 与 xllm 方案能否在 rtp-llm 复用

> 在 `01`/`02`/`03` 三份分析基础上，给出明确结论

---

## 1. 三框架对照

| 维度 | vllm-ascend | xllm | rtp-llm（现状） |
|------|-------------|------|-----------------|
| 主语言 | Python（vLLM 生态） | C++ 主循环 + Python 模型 | C++ 主循环 + Python 模型（xllm 后继） |
| torch_npu profiling API | ✅ `torch_npu.profiler.profile` | ❌ 用 C 级 `mspti`/`mstx`（非 `torch_npu.profiler`） | ❌ 无（CUDA Kineto only） |
| 触发面 | vLLM HTTP `/start_profile` | HTTP `/start_profile`（CUDA gate） | gRPC `StartProfile` + per-req `gen_timeline` |
| 引擎循环接入 | vLLM `WorkerProfiler` step gate | `WorkerImpl` fan-out 单例 | `StepWindowProfiler.tick()`（已接入 step 循环） |
| Ascend 状态 | **生产可用** | 死代码（mspti_helper 未接线） | 显式 no-op TODO（`ExecOps.cc`） |
| 输出 | `ascend_pt` → `analyse()` → Chrome trace | Kineto JSON（CUDA）/ mspti log→trace（NPU 未接） | Kineto JSON（CUDA only） |

---

## 2. vllm-ascend 方案能否直接复用？

### 结论：**部分可复用 —— 配方可搬，外围脚手架不可搬**

**可复用部分（核心价值）**：
- ✅ `torch_npu.profiler.profile(...)` 调用块（含 `_ExperimentalConfig`、`ProfilerActivity.CPU+NPU`、`tensorboard_trace_handler`）是**自包含配方**
- ✅ dump 后处理 `torch_npu.profiler.profiler.analyse(...)` 是独立能力
- ✅ `torch_npu` 2.9.0 已是 rtp-llm 依赖，版本兼容，API 可直接调

**不可直接搬部分（耦合 vLLM）**：
- ❌ `TorchNPUProfilerWrapper` 继承 vLLM `WorkerProfiler`，依赖 `vllm.config.ProfilerConfig` 与 vLLM 的 `delay/max` step gate —— rtp-llm 有自己的 C++ `StepWindowProfiler`，不该也不需引入 vLLM 类
- ❌ `NPUWorker` 是 vLLM worker，rtp-llm 用 C++ `NormalEngine`
- ❌ trace_name 的 vLLM rank 格式 `dp{X}_pp{X}_..._rank{X}` 是 vLLM 专属

### 为什么 rtp-llm 不该照搬 vllm-ascend 的外围
rtp-llm 引擎主循环在 **C++**，模型 forward 才回到 Python（`PyWrappedModel.cc:344/388`）。vllm-ascend 的 `WorkerProfiler` 是纯 Python、绑定 vLLM Python worker。把整套搬来会引入 vLLM 强耦合，且与 rtp-llm 已有的 C++ `StepWindowProfiler`/gRPC `StartProfile`/`gen_timeline` 重复造轮子。

### 正确复用姿势
**保留 vllm-ascend 的「torch_npu 调用配方」，套到 rtp-llm 已有的「C++ profiler 框架」上**：
- 窗口控制：沿用 rtp-llm `StepWindowProfiler`（start_step/num_steps/gRPC 触发）
- 数据采集：把 vllm-ascend 的 `torch_npu.profiler.profile(...)` 块放进 Python 侧（或经 pybind 由 C++ 触发），替换 `cudaProfilerBegin/End` 的 Ascend no-op

---

## 3. xllm 方案能否复用？

### 结论：**控制流脚手架可参考，但 Ascend 数据采集不是 torch_npu 路径，且为死代码**

- xllm **不用** `torch_npu.profiler` —— 走 C 级 `mspti`/`mstx`，与用户问题"基于 torch_npu 接口"不符
- xllm 的 mspti 脚手架（`mspti_helper.*`）**未接线**（`register_subscriber` 从不调用、`LLM_MSTX_RANGE()` 全注释）
- xllm 的在线 profiling endpoint 在 NPU 构建下直接返回 false（CUDA gate）
- 唯一可参考：HTTP fan-out、单例幂等、`npu_timeline.py` log→trace 转换器

### 与 rtp-llm 的关系
rtp-llm 是 xllm 后继，已有等价（甚至更完整）的 C++ profiler 框架（`StepWindowProfiler` 已接入 step 循环 + gRPC + per-req flag）。**无需从 xllm 借鉴控制流** —— rtp-llm 已具备。

---

## 4. 最终判断

| 问题 | 答案 |
|------|------|
| vllm-ascend 方案能否在 rtp-llm 上用？ | **配方能，外围不能。** 取 vllm-ascend 的 torch_npu profiler 调用块，套进 rtp-llm 已有 C++ profiler 框架 |
| xllm 方案是否需要？ | **不需要。** rtp-llm 已有等价 C++ 控制流；xllm 的 Ascend 数据采集非 torch_npu 路径且为死代码 |
| 推荐路径 | **vllm-ascend 的 torch_npu.profiler 配方 + rtp-llm 已有 StepWindowProfiler/gRPC/`gen_timeline` 框架** |

---

## 5. 推荐方案的三个关键理由

1. **最小改动**：rtp-llm 已有完整的 profiler 窗口控制 + 触发链 + 异步落盘，只缺 Ascend 后端实现
2. **稳定性**：`torch_npu.profiler.profile(...)` 是 Huawei 官方稳定 Python 接口（vllm-ascend 生产在用），比 C 级 `mspti`（xllm 死代码）更稳
3. **输出对齐**：vllm-ascend 产出的 `trace_view.json`/`kernel_details.csv` 等可被 MindStudio Insight 打开，与 Ascend 生态工具链一致
