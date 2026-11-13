# RTP-LLM Profiling 采集方案（最终推荐方案）

> ✅ **已落地验证**：编译通过、端到端采集成功（12 万事件，含 NPU kernel + host 算子 + 引擎 scope）。完整调测记录见 `06_validation_record.md`。
>
> 基于 `01`~`04` 分析结论：**取 vllm-ascend 的 torch_npu.profiler 调用配方，套进 rtp-llm 已有的 C++ StepWindowProfiler 框架**，最小改动、零新依赖、与现有触发链（gRPC `StartProfile` / per-req `gen_timeline`）无缝衔接。

---

## 0. 方案一句话总结

> **保留 rtp-llm 已有的 C++ `StepWindowProfiler`（窗口控制 + gRPC/per-req 触发 + 异步落盘），在 `TorchProfile` 的 Ascend 分支里经 pybind 调用 vllm-ascend 同款 `torch_npu.profiler.profile(...)` 配方**，填掉 `ExecOps.cc` 与 `TorchProfile` 的两处 CUDA-only 缺口。

---

## 1. 设计依据（为什么这样设计）

### 1.1 现状可复用资产（无需改动）
| 资产 | 位置 | 作用 |
|------|------|------|
| `StepWindowProfiler` | `rtp_llm/cpp/engine_base/TorchProfiler.{h,cc}` | 窗口控制（start_step/num_steps）、线程安全 configure/tick、异步落盘 |
| 引擎 step 接入点 | `rtp_llm/cpp/normal_engine/NormalEngine.cc:466-489` | tick() 在 process() 前后各一次，已正确括号 |
| gRPC 触发 | `LocalRpcServer.cc:451-510`（`StartProfile`/`StartProfileInternal`，含 TP 广播） | 外部触发，原样可用 |
| per-req 触发 | `gen_timeline`/`profile_step`/`profile_trace_name`（`GenerateConfig.h:81`） | 单请求触发，原样可用 |
| torch_npu 依赖 | Python wheel（`requirements_lock_ascend.txt:346`）+ C++ lib（`BUILD.torch_npu`） | 已就位，零新依赖 |

### 1.2 缺口（本方案填补）
1. `TorchProfile::activities_` 硬编码 `{CPU, CUDA}`（`TorchProfiler.h:39`）—— Kineto 无 NPU 活动
2. `cudaProfilerBegin/End` 在 Ascend 是 no-op + TODO（`ExecOps.cc:383-399`）
3. 全代码库零 `torch_npu.profiler` 集成

### 1.3 关键安全前提（已核实）
**引擎循环线程 == 调 `tick()` 的线程 == 调 Python `forward_micro_batch` 的线程**（都在 `NormalEngine::step()` 内，`NormalEngine.cc:471-489`）。
⇒ 在 `TorchProfile::start()/stop()` 里获取 GIL 调 Python torch_npu profiler **无死锁**，且 profiler 会在随后的 `forward_micro_batch`（npu op 发射处）期间处于激活态 —— 与现有 Kineto start/stop 时序完全一致。

---

## 2. 总体架构

```
┌─────────────────── 触发层（不变） ───────────────────┐
│ gRPC StartProfile  /  per-req gen_timeline  /  env   │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌─────────────── 窗口控制层（不变，C++） ───────────────┐
│ StepWindowProfiler.configure()/tick()                 │
│   按 start_step/num_steps 门控，在 process() 前后 tick │
│   tick() 内调 profiler_->start() / stopAndCollect()   │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────── 数据采集层（本方案改造点） ──────────────┐
│ TorchProfile::start()/stopAndCollect()                │
│   #if USING_CUDA  : Kineto enable/disableProfiler     │  ← 不变
│   #if USING_ASCEND: GIL → Python torch_npu profiler   │  ← 新增
│                      (vllm-ascend 同款配方)            │
└──────────────────────────┬───────────────────────────┘
                           ▼
┌──────────────────── Python 侧（新增） ───────────────┐
│ AscendTorchNpuProfiler (新模块)                        │
│   torch_npu.profiler.profile(CPU+NPU, _Experimental   │
│   Config, on_trace_ready=tensorboard_trace_handler)   │
│   .start() / .stop() → 写 *_ascend_pt/                │
└──────────────────────────┬───────────────────────────┘
                           ▼
                  torch_npu.profiler.profiler.analyse()
                  → ASCEND_PROFILER_OUTPUT/trace_view.json
                  → MindStudio Insight 打开
```

---

## 3. 详细改动清单

### 改动 1：新增 Python profiling 模块（vllm-ascend 配方）

**新文件**：`rtp_llm/models_py/profiling/ascend_profiler.py`

移植 vllm-ascend `vllm_ascend/profiler/torch_npu_profiler.py` 的核心配方，去 vLLM 耦合：

```python
import torch_npu

class AscendTorchNpuProfiler:
    """rtp-llm 的 Ascend torch_npu profiler 单例。
    由 C++ TorchProfile 经 pybind 调用 start()/stop()。
    """
    def __init__(self, output_dir: str, trace_name: str,
                 with_stack: bool = False, with_memory: bool = False):
        if not output_dir:
            raise RuntimeError("Ascend profiling requires non-empty output_dir")
        self.output_dir = output_dir
        self.trace_name = trace_name
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=False,
            data_simplification=True,
            record_op_args=False,
            gc_detect_threshold=None,
        )
        self._profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            with_stack=with_stack,
            profile_memory=with_memory,
            with_modules=with_stack,          # torch_npu 视同 with_stack
            experimental_config=experimental_config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir, worker_name=trace_name,
            ),
        )

    def start(self):
        self._profiler.start()

    def stop(self):
        # on_trace_ready 回调会在 stop 时写出 *_ascend_pt/ 目录
        self._profiler.stop()
```

**注册到 pybind**：在 `rtp_llm/cpp/pybind/init.cc`（或 `RtpLLMOp.cc` 的 pybind 模块内）暴露工厂：

```cpp
m.def("create_ascend_profiler", [](const std::string& dir,
                                   const std::string& trace_name,
                                   bool with_stack, bool with_memory) {
    py::object cls = py::module_::import("rtp_llm.models_py.profiling.ascend_profiler")
                         .attr("AscendTorchNpuProfiler");
    return cls(dir, trace_name, with_stack, with_memory);
});
```

> 说明：`PyWrappedModel` 已用 `py::object` 持有 Python 对象（`PyWrappedModel.h` 持 `py_model_`），同模式可直接复用。

---

### 改动 2：`TorchProfile` 增加 Ascend 分支

**文件**：`rtp_llm/cpp/engine_base/TorchProfiler.h`

```cpp
#pragma once
#include <atomic>
#include <mutex>
#include <thread>
// ... 原有 include ...
#include "torch/csrc/autograd/profiler_kineto.h"
#if USING_ASCEND
#include <pybind11/pybind11.h>   // 新增
namespace py = pybind11;
#endif

namespace rtp_llm {
namespace tpi = torch::profiler::impl;

class TorchProfile {
public:
    TorchProfile(const std::string& prefix, std::string output_dir = "");
    ~TorchProfile();
    void start();
    std::pair<std::unique_ptr<torch::autograd::profiler::ProfilerResult>, std::string> stopAndCollect();
    void stop();
    TorchProfile(const TorchProfile&) = delete;
    TorchProfile& operator=(const TorchProfile&) = delete;

private:
    std::string prefix_;
    std::string output_dir_;
    static std::atomic<size_t> count_;
    tpi::ProfilerConfig config_ = tpi::ProfilerConfig(tpi::ProfilerState::KINETO, true);
    std::set<tpi::ActivityType> activities_{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
    bool stopped_ = true;

#if USING_ASCEND
    // Ascend: 持有 Python torch_npu profiler 对象；start/stop 经 GIL 调用
    py::object ascend_profiler_;          // 由 init 时构造
    bool       ascend_mode_ = false;
    void startAscend();                    // GIL → ascend_profiler_.attr("start")()
    void stopAscend();                     // GIL → ascend_profiler_.attr("stop")()
#endif
};
}
```

**文件**：`rtp_llm/cpp/engine_base/TorchProfiler.cc`

```cpp
void TorchProfile::start() {
    count_ += 1;
    stopped_ = false;
#if USING_ASCEND
    ascend_mode_ = true;
    try {
        // 在引擎循环线程（与 forward 同线程）取 GIL 安全
        py::gil_scoped_acquire acquire;
        ascend_profiler_ = py::module_::import("rtp_llm.models_py.profiling.ascend_profiler")
                              .attr("AscendTorchNpuProfiler")(output_dir_, prefix_);
        ascend_profiler_.attr("start")();
        RTP_LLM_LOG_INFO("ascend torch_npu profiler started: dir=%s prefix=%s",
                         output_dir_.c_str(), prefix_.c_str());
        return;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("ascend profiler start failed, fallback disabled: %s", e.what());
        ascend_mode_ = false;
    }
#endif
    // CUDA / fallback 路径（原逻辑）
    namespace tap = torch::autograd::profiler;
    tap::prepareProfiler(config_, activities_);
    tap::enableProfiler(config_, activities_);
}

std::pair<std::unique_ptr<tap::ProfilerResult>, std::string>
TorchProfile::stopAndCollect() {
    if (stopped_) return {nullptr, ""};
#if USING_ASCEND
    if (ascend_mode_) {
        try {
            py::gil_scoped_acquire acquire;
            ascend_profiler_.attr("stop")();   // on_trace_ready 写 ascend_pt
            ascend_profiler_ = py::none();
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("ascend profiler stop failed: %s", e.what());
        }
        stopped_ = true;
        // Ascend 由 torch_npu 自行写盘，无需 ProfilerSaveWorker
        return {nullptr, ""};
    }
#endif
    auto res = tap::disableProfiler();
    std::string file_name = output_dir_ + "/" + prefix_ + std::to_string(count_) + ".json";
    stopped_ = true;
    return {std::move(res), std::move(file_name)};
}
```

> `StepWindowProfiler::tick()` 中 `if (res) { save_worker_.enqueue(...) }` 已天然跳过 Ascend（res 为 nullptr）—— **`StepWindowProfiler` 无需任何改动**。

---

### 改动 3：补齐 `cudaProfilerBegin/End` Ascend shim（可选，对应 capture-range 风格）

**文件**：`rtp_llm/models_py/bindings/core/ExecOps.cc:381-399`

若需保留 nsys 式"外部 capture"语义，可在此接 `aclrtProfilerStart/Stop` 或经 pybind 触发 torch_npu start/stop；当前主路径走改动 1+2 即可覆盖，本项可作为低优先 TODO：

```cpp
void cudaProfilerBegin() {
#if USING_CUDA
    check_cuda_value(cudaProfilerStart());
#elif USING_ASCEND
    // 接入 torch_npu/msprof capture-range（如需进程外 msprof 流程）
    // 当前由 TorchProfile 的 Ascend 分支覆盖，此处保留为后续扩展点
#endif
}
```

---

### 改动 4：输出目录与配置

复用现有 `torch_cuda_profiler_dir`（已绑，`ConfigInit.cc:441,456,475`，CLI `server_args/profile_debug_logging_group_args.py:52`），无需新增 flag。该目录即作为 `ascend_profiler` 的 `output_dir`。

建议在 `rtp_llm/cpp/normal_engine/NormalEngine.cc:62` 构造 `StepWindowProfiler` 时传入的 `torch_cuda_profiler_dir` 同时供 Ascend 路径使用 —— **已天然满足**（`TorchProfile` 构造已接收 `output_dir_`）。

---

## 4. 使用方式（面向用户）

### 方式 A：gRPC（已有，推荐）
```bash
# TP=1 或 enable_all_rank=true 时向任意 rank 发；多卡广播
grpcurl ... StartProfile '{trace_name:"mytrace", start_step:5, num_steps:3, enable_all_rank:true}'
# 跑若干请求……
grpcurl ... StopProfile '{}'
```
profiler 在第 5 步启动，跑 3 步后自动 stop，写出 `<torch_cuda_profiler_dir>/mytrace_wr0_*_ascend_pt/`。

### 方式 B：per-request（已有，单请求调试）
在请求体里设：
```json
{"gen_timeline": true, "profile_step": 3, "profile_trace_name": "debug_req"}
```
引擎检测到 `genTimeline()` → `step_profiler_.configure(...)` → 立即 start。

### 方式 C：环境变量快速抓取（无代码路径，msprof 兜底）
不改代码时，可临时用 torch_npu 的 dynamic msprof（参考 xllm 文档 `PROFILING_MODE=dynamic`），但**不推荐**——经集成路径（方式 A/B）才能拿到与请求/步对齐的 trace。

### 结果分析
```python
from torch_npu.profiler.profiler import analyse
analyse("./<torch_cuda_profiler_dir>/<host>_<trace>_ascend_pt/")
# 产物：ASCEND_PROFILER_OUTPUT/trace_view.json（Chrome Tracing）
#       kernel_details.csv / op_statistic.csv / step_trace_time.csv / analysis.db
```
用 **MindStudio Insight** 或 `chrome://tracing` 打开 `trace_view.json`。

---

## 5. 改动影响面与回归风险

| 文件 | 改动类型 | 风险 |
|------|----------|------|
| `rtp_llm/models_py/profiling/ascend_profiler.py`（新） | 新增 | 低（隔离模块） |
| `rtp_llm/cpp/engine_base/TorchProfiler.{h,cc}` | 加 `#if USING_ASCEND` 分支 | 低（CUDA 路径不变） |
| `rtp_llm/cpp/pybind/init.cc`（或 RtpLLMOp.cc） | 暴露工厂/或直接 import | 低 |
| `StepWindowProfiler` | **不改** | 无 |
| `NormalEngine.cc` step 循环 | **不改** | 无 |
| `LocalRpcServer.cc` gRPC | **不改** | 无 |
| `ExecOps.cc`（可选） | 填 Ascend shim | 低 |

**CUDA/ROCm 构建零影响**：所有新增均在 `#if USING_ASCEND` 守卫内。

**GIL 安全性**：`TorchProfile::start()/stop()` 在引擎循环线程调用，与 `forward_micro_batch`（同线程取 GIL）无嵌套冲突；`StepWindowProfiler::tick()` → `process()` → `forward` 顺序保证 profiler 在 npu op 发射期间激活。

---

## 6. 与 vllm-ascend / xllm 方案的对照

| 维度 | vllm-ascend | xllm | **rtp-llm（本方案）** |
|------|-------------|------|----------------------|
| 数据采集 | `torch_npu.profiler.profile`（Python） | C 级 mspti/mstx（死代码） | **`torch_npu.profiler.profile`（Python，同 vllm-ascend）** |
| 窗口控制 | vLLM `WorkerProfiler`（Python） | `WorkerImpl` 单例（C++） | **`StepWindowProfiler`（C++，已有）** |
| 触发 | vLLM HTTP | HTTP（CUDA gate） | **gRPC `StartProfile` + per-req `gen_timeline`（已有）** |
| 引擎耦合 | 强（vLLM worker） | 弱 | **无新增耦合（复用已有 C++ 框架）** |
| 输出 | ascend_pt → analyse → Chrome trace | Kineto JSON / mspti log | **ascend_pt → analyse → Chrome trace（同 vllm-ascend）** |

---

## 7. 实施步骤（建议顺序）

1. **新增** `rtp_llm/models_py/profiling/ascend_profiler.py`（改动 1）
2. **改** `TorchProfiler.h/.cc` 加 `#if USING_ASCEND` 分支（改动 2）
3. **（如需）** 在 `pybind/init.cc` 暴露工厂；或直接在 C++ 内 `py::module_::import`（改动 1 已用此法，可不改 init.cc）
4. **Ascend 构建**（`USING_ASCEND=1`）单测验证 start/stop/trace 落盘
5. **多卡**：验证 `enable_all_rank=true` 的 TP 广播（`LocalRpcServer.cc:476-497` 已实现，每 rank 各写一份带 `wr<rank>` 前缀）
6. **文档**：在 `docs/build/en/_sources/references/profiling.md` 增补 "Ascend Profiling" 章节（输出格式、analyse、MindStudio Insight）
7. （低优先）补 `ExecOps.cc` 的 Ascend shim（改动 3）

---

## 8. 结论

- **vllm-ascend 方案**：配方（`torch_npu.profiler.profile` 调用块）**可直接复用**；外围 `WorkerProfiler`/vLLM 耦合**不可也不必搬**
- **xllm 方案**：**无需借鉴**（rtp-llm 已有等价且更完整的 C++ profiler 框架；xllm 的 Ascend 数据采集非 torch_npu 路径且为死代码）
- **rtp-llm 方案**：在已有 `StepWindowProfiler` + gRPC + `gen_timeline` 框架上，经 pybind 接入 vllm-ascend 同款 torch_npu profiler 配方，**改动集中在 2 个文件（1 新增 + 1 修改），CUDA 路径零影响，零新依赖，多卡/触发链/异步落盘全部复用**

> 详细分析见同目录 `01_vllm_ascend_profiling_analysis.md`、`02_rtp_llm_current_profiling_state.md`、`03_xllm_profiling_analysis.md`、`04_reuse_assessment.md`。
