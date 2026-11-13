# RTP-LLM 框架现状与已有 Profiling 基础设施分析

> 分析对象：`/home/d30033799/rtp-llm`
> 目的：搞清框架架构、torch_npu 依赖现状、已有 profiling 基础设施，为移植方案定位插入点

---

## 1. 总体结论

RTP-LLM 是 **C++/Python 混合** 的 LLM 推理引擎（Bazel 构建）：
- **性能关键引擎主循环跑在 C++**（`rtp_llm/cpp/`）
- 通过 **pybind11** 调用 **Python 模型实现**（`rtp_llm/models_py/`）
- `torch_npu` **已经是 Ascend 构建的一等依赖**（Python wheel + C++ so + headers 均已接入）
- 已有 **成熟的、sglang 风格的 C++ profiling 框架**（`StepWindowProfiler`，基于 PyTorch Kineto），通过 **gRPC** 和 **per-request flag** 暴露
- **但是**：Kineto 活动类型硬编码为 `{CPU, CUDA}`；`cudaProfilerBegin/End` shim 在 Ascend 上是 **no-op + 显式 TODO**；全代码库 **没有任何 `torch_npu.profiler` / `msprof` / `aclprof` 集成**

> 这正是 vllm-ascend 的 torch_npu profiling 方案要填补的缺口。

---

## 2. 顶层架构

**语言**：C++（engine core、device、graph runner、RPC/HTTP server）+ Python（model 定义、server 入口、device 检测、config）

**构建**：Bazel（`WORKSPACE`、`BUILD*`、`def.bzl`、`.bazelrc`），多后端：`cuda_configure` / `rocm_configure` / **`ascend_configure`**（`WORKSPACE:5-12`）

| 路径 | 角色 |
|------|------|
| `rtp_llm/cpp/` | C++ engine core |
| `rtp_llm/models_py/` | Python 模型实现 + pybind op 绑定（ascend/cuda/rocm） |
| `rtp_llm/server/`、`rtp_llm/start_*.py` | Python server 入口 & 参数解析 |
| `rtp_llm/device/` | Python device 抽象（`device_type.py`） |
| `rtp_llm/model_loader/` | Python 权重加载 |
| `rtp_llm/config/` | Python server config |

**C++ engine 关键子目录**：
- `engine_base/` — `EngineBase`、scheduler、streams、**`TorchProfiler.{h,cc}`** ← 核心
- `normal_engine/` — 主 decode/prefill 引擎（`NormalEngine`、`NormalExecutor`）
- `models/` — `PyWrappedModel`（C++↔Python 模型边界）、`Sampler`
- `model_rpc/` — gRPC server（`LocalRpcServer`）+ `.proto`
- `api_server/` — C++ HTTP server（`HttpApiServer`）
- `pybind/` — `init.cc` 定义 `PYBIND11_MODULE(libth_transformer, m)`
- **`ascend_graph/`** — Ascend ACL Graph（NPUGraph）decode runner
- `cuda_graph/`、`cache/`、`metrics/` 等

---

## 3. 框架调用方式 & 引擎主循环

**入口**：`rtp_llm/start_server.py` → `rtp_llm/start_backend_server.py`。后端 fork rank 进程；每个 rank import torch_npu 并 set device：
- `rtp_llm/start_backend_server.py` — `torch.npu.device_count()` 计算 world-size
- 设备检测：`rtp_llm/device/device_type.py:27-33`（`import torch_npu; torch.npu.is_available()` → `DeviceType.Ascend`）

**Engine + RPC 启动**（C++ 入口，经 pybind 创建）：
- `rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.cc:291-326` — 创建 `LocalRpcServiceImpl`/`RemoteRpcServiceImpl`，构建 `HttpApiServer`、注册 gRPC service。
- Python `RtpLLMOp`（pybind class）包裹 C++ `NormalEngine`。engine 在自有循环线程跑 `NormalEngine::loop()` → `step()`。

> **Python 是启动器 & 模型实现者；推理循环、调度、batching、KV-cache 全在 C++**；C++ 仅在模型 `forward` 时回调 Python。

---

## 4. torch_npu 依赖现状（已完整接入）

### Python 侧（`deps/requirements_lock_ascend.txt`）
- `:346` — `torch-npu @ .../torch_npu-2.9.0-cp310-...aarch64.whl`
- `:338` — 匹配的 `torch` 2.9.0+cpu aarch64 wheel
- 通过 `@pip_ascend_torch` 在 `WORKSPACE:63-67` 锁定

### C++ 侧（`BUILD.torch_npu:1-17`）
`cc_library(name="torch_npu")` glob 了 `torch_npu.libs/libtorch_npu*.so*` + headers（`torch_npu/include/**`），依赖 `@torch_cpu_ascend//:torch`。已链接进 `libth_transformer`。

### torch_npu C++ headers 已 include 处
- `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h:21-22` — `NPUGraph.h`、`NPUStream.h`
- `rtp_llm/cpp/models/PyWrappedModel.h:28` / `.cc:22` — `NPUStream.h`
- `rtp_llm/models_py/bindings/ascend/ops/op_api_common.h:35` — `NPUStream.h` + `CalcuOpUtil.h`

### Python `import torch_npu` 处
- `rtp_llm/device/device_type.py:28`
- `rtp_llm/model_loader/loader.py:444`
- `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py:2`、`ascend_decode.py:2`、`ascend_kv_cache_write_op.py:2`

### 已用的 Ascend torch_npu op（Python）
`torch_npu.npu_fused_infer_attention_score_v2`、`torch_npu.npu_scatter_pa_kv_cache`、`torch.npu.graph_task_group_begin/end`、`torch.npu.current_stream`。

---

## 5. 已有 Profiling 基础设施（核心）

### 5.1 核心 profiler：`StepWindowProfiler`（C++，Kineto，sglang 风格）

**头文件**：`rtp_llm/cpp/engine_base/TorchProfiler.h`
- `TorchProfile` 类（`:18-41`）—— Kineto 低层封装；config=`KINETO`、`report_input_shapes=true`；**activities 硬编码 `{CPU, CUDA}`（`:39`）** ⚠️ 无 NPU 活动类型
- `ProfilerSaveWorker`（`:44-68`）—— 后台线程异步 `ProfilerResult->save()` 写 JSON
- `StepWindowProfiler`（`:74-116`）—— 可配 start_step/num_steps，原子 `configure()`（gRPC 线程安全）+ `tick()`（引擎循环线程内调用，满足 Kineto 线程亲和性）

**实现**：`rtp_llm/cpp/engine_base/TorchProfiler.cc`
- `start()`（`:22-27`）→ `tap::enableProfiler(config_, activities_)`
- `stopAndCollect()`（`:29-37`）→ `disableProfiler()`，生成 `"<dir>/<prefix><count>.json"`
- trace 经 `ProfilerSaveWorker`（`:67-87`）异步落盘，不阻塞推理

```cpp
// TorchProfile.h:38-39 —— 硬编码 CUDA，这是 Ascend 的关键缺口
tpi::ProfilerConfig config_ = tpi::ProfilerConfig(tpi::ProfilerState::KINETO, true);
std::set<tpi::ActivityType> activities_{tpi::ActivityType::CPU, tpi::ActivityType::CUDA};
```

### 5.2 引擎主循环接入（已有现成的插入点）

**NormalEngine** 持有 `StepWindowProfiler step_profiler_`（`NormalEngine.h:85`），用 `torch_cuda_profiler_dir` 作输出目录（`NormalEngine.cc:62`）。

**step 循环内的插入点**（`rtp_llm/cpp/normal_engine/NormalEngine.cc:466-489`）：
```cpp
// 插入点 #1：pre-process，configure + tick（启动 profiler）
if (!step_profiler_.enabled()) {
    for (const auto& stream : streams) {
        if (stream && stream->genTimeline()) {
            step_profiler_.configure(true, cfg->profile_trace_name, 0, cfg->profile_step);
            step_profiler_.tick();   // 立即启动（start_step=0）
            break;
        }
    }
}
status = executor_->process(streams);   // 实际 model forward（插入点 #2 之间）
step_profiler_.tick();                   // 插入点 #2：post-process，计数/停止
```

`EmbeddingEngine` 同构（`EmbeddingEngine.cc:104`）。基类虚函数：`EngineBase.h:89`（`startTimelineProfiling`）。

### 5.3 触发面（已有完整触发链）

1. **gRPC RPC**（主外部触发）：
   - proto：`rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto`（`StartProfileRequestPB{trace_name, start_step, num_steps, enable_all_rank}`）
   - handler：`rtp_llm/cpp/model_rpc/LocalRpcServer.cc:451-510` —— 支持 TP-group 广播（`enable_all_rank`）
2. **per-request flag**：`gen_timeline` / `profile_step` / `profile_trace_name`（`GenerateConfig.h:81`）
3. **服务级**：`--gen_timeline_sync` / env `GEN_TIMELINE_SYNC=1`
4. **legacy nsight dir**：`torch_cuda_profiler_dir`

### 5.4 CUDA-only profiler shim（Ascend 缺口，明确 TODO）

`rtp_llm/models_py/bindings/core/ExecOps.cc:381-399`：
```cpp
void cudaProfilerBegin() {
#if USING_CUDA
    check_cuda_value(cudaProfilerStart());
#else
    // no-op on ROCm / Ascend
    // TODO: Ascend - fix profiler on Ascend     ← 显式缺口
#endif
}
void cudaProfilerEnd() { /* 同上，cudaProfilerStop() */ }
```
声明在 `ExecOps.h:56-57`。这是替换 Ascend profiler start/stop 的最干净 swap 点。

### 5.5 文档
`docs/build/en/_sources/references/profiling.md` —— 仅文档化 Kineto timeline（`gen_timeline`）+ NVIDIA Nsight/nsys，**无任何 Ascend/msprof 章节**。

---

## 6. 请求/推理生命周期与 Profiling 插入点

端到端流程（C++/Python 边界显式标注）：

1. **Client → gRPC/HTTP** —— 请求到 `LocalRpcServer`（gRPC）或 `HttpApiServer`（C++ HTTP）
2. **RtpLLMOp**（pybind）→ `engine->enqueue(stream)`
3. **NormalEngine.loop()** 在自有线程跑 `loop()` → `step()`
4. **step()**（`NormalEngine.cc:438-499`）：
   - **[插入点 #1]** `:471-480` —— pre-process profiler configure+tick
   - `scheduler_->schedule()` → streams batch
   - `executor_->process(streams)` → `PyWrappedModel::forward()`
   - **[插入点 #2]** `:489` —— post-process `tick()`
5. **PyWrappedModel::forward / forwardMicroBatched**（`PyWrappedModel.cc:324,344,388`）—— 获取 **GIL** 调 `py_model_.attr("forward_micro_batch")(...)` —— **这是 C++→Python 穿越**，所有 `torch_npu.*` op 在此执行
6. 输出回流 → client

### Ascend profiling 最优插入点
- **C++ 层（engine step 粒度）**：已通过 `StepWindowProfiler.tick()` 现成接入（`NormalEngine.cc:471-489`），仅 backend（Kineto activities + save）是 CUDA 专属
- **Python 层（per-forward 粒度）**：在 `models_py` 的 `forward_micro_batch` 路径内，或包裹 `PyWrappedModel::forwardMicroBatched`
- **no-op shim**：`cudaProfilerBegin/End`（`ExecOps.cc:383-399`）—— 替换为 ascend profiler start/stop 的最干净点

---

## 7. Python vs C++ 边界 —— torch_npu profiling 该放哪

| 关注点 | 位置 |
|--------|------|
| pybind module | `rtp_llm/cpp/pybind/init.cc:51`（`libth_transformer`） |
| C++→Python 模型调用 | `rtp_llm/cpp/models/PyWrappedModel.cc:344,388`（GIL 下 `forward_micro_batch`） |
| Python 模型代码 | `rtp_llm/models_py/modules/...`（ascend attention 实现在 `factory/attention/ascend_impl/`） |
| 已有 C++ Kineto profiler | `rtp_llm/cpp/engine_base/TorchProfiler.{h,cc}` |

**含义**：Python 级 torch_npu profiling（`torch_npu.npu.profile(...)`）有两种落法：
1. **在 Python 内包裹 model forward**（`models_py`）—— 但引擎循环在 C++，需每次 forward 起/停 profiler
2. **从 C++ 引擎驱动**（现有 `StepWindowProfiler` 模式）—— 通过 pybind 从 `tick()` 调用 Python profiler API —— **匹配现有架构与 gRPC `StartProfile` 面**

由于 `import torch_npu` 已在 Python 完成、C++ 侧已链接 `libtorch_npu`，**两条路径机制上都可行**，无需新增依赖接线。

---

## 8. Ascend/ACL 集成点

| 文件 | 作用 |
|------|------|
| `rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.cc` | `aclrtMemcpyAsync`、`aclrtSynchronizeDevice/Stream`、`aclrtGetMemInfo`、`c10_npu::NPUGraph` capture/replay |
| `rtp_llm/cpp/ascend_graph/ascend_graph_runner.{h,cc}` | `AscendGraphRunner`（decode-only ACL Graph runner） |
| `rtp_llm/cpp/models/PyWrappedModel.{h,cc}` | `#if USING_ASCEND` 分支用 `c10_npu::getCurrentNPUStream()` |
| `rtp_llm/models_py/bindings/ascend/ascend_host_utils.cc` | `aclrtSynchronizeDevice`、`aclrtGetDevice`、`aclrtGetDeviceCount`、`aclrtGetMemInfo(ACL_HBM_MEM,...)` |

**编译开关**：`USING_ASCEND`（`3rdparty/gpus/ascend_configure.bzl`，`WORKSPACE:5,12`）。

**关键观察**：全代码库 **零** `aclprof*` / `msprof*` / `torch_npu.npu.profile` / `experimental.export_chrome_trace` / `ProfilerActivity` 使用（已穷举 grep 验证）。唯一的 Ascend "profiling" 面就是 no-op 的 `cudaProfilerBegin/End` TODO。

---

## 9. 结论 —— 现状对移植方案的意义

1. **依赖管线已就位**：`torch_npu` 已是 Python + C++ 依赖
2. **profiler 框架已存在且形状正确**：`StepWindowProfiler`（sglang 风格、步进窗口、异步落盘、线程安全）已接入 step 循环并经 gRPC `StartProfile` 暴露
3. **缺口很窄**：`TorchProfile` 硬编码 `ActivityType::{CPU,CUDA}`；`cudaProfilerBegin/End` 在 Ascend 是显式 no-op。vllm-ascend 的 torch_npu 方案正好对上这两个点
4. **两条干净插入策略**（无需新接线）：
   - 扩展 C++ `StepWindowProfiler`/`TorchProfile` 注册 NPU 活动类型，或
   - 用 `torch_npu` profiler（Python 侧或经 pybind）实现 `cudaProfilerBegin/End` shim
5. **触发面可原样复用**：gRPC `StartProfile` + per-request `gen_timeline` 无需改，只换 profiler backend 实现
