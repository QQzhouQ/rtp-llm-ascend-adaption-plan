# xLLM Profiling 采集方案分析

> 分析对象：`/home/d30033799/xllm`
> 关注点：是否使用 torch_npu 采集 profiling、方案是否可复用

---

## 1. 总体结论（关键纠正）

> **xLLM 并未使用 Python `torch_npu.profiler` API。** 全代码库零 `torch_npu.profiler` / `ProfilerActivity` / `export_chrome_trace` 使用。仅有的 `torch_npu` Python import 是 HCCL（`ProcessGroupHCCL`）和一个 bootstrap shim。

xLLM 的 profiling 分为**两套互不重叠的子系统**：

| 子系统 | 状态 | 后端 | 是否用 torch_npu |
|--------|------|------|------------------|
| **A. 在线 timeline profiling**（`/start_profile`、`/stop_profile` HTTP） | **已实现、已接线、仅 CUDA** | libtorch Kineto 或 `cudaProfilerStart/Stop` | **否**（`#if defined(USE_CUDA)` 守卫） |
| **B. Ascend MSPTI + mstx**（`mspti_helper.{h,cpp}`、`npu_timeline.py`） | **死代码 / 未接线** | Ascend `mspti`(CUPTI 对应) + `mstx`(NVTX 对应) | 仅链接 torch_npu headers，**从不调用** |

**对 rtp-llm 的意义**：在线 profiling 的控制流管线（HTTP→Master→Engine→Worker fan-out→幂等单例）**完整且可复用**，但在非-CUDA 上短路。Ascend 数据采集原语（`MsptiMetrics`、`MstxRange`）作为脚手架+文档配方存在，**未接入**该管线。

---

## 2. 核心文件清单

### 平台 / profiler 后端
- `xllm/core/platform/torch_profiler.{h,cpp}` — Kineto C++ profiler（默认 backend）
- `xllm/core/platform/cuda_profiler.{h,cpp}` — `cudaProfilerStart/Stop`（"cuda" backend）

### Ascend 专属 profiling 原语（未使用的脚手架）
- `xllm/core/common/mspti_helper.h`（76 行）— `MstxRange`、`LLM_MSTX_RANGE()` 宏、`MsptiMetrics`
- `xllm/core/common/mspti_helper.cpp`（208 行）— **唯一接触 torch_npu profiler headers 的文件**

### 配置
- `xllm/core/framework/config/profile_config.{h,cpp}` — `enable_online_profile`、`profile_backend`、`profile_dir`
- `xllm/core/common/global_flags.h:244-266` — gflags `DECLARE_*`

### HTTP API + 调用链
- `xllm/server/xllm_server.cpp:39-60` — 路由表注册
- `xllm/api_service/api_service.cpp:1304-1374` — `StartProfileHttp`/`StopProfileHttp`
- `xllm/core/distributed_runtime/llm_engine.cpp:1337-1383` — `LLMEngine` fan-out
- `xllm/core/runtime/worker_impl.cpp:1351-1392` — **`WorkerImpl` 分发器；CUDA-only 守卫**

### 文档与工具
- `docs/.../online_profiling.md` — 权威设计文档
- `tools/README.md` — MSPTI 集成配方
- `tools/npu_timeline.py`（516 行）— log → Chrome-trace 转换器

---

## 3. torch_npu / 头文件使用情况

`mspti_helper.cpp:19-27` 是 torch_npu profiler-邻接头文件唯一出现处：
```cpp
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include <acl/acl.h>
#include <torch_npu/csrc/libs/init_npu.h>
```

实际 profiling 调用**不是** torch_npu 调用，而是 Ascend C 级 **`mstx`**(标注) 与 **`mspti`**(活动 trace) 库：

### mstx（NVTX 对应，`USE_NPU` 下编译）
```cpp
MstxRange::MstxRange(const char* name) : name_(name) {
  aclrtGetDevice(&dev_id);
  stream_ = c10_npu::getCurrentNPUStream(dev_id);   // 唯一 torch_npu 运行时 API
  mstx_id_ = mstxRangeStartA(name, stream_);
}
MstxRange::~MstxRange() { aclrtSynchronizeStream(stream_); mstxRangeEnd(mstx_id_); }
```

### mspti（CUPTI 对应，`USE_MSPTI` 下编译）
```cpp
msptiSubscribe(&subscriber_, nullptr, nullptr);
msptiActivityRegisterCallbacks(user_buffer_request, user_buffer_complete);
msptiActivityEnable(MSPTI_ACTIVITY_KIND_MARKER);
msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
// ... msptiActivityGetNextRecord(buffer, validSize, &pRecord);
```

> 没有 `torch_npu.profiler.*`、`ProfilerActivity.NPU`、`experimental.*`、`export_chrome_trace`。

---

## 4. 实际后端：libtorch Kineto（C++），非 torch_npu

在线 endpoint 实际驱动的是 `torch_profiler.cpp`（libtorch Kineto）：
```cpp
const std::set<tp::ActivityType> activities = {tp::ActivityType::CPU,
                                               tp::ActivityType::CUDA};  // 硬编码 CUDA
ap::enableProfiler(config, activities);
// stop: auto result = ap::disableProfiler(); result->save(path);
```
且整个 `WorkerImpl::start_profile` 被 `#if defined(USE_CUDA)` 守卫 —— NPU 构建下 endpoint 总返回 false。

---

## 5. 触发与调用链（与 rtp-llm 同源，可复用）

```
HTTP POST /start_profile
  → APIService::StartProfileHttp       api_service.cpp:1304
  → Master::start_profile              master.h:54
  → LLMEngine::start_profile           llm_engine.cpp:1337
       folly::collectAll over all worker_clients_:
       → WorkerClient::start_profile_async  worker_client.cpp:189 (本地)
       → WorkerImpl::start_profile          worker_impl.cpp:1351  ← CUDA 守卫
            if profile_backend=="cuda": CudaProfiler::start()
            else (默认 "torch"): schedule 到 compute 线程 → TorchProfiler::start()
       (remote workers 走 brpc → WorkerService::StartProfile → WorkerImpl)
```
`/stop_profile` 对称。**关键 CUDA 守卫**（`worker_impl.cpp:1367-1370`）：
```cpp
#else
  LOG(ERROR) << "Online timeline profiling is only supported on CUDA.";
  return false;
#endif
```

---

## 6. 输出格式与位置

- **`torch` backend**：Chrome Trace JSON，文件名 `xllm_rank<rank>_<pid>_<ms_timestamp>.pt.trace.json`，目录 `--profile_dir`，用 Perfetto/chrome://tracing 打开。
- **`cuda` backend**：xllm 不写文件，nsys 进程退出时写 `xllm_profile.nsys-rep`。
- **MSPTI 子系统**：活动记录经 `LOG(INFO)` 序列化到 glog 文件（如 `log/node_0.log`），后用 `tools/npu_timeline.py`（`-i ./node_0.log -o mspti_chrome_trace.json`）转 Chrome trace。

---

## 7. 与推理生命周期的集成

1. **进程模型**：单进程内每设备一 worker 线程；profiler 是**进程级幂等单例**（`get_instance()` + mutex），重复 start 合并、多余 stop no-op。
2. **CPU-op 采集正确性**：Kineto 经线程局部 `RecordFunction` 记录 host op，故 start/stop **schedule 到 worker compute 线程**（非 RPC handler 线程）。
3. **广播语义**：`LLMEngine` 对每个 worker 并发 fan-out，等所有 ack。
4. **与推理正交**：profiling 纯 toggle，用户用 HTTP 手动括号窗口，推理路径不触碰。

> 注：`ProfileManager`（`profile_manager.cpp`，1011 行）名字相似但属于**另一子系统** —— 预填充/解码延迟预测（用于调度器），跑 dummy request 测 wall-clock 训练 `TimePredictor`，并做 ACL-graph warmup。**不属** timeline profiling 流。

---

## 8. 对 rtp-llm 的可复用性评估

由于 rtp-llm 是 xllm 的直接后继，映射基本 1:1。

### 可原样复用
1. **整个 HTTP→Master→Engine→Worker fan-out 管线**（泛型）
2. **单例 profiler 模式**（`get_instance()` + mutex + 幂等 start/stop）—— 对 Ascend 也正确（mspti/mstx 进程级）
3. **start/stop 的 compute 线程调度**
4. **config 管线**（`ProfileConfig`、三 flag、JSON config 支持）
5. **`tools/npu_timeline.py`** —— 现成 log→Chrome-trace 转换器，schema 匹配 `MsptiMetrics`

### 需填补的缺口
1. `WorkerImpl::start_profile/stop_profile` 去掉 `#if defined(USE_CUDA)`，加 `#elif defined(USE_NPU)`（或新 `profile_backend="npu"`）
2. `MsptiMetrics`、`MstxRange` 是**死代码**：
   - `mspti_helper.h` 在 `mspti_helper.cpp` 外**无处** include
   - `MsptiMetrics::register_subscriber/release_subscriber` **从不调用**
   - 所有 `LLM_MSTX_RANGE()` 站点被**注释**（如 `npu_qwen2_decoder_layer_impl.cpp:267-276`）
   - 激活需：`-DUSE_MSPTI=ON` 构建 + 在 NPU 分支 `start_profile` 接 `register_subscriber()`、`stop_profile` 接 `release_subscriber()`+log flush + 取消注释 `LLM_MSTX_RANGE()`

### 两种可行 Ascend 策略
- **Option 1 — 进程内 MSPTI**（`mspti_helper.*`+`npu_timeline.py` 原生设计）：subscribe KERNEL/HCCL/MEMORY/MARKER，log JSON，后处理 Chrome trace。全进程内，类 "torch" backend。
- **Option 2 — 进程外 msprof**（`PROFILING_MODE=dynamic`+`xllm-npu-profiler`）：HTTP 仅括号 capture 窗口，msprof 外部驱动。需 `aclrt*` capture-range 等价物 —— **源码中不存在**（无 `aclrtProfiler*`）。

---

## 9. 关键告诫

> "xllm 用 torch_npu API 采集 profiling" 的前提**字面不成立**。

- xllm **活跃** profiling 用 libtorch Kineto/CUDA（仅 CUDA）
- xllm **Ascend** profiling 用 C 级 `mspti`/`mstx` 库（与 `torch_npu` 一起链接，用 `c10_npu::getCurrentNPUStream` 取 stream），**而非** `torch_npu.profiler` Python 模块
- 若 rtp-llm 明确要 `torch_npu.profiler.profile(...)` / `ProfilerActivity.CPU+NPU` / `export_chrome_trace` 这条 Python 路径（vLLM-Ascend 走法），**xllm 无先例** —— 属净新代码
- xllm *提供*的是周边控制流脚手架（HTTP endpoint、fan-out、config、单例幂等、log→trace 转换器），可接入 MSPTI C++ backend 或 `torch_npu.profiler`-via-pybind backend
