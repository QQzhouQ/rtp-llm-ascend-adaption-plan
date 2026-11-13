# vLLM-Ascend Profiling 采集方案分析（基于 torch_npu 接口）

> 分析对象：`/home/d30033799/vllm-ascend`
> 关注点：基于 `torch_npu` 接口的性能 Profiling 采集方案

---

## 1. 总体结论

vllm-ascend 的 profiling 体系由 **三套相互独立、互斥感知** 的机制构成，其中核心是 **Ascend PyTorch Profiler（`torch_npu.profiler.*`）**：

| 机制 | 角色 | 是否用 torch_npu | 状态 |
|------|------|------------------|------|
| **A. Ascend PyTorch Profiler** (`torch_npu.profiler.profile`) | 算子级性能 trace（核心方案） | ✅ 是 | 主力，本文件重点 |
| **B. MS Service Profiler** (`ms_service_profiler`) | 框架/服务级 trace（请求、kvcache、batch） | ❌ 否（外部工具） | 辅助，可叠加 |
| **C. msMonitor** (`torch_npu.profiler.dynamic_profile`) | 常驻轻量监控 | ✅ 是（`dp.step()`） | 与 A **互斥** |

> 另有 `msprobe`（精度 dump，非性能 trace）和 `ProfilingChunk`（动态分块，非 trace），不在本方案讨论范围。

---

## 2. 核心文件清单

| 文件 | 作用 |
|------|------|
| `vllm_ascend/profiler/torch_npu_profiler.py` | **核心**：`TorchNPUProfilerWrapper`，构造并驱动 `torch_npu.profiler.profile(...)` |
| `vllm_ascend/profiler/__init__.py` | 导出 `TorchNPUProfilerWrapper` |
| `vllm_ascend/worker/worker.py` | `NPUWorker` 拥有 profiler 生命周期（`profile()` / 懒加载 / `execute_model` 步进） |
| `vllm_ascend/envs.py` | `MSMONITOR_USE_DAEMON` 环境变量 |
| `vllm_ascend/ascend_config.py` | `msmonitor_use_daemon` 解析、`ProfilingChunkConfig` |
| `vllm_ascend/profiling_config.py` | 自动生成 `service_profiling_symbols.<vllm_version>.yaml` |
| `docs/.../service_profiling_guide.md` | 官方用户指南 |

---

## 3. torch_npu 接口调用（核心代码）

整个 torch_npu profiler 调用集中在 `vllm_ascend/profiler/torch_npu_profiler.py`，构造逻辑如下（lines 49-76）：

```python
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

return torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    with_stack=False,
    profile_memory=torch_profiler_with_memory,
    with_modules=torch_profiler_with_stack,
    experimental_config=experimental_config,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        torch_profiler_dir,
        worker_name=trace_name,
    ),
)
```

> #### ⚠️ 常见误解澄清：`_ExperimentalConfig` 不管「Python 还是 C++」
>
> `_ExperimentalConfig` 是一个**配置结构体**（不是采集函数），它配置的全是**设备侧（Ascend NPU）的采集粒度 + op 级细节 + 用户标注**，与「Python 侧还是 C++ 侧」完全无关。常见误以为「它只能采 Python 信息」是错的。
>
> | 字段 | 控制 | 侧 |
> |------|------|-----|
> | `profiler_level` (Level0/1/2) | NPU 采集深度 | 设备 |
> | `aic_metrics` | AICore 性能计数器（PIPE_UTILIZATION 等） | 设备 |
> | `l2_cache` | L2 cache 统计 | 设备 |
> | `op_attr` / `record_op_args` | op 属性 / 输入参数 | op |
> | `sys_io` / `sys_interconnection` | 系统 I/O / 互联 | 设备 |
> | `msprof_tx` / `mstx` | 用户标注（mstx ≈ NVTX，C++ 侧打点也可采） | 标注 |
> | `export_type` / `data_simplification` | dump 格式 / 数据简化 | 输出 |
> | `host_sys` / `gc_detect_threshold` | host 系统 / GC | host |
>
> **真正决定 host 侧（含 C++）能否被采的是**：`activities=[ProfilerActivity.CPU]`。CPU 活动启用后，Kineto 注册**全局 `at::RecordFunction` 回调**，捕获所有线程的 `before()/end()` 调用。任何用 `at::RecordFunction` / `RECORD_FUNCTION` / `RTP_LLM_PROFILE_SCOPE` 标注的 C++ 代码都会被采到——与 `_ExperimentalConfig` 如何配置无关。
>
> 实证：在 rtp-llm 端到端调测中（见 `06_validation_record.md`），使用上述 `Level1 / AiCoreNone` 配置，trace 里仍出现了纯 C++ 事件 `engine.normal.step_work`(27ms)、`executor.model_forward`、`KVCacheManager::malloc` 等，证明 C++ 侧被完整采集。
>
> 若需要**更多细节**，调的是这些（而非纠结 Python/C++）：`profiler_level=Level2`（更深 kernel）、`op_attr=True`、`record_op_args=True`、`aic_metrics=ACL_AICORE_PIPE_UTILIZATION`、`l2_cache=True`、`mstx=True`。

### torch_npu API 完整清单

| API | 用途 |
|-----|------|
| `torch_npu.profiler._ExperimentalConfig(...)` | Ascend 扩展配置（export_type / profiler_level / aic_metrics / data_simplification 等） |
| `torch_npu.profiler.profile(activities=[CPU,NPU], on_trace_ready=...)` | 构造 profiler 上下文对象 |
| `torch_npu.profiler.ProfilerActivity.CPU` / `.NPU` | 采集 CPU 侧 + NPU 侧事件 |
| `torch_npu.profiler.tensorboard_trace_handler(dir, worker_name=...)` | trace 完成回调，写出 dump |
| `torch_npu.profiler.dynamic_profile.step()` (`dp.step()`) | msMonitor 路径的常驻步进 |
| `profiler.start()` / `profiler.stop()` | 启停采集 |

### Wrapper 生命周期钩子（覆盖 vLLM 父类 `WorkerProfiler`）

```python
def _start(self):          self.profiler.start()    # torch_npu_profiler.py:78-79
def _stop(self):           self.profiler.stop()     # torch_npu_profiler.py:81-82
def _profiler_step(self):  return True              # torch_npu_profiler.py:84-85
```

> 注意：**没有** 用 `torch_npu.profiler.schedule(wait/warmup/active/repeat)`，步进门控完全依赖父类 `WorkerProfiler` 的 `delay_iterations`/`max_iterations` + HTTP on/off 控制。

---

## 4. 触发方式与调用链

### 触发入口

**(A) HTTP API（主推，在线服务）** —— 由 vLLM core 的 OpenAI server 提供：
```bash
curl -X POST http://localhost:8080/start_profile
curl -X POST http://localhost:8080/stop_profile
```

**(B) CLI flag（预置能力）** —— 通过 `--profiler-config`：
```bash
--profiler-config '{"profiler": "torch",
                    "torch_profiler_dir": "./vllm_profile",
                    "torch_profiler_with_stack": false}'
```

### 调用链

```
HTTP /start_profile?profile_prefix=...
   │  (vLLM core 路由 → engine → executor → worker RPC)
   ▼
NPUWorker.profile(is_start=True, profile_prefix=...)          worker.py:687-710
   │  - 校验 profiler_config 已启用
   │  - trace_name = f"{profile_prefix}_{rank_suffix}"        # dp{X}_pp{X}_tp{X}_..._rank{X}
   │  - 懒加载：首次 start 才构造 TorchNPUProfilerWrapper      worker.py:703-706
   │        → _create_profiler() → torch_npu.profiler.profile(...)
   │  - self.profiler.start()
   ▼
... 每个 request ...
NPUWorker.execute_model()                                     worker.py:400-435
   │  - if msmonitor_use_daemon: dp.step()                    worker.py:405-406
   │  - if self.profiler: self.profiler.step()                worker.py:432-433
   ▼
WorkerProfiler.step()  (父类，按 delay/max gate)
   │  - 到 delay → _start() → torch_npu.profiler.profile.start()
   │  - 超 max_iterations → _stop()
   ▼
HTTP /stop_profile
   ▼
NPUWorker.profile(is_start=False) → profiler.stop() → _stop() → profile.stop()
   ▼
on_trace_ready: torch_npu.profiler.tensorboard_trace_handler(dir, worker_name=trace_name)
   → 写出 *_ascend_pt 目录
```

**关键行为**：
- **懒初始化**：`self.profiler` 初始为 `None`，首次 `start_profile` 才构造 `torch_npu.profiler.profile` 对象。
- **重启复用**：stop 后再 start 复用同一 profiler 对象，只重新 `start()`，保留首个 trace_name。
- **msMonitor 互斥**：若 `msmonitor_use_daemon` 开启，构造 torch profiler 会抛 `RuntimeError`。

---

## 5. 环境变量与配置项

| 项 | 默认 | 说明 |
|----|------|------|
| `MSMONITOR_USE_DAEMON` | `0` | 启用 msMonitor 常驻 daemon；与 torch profiler **互斥** |
| `profiler` | — | `ProfilerConfig` 中须为 `"torch"` |
| `torch_profiler_dir` | — | 输出目录，非空必填 |
| `torch_profiler_with_memory` | — | `profile_memory=` |
| `torch_profiler_with_stack` | — | 映射到 `with_modules=`（torch_npu 视同 with_stack） |
| `delay_iterations` | 0 | 起始前跳过的步数 |
| `max_iterations` | 0 | 自动停止步数（0=不自动停） |

外部工具相关（文档提及）：`SERVICE_PROF_CONFIG_PATH`、`PROFILING_SYMBOLS_PATH`、`LD_PRELOAD=.../libmspti.so`（仅 acl_task_time=2）。

---

## 6. 输出格式与位置

### Ascend PyTorch Profiler（torch_npu 路径）
- **位置**：`<torch_profiler_dir>`（如 `./vllm_profile`）
- **子目录**：`<host>_<trace_name>_ascend_pt/`，trace_name 含 rank 后缀
- **原始格式**：ascend_pt 文本格式（`ExportType.Text` 强制）
- **后处理**（用户手动执行）：
  ```python
  from torch_npu.profiler.profiler import analyse
  analyse("./vllm_profile/localhost.localdomain_*_ascend_pt/")
  ```
- **产物**（`ASCEND_PROFILER_OUTPUT/`）：`trace_view.json`（Chrome Tracing）、`kernel_details.csv`、`op_statistic.csv`、`step_trace_time.csv`、`analysis.db` 等 → 用 MindStudio Insight 打开。

---

## 7. 框架耦合点（哪些是 vllm-ascend 专属、哪些通用）

### vllm-ascend 专属（不可直接搬）
1. **继承 vLLM `WorkerProfiler`**：`TorchNPUProfilerWrapper` 插桩于 vLLM 的 worker-profiler 抽象，`_start/_stop/_profiler_step` 覆盖父类；`delay/max` 门控继承自上游 vLLM。耦合 `vllm.profiler.wrapper.WorkerProfiler` 与 `vllm.config.ProfilerConfig`。
2. **`NPUWorker` 拥有生命周期**：profiler 绑在 `vllm_ascend.worker.worker.NPUWorker`（仅 platform=Ascend 时选中）。
3. **trace_name 用 vLLM rank 格式**：`dp{X}_pp{X}_tp{X}_dcp{X}_ep{X}_rank{X}`。
4. **msMonitor 互斥策略** 是 Ascend 专属。

### 通用可复用（核心配方）
- **`torch_npu.profiler.profile(...)` 调用块**（含 `_ExperimentalConfig`）是自包含配方，可lift到任何 PyTorch+Ascend 应用；Ascend 旋钮全部集中在 `_ExperimentalConfig`。
- **dump 分析步骤** `torch_npu.profiler.profiler.analyse(...)` 是独立能力。

---

## 8. 小结

vllm-ascend 的 torch_npu profiling 方案本质是：**一层薄薄的、策略丰富的 `WorkerProfiler` 子类包裹 `torch_npu.profiler.profile`**，通过标准 vLLM HTTP `start_profile/stop_profile` 触发，dump 出每 worker 的 `ascend_pt` trace，再用 `torch_npu.profiler.profiler.analyse` 后处理。

- **配方本身（torch_npu 调用块）可移植**
- **外围脚手架（WorkerProfiler/vLLM config/HTTP endpoint）是 vLLM 专属，不可直接搬**
