# getTorchCudaDevice 改回 index=-1 的决策记录

> 日期：2026-08-29
> 涉及：`rtp_llm/models_py/bindings/core/ExecOps.cc` getTorchCudaDevice()
> 背景：上游 PR #1349 checklist C-6（状态不变量：多卡落卡无保证）曾建议带 index

## 变更

```cpp
// 曾改为（已撤销）：
return torch::Device(torch::kPrivateUse1, static_cast<c10::DeviceIndex>(ascend::getDevice()));
// 现恢复为（上游原始形态）：
return torch::Device(torch::kPrivateUse1);   // index = -1
```

## 依据（实测）

```
torch_npu 对 index=-1 的解析与 CUDA 一致 = "当前设备"：
  torch.device("npu") → 张量实际分配在 npu:0（= set_device 设定的目标）
```

## 已知约束（遗留的防御性缺口，记录备查）

**多卡 TP 场景下，index=-1 的正确性依赖一条前置不变量**：

> 进程必须先完成 `set_device(local_rank)`，之后 `getTorchCudaDevice()` 才会落到正确的卡。

该不变量由 `setup_cuda_device_and_accl_env()`（server_config_setup.py）在引擎启动前建立。

**风险链**：本 PR 将该处的 `torch.npu.set_device` 失败处理从静默 pass 改为
`logging.warning` 后继续（非 fatal）。这意味着：

```
set_device 失败（如 torch_npu 库缺失 OSError / torch.npu 未注册 AttributeError）
  → 仅打印 warning，进程继续启动
  → getTorchCudaDevice() 的 index=-1 解析为"当前设备" = 默认设备 0
  → 多卡 TP 下该 rank 的张量分配/流操作全部落卡 0
  → 与其它 rank 数据竞争或错卡执行，且无报错（静默错误）
```

（带 `ascend::getDevice()` 显式 index 的写法可以堵住此链——即使 set_device 失败，
getDevice() 返回的仍是当前实际设备；但当时为最小化 diff 恢复上游形态而撤销。）

## 后续建议

若将来启用 Ascend 多卡 TP，二选一加固：
1. 恢复显式 index：`Device(kPrivateUse1, ascend::getDevice())`
2. 或将 `setup_cuda_device_and_accl_env` 中 set_device 失败升级为 fail-fast（RTP_LLM_FAIL）

任一改动需配套多卡（≥2 NPU）启动 + TP 推理用例验证。
