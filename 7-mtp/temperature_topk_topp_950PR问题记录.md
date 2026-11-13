# Temperature > 0 (Top-K/Top-P 采样) 在 950PR 上的问题记录

> 时间：2026-09-28 ~ 2026-09-29
> 影响范围：所有 `temperature > 0` 且触发 `top_k` / `top_p` 过滤的请求
> 状态：**未修复**——950PR 缺少兼容的 AscendC kernel，需要算子适配或替代方案

---

## 一、问题概述

在 Ascend950PR 上，当请求携带 `top_p < 1.0` 或 `top_k > 0`（无论 temperature 是否 > 0）时，服务进程直接崩溃（`terminate`），而非优雅报错。greedy / `temperature = 0` 请求不受影响。

---

## 二、调用链

```
temperature > 0 / top_p < 1.0 / top_k > 0 请求
  → rtp_llm::Sampler::forward
    → rtp_llm::sampleGreedy (CudaSampleOp.cc, USING_ASCEND 分支)
      → has_top_k / has_top_p 检测为 true
        → rtp_llm::ascend::applyTopKTopP (ascend_apply_top_k_top_p_custom.h:53)
          → EXEC_NPU_CMD 宏
            → dlopen("libopapi.so") → dlsym("aclnnApplyTopKTopPCustom")
              → 调 aclnnApplyTopKTopPCustomGetWorkspaceSize()
                → 加载 AscendC kernel binary (.o 文件)
                  → ★ 这里崩溃 ★
```

---

## 三、报错信息（按时间顺序）

### 3.1 第一阶段：库缺失（Phase 1 调试期间）

```
aclnnApplyTopKTopPCustom or aclnnApplyTopKTopPCustomGetWorkspaceSize
not in libopapi.so, or libopapi.so not found.
```

**原因**：`libcust_opapi.so`（含自定义算子 host wrapper）未在 `LD_LIBRARY_PATH` 中。

**当时修复**：加 `LD_LIBRARY_PATH` 环境变量后解决（但仅解决了库加载层面）。

---

### 3.2 第二阶段：kernel 执行失败（当前问题）

```
terminate called after throwing an instance of 'c10::Error'
  what():  call aclnnApplyTopKTopPCustom failed, detail:
[PID: 240300] 2026-09-29-06:34:53.682.338
AclNN_Parameter_Error(EZ1001): The workspaceSize or executor is nullptr.

Exception raised from applyTopKTopP at
./rtp_llm/models_py/bindings/ascend/ops/ascend_apply_top_k_top_p_custom.h:53

Backtrace:
  frame #2:  rtp_llm::ascend::applyTopKTopP
  frame #3:  rtp_llm::sampleGreedy
  frame #5:  rtp_llm::Sampler::forward
  frame #6:  rtp_llm::MtpExecutor::prefillStep
  frame #8:  rtp_llm::NormalEngine::step
  frame #9:  rtp_llm::NormalEngine::loop
```

**含义**：host 侧 wrapper 函数加载成功（`libcust_opapi.so` 在 LD_LIBRARY_PATH 中），但调用 `aclnnApplyTopKTopPCustomGetWorkspaceSize()` 时返回 nullptr——运行时无法为 910b 编译的 kernel binary 在 950PR 硬件上创建执行上下文。

---

## 四、根因分析

### 4.1 算子来源

`applyTopKTopP` 是 rtp-llm 仓 `3rdparty/aclnn_custom_ops/` 中的 **CANN Open Software 自定义算子**（Copyright Huawei Technologies, CANN OSL）。包含：

```
3rdparty/aclnn_custom_ops/
├── BUILD                                    ← bazel genrule，SoC 硬编码 ascend910b
├── build_for_bazel.sh                       ← 透传 SoC 给 cmake
├── build.sh                                 ← cmake + make 入口
├── cmake/config.cmake                       ← 已支持 ASCEND950 检测
├── apply_top_k_top_p_custom/
│   ├── op_host/                             ← host 侧 wrapper（纯 C++，与 SoC 无关）
│   │   ├── apply_top_k_top_p_custom_def.cpp
│   │   └── aclnn_recurrent_gated_delta_rule.cpp
│   └── op_kernel/                           ← AscendC kernel 源码（SoC 相关！）
│       ├── apply_top_k_top_p_custom.cpp     ← 910b 架构特有编程模型
│       └── apply_top_k_top_p_custom.h
```

### 4.2 为什么 910b kernel 不能在 950PR 上运行

| 维度 | 910b | 950PR |
|---|---|---|
| AI Core 代际 | 上一代 | 新一代 |
| 向量指令集 | 不同 | 不同 |
| UB（Unified Buffer）大小/布局 | 不同 | 不同 |
| AscendC 编程模型参数 | 不同 | 不同 |
| **kernel binary 兼容性** | — | **不兼容** |

kernel binary（.o 文件）包含的是特定 SoC AI Core 的机器码。910b 和 950PR 的 AI Core 架构不同代际，**二进制不兼容**。

### 4.3 为什么不能为 950PR 编译

尝试将 BUILD 文件的 SoC 从 `ascend910b` 改为 `ascend950` 后：

| 步骤 | 结果 |
|---|---|
| CMake 配置 | ✅ 成功（`platform_ascendc::SocVersion::ASCEND950` 检测通过） |
| Host 侧库（.so 文件） | ✅ 编译成功（纯 C++，与 SoC 无关） |
| **AscendC kernel（.o 二进制）** | ❌ **被静默跳过**（0 个 .o 文件生成） |
| ops-info.ini 生成 | ❌ 失败（依赖 kernel 编译产物） |

**kernel 被跳过的原因**：`op_kernel/apply_top_k_top_p_custom.cpp` 使用 910b 特有的 AscendC 编程模型（向量操作模板、UB 管理、tiling 方案）。CANN 9.2.0 的 bisheng 编译器在为 ascend950 编译时**静默跳过**而非报错，导致 host 库看似编译成功但 kernel 缺失。

---

## 五、已验证的替代方案

### 5.1 `torch_npu.npu_top_k_top_p`（推荐，未实施）

op-plugin 提供的**原生算子**，Atlas A2/A3 系列（含 950PR）原生支持。

**已验证可用**（950PR 实机测试通过）：
```python
import torch_npu
out = torch_npu.npu_top_k_top_p(logits, p, k)  # 直接可用
```

文档：`/home/d30033799/op-plugin/docs/context/torch_npu-npu_top_k_top_p.md`

**问题**：在 C++ 代码（CudaSampleOp.cc）中调用该 Python 层算子需要 C++ dispatcher（`c10::Dispatcher::singleton().findSchemaOrThrow`），引入链接复杂度。可通过以下方式解决：
- 在 Python 层包装（如在 sampler Python 入口处预处理 top_k/top_p）
- 或在 C++ 中用 `#include <torch/script.h>` + `torch::jit::` API 调用

### 5.2 纯 torch fallback（已实施但回退）

用 `torch::sort` / `gather` / `where` / `cumsum` / `softmax` 实现等效过滤。已实现并编译通过，但 scatter 的 dtype 不匹配导致崩溃（可修复），用户选择回退到原始代码。

---

## 六、修复路径（待选）

| # | 方案 | 优点 | 缺点 |
|---|---|---|---|
| 1 | **改用 `torch_npu.npu_top_k_top_p`**（Python 层或 C++ dispatcher） | 原生 NPU 加速、无自定义算子依赖、官方支持 | 需要解决 C++ 调用方式 |
| 2 | **纯 torch fallback**（修复 scatter dtype 问题） | 零外部依赖、任意 SoC 可用 | 性能略差于融合 kernel |
| 3 | **为 950PR 编写 AscendC kernel** | 性能最优 | 需要适配 950 架构，工作量大 |
| 4 | **获取 Huawei 提供的 950 兼容版本算子** | 官方支持 | 需要联系 Huawei/查找 CANN 更新 |

---

## 七、临时规避

在算子适配完成前：
- **greedy / temperature = 0**：不受影响，正常使用
- **temperature > 0 但不设 top_k / top_p**：默认 top_p=1.0 / top_k=0，不触发 `applyTopKTopP`，正常使用
- **显式设置 top_p < 1.0 或 top_k > 0**：**会崩溃**，暂不可用

---

## 八、相关文件

| 文件 | 位置 |
|---|---|
| 崩溃点 | `rtp_llm/models_py/bindings/ascend/ops/ascend_apply_top_k_top_p_custom.h:53` |
| 调用方 | `rtp_llm/models_py/bindings/core/CudaSampleOp.cc` sampleGreedy Ascend 分支 |
| 自定义算子 | `3rdparty/aclnn_custom_ops/apply_top_k_top_p_custom/` |
| 编译配置 | `3rdparty/aclnn_custom_ops/BUILD`（SoC 硬编码 ascend910b） |
| 替代算子文档 | `/home/d30033799/op-plugin/docs/context/torch_npu-npu_top_k_top_p.md` |
| 替代算子测试 | `/home/d30033799/op-plugin/test/test_custom_ops/test_npu_top_k_top_p.py` |
