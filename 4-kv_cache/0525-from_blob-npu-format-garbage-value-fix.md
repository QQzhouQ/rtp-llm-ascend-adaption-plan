# rtp-llm NPU KV Cache Format 问题分析

## 1. 问题现象

启动 rtp-llm 服务加载 Qwen3-0.6B 模型后，发送推理请求时服务崩溃，返回：

```json
{"error_code":514,"error_code_str":"514_UNKNOWN_ERROR","message":"Socket closed"}
```

## 2. 错误链路

```
用户请求 → frontend → gRPC → backend_manager → Python forward() → C++ PyWrappedModel::forward()
  → AscendPrefillImpl.forward()
    → AscendKVCacheWriteOp.forward()
      → torch_npu._npu_reshape_and_cache()
        → RuntimeError: unknown format type:1752432660  ← 根因
```

### 关键日志

**engine.log:**
```
[ERROR] Python error during forward call on Python instance: RuntimeError: unknown format type:1752432660

At:
  ascend_kv_cache_write_op.py(45): _scatter_write_cache
  ascend_kv_cache_write_op.py(35): forward
  ascend_prefill.py(81): forward
  causal_attention.py(90): forward
  qwen3.py(66): forward
  qwen3.py(130): forward
```

**main_0.log:**
```
[ERROR] backend_server health check failed
[ERROR] frontend_server health check failed
[ERROR] Health checks failed
[INFO] proc.name [backend_manager] pid[xxx] is not alived
```

## 3. 根因分析

### 3.1 核心问题

`1752432660`（十六进制 `0x68740014`）不是一个合法的 ACL format 枚举值，而是**未初始化的内存垃圾值**。

标准 ACL format 枚举：
| 值 | 名称 |
|---|---|
| -1 | ACL_FORMAT_UNDEFINED |
| 0 | ACL_FORMAT_NCHW |
| 1 | ACL_FORMAT_NHWC |
| 2 | ACL_FORMAT_ND |
| 3 | ACL_FORMAT_NC1HWC0 |
| 4 | ACL_FORMAT_FRACTAL_Z |

### 3.2 问题根源：C++ 端 `torch::from_blob` 不设置 NPU storage format

KV Cache tensor 的创建链路：

```
BlockPool::initializeCacheBuffer()
  → torch::empty({bytes}, options)  // 创建 1D uint8 buffer，NPU format=ND ✓
  → BlockPool::initializeLayoutStrategies()
    → MemoryLayoutStrategy::processKVTensor()
      → torch::from_blob(layer_ptr, {block_num, stride}, kv_options)  // 创建 2D view，NPU format=垃圾值 ✗
```

**关键代码** (`MemoryLayoutStrategy.cc:73`):

```cpp
// separate_kv_cache_ 分支
auto layer_tensor = torch::from_blob(
    layer_ptr, {block_num, stride}, kv_options);  // ← NPU format 未初始化！
layer_kv_tensors_.push_back(layer_tensor);
```

**V cache 同样问题** (`BlockPool.cc:221`):

```cpp
auto v_view = torch::from_blob(v_ptr, k_layer_tensor.sizes(),
                                k_layer_tensor.strides(), v_options);  // ← NPU format 未初始化！
global_layer_v_tensors_[global_layer] = v_view;
```

### 3.3 为什么 `torch::from_blob` 不设置 format

PyTorch 标准的 `torch::from_blob` 仅创建一个 tensor view，不涉及 NPU storage format 的设置。
而 torch_npu 提供了自己的 `at_npu::native::from_blob` 实现（定义在 `torch_npu/csrc/aten/common/from_blob.h`），
该实现会正确处理 NPU format。rtp-llm 使用的是标准 `torch::from_blob`，导致 format 字段为未初始化值。

### 3.4 影响范围

`_npu_reshape_and_cache` 在 C++ 层检查 `key_cache` 和 `value_cache` tensor 的 ACL format，
发现不是合法值后直接报错 `unknown format type:1752432660`。

该错误在 C++ 层面抛出，**Python 端的 try/except 无法可靠捕获**（异步 NPU 执行模式下错误可能延迟到后续操作才触发）。

## 4. 修复方案

### 4.1 C++ 端修复（推荐，根本解决）

在 `torch::from_blob` 创建 tensor 后，对 NPU device 的 tensor 调用 `at_npu::native::npu_format_cast` 修正 format 为 `ACL_FORMAT_ND`。

#### 修改文件 1: `rtp_llm/cpp/cache/MemoryLayoutStrategy.cc`

**添加头文件：**
```cpp
#if USING_ASCEND
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <third_party/acl/inc/acl/acl_base.h>
#endif
```

**添加辅助函数：**
```cpp
#if USING_ASCEND
static void fix_npu_format(torch::Tensor& tensor) {
    if (tensor.is_privateuseone()) {
        auto fmt = at_npu::native::get_npu_format(tensor);
        if (fmt != ACL_FORMAT_ND) {
            tensor = at_npu::native::npu_format_cast(tensor, ACL_FORMAT_ND);
        }
    }
}
#endif
```

**在所有 `torch::from_blob` 调用后添加修正：**

```cpp
// processKVTensor 中 separate_kv_cache_ 分支
auto layer_tensor = torch::from_blob(
    layer_ptr, {block_num, stride}, kv_options);
#if USING_ASCEND
fix_npu_format(layer_tensor);
#endif
layer_kv_tensors_.push_back(layer_tensor);

// processKVTensor 中非 separate 分支
torch::Tensor kv_cache_typed = torch::from_blob(
    kv_cache_tensor.data_ptr(), {kv_typed_numel}, kv_options);
#if USING_ASCEND
fix_npu_format(kv_cache_typed);
#endif

// processScaleTensor 中所有 from_blob 同样处理
```

#### 修改文件 2: `rtp_llm/cpp/cache/BlockPool.cc`

**添加头文件：**
```cpp
#if USING_ASCEND
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <third_party/acl/inc/acl/acl_base.h>
#endif
```

**V cache from_blob 后添加修正：**
```cpp
auto v_view = torch::from_blob(v_ptr, k_layer_tensor.sizes(),
                                k_layer_tensor.strides(), v_options);
#if USING_ASCEND
if (v_view.is_privateuseone()) {
    auto fmt = at_npu::native::get_npu_format(v_view);
    if (fmt != ACL_FORMAT_ND) {
        v_view = at_npu::native::npu_format_cast(v_view, ACL_FORMAT_ND);
    }
}
#endif
global_layer_v_tensors_[global_layer] = v_view;
```

#### 修改文件 3: `rtp_llm/cpp/cache/BUILD`

**在 `block_pool` 的 deps 中添加 torch_npu_api 依赖：**
```python
deps = [
    ":cache_types",
    "//rtp_llm/models_py/bindings/core:exec_ops_hdr",
    ...
] + select({
    "@//:using_ascend": [
        "@torch_npu_ascend//:torch_npu_api",
    ],
    "//conditions:default": [],
}),
```

### 4.2 Python 端临时修复（备选，不推荐）

在 `ascend_kv_cache_write_op.py` 中调用 `_npu_reshape_and_cache` 前，
用 `torch_npu.npu_format_cast(kv_cache, 2)` 修正 format。

**缺点：**
- `npu_format_cast` 可能返回新的 tensor（copy），而非原位修改，导致写入不生效
- 异步 NPU 执行模式下错误捕获不可靠
- 每次推理都需 format cast，有性能开销

### 4.3 备选方案：使用 `at_npu::native::from_blob` 替代 `torch::from_blob`

torch_npu 提供了自己的 `from_blob` 实现（`at_npu::native::from_blob`），能正确设置 NPU format。
将所有 `torch::from_blob` 替换为 `at_npu::native::from_blob` 可从根本上避免 format 未初始化问题。

**缺点：** API 签名不同，需要适配 deleter 和 device 参数。

## 5. 环境信息

| 项目 | 值 |
|---|---|
| NPU | 910B3 × 8 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0 |
| CANN | 25.3.rc1 |
| 模型 | Qwen3-0.6B |
| rtp-llm 路径 | /home/w30067200/rtp-llm |
| kv_cache shape | [521088, 1024] (2D, bfloat16) |
| key/value shape | [num_tokens, 8, 128] (3D, bfloat16) |

## 6. 编译验证

```bash
# 编译 block_pool 库
bazel build //rtp_llm/cpp/cache:block_pool --config=ascend

# 编译完整 rtp_llm 包
bazel build //rtp_llm:rtp_llm_package --config=ascend
```

## 7. 其他已知问题

### 7.1 vipserver DNS 解析失败（可忽略）

```
failed to refresh vipserver server list: Failed to resolve 'jmenv.tbsite.net'
```

这是内网服务发现域名，在独立部署环境不可用，不影响推理功能。

### 7.2 `aclnnMean` 错误（关联问题）

在 format 修正后可能触发：
```
RuntimeError: The Inner error is reported as above. The current working operator name is aclnnMean.
```
这是异步 NPU 执行模式下，之前操作的错误延迟到后续算子才报出。设置 `ASCEND_LAUNCH_BLOCKING=1` 可获取准确 stacktrace。

### 7.3 agent client port 4141 连接失败（可忽略）

```
agent client failed, host[127.0.0.1] port[4141]
```
Thrift agent 连接失败，与 kv_cache format 问题无关。
