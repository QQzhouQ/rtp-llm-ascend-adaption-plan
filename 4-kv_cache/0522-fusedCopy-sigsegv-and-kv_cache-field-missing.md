# rtp-llm Ascend NPU 适配问题排查与修复总结

## 环境

- **平台**: Huawei Ascend NPU (aarch64)
- **CANN**: 8.5.1
- **PyTorch**: torch_npu 2.9.0 + torch 2.9.0+cpu
- **模型**: Qwen3-0.6B
- **编译**: Bazel `--config=ascend`
- **Docker容器**: rtp_llm

---

## 问题一：fusedCopy SIGSEGV 段错误

### 现象

服务启动后收到推理请求，backend_server 进程在 `fusedCopy()` 处崩溃：

```
SIGSEGV (@0x12c04122aa00) received by PID
rtp_llm::fusedCopy()
rtp_llm::PyWrappedModel::forward()
rtp_llm::NormalExecutor::process()
rtp_llm::NormalEngine::step()
```

### 根因

`FusedCopyOp.cc` 中没有 `USING_ASCEND` 编译分支。在 Ascend 编译时走的是 `#else` 的 `memcpy`，而 `FusedD2DCopyParams` 里的 `src/dst` 指针是 **NPU Device 内存地址**，CPU `memcpy` 访问 Device 内存必然段错误。

```cpp
// 修改前：Ascend编译走#else分支
void fusedCopy(const FusedD2DCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
    invokeFusedCopy(params, stream);
#else
    // ❌ Ascend也走这里，memcpy对Device内存访问必然SIGSEGV
    for (int i = 0; i < params.num_copies; ++i) {
        memcpy(params.dst[i], params.src[i], params.size[i]);
    }
#endif
}
```

### 修复

在 `FusedCopyOp.cc` 中添加 `#elif USING_ASCEND` 分支，使用 `aclrtMemcpyAsync` 进行 NPU Device-to-Device 异步拷贝。

**修改文件**: `rtp_llm/models_py/bindings/common/FusedCopyOp.cc`

```cpp
// 修改后
#if USING_ASCEND
#include <acl/acl.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <torch_npu/csrc/core/npu/NPUStream.h>
#pragma GCC diagnostic pop
#include "rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h"
#endif

void fusedCopy(const FusedD2DCopyParams& params) {
#if USING_CUDA
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    invokeFusedCopy(params, stream);
#elif USING_ROCM
    hipStream_t stream = at::hip::getCurrentHIPStream();
    invokeFusedCopy(params, stream);
#elif USING_ASCEND
    // ✅ 使用aclrtMemcpyAsync进行NPU D2D异步拷贝
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    for (int i = 0; i < params.num_copies; ++i) {
        ASCEND_CHECK(aclrtMemcpyAsync(params.dst[i], params.size[i],
                                       const_cast<void*>(params.src[i]), params.size[i],
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    }
#else
    for (int i = 0; i < params.num_copies; ++i) {
        memcpy(params.dst[i], params.src[i], params.size[i]);
    }
#endif
}

// fusedStridedCopy同理，逐行调用aclrtMemcpyAsync
```

**修改文件**: `rtp_llm/models_py/bindings/common/BUILD`

`fuse_copy_op` 的 `using_ascend` 依赖增加 `ascend_types_hdr`（提供 `ASCEND_CHECK` 宏）：

```python
"@//:using_ascend": [
    "@local_config_ascend//ascend:ascend_headers",
    "//rtp_llm/models_py/bindings/ascend:ascend_types_hdr",  # 新增
],
```

> **注意**: `#pragma GCC diagnostic push/pop` 包裹 `NPUStream.h` 是为了抑制 torch_npu 头文件中的 `-Wunused-function` 警告（项目开启了 `-Werror`）。

---

## 问题二：PyAttentionInputs 缺少 kv_cache 属性

### 现象

fusedCopy 修复后，backend 在 `PyWrappedModel::forward()` 中抛出 Python 异常：

```
AttributeError: 'librtp_compute_ops.PyAttentionInputs' object has no attribute 'kv_cache'

At:
  rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py(95): support
  rtp_llm/models_py/modules/factory/attention/attn_factory.py(162): get_fmha_impl
```

### 根因

`PyAttentionInputs` C++ 结构体中没有 `kv_cache` 成员，pybind11 绑定中也没有暴露该属性。但 Ascend 的 `AscendPrefillImpl.support()` 和 `AscendDecodeImpl.support()` 需要通过 `attn_inputs.kv_cache` 访问 KV cache 信息：

```python
# ascend_prefill.py:92-96
@staticmethod
def support(attn_configs, attn_inputs):
    return attn_inputs.is_prefill and \
           not attn_configs.use_mla and \
           attn_inputs.kv_cache is not None and \      # ❌ 属性不存在
           attn_inputs.kv_cache.separate_kv_cache
```

### 修复

三处修改：

**1. 修改文件**: `rtp_llm/models_py/bindings/OpDefs.h`

在 `PyAttentionInputs` 结构体中添加 `kv_cache` 成员：

```cpp
struct PyAttentionInputs {
    std::optional<KVCache> kv_cache;  // 新增
    bool          is_prefill{false};
    // ... 其余成员不变
};
```

**2. 修改文件**: `rtp_llm/models_py/bindings/OpDefs.cc`

在 pybind11 绑定中暴露 `kv_cache` 属性：

```cpp
pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
    .def(pybind11::init<>())
    .def_readwrite("kv_cache", &PyAttentionInputs::kv_cache)  // 新增
    .def_readwrite("is_prefill", &PyAttentionInputs::is_prefill)
    // ... 其余绑定不变
```

**3. 修改文件**: `rtp_llm/models_py/model_desc/module_base.py`

在 `prepare_fmha_impl` 调用 `AttnImplFactory.get_fmha_impl` 之前，将 `self.kv_cache` 设置到 `attn_inputs` 上：

```python
def prepare_fmha_impl(
    self, inputs: PyModelInputs, is_cuda_graph: bool = False
) -> Any:
    inputs.attention_inputs.kv_cache = self.kv_cache  # 新增
    fmha_impl = AttnImplFactory.get_fmha_impl(
        self.config,
        self.parallelism_config,
        self.weight,
        inputs.attention_inputs,
        self.fmha_config,
        is_cuda_graph,
    )
    return fmha_impl
```

---

## 问题三：AttentionConfigs 缺少 scale 属性

### 现象

kv_cache 修复后，`AscendPrefillImpl` 实例化失败：

```
Failed to instantiate AscendPrefillImpl: 'libth_transformer_config.AttentionConfigs' object has no attribute 'scale'
```

导致所有 FMHA 实现都不可用，最终抛出 `can not find mha type`。

### 根因

`AttentionConfigs` C++ 绑定中没有 `scale` 属性（有 `softmax_extra_scale` 和 `q_scaling`），但 Ascend 实现中访问了 `attn_configs.scale`：

```python
# ascend_prefill.py & ascend_decode.py
self.scale = attn_configs.scale if attn_configs.scale else \
             self.head_dim ** -0.5   # ❌ attn_configs没有scale属性
```

### 修复

**修改文件**:
- `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py`
- `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py`

使用 `getattr` 安全访问，若属性不存在则回退到 `head_dim ** -0.5`：

```python
# 修改后
self.scale = getattr(attn_configs, "scale", None) or \
             self.head_dim ** -0.5
```

---

## 问题四（待解决）：_npu_reshape_and_cache 算子段错误

### 现象

以上三个问题修复后，Ascend FMHA 实现被正确选中并执行，但在 KV cache 写入阶段崩溃：

```
SIGSEGV received by PID
atb::_npu_reshape_and_cache()
c10::impl::make_boxed_from_unboxed_functor<>::call()
c10::Dispatcher::callBoxed()
torch::jit::invokeOperatorFromPython()
```

dmesg 也有对应记录：

```
[ascend] [devmm] [ERROR] Va is error, can not fault.
(start=0x12c2bec00000, bitmap=0xa88)
```

### 可能原因

- 传入 `torch_npu._npu_reshape_and_cache` 的 k/v cache tensor shape/stride 与算子预期不匹配
- slot_mapping 索引越界
- KV cache buffer 内存未正确分配或对齐

### 排查方向

1. 在 `AscendKVCacheWriteOp.forward()` 中添加输入参数打印（k/v shape、slot_mapping range、kv_cache shape）
2. 对比 CUDA 路径下 `FusedRopeKVCacheOp` 的参数格式
3. 检查 `separate_kv_cache=True` 时 `k_cache_base` / `v_cache_base` 的 layout 是否符合 `_npu_reshape_and_cache` 预期

---

## 修改文件清单

| # | 文件路径 | 修改类型 | 解决问题 |
|---|---------|---------|---------|
| 1 | `rtp_llm/models_py/bindings/common/FusedCopyOp.cc` | C++ 源码 | fusedCopy SIGSEGV |
| 2 | `rtp_llm/models_py/bindings/common/BUILD` | Bazel 依赖 | 编译依赖缺失 |
| 3 | `rtp_llm/models_py/bindings/OpDefs.h` | C++ 头文件 | kv_cache 属性缺失 |
| 4 | `rtp_llm/models_py/bindings/OpDefs.cc` | pybind11 绑定 | kv_cache 属性缺失 |
| 5 | `rtp_llm/models_py/model_desc/module_base.py` | Python 源码 | kv_cache 未传入 |
| 6 | `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py` | Python 源码 | scale 属性不存在 |
| 7 | `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py` | Python 源码 | scale 属性不存在 |

## 修复验证

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| fusedCopy | SIGSEGV in `rtp_llm::fusedCopy()` | ✅ 不再出现 |
| kv_cache | `AttributeError: no attribute 'kv_cache'` | ✅ 不再出现 |
| scale | `AttentionConfigs has no attribute 'scale'` | ✅ 不再出现 |
| FMHA选择 | `can not find mha type` | ✅ AscendPrefillImpl/AscendDecodeImpl 被正确选中 |
| reshape_and_cache | 未触达 | ❌ `_npu_reshape_and_cache` 内部 SIGSEGV（待解决） |
