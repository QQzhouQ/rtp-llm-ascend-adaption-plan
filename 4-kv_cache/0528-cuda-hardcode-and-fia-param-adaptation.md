# rtp-llm Ascend NPU 适配：问题排查与代码修改记录

## 环境

- **硬件**: 5x Ascend 910B3 NPU，~64GB HBM 每个
- **软件**: PyTorch 2.9.0+cpu, torch_npu 2.9.0, CANN 8.5.1, ATB 8.5.1
- **模型**: Qwen3-0.6B (head_dim=128, num_heads=16, kv_heads=8, 28 layers, bfloat16)
- **命令**: `python -m rtp_llm.start_server --checkpoint_path=... --model_type=qwen_3 --start_port=9000`

---

## 一、全局性硬编码 CUDA 问题

### 1.1 问题

rtp-llm 代码中大量硬编码 `"cuda"` 设备字符串和 `torch.cuda` API 调用，Ascend NPU 环境下直接报错。

### 1.2 涉及文件

| 文件 | 问题 |
|------|------|
| `rtp_llm/models/base_model.py` | `_get_device_str()` 硬编码 `"cuda:"` |
| `rtp_llm/lora/lora_manager.py` | 硬编码 `f"cuda:{local_rank}"` |
| `rtp_llm/model_loader/weight_manager.py` | 使用 `torch.cuda.Stream`、`torch.cuda.stream` |
| `rtp_llm/model_loader/loader.py` | 仅调用 `torch.cuda.synchronize()` 清理显存 |
| `rtp_llm/models_py/distributed/symm_mem.py` | 硬编码 `f"cuda:{device}"` |
| `rtp_llm/models_py/distributed/user_buffers.py` | 硬编码 `f"cuda:{local_rank}"` |
| `rtp_llm/models_py/model_desc/disaggregate_qwen3.py` | 硬编码 `"cuda:" + str(...)` |
| `rtp_llm/start_backend_server.py` | `torch.cuda.device_count()` 无 NPU 回退 |
| `rtp_llm/tools/convert/weights_convert.py` | 硬编码 `f"cuda:{local_rank}"` |
| `rtp_llm/utils/database.py` | 硬编码 `f"cuda:{pg.rank()}"` |

### 1.3 解决方案

统一通过 `get_device_type()` 判断设备类型，动态选择 `"npu"` / `"hip"` / `"cuda"` 前缀：

```python
from rtp_llm.device.device_type import get_device_type, DeviceType
_dt = get_device_type()
_dev_name = "npu" if _dt == DeviceType.Ascend else ("hip" if _dt == DeviceType.ROCm else "cuda")
device_str = f"{_dev_name}:{local_rank}"
```

`weight_manager.py` 中 Stream 相关：
```python
_StreamCls = torch.npu.Stream if get_device_type() == DeviceType.Ascend else torch.cuda.Stream
```

`start_backend_server.py` 中设备计数：
```python
_dev_count = torch.npu.device_count() if (not torch.cuda.is_available()) else torch.cuda.device_count()
```

`loader.py` 中显存清理增加 NPU 分支：
```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
else:
    try:
        import torch_npu
        if torch.npu.is_available():
            torch.npu.synchronize()
            torch.npu.empty_cache()
    except ImportError:
        pass
```

---

## 二、C++ 端编译与运行时问题

### 2.1 FusedCopy 算子缺少 Ascend 实现

**问题**: `FusedCopyOp.cc` 和 fused kernel 仅支持 CUDA/ROCm，Ascend 构建时无实现。

**修改文件**: `rtp_llm/models_py/bindings/common/FusedCopyOp.cc`

**方案**: 使用 CANN `aclrtMemcpyAsync` API 实现 `fusedCopy` 和 `fusedStridedCopy`：

```cpp
#elif USING_ASCEND
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h"

void fusedCopy(const FusedD2DCopyParams& params) {
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    for (int i = 0; i < params.num_copies; ++i) {
        aclrtMemcpyKind kind = rtp_llm::ascend::getMemcpyKind(params.src[i], params.dst[i]);
        ASCEND_CHECK(aclrtMemcpyAsync(params.dst[i], params.size[i],
                                       const_cast<void*>(params.src[i]), params.size[i],
                                       kind, stream));
    }
}
```

同时在 `ascend_types_hdr.h` 中增加 `getMemcpyKind()` 辅助函数，通过 `aclrtPointerGetAttributes` 判断源/目标地址类型。

### 2.2 Sample/BeamSearch 算子缺少 Ascend 实现

**问题**: `CudaSampleOp.cc` 和 `CudaBeamSearchOp.cc` 的 Ascend 分支直接 `throw ERROR_UNIMPLEMENTED`。

**修改文件**:
- `rtp_llm/models_py/bindings/core/CudaSampleOp.cc`
- `rtp_llm/models_py/bindings/core/CudaBeamSearchOp.cc`

**方案**:

`sampleGreedy`: 纯 PyTorch CPU 回退实现，将 logits 搬到 CPU 上执行 temperature 缩放、top-k/top-p 过滤、multinomial 采样，再搬回 NPU。约 100 行 C++/PyTorch 代码。

`chainSpeculativeSampling`: 简化实现，直接接受 draft tokens。

`sampleBeamSearch`: 简化为 greedy 风格的 beam search。

### 2.3 KV Cache 内存布局处理（separate_kv_cache 路径）

**问题**: `MemoryLayout.cc` 的 `processKVTensor` / `processScaleTensor` 对 Ascend 分离 K/V cache 场景处理不正确，使用了错误的 stride 和 `torch::from_blob`（Ascend 需要用 `at_npu::native::from_blob` 注册 NPU pointer）。

**修改文件**: `rtp_llm/cpp/models/MemoryLayout.cc`

**方案**:
1. Ascend 平台使用 `at_npu::native::from_blob` 替代 `torch::from_blob`（通过宏 `CACHE_FROM_BLOB`）
2. `separate_kv_cache` 路径独立处理：按 layer 切分 cache，使用 `k_block_stride_bytes` 计算正确的 stride
3. Scale tensor 同样按 layer 独立切分

### 2.4 ComputeInit 注册缺失

**问题**: `registerExecCtxOps` 仅在 `USING_CUDA || USING_ROCM` 时编译注册，Ascend 缺少 `get_device_id` 等函数符号。

**修改文件**:
- `rtp_llm/cpp/pybind/ComputeInit.cc`: 扩展条件为 `USING_CUDA || USING_ROCM || USING_ASCEND`
- `rtp_llm/cpp/pybind/BUILD`: 添加 `//rtp_llm/models_py/bindings/core:exec_ctx_ops` 依赖
- `rtp_llm/ops/compute_ops.py`: Python 层注入 stub（`get_device_id`、`preprocess_gemm_weight_by_key` 等）

### 2.5 AscendRegister.cc 空实现

**问题**: 原 `AscendRegister.cc` 为空文件，未提供 `registerPyModuleOps` 函数。

**修改**: 提供空函数体：
```cpp
namespace rtp_llm {
void registerPyModuleOps(pybind11::module& m) {}
}
```

### 2.6 GenerateStream 预处理器条件

**问题**: `#if defined(USING_CUDA)` 在 Ascend 构建中不匹配。

**修改**: `GenerateStream.cc` 中改为 `#if USING_CUDA || USING_ROCM`。

---

## 三、Python 端 Attention 核心问题（重点）

### 3.1 Decode 路径 `context_lens` 错误

**问题**: `ascend_decode.py` 的 `AscendDecodeImpl.forward()` 中：
```python
self.fmha_impl.context_lens = self.attn_inputs.prefix_lengths + self.attn_inputs.input_lengths
```
在纯 decode（非 prefix）批次中，`prefix_lengths` 和 `input_lengths` 可能为空或全零，导致 `_npu_paged_attention` 收到错误的 context_lens。

**修改**（`ascend_decode.py:87`）:
```python
self.fmha_impl.context_lens = self.attn_inputs.sequence_lengths
```

### 3.2 `block_table` 和 `context_lens` 设备问题

**问题**: ATB 的 `_npu_paged_attention` Setup 阶段需要读取 `block_table` 和 `context_lens` 的 host 数据（`tensor.hostData is null` 错误）。原代码可能将它们搬到 NPU 上。

**方案**: `AscendDecodeAttnOp.forward()` 中不再对 `block_table` / `context_lens` 调用 `.to(q.device)`，保持它们在 CPU 上（C++ 端创建的 pinned tensor）。

### 3.3 `block_table` 使用 kernel block table

**问题**: 原代码使用 `kv_cache_block_id_host`（物理 block table），但 ATB 的 paged attention 需要逻辑/内核 block table。

**修改**（`ascend_decode.py:119`）:
```python
self.block_table = attn_inputs.kv_cache_kernel_block_id_host  # 而非 kv_cache_block_id_host
```

### 3.4 KV Cache 布局（未解决的核心问题）

**问题**: 这是当前最大的阻塞点。ATB 的 `_npu_paged_attention` 对 KV cache 布局有严格约束：

- **BSND** `[blocks, block_size, kv_heads, head_dim]`：Setup 校验通过（`dim2=kv_heads` 满足 `kvHead == dim2`），但实际 kernel 执行时报 **MPU address access is invalid**（aicore 错误）
- **BNSD** `[blocks, kv_heads, block_size, head_dim]`：Setup 报 **`param kvHead must equal key_cache dim2`**（`dim2=block_size != kvHead`）

ATB 内部日志显示，对于 BSND 输入 `[2,8,2,32]`，内核将 `headSize` 解释为 `dim2=2`（即 kv_heads），而非实际的 head_dim=32。这表明 ATB 内部对维度的解释与文档描述的 BSND 不一致。

此外，测试发现 **`block_size >= 16` 时 BSND 必定 MPU 崩溃**，`block_size <= 15` 可以正常运行（但结果正确性存疑）。

**当前状态**: `ascend_decode.py` 中暂时使用 BSND（无 permute），但 decode 执行仍会 aicore 错误。

**可能的方向**:
1. 确认 ATB/CANN 版本对 `_npu_paged_attention` 的实际 cache 布局要求
2. 考虑使用 `npu_fused_infer_attention_score`（prefill 用的同一个 op）来替代 `_npu_paged_attention` 做 decode
3. 确认 `_npu_reshape_and_cache` 的写入布局与读取布局是否兼容
4. 检查是否需要 NZ (NzFormat) 格式而非 ND 格式

### 3.5 Prefill 路径

**修改**: `ascend_prefill.py` 中：
1. KV cache permute 为 BNSD：`kv_cache.k_cache_base.permute(0, 2, 1, 3).contiguous()`
2. `actual_seq_q/actual_seq_kv` 从 `torch.cat([zeros, cumsum])` 改为直接 `torch.cumsum`（无前导零）
3. `block_table` 使用 `kv_cache_kernel_block_id_host`
4. 增加 causal mask 生成
5. `block_table`、`actual_seq_q`、`actual_seq_kv` 按需搬到 NPU（`npu_fused_infer_attention_score` 要求同设备）
6. 添加调试日志

### 3.6 scale 计算

**问题**: 原代码 `self.scale = attn_configs.scale if attn_configs.scale else self.head_dim ** -0.5`，`attn_configs.scale` 可能为 0 而非 None。

**修改**: `ascend_decode.py` 和 `ascend_prefill.py` 统一改为：
```python
self.scale = attn_configs.q_scaling * self.head_dim ** -0.5
```

---

## 四、PyAttentionInputs 缺少 kv_cache 字段

**问题**: Python 层 `prepare_fmha_impl` 调用时，`attn_inputs.kv_cache` 为 None，导致 Ascend attention 实现无法获取 cache 信息。

**修改**:
- `OpDefs.h`: 添加 `std::optional<KVCache> kv_cache` 字段
- `OpDefs.cc`: 暴露 `.def_readwrite("kv_cache", ...)`
- `module_base.py`: 在 `prepare_fmha_impl` 中设置 `inputs.attention_inputs.kv_cache = self.kv_cache`

---

## 五、OpDefs.h 中 KV cache reshape

**问题**: `KVCache::initialize` 对 separate_kv_cache 路径的 reshape 格式化不良，且缺少 block_table 的 ndim 校验。

**修改**: 格式化 reshape 参数，确保 `k_cache_base` 和 `v_cache_base` reshape 为 `[kernel_block_num, kernel_seq_size_per_block, num_kv_heads, head_dim]`。

---

## 六、其他修改

### 6.1 `ascend_attn_params.py`

- `build_ascend_params`: 添加 `block_table` ndim 校验（reshape 为 2D）
- `compute_ascend_attn_params`: 添加 block_table reshape 和调试日志

### 6.2 `__init__.py` (modules/base)

- 所有平台统一导出 `LayerNorm`（从对应平台的 norm 模块导入）

### 6.3 `attn_factory.py`

- 添加 debug 日志打印 FMHA impl 的 support 检查结果

### 6.4 `ExecOps.cc`

- Ascend 运行时初始化中，MLA ops type 默认设为 `FLASH_INFER`

### 6.5 `BUILD` 文件

- `common/BUILD`: Ascend 条件添加 `ascend_types_hdr` 依赖
- `pybind/BUILD`: Ascend 条件添加 `exec_ctx_ops` 依赖

---

## 七、当前阻塞问题总结

| 问题 | 状态 | 说明 |
|------|------|------|
| `_npu_paged_attention` KV cache 布局 | **未解决** | BSND Setup 通过但执行 MPU 错误，BNSD Setup 不通过 |
| `_npu_paged_attention` block_size >= 16 | **未解决** | 小 block_size 可运行，但实际模型需要 block_size=64 |
| prefill 路径是否正常 | **未验证** | 需要先解决 decode 才能完整测试 |

---

## 八、关键文件清单

| 文件 | 修改类型 |
|------|----------|
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py` | decode attention 核心修改 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py` | prefill attention 核心修改 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py` | 参数计算修改 |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py` | KV cache 写入（未改） |
| `rtp_llm/models_py/bindings/OpDefs.h` / `OpDefs.cc` | KV cache 字段暴露 |
| `rtp_llm/models_py/model_desc/module_base.py` | kv_cache 传递 |
| `rtp_llm/cpp/models/MemoryLayout.cc` | KV cache 内存布局 |
| `rtp_llm/models_py/bindings/common/FusedCopyOp.cc` | Ascend memcpy 实现 |
| `rtp_llm/models_py/bindings/core/CudaSampleOp.cc` | Ascend sample 回退 |
| `rtp_llm/models_py/bindings/core/CudaBeamSearchOp.cc` | Ascend beam search 回退 |
| `rtp_llm/models_py/bindings/core/ExecOps.cc` | 运行时初始化 |
| `rtp_llm/cpp/pybind/ComputeInit.cc` | 算子注册条件 |
| `rtp_llm/ops/compute_ops.py` | Python stub 注入 |
| `rtp_llm/models/base_model.py` | 设备字符串 |
| `rtp_llm/start_backend_server.py` | NPU 设备计数 |
| `rtp_llm/model_loader/weight_manager.py` | Stream 适配 |
| `rtp_llm/model_loader/loader.py` | 显存清理 |
