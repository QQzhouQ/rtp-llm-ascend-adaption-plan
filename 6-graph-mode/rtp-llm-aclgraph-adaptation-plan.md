# rtp-llm ACL Graph 适配方案

## 1. 概述

### 1.1 目标

将 rtp-llm 现有的 CUDA Graph 模式（decode 阶段 kernel launch 加速）适配到华为 Ascend NPU 的 ACL Graph（`c10_npu::NPUGraph`），使 Ascend NPU 上的 decode 推理也能受益于图模式加速。

### 1.2 参考实现

| 实现 | 位置 | 核心思路 |
|------|------|---------|
| rtp-llm CUDA Graph | `rtp_llm/cpp/cuda_graph/` | `GraphBase` → `CudaGraphRunner`，per-instance pinned memory buffer，custom copy kernels |
| xLLM ACL Graph | `xllm/core/runtime/acl_graph_executor_impl.h/.cpp` | `ExecutorImpl` → `AclGraphExecutorImpl`，shared `GraphPersistentParam`，ATB tiling |

### 1.3 核心架构差异

| 维度 | rtp-llm CUDA Graph | xLLM ACL Graph | 适配策略 |
|------|-------------------|----------------|---------|
| 基类 | `GraphBase` → `CudaGraphRunner` | `ExecutorImpl` → `AclGraphExecutorImpl` | 新增 `AscendGraphRunner` 继承 `GraphBase` |
| Buffer 管理 | 每个 graph instance 独立 `CaptureMemoryHold`（global max buffer slice） | 共享 `GraphPersistentParam`，所有 graph 用同一组 device buffer | 采用 xLLM 的共享 persistent buffer 模式 |
| 数据拷贝 | custom CUDA copy kernel（Small2Large/Large2Small） | `aclrtMemcpyAsync` 直拷 + `torch::Tensor::copy_` | 使用 `aclrtMemcpyAsync` + tensor copy |
| Attention 集成 | Python 侧 `prepare_cuda_graph()` | C++ ATB `Setup` + H2D tiling | 引入 ATB tiling 到 C++ 层，保留 Python `prepare_cuda_graph()` 兼容 |
| 图 API | `at::cuda::CUDAGraph` | `c10_npu::NPUGraph` | 新建 `ascend_graph_device_shims.h`，不修改原始 CUDA shim |
| Bucket 维度 | batch_size（decode）/ seq_len（prefill） | num_tokens | batch_size 维度，与 rtp-llm decode 语义完全对齐 |

---

## 2. 整体架构

### 2.1 新增文件清单

> **关于 device shim 的策略说明**: Ascend 的图捕获 API（`c10_npu::NPUGraph::capture_begin`）与 CUDA 的 `at::cuda::CUDAGraph::capture_begin` 签名和语义差异较大（如 `graphCaptureBegin` 无法通过简单的 `#if` 分支实现统一），因此 **不修改原始 `cuda_graph_device_shims.h/.cc`**，而是新建独立的 `ascend_graph_device_shims.h/.cc`。

```
rtp_llm/cpp/ascend_graph/                    # [新增] Ascend Graph 核心目录
├── BUILD                                    # Bazel 构建
├── ascend_graph_runner.h                    # AscendGraphRunner 声明
├── ascend_graph_runner.cc                   # AscendGraphRunner 实现
├── ascend_graph_persistent_param.h          # GraphPersistentParam（基于 xLLM）
├── ascend_graph_persistent_param.cc         # 持久化参数实现
├── ascend_graph_device_shims.h              # [新增] Ascend 专用 device shim
├── ascend_graph_device_shims.cc             # [新增] Ascend 专用 device shim 实现
├── ascend_graph_utils.h                     # 辅助类（AscendGraphInstance, CaptureGuard）
└── ascend_graph_capture.cc                  # 捕获 + 重放逻辑

rtp_llm/cpp/ascend/                          # [已有] Ascend 基础实现目录
└── ascend_shims.h                           # aclrtStream/aclrtEvent 基础宏映射（Phase 0 产物）

rtp_llm/models_py/modules/factory/attention/ascend_impl/
├── __init__.py
├── ascend_prefill.py                        # Ascend Prefill Attention
└── ascend_decode.py                         # Ascend Decode Attention（含 prepare_cuda_graph）
```

### 2.2 类设计

```
GraphBase (cuda_graph_base.h, 不变)
  ▲
  ├── CudaGraphRunner (已有, 不变)
  │
  └── AscendGraphRunner (新增)
        ├── 配置字段（与 CudaGraphRunner 对齐）:
        │     enable_graph_, enable_graph_debug_mode_, max_bs_
        │     max_seq_len_, hidden_size_, seq_size_per_block_, kernel_seq_size_per_block_
        │     model_data_type_, num_tokens_per_bs_, sp_steps_
        │     is_prefill_cuda_graph_mode_, is_target_verify_
        │     decode_capture_batch_sizes_, capture_range_
        │     kv_cache_layer_to_group_, kv_cache_group_num_
        │
        ├── 运行时组件:
        │     std::unique_ptr<GraphPersistentParam> persistent_param_   // 共享持久化 buffer
        │     std::unordered_map<int, AscendGraphInstance> graph_instances_  // bucket→graph
        │     std::optional<c10_npu::NPUStream> capture_stream_         // 捕获专用 stream
        │     torch::Event forward_event_                               // 完成事件
        │     py::object py_forward_method_, py_attn_pyobj_method_     // Python 方法
        │
        ├── initCapture()
        ├── canRun(inputs, state) -> bool
        ├── forward(inputs, state) -> PyModelOutputs
        ├── prepareInputs(inputs, state)        // 用 aclrtMemcpyAsync 更新 persistent buffer
        ├── captureOneGraphInstance(key, type)  // NPUGraph::capture_begin/end
        └── replayGraph(key)                    // NPUGraph::replay()

GraphPersistentParam (新增, 参考 xLLM + 补齐 rtp-llm 特有字段)
        ├── 持久化 device tensor（DECODE 阶段完整字段）:
        │     persistent_tokens_              [max_tokens]         int32
        │     persistent_positions_           [max_tokens]         int32
        │     persistent_input_hiddens_       [max_tokens, hidden] model_dtype  (spec decode 用)
        │     persistent_block_tables_        [max_batch, max_blocks] int32
        │     persistent_block_tables_by_group_[kv_cache_group_num]   (hybrid cache)
        │     persistent_input_lengths_       [max_batch]          int32
        │     persistent_sequence_lengths_    [max_batch]          int32
        │     persistent_sequence_lengths_plus_1_d_ [max_batch]    int32
        │     persistent_prefix_lengths_      [max_batch]          int32
        │     persistent_decode_cu_seqlens_d_ [max_batch + 1]      int32
        │     persistent_context_total_kv_length_                    int32
        │     persistent_kv_cache_layer_to_group_ [max_layers]      int32  (hybrid cache)
        │     persistent_mask_                [max_tokens, max_seq_len] float
        │     hidden_states_                  [max_tokens, hidden]  model_dtype
        │     tiling_data_                    [1024*256]            int32 (ATB tiling)
        │
        ├── update(inputs, batch_size, return_capture_params)
        │       → 完整 PyModelInputs → persistent tensor 映射拷贝 + 清空 + padding
        └── plan_paged_attention_tiling(...)  // ATB Setup + H2D (xLLM 模式)

AscendGraphInstance (新增)
        ├── c10_npu::NPUGraph graph_          // ACL 图句柄
        ├── py::object attn_pyobj_            // 该 instance 的 attention Python 对象
        └── torch::Tensor output_hidden_states_  // 输出 buffer slice (指向 persistent_param_.hidden_states_)
```

### 2.3 `PyModelInputs` → `GraphPersistentParam` 字段映射表

这是正确实现 `update()` 的核心。下表列出了 rtp-llm decor decode 阶段所有需要映射的字段：

| 序号 | PyModelInputs 字段 | Persistent Tensor | 形状 | 拷贝方向 | 备注 |
|------|--------------------|--------------------|------|---------|------|
| 1 | `input_ids` | `persistent_tokens_` | `[max_tokens]` int32 | D2D copy | decode 时 input_ids.size(0) == batch_size |
| 2 | `input_hiddens` | `persistent_input_hiddens_` | `[max_tokens, hidden]` | D2D copy | speculative decoding 时有效 |
| 3 | `attn.input_lengths` | `persistent_input_lengths_` | `[max_batch]` int32 | H2D copy | CPU（pinned）→ NPU |
| 4 | `attn.sequence_lengths` | `persistent_sequence_lengths_` | `[max_batch]` int32 | H2D copy | decode 特有 |
| 5 | `attn.sequence_lengths_plus_1_d` | `persistent_sequence_lengths_plus_1_d_` | `[max_batch]` int32 | D2D copy | NPU→NPU |
| 6 | `attn.decode_cu_seqlens_d` | `persistent_decode_cu_seqlens_d_` | `[max_batch+1]` int32 | D2D copy | NPU→NPU，[0,1,2,...,bs] |
| 7 | `attn.prefix_lengths` | `persistent_prefix_lengths_` | `[max_batch]` int32 | H2D copy | CPU→NPU |
| 8 | `attn.kv_cache_kernel_block_id_device` | `persistent_block_tables_` | `[max_batch, max_blocks]` int32 | D2D copy | 主 block table |
| 9 | `attn.kv_cache_block_id_device` | 同上（如有）| `[max_batch, max_blocks]` int32 | D2D copy | 备选 block table |
| 10 | `attn.kv_cache_kernel_block_id_device_by_group[g]` | `persistent_block_tables_by_group_[g]` | `[max_batch, max_blocks]` int32 | D2D copy | hybrid cache per-group |
| 11 | `attn.kv_cache_layer_to_group` | `persistent_kv_cache_layer_to_group_` | `[max_layers]` int32 | H2D copy | layer→group 映射表 |
| 12 | `attn.context_total_kv_length` | `persistent_context_total_kv_length_` | scalar tensor | copy | 总 KV 长度 |
| 13 | `attn.dtype` | (直接赋值) | — | — | 不做拷贝，capture 时写入 |
| 14 | `attn.is_prefill` | (直接赋值) | — | — | 常量 false（decode-only） |
| 15 | `attn.is_target_verify` | (直接赋值) | — | — | 常量（spec decode 场景） |

**关于 `copySmallerIntoLarger` 的逻辑**: rtp-llm 的 `copySmallerIntoLarger` 会先 slice target 到 source 的维度，然后 `copy_`。Ascend 版本直接使用相同的 pattern：`persistent_xxx_.slice(0, 0, actual).copy_(inputs.xxx, true)`。

### 2.4 集成架构图

```
PyWrappedModel
  │
  ├── enable_cuda_graph_ && graph_runner_->canRun()
  │     └── graph_runner_->forward()      // GraphBase* 多态
  │           ├── AscendGraphRunner::prepareInputs()
  │           │     ├── 清零 persistent block_tables（防 stale data）            [NEW]
  │           │     ├── GraphPersistentParam::update()
  │           │     │     ├── 按字段映射表逐个 copy（H2D / D2D）                  [NEW: 完整 15 字段]
  │           │     │     ├── padding 区域清零（> actual_batch 部分）             [NEW]
  │           │     │     ├── ATB Setup → tiling_data H2D
  │           │     │     └── aclrtSynchronizeStream()
  │           │     └── attn_pyobj.attr("prepare_cuda_graph")(attn_inputs)     [NEW]
  │           │
  │           ├── AscendGraphRunner::replayGraph()
  │           │     └── graph_.replay()
  │           │
  │           └── persistent_param_->hidden_states(actual).clone() 输出          [UPDATED]
  │
  └── else: 普通 Python forward (fallback)
```

### 2.5 输出路径的地址保证

**关键问题**: NPUGraph replay 后，`model->forward()` 的 output tensor 地址是否仍指向 persistent buffer？

- **xLLM 模式**: 在 capture 时 `set_hidden_states(result)` 把结果 `copy_` 到 `persistent_param_.hidden_states_`，replay 时 output 由 NPUGraph mempool 分配，**但不保证地址不变**。因此 xLLM 在 capture 和 replay 时都从 `persistent_param_.hidden_states_` 读取。
- **rtp-llm CUDA Graph 模式**: capture 时 `decoder_layer_hidden_states_.copy_(outputs.hidden_states)`，replay 时直接 slice `decoder_layer_hidden_states_`（因为 CUDA Graph 保证 replay 时写入同一地址）。

**建议方案**: **保守策略**——capture 时 `copy_` 到 persistent buffer。replay 后也从 persistent buffer 读取。

```cpp
// capture 时：
auto outputs = py_forward_method_(...).cast<PyModelOutputs>();
persistent_param_->set_hidden_states(outputs.hidden_states);  // copy_ 到固定地址

// replay 后：
auto result = persistent_param_->hidden_states(actual_tokens);  // 从固定地址读取
```

如果后续验证确认 NPUGraph replay 的 output tensor 地址确实稳定，可优化为直接读取 replay 输出（省一次 copy）。

---

## 3. 分阶段实施

### Phase 1: 设备抽象层（新建文件，2-3天）

**目标**: 创建 `ascend_graph_device_shims.h/.cc`，提供 Ascend NPU 上的 stream/event/memcpy/graph 等基础抽象。**不修改**原始的 `cuda_graph_device_shims.h/.cc`，保持 CUDA/ROCm 代码隔离。

#### 3.1 `ascend_graph_device_shims.h`

```cpp
// ascend_graph_device_shims.h — Ascend NPU 专用 shim，不修改原始 cuda_graph_device_shims.h
#pragma once

#include <acl/acl.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>

#include "torch_npu/csrc/core/npu/NPUGraph.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

#define GRAPH_DEVICE_TYPE c10::DeviceType::PrivateUse1

// 类型定义（独立命名空间 ascend_graph，不混入 cuda_graph 命名空间）
namespace rtp_llm {
namespace ascend_graph {

using GraphStream = c10_npu::NPUStream;
struct GraphPoolHandle {};  // Ascend: NPUGraph 使用 {0,0} 自动创建 mempool

inline GraphStream graphGetStreamFromPool(bool is_high_priority) {
    return c10_npu::getStreamFromPool(is_high_priority, c10_npu::current_device());
}

inline GraphStream graphGetCurrentStream() {
    return c10_npu::getCurrentNPUStream(c10_npu::current_device());
}

inline void graphSetCurrentStream(GraphStream stream) {
    c10_npu::setCurrentNPUStream(stream);
}

inline torch::Event makeGraphEvent() {
    return torch::Event(GRAPH_DEVICE_TYPE);
}

void graphMemcpyAsync(void* dst, const void* src, size_t size,
                      GraphMemcpyKind kind, void* stream);
void graphDeviceSynchronize();
void graphMemGetInfo(size_t* free_bytes, size_t* total_bytes);

// Ascend 专用: NPUGraph capture wrapper
void npuGraphCaptureBegin(c10_npu::NPUGraph& graph,
                          aclmdlRICaptureMode mode);

}  // namespace ascend_graph
}  // namespace rtp_llm
```

#### 3.2 `ascend_graph_device_shims.cc`

```cpp
// ascend_graph_device_shims.cc
#include "rtp_llm/cpp/ascend_graph/ascend_graph_device_shims.h"

namespace rtp_llm {
namespace ascend_graph {

void graphMemcpyAsync(void* dst, const void* src, size_t size,
                      GraphMemcpyKind kind, void* stream) {
    aclrtMemcpyKind acl_kind;
    switch (kind) {
        case GraphMemcpyKind::D2D: acl_kind = ACL_MEMCPY_DEVICE_TO_DEVICE; break;
        case GraphMemcpyKind::D2H: acl_kind = ACL_MEMCPY_DEVICE_TO_HOST; break;
        case GraphMemcpyKind::H2D: acl_kind = ACL_MEMCPY_HOST_TO_DEVICE; break;
        default: throw std::runtime_error("Unsupported memcpy kind");
    }
    aclrtMemcpyAsync(dst, size, src, size, acl_kind, static_cast<aclrtStream>(stream));
}

void graphDeviceSynchronize() {
    aclrtSynchronizeDevice();
}

void graphMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    size_t free_hbm, total_hbm;
    aclrtGetMemInfo(ACL_HBM_MEM, &free_hbm, &total_hbm);
    if (free_bytes) *free_bytes = free_hbm;
    if (total_bytes) *total_bytes = total_hbm;
}

void npuGraphCaptureBegin(c10_npu::NPUGraph& graph, aclmdlRICaptureMode mode) {
    graph.capture_begin({0, 0}, mode);
}

}  // namespace ascend_graph
}  // namespace rtp_llm
```

#### 3.3 `GraphPoolHandle` 处理

Ascend 的 `c10_npu::NPUGraph::capture_begin()` 使用 `{0,0}` 作为 mempool id（自动创建新 pool），不支持 CUDA 式的显式 mempool 共享。因此 `GraphPoolHandle` 在 Ascend 上为空结构体。

#### 3.4 构建系统修改

**`rtp_llm/cpp/cuda_graph/BUILD`**:

```python
cc_library(
    name = "cuda_graph_impl",
    ...
    deps = torch_deps() + [
        ...
    ] + select({
        "//:using_cuda": [...],
        "@//:using_rocm": [...],
        "@//:using_ascend": [      # [新增]
            "@local_config_ascend//:ascend_headers",
            "@local_config_ascend//:ascend_libs",
        ],
        "//conditions:default": [],
    }),
)
```

**全局 Bazel 配置**:

```python
# def.bzl 新增
def if_ascend():
    return select({
        "//:using_ascend": True,
        "//conditions:default": False,
    })
```

---

### Phase 2: GraphPersistentParam 实现（4-5天）

**目标**: 实现 `GraphPersistentParam`，统一管理所有持久化 device buffer，包含完整的字段映射、block table 清零、padding 处理。

#### 2.1 完整类声明

```cpp
// ascend_graph_persistent_param.h
class GraphPersistentParam {
public:
    GraphPersistentParam(const GraphParams& params, torch::Device device);

    // 更新所有持久化 buffer：拷贝输入数据 + 清零 padding + 计算 tiling
    // batch_size: 当前实际 batch_size（<= max_batch）
    // return_capture_params: capture 阶段返回引用 persistent tensor 的 PyModelInputs
    std::optional<PyModelInputs> update(
        const PyModelInputs& inputs,
        int batch_size,
        bool return_capture_params = false);

    // Getter
    torch::Tensor persistent_tokens(uint32_t actual = 0) const;
    torch::Tensor persistent_positions(uint32_t actual = 0) const;
    torch::Tensor persistent_input_hiddens(uint32_t actual = 0) const;
    torch::Tensor persistent_block_tables(uint32_t actual_batch = 0) const;
    torch::Tensor persistent_block_tables_by_group(int group, uint32_t actual_batch = 0) const;
    torch::Tensor persistent_mask(uint32_t actual = 0) const;
    const torch::Tensor& tiling_data() const { return tiling_data_; }
    torch::Tensor hidden_states(uint32_t actual = 0) const;
    void set_hidden_states(const torch::Tensor& value);

    // decode 特有 getter
    torch::Tensor persistent_sequence_lengths(uint32_t actual = 0) const;
    torch::Tensor persistent_decode_cu_seqlens_d(uint32_t actual = 0) const;
    torch::Tensor persistent_context_total_kv_length() const;

private:
    // ============ 输入持久化 tensors ============
    torch::Tensor persistent_tokens_;              // [max_tokens] int32
    torch::Tensor persistent_positions_;           // [max_tokens] int32
    torch::Tensor persistent_input_hiddens_;       // [max_tokens, hidden] model_dtype
    torch::Tensor persistent_block_tables_;        // [max_batch, max_blocks] int32
    std::vector<torch::Tensor> persistent_block_tables_by_group_;  // per-group block tables
    torch::Tensor persistent_input_lengths_;       // [max_batch] int32
    torch::Tensor persistent_sequence_lengths_;    // [max_batch] int32
    torch::Tensor persistent_sequence_lengths_plus_1_d_;  // [max_batch] int32
    torch::Tensor persistent_prefix_lengths_;      // [max_batch] int32
    torch::Tensor persistent_decode_cu_seqlens_d_; // [max_batch + 1] int32  [0,1,2,...,bs]
    torch::Tensor persistent_context_total_kv_length_;            // int32 scalar
    torch::Tensor persistent_kv_cache_layer_to_group_;  // [max_layers] int32
    torch::Tensor persistent_mask_;                  // [max_tokens, max_seq_len] float

    // ============ 输出持久化 tensors ============
    torch::Tensor hidden_states_;                    // [max_tokens, hidden] model_dtype

    // ============ ATB tiling ============
    torch::Tensor tiling_data_;                      // [1024*256] int32
    atb::Context* context_for_plan_;
    atb::Operation* custom_pa_op_for_plan_;
    aclrtStream stream_for_plan_;

    bool need_update_attention_plan_;
    bool need_update_attn_mask_;

    // 配置
    int max_batch_;
    int max_tokens_;
    int max_blocks_;
    int kv_cache_group_num_;
    size_t hidden_size_;
    c10::ScalarType model_data_type_;
};
```

#### 2.2 update() 完整逻辑（含清零 + padding）

```cpp
std::optional<PyModelInputs> GraphPersistentParam::update(
    const PyModelInputs& inputs,
    int batch_size,
    bool return_capture_params) {

    const int num_tokens = inputs.input_ids.size(0);
    const auto& attn = inputs.attention_inputs;

    // ============ 1. 清零所有 persistent block_tables（防 stale KV cache block ID）============
    persistent_block_tables_.fill_(0);
    for (auto& tbl : persistent_block_tables_by_group_) {
        tbl.fill_(0);
    }

    // ============ 2. 拷贝输入数据（按 2.3 字段映射表逐字段拷贝）============

    // input_ids: D2D copy
    persistent_tokens_.slice(0, 0, num_tokens).copy_(inputs.input_ids, true);

    // input_hiddens (speculative decoding)
    if (inputs.input_hiddens.defined() && inputs.input_hiddens.numel() > 0) {
        persistent_input_hiddens_.slice(0, 0, num_tokens).copy_(inputs.input_hiddens, true);
    }

    // positions (若有，拷贝到 persistent_positions_)
    if (inputs.attention_inputs.position_ids.defined()) {
        persistent_positions_.slice(0, 0, num_tokens)
            .copy_(inputs.attention_inputs.position_ids, true);
    }

    // input_lengths: H2D copy (CPU pinned → NPU)
    persistent_input_lengths_.slice(0, 0, batch_size)
        .copy_(attn.input_lengths.slice(0, 0, batch_size), true);

    // sequence_lengths: H2D copy
    persistent_sequence_lengths_.slice(0, 0, batch_size)
        .copy_(attn.sequence_lengths.slice(0, 0, batch_size), true);

    // sequence_lengths_plus_1_d: D2D copy
    if (attn.sequence_lengths_plus_1_d.defined()) {
        persistent_sequence_lengths_plus_1_d_.slice(0, 0, batch_size)
            .copy_(attn.sequence_lengths_plus_1_d.slice(0, 0, batch_size), true);
    }

    // decode_cu_seqlens_d: D2D copy [0, 1, 2, ..., batch_size]
    persistent_decode_cu_seqlens_d_.slice(0, 0, batch_size + 1)
        .copy_(attn.decode_cu_seqlens_d.slice(0, 0, batch_size + 1), true);

    // prefix_lengths: H2D copy
    if (attn.prefix_lengths.defined()) {
        persistent_prefix_lengths_.slice(0, 0, batch_size)
            .copy_(attn.prefix_lengths.slice(0, 0, batch_size), true);
    }

    // context_total_kv_length
    if (attn.context_total_kv_length.defined()) {
        persistent_context_total_kv_length_.copy_(attn.context_total_kv_length, true);
    }

    // block_table: D2D copy（使用 copySmallerIntoLarger 等价逻辑）
    copySmallerIntoLarger(attn.kv_cache_kernel_block_id_device,
                          persistent_block_tables_, batch_size);

    // kv_cache_block_id_device (备选 block table)
    if (attn.kv_cache_block_id_device.defined()) {
        copySmallerIntoLarger(attn.kv_cache_block_id_device,
                              persistent_block_tables_alt_, batch_size);
    }

    // hybrid cache: per-group block tables
    if (!attn.kv_cache_kernel_block_id_device_by_group.empty()) {
        for (size_t g = 0; g < attn.kv_cache_kernel_block_id_device_by_group.size(); ++g) {
            copySmallerIntoLarger(attn.kv_cache_kernel_block_id_device_by_group[g],
                                  persistent_block_tables_by_group_[g], batch_size);
        }
    }

    // kv_cache_layer_to_group
    if (persistent_kv_cache_layer_to_group_.defined()
        && attn.kv_cache_layer_to_group.defined()) {
        persistent_kv_cache_layer_to_group_.copy_(attn.kv_cache_layer_to_group, true);
    }

    // ============ 3. Padding 区域清零（batch_size < max_batch 的额外 slots）============
    if (batch_size < max_batch_) {
        // block_tables padding 行清零
        persistent_block_tables_.slice(0, batch_size, max_batch_).fill_(0);
        for (auto& tbl : persistent_block_tables_by_group_) {
            tbl.slice(0, batch_size, max_batch_).fill_(0);
        }
        // input_lengths padding 清零
        persistent_input_lengths_.slice(0, batch_size, max_batch_).fill_(0);
        // sequence_lengths padding 清零
        persistent_sequence_lengths_.slice(0, batch_size, max_batch_).fill_(0);
        // prefix_lengths padding 清零
        persistent_prefix_lengths_.slice(0, batch_size, max_batch_).fill_(0);
    }

    // ============ 4. ATB tiling plan ============
    if (need_update_attention_plan_) {
        plan_paged_attention_tiling(inputs, stream_for_plan_);
    }

    // ============ 5. 同步 H2D 拷贝 ============
    aclrtSynchronizeStream(stream_for_plan_);

    // ============ 6. 返回 capture 参数（引用 persistent tensor）============
    if (return_capture_params) {
        PyModelInputs capture_inputs;
        capture_inputs.input_ids = persistent_tokens(num_tokens);
        capture_inputs.input_hiddens = persistent_input_hiddens(num_tokens);
        // attention 字段
        capture_inputs.attention_inputs.input_lengths = persistent_input_lengths(batch_size);
        capture_inputs.attention_inputs.sequence_lengths = persistent_sequence_lengths(batch_size);
        capture_inputs.attention_inputs.sequence_lengths_plus_1_d =
            persistent_sequence_lengths_plus_1_d(batch_size);
        capture_inputs.attention_inputs.decode_cu_seqlens_d =
            persistent_decode_cu_seqlens_d(batch_size + 1);
        capture_inputs.attention_inputs.prefix_lengths = persistent_prefix_lengths(batch_size);
        capture_inputs.attention_inputs.context_total_kv_length =
            persistent_context_total_kv_length();
        capture_inputs.attention_inputs.kv_cache_kernel_block_id_device =
            persistent_block_tables(batch_size);
        capture_inputs.attention_inputs.kv_cache_layer_to_group =
            persistent_kv_cache_layer_to_group_;
        capture_inputs.attention_inputs.is_prefill = false;
        capture_inputs.attention_inputs.dtype = model_data_type_;
        capture_inputs.attention_inputs.is_s_padded = true;
        // hybrid cache
        if (!persistent_block_tables_by_group_.empty()) {
            for (int g = 0; g < kv_cache_group_num_; ++g) {
                capture_inputs.attention_inputs
                    .kv_cache_kernel_block_id_device_by_group.push_back(
                        persistent_block_tables_by_group_[g].slice(0, 0, batch_size));
            }
        }
        // tiling data
        capture_inputs.attention_inputs.tiling_data = tiling_data_;
        return capture_inputs;
    }
    return std::nullopt;
}
```

#### 2.3 ATB Tiling Plan（关键，参考 xLLM）

xLLM 在 `plan_paged_attention_tiling()` 中使用 ATB (Ascend Tensor Boost) 计算 paged attention 的 tiling 参数并 H2D 拷贝到 `tiling_data_`。rtp-llm 需要引入相同机制。

```cpp
void GraphPersistentParam::plan_paged_attention_tiling(
    const PyModelInputs& inputs, aclrtStream stream) {
    // 1. 构造 ATB VariantPack (k_cache, v_cache, block_tables, context_lens, tiling_data)
    // 2. custom_pa_op_for_plan_->Setup(variantPack, workspace_size, context)
    // 3. GetHostTilingBufferFromCustomPagedAttentionOperation()
    // 4. aclrtMemcpyAsync(host_tiling → tiling_data_, ACL_MEMCPY_HOST_TO_DEVICE)
}
```

**前提条件**: rtp-llm 的 Attention 算子需要支持 ATB 接口，或者能通过 Ascend C 自定义算子暴露 tiling 参数。

- 若 rtp-llm 的 Ascend 注意力使用 ATB 算子 → 直接复用 xLLM 的 tiling plan 逻辑
- 若使用 `aclnnFlashAttentionScore` → 无需显式 tiling，CANN 内部处理
- 若使用 Ascend C 自定义算子 → 需要在算子实现中暴露 tiling 信息接口

---

### Phase 3: AscendGraphRunner 核心实现（5-7天）

**目标**: 实现 `AscendGraphRunner`，完成 capture/replay 全流程。

#### 3.1 完整类声明（补齐所有缺少字段）

```cpp
// ascend_graph_runner.h
class AscendGraphRunner : public GraphBase {
public:
    AscendGraphRunner(const GraphParams& graph_params, py::object py_instance);
    ~AscendGraphRunner() override;

    void initCapture() override;
    PyModelOutputs forward(const PyModelInputs& inputs, CudaGraphState& state) override;
    bool canRun(const PyModelInputs& inputs, CudaGraphState& state) override;

    void setPositionEncoding(torch::Tensor position_encoding) override;
    void setTokenTypeEmbedding(torch::Tensor token_type_embedding) override;
    void setInputEmbeddingScalar(float input_embedding_scalar) override;

private:
    void captureDecode();
    void captureOneGraphInstance(int key, const char* key_type);

    void prepareInputs(const PyModelInputs& inputs, CudaGraphState& state);
    void replayGraph(int key);

    bool tryGetRealGraphDecodeBatchSize(const PyModelInputs& inputs, CudaGraphState& state);
    std::vector<int> getDecodeBatchSizesToCapture();
    void initCaptureAttentionInputs(PyModelInputs& inputs, int max_bs);
    void prepareCaptureInputs(PyModelInputs& inputs, int batch_size, int num_tokens);
    void copySmallerIntoLarger(const torch::Tensor& src, torch::Tensor& dst, int batch_size);

    // Python 对象
    py::object py_forward_method_;
    py::object py_attn_pyobj_method_;

    // 配置 — 与 CudaGraphRunner 完全对齐
    bool     enable_graph_{false};
    bool     enable_graph_debug_mode_{false};
    bool     is_prefill_cuda_graph_mode_{false};
    bool     is_target_verify_{false};
    size_t   max_bs_{1};
    int      max_seq_len_{0};
    int      hidden_size_{0};
    int      seq_size_per_block_{0};
    int      kernel_seq_size_per_block_{0};
    int      sp_steps_{0};
    int      num_tokens_per_bs_{1};            // spec decode 时 > 1
    int      max_num_token_{1};
    c10::ScalarType model_data_type_;

    // Bucket
    std::vector<int> capture_range_;
    std::vector<int> decode_capture_batch_sizes_;
    std::unordered_map<int, AscendGraphInstance> graph_instances_;

    // Hybrid KV cache 支持
    std::vector<int32_t> kv_cache_layer_to_group_;
    int32_t              kv_cache_group_num_{0};

    // 持久化参数（共享）
    std::unique_ptr<GraphPersistentParam> persistent_param_;

    // Capture stream
    std::optional<c10_npu::NPUStream> capture_stream_;

    // 同步事件
    torch::Event forward_event_{c10::DeviceType::PrivateUse1};

    // TensorOptions
    at::TensorOptions options_npu_int32_;
    at::TensorOptions options_cpu_int32_;
    at::TensorOptions options_npu_float_;

    // Bert Embedding（与 CUDA 版本对齐）
    torch::Tensor position_encoding_;
    torch::Tensor token_type_embedding_;
    float         input_embedding_scalar_{0.0f};
};
```

#### 3.2 构造与初始化

```cpp
AscendGraphRunner::AscendGraphRunner(const GraphParams& params, py::object py_instance)
    : GraphBase(std::move(py_instance)),
      enable_graph_(params.enable_cuda_graph),
      enable_graph_debug_mode_(params.enable_cuda_graph_debug_mode),
      is_prefill_cuda_graph_mode_(params.is_prefill_cuda_graph_mode),
      is_target_verify_(params.is_target_verify),
      max_bs_(params.concurrency_limit),
      max_seq_len_(params.max_seq_len),
      hidden_size_(params.hidden_size),
      seq_size_per_block_(params.tokens_per_block),
      kernel_seq_size_per_block_(params.kernel_tokens_per_block),
      sp_steps_(params.sp_steps),
      num_tokens_per_bs_(params.num_tokens_per_bs),
      max_num_token_(params.concurrency_limit * params.num_tokens_per_bs),
      model_data_type_(params.model_data_type),
      decode_capture_batch_sizes_(params.decode_capture_batch_sizes),
      kv_cache_layer_to_group_(params.kv_cache_layer_to_group),
      kv_cache_group_num_(params.kv_cache_group_num),
      forward_event_(c10::DeviceType::PrivateUse1) {

    // 创建捕获专用 stream
    capture_stream_ = c10_npu::getStreamFromPool(true, c10_npu::current_device());

    // Python 方法引用
    {
        py::gil_scoped_acquire gil;
        if (!py_instance_ || py_instance_.is_none()) {
            throw std::runtime_error("AscendGraphRunner: Python instance is null or none.");
        }
        py_forward_method_    = py_instance_.attr("forward");
        py_attn_pyobj_method_ = py_instance_.attr("prepare_fmha_impl");
    }

    // Tensor options
    auto device = torch::Device(torch::kPrivateUse1);
    options_npu_int32_ = torch::TensorOptions()
        .dtype(torch::kInt32).device(device).requires_grad(false);
    options_cpu_int32_ = torch::TensorOptions()
        .dtype(torch::kInt32).device(torch::kCPU).requires_grad(false);
    options_npu_float_ = torch::TensorOptions()
        .dtype(model_data_type_).device(device).requires_grad(false);
}
```

#### 3.3 initCapture()

```cpp
void AscendGraphRunner::initCapture() {
    // 1. 创建共享 GraphPersistentParam
    persistent_param_ = std::make_unique<GraphPersistentParam>(graph_params, device);

    // 2. 确定捕获范围
    capture_range_ = getDecodeBatchSizesToCapture();  // [1, 8, 16, 24, 32, 48, ...]

    // 3. 分配 max-size buffer 并 warmup forward
    PyModelInputs inputs;
    inputs.input_ids = torch::zeros({max_bs_}, options_npu_int32_);
    // ... 分配 attention inputs

    // 4. Warmup forward（确定 output dtype）
    auto attn_pyobj = py_attn_pyobj_method_(inputs, true);
    py_forward_method_(inputs, attn_pyobj);

    // 5. Capture decode graphs
    captureDecode();
}
```

#### 3.4 captureDecode() 与 captureOneGraphInstance()

```cpp
void AscendGraphRunner::captureDecode() {
    // 从大到小捕获（优化 mempool 分配）
    for (auto it = capture_range_.rbegin(); it != capture_range_.rend(); ++it) {
        int bs = *it;

        // 1. 构造该 batch_size 的输入（取 persistent buffer 的 bs 部分）
        PyModelInputs inputs;
        inputs.input_ids = persistent_param_->persistent_tokens(bs);
        inputs.attention_inputs.input_lengths = persistent_param_->persistent_input_lengths(bs);
        inputs.attention_inputs.sequence_lengths = persistent_param_->persistent_sequence_lengths(bs);
        inputs.attention_inputs.decode_cu_seqlens_d =
            persistent_param_->persistent_decode_cu_seqlens_d(bs + 1);
        inputs.attention_inputs.kv_cache_kernel_block_id_device =
            persistent_param_->persistent_block_tables(bs);
        inputs.attention_inputs.sequence_lengths_plus_1_d =
            persistent_param_->persistent_sequence_lengths_plus_1_d(bs);
        inputs.attention_inputs.prefix_lengths = persistent_param_->persistent_prefix_lengths(bs);
        inputs.attention_inputs.context_total_kv_length =
            persistent_param_->persistent_context_total_kv_length();
        inputs.attention_inputs.is_prefill = false;
        inputs.attention_inputs.dtype = model_data_type_;
        inputs.attention_inputs.is_s_padded = true;
        // hybrid cache
        if (kv_cache_group_num_ > 1) {
            for (int g = 0; g < kv_cache_group_num_; ++g) {
                inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.push_back(
                    persistent_param_->persistent_block_tables_by_group(g, bs));
            }
        }
        // tiling data
        inputs.attention_inputs.tiling_data = persistent_param_->tiling_data();

        // 2. 创建 attention impl（is_cuda_graph=true 触发 graph 模式初始化）
        auto attn_pyobj = py_attn_pyobj_method_(inputs, true);

        // 3. 存储 graph instance
        AscendGraphInstance instance;
        instance.attn_pyobj_ = attn_pyobj;
        graph_instances_[bs] = std::move(instance);

        // 4. 捕获
        captureOneGraphInstance(bs, "decode batch size");
    }
}

void AscendGraphRunner::captureOneGraphInstance(int key, const char* key_type) {
    auto& instance = graph_instances_[key];

    // 构造该 key 对应的 capture 输入（从 persistent buffer slice）
    PyModelInputs inputs;
    inputs.input_ids = persistent_param_->persistent_tokens(key);
    inputs.attention_inputs.input_lengths =
        persistent_param_->persistent_input_lengths(key);
    // ... 构造完整 inputs（prepareCaptureInputs 风格）

    // Warmup（reset lazy init + 预热 kernel launch 路径）
    py_forward_method_(inputs, instance.attn_pyobj_);
    py_forward_method_(inputs, instance.attn_pyobj_);

    // 同步
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream());

    // 切换到 capture stream（NPUGraph 要求非默认 stream）
    bool need_restore = false;
    if (c10_npu::getCurrentNPUStream() == c10_npu::getDefaultNPUStream()) {
        c10_npu::setCurrentNPUStream(capture_stream_.value());
        aclrtSynchronizeStream(capture_stream_.value().stream());
        need_restore = true;
    }

    // Capture
    {
        // 加锁（参考 xLLM DeviceCaptureLock）
        std::lock_guard<std::mutex> lock(get_device_capture_lock());

        instance.graph_.capture_begin(
            {0, 0},
            aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);

        auto outputs = py_forward_method_(inputs, instance.attn_pyobj_)
            .cast<PyModelOutputs>();

        // ★ 关键：将 forward 输出 copy_ 到 persistent buffer（保证 replay 后可读取）
        persistent_param_->set_hidden_states(outputs.hidden_states);

        instance.graph_.capture_end();
    }

    // 恢复 stream
    if (need_restore) {
        c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream());
    }

    // ★ 关键：验证 replay 并同步，确保 capture 后的 replay 也能正确写入输出
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream());
    instance.graph_.replay();
    // replay 后从 persistent buffer 读取结果（不依赖 replay 的隐式写入地址）
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream());
}
```

#### 3.5 canRun() 与 prepareInputs()

```cpp
bool AscendGraphRunner::canRun(const PyModelInputs& inputs, CudaGraphState& state) {
    if (!enable_graph_) return false;

    // ============ speculative decoding: target-verify 检查 ============
    // 同 rtp-llm CudaGraphRunner::canRun() L301-L308
    if (is_target_verify_) {
        if (inputs.attention_inputs.is_target_verify) {
            return tryGetRealGraphDecodeBatchSize(inputs, state);
        }
        return false;
    }

    // ============ 仅 decode: prefill 走 eager ============
    if (inputs.attention_inputs.is_prefill) return false;

    // ============ hybrid KV cache group 数检查 ============
    if (!inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.empty()) {
        const size_t group = inputs.attention_inputs.kv_cache_kernel_block_id_device_by_group.size();
        if (kv_cache_group_num_ <= 0) {
            RTP_LLM_LOG_WARNING(
                "Hybrid kv cache detected but kv_cache_group_num_ is not set, fallback to normal run.");
            return false;
        }
        if (group != static_cast<size_t>(kv_cache_group_num_)) {
            RTP_LLM_LOG_WARNING(
                "Hybrid kv cache group size mismatch: inputs=%zu, captured=%d, fallback to normal run.",
                group, kv_cache_group_num_);
            return false;
        }
    }

    int batch_size = inputs.attention_inputs.input_lengths.size(0);
    state.current_batch_size = batch_size;

    // lower_bound 找匹配 bucket
    auto it = std::lower_bound(capture_range_.begin(), capture_range_.end(), batch_size);
    if (it == capture_range_.end()) return false;

    state.current_real_graph_bs = *it;
    state.seq_len_sum = batch_size;
    return true;
}

void AscendGraphRunner::prepareInputs(const PyModelInputs& inputs, CudaGraphState& state) {
    // 同步上一次 forward 完成
    forward_event_.synchronize();

    int key = state.current_real_graph_bs;

    // 通过 shared persistent_param 更新所有输入数据（含清零 + padding）
    persistent_param_->update(inputs, state.current_batch_size,
                              /*return_capture_params=*/false);

    // 调用 attention 的 prepare_cuda_graph()（更新 attn 内部参数）
    // 对应 rtp-llm CudaGraphRunner::prepareInputs() L151/L184
    graph_instances_[key].attn_pyobj_.attr("prepare_cuda_graph")(
        persistent_param_->get_attention_inputs_for_bs(key));
}
```

#### 3.6 forward()

```cpp
PyModelOutputs AscendGraphRunner::forward(const PyModelInputs& inputs, CudaGraphState& state) {
    prepareInputs(inputs, state);

    int key = state.current_real_graph_bs;
    replayGraph(key);

    // ★ 关键：从 persistent buffer 的固定地址读取输出（而非依赖 NPUGraph replay 隐式写入）
    PyModelOutputs outputs;
    outputs.hidden_states = persistent_param_->hidden_states(state.seq_len_sum).clone();
    // CUDA Graph 版本也是 .clone()，确保独立 tensor

    // 记录完成事件
    forward_event_.record(c10_npu::getCurrentNPUStream());

    return outputs;
}
```

---

### Phase 4: PyWrappedModel 集成（2-3天）

**目标**: 在 `PyWrappedModel` 中支持创建和使用 `AscendGraphRunner`。

#### 4.1 PyWrappedModel.h 修改

```cpp
#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#elif USING_ASCEND
#include "rtp_llm/cpp/ascend_graph/ascend_graph_runner.h"
#endif
```

#### 4.2 PyWrappedModel.cc 条件编译（constructor, ~246行附近）

```cpp
if (enable_cuda_graph_) {
#if USING_CUDA || USING_ROCM
    graph_runner_ = new CudaGraphRunner(graph_params, py_instance);
#elif USING_ASCEND
    graph_runner_ = new AscendGraphRunner(graph_params, py_instance);
#else
    RTP_LLM_CHECK_WITH_INFO(false, "Graph mode is only supported on CUDA/ROCm/Ascend");
#endif
    // ... 设置 weights（设备无关）
    if (weights_.position_encoding) {
        graph_runner_->setPositionEncoding(weights_.position_encoding->kernel.cuda());
    }
    // ...
    graph_runner_->initCapture();
}
```

**注意**: `weights_.position_encoding->kernel.cuda()` 中的 `.cuda()` 调用需要参数化为 `.to(device_)`，这在 Phase 0 的全局设备适配中已有计划。

#### 4.3 forward() 调用路径（已有，无需修改）

```cpp
// PyWrappedModel.cc L384-396，无需修改
CudaGraphState graph_state;
if (enable_cuda_graph_ && graph_runner_->canRun(py_model_inputs, graph_state)) {
    py_model_inputs.attention_inputs.is_s_padded = true;
    py_model_outputs = graph_runner_->forward(py_model_inputs, graph_state);
    hidden_states = py_model_outputs.hidden_states.clone();
} else {
    // normal forward
}
```

`GraphBase*` 多态已经提供了正确的抽象。

---

### Phase 5: Attention 集成（5-7天，核心难点）

**目标**: 使 Ascend 上的 Attention 实现支持 graph 模式。

#### 5.1 方案选择

| 方案 | 描述 | 复杂度 | 风险 |
|------|------|--------|------|
| **A: ATB + tiling（推荐）** | C++ 层使用 ATB 算子，capture/replay 时 tiling 通过 persistent buffer 更新 | 中 | ATB 依赖 |
| **B: Python prepare_cuda_graph** | 纯 Python 侧 Attention 实现，capture 时构造固定参数，replay 前更新 | 低 | 图捕获时 Python 调用不能被录制 |
| **C: aclnnFlashAttentionScore** | 使用 CANN 内置 FlashAttention 算子，CANN 内部处理图捕获 | 低 | 依赖 CANN 版本 |

**推荐方案 A+B 混合**:
- C++ 层使用 ATB 管理 paged attention tiling（参考 xLLM）
- Python 侧 `prepare_cuda_graph()` 更新非 tiling 参数（与其他 rtp-llm Attention 实现保持一致）

#### 5.2 Ascend Decode Attention 实现

```python
# rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py

class AscendDecodeImpl(FMHAImplBase):
    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        """图模式重放前更新 attention 参数"""
        # 从 persistent buffer 读取更新后的参数
        # 更新 attention 算子的内部状态
        pass

    def support_cuda_graph(self) -> bool:
        return True
```

**注意**: 如果 Attention 使用 `aclnnFlashAttentionScore`，则 capture 时该算子会被录制到 NPU graph 中，replay 时自动执行，无需特殊的 prepare_cuda_graph 逻辑。此时仅需在 capture 前做好所有参数设置。

#### 5.3 KV Cache Block Table 处理

在 `prepareInputs()` 中，需要把 KV cache block table 更新到 persistent buffer。rtp-llm 的 `kv_cache_kernel_block_id_device` 需要映射到 Ascend 的 block table 格式。

```cpp
// 在 GraphPersistentParam::update() 中
persistent_block_tables_.slice(0, 0, batch_size).copy_(
    inputs.attention_inputs.kv_cache_kernel_block_id_device, true);
```

---

### Phase 6: 构建系统与条件编译（1-2天）

**目标**: 使 Ascend Graph 编译通过，与 CUDA/ROCm 共存。

#### 6.1 Bazel BUILD

```python
# rtp_llm/cpp/ascend_graph/BUILD
load("//bazel:arch_select.bzl", "torch_deps")

cc_library(
    name = "ascend_graph",
    srcs = [
        "ascend_graph_runner.cc",
        "ascend_graph_persistent_param.cc",
        "ascend_graph_capture.cc",
        "ascend_graph_device_shims.cc",
    ],
    hdrs = [
        "ascend_graph_runner.h",
        "ascend_graph_persistent_param.h",
        "ascend_graph_device_shims.h",
        "ascend_graph_utils.h",
    ],
    deps = torch_deps() + [
        "//rtp_llm/cpp/cuda_graph:cuda_graph_base",
        "//rtp_llm/cpp/utils:core_utils",
        "//rtp_llm/models_py/bindings:op_defs",
        "@local_config_ascend//:ascend_headers",
    ],
    copts = ["-DUSING_ASCEND=1"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
```

#### 6.2 PyWrappedModel BUILD 条件依赖

```python
# rtp_llm/cpp/models/BUILD
cc_library(
    name = "py_wrapped_model",
    ...
    deps = [
        ...
    ] + select({
        "//:using_cuda": ["//rtp_llm/cpp/cuda_graph:cuda_graph_impl"],
        "//:using_rocm": ["//rtp_llm/cpp/cuda_graph:cuda_graph_impl"],
        "//:using_ascend": ["//rtp_llm/cpp/ascend_graph:ascend_graph"],
        "//conditions:default": [],
    }),
)
```

---

## 4. 关键设计决策

### 4.1 为什么新建 `AscendGraphRunner` 而非在 `CudaGraphRunner` 中加 `#if USING_ASCEND`

| 方案 | 优劣 |
|------|------|
| 在 CudaGraphRunner 加 #if | 代码膨胀，CaptureMemoryHold 与 GraphPersistentParam 差异大；`graphCaptureBegin` 签名不同，shim 层无法统一 |
| 新建 AscendGraphRunner | 代码隔离，可独立测试，充分利用 xLLM 成熟模式 |

rtp-llm 的 `GraphBase*` 多态机制已经为这种扩展做好了准备。

**同理，device shim 层也新建 `ascend_graph_device_shims.h/.cc` 而非在原始 `cuda_graph_device_shims.h` 中加分支**。原因：
- `graphCaptureBegin` 的签名完全不同（`at::cuda::CUDAGraph&` vs `c10_npu::NPUGraph&`），无法通过同一切换实现
- 命名空间 `ascend_graph` 与 `cuda_graph` 隔离，避免符号冲突
- 未来维护时修改 Ascend 代码不影响 CUDA/ROCm 路径

### 4.2 为什么使用共享 Persistent Buffer（xLLM 模式）

| 维度 | rtp-llm 模式（per-instance） | xLLM 模式（shared） |
|------|----------------------------|-------------------|
| 内存占用 | O(n_buckets × max_size) | O(max_size) |
| 数据拷贝 | 每个 replay 需拷贝到指定 instance | 所有 instance 共享同一份数据 |
| 复杂度 | 需维护 per-instance CaptureMemoryHold | 只需一个 persistent_param_ |

对于 Ascend NPU，使用共享 persistent buffer 更简洁且内存高效。

### 4.3 Bucket 策略

**decode 阶段**: 沿用 rtp-llm 的 batch_size bucket 策略（`[1, 8, 16, 24, 32, 48, 64, ...]`），通过 `std::lower_bound` 向上对齐。

**关于 batch_size 与 num_tokens 的语义差异**: 

- **正常 decode** (`num_tokens_per_bs_ == 1`): `batch_size == num_tokens`，无差异，bucket 策略完全等价。
- **Speculative decoding** (`num_tokens_per_bs_ > 1`): `num_tokens = batch_size × num_tokens_per_bs`，此时 batch_size 维度仍是正确的 bucket 维度（因为 graph 捕获的是 batch_size 决定的所有 tensor 形状）。例如 batch_size=8, num_tokens_per_bs=4 → num_tokens=32，但 graph 的 block_tables 是 `[8, ...]` 而非 `[32, ...]`。

因此 `batch_size` 维度是语义正确的 bucket key。

**暂不支持 prefill graph**: xLLM 的 ACL Graph 只支持 decode。rtp-llm 的 prefill CUDA graph 使用 piecewise capture + 自定义 copy kernel，这部分在 Ascend 上的适配复杂度高，建议后续再支持。

### 4.4 同步机制

```cpp
// 1. forward 完成事件（保证 prepareInputs 前一次 forward 已完成）
forward_event_.record(c10_npu::getCurrentNPUStream());
// ... 在下一轮 prepareInputs 中
forward_event_.synchronize();

// 2. Capture 锁（防止多线程冲突，参考 xLLM DeviceCaptureLock）
std::lock_guard<std::mutex> lock(get_device_capture_lock());

// 3. H2D 拷贝同步
aclrtSynchronizeStream(c10_npu::getCurrentNPUStream().stream());
```

---

## 5. 与 xLLM ACL Graph 的关键差异

| 维度 | xLLM | rtp-llm（本方案） | 原因 |
|------|------|-------------------|------|
| Bucket 维度 | num_tokens（`1,2,4,8,16,...`） | batch_size（`1,8,16,24,32,...`） | 与 rtp-llm 现有逻辑对齐 |
| 输入源 | C++ ModelInputParams（直接 tensor） | PyModelInputs（Python→C++） | rtp-llm 通过 Python binding 传递输入 |
| Attention | ATB 自定义 PagedAttention | `prepare_cuda_graph()` Python 接口 + ATB tiling | 兼容 rtp-llm 现有 Attention Factory 机制 |
| 输出 | ModelOutput（C++ struct） | PyModelOutputs（Python binding） | rtp-llm 输出格式 |
| 多 stream | model forward 在 capture stream 上 | 切换 stream（与 CUDA 版本一致） | 兼容 rtp-llm 现有模式 |
| Prefill | 不支持 | 不支持（同 xLLM） | Ascend ACL Graph 当前只支持 decode |

---

## 6. 工作量估算

| Phase | 内容 | 工作日 | 依赖 |
|-------|------|--------|------|
| **P1** | 设备抽象层扩展 | 2-3 | Bazel Ascend 编译环境就绪 |
| **P2** | GraphPersistentParam | 3-4 | ATB 库可用 |
| **P3** | AscendGraphRunner 核心 | 5-7 | P1+P2 |
| **P4** | PyWrappedModel 集成 | 2-3 | P3 |
| **P5** | Attention 集成 | 5-7 | P3 + Ascend Attention 算子就绪 |
| **P6** | 构建系统 | 1-2 | P1-P5 |
| **集成测试** | 端到端验证 | 3-5 | P6 |
| **总计** | | **~23-34** | |

---

## 7. 测试验证方案

### 7.1 单元测试

```cpp
// GraphPersistentParam
TEST(GraphPersistentParamTest, UpdateCopiesAllFields) {
    // 验证 update() 中 15 个字段全部正确拷贝
}
TEST(GraphPersistentParamTest, BlockTableClearedBeforeCopy) {
    // 验证 replay 前 block_tables 被清零（防 stale data）
}
TEST(GraphPersistentParamTest, PaddingSlotsCleared) {
    // 验证 batch_size < max_batch 时多余行被清零
}
TEST(GraphPersistentParamTest, TilingDataUpdated) {
    // 验证 ATB tiling plan H2D 拷贝后 tiling_data_ 非零
}

// AscendGraphRunner
TEST(AscendGraphRunnerTest, CanRunDecodeOnly) {
    // prefill 输入返回 false
}
TEST(AscendGraphRunnerTest, CanRunBatchSizeOutOfRange) {
    // 超出 max capture 的 batch_size 返回 false
}
TEST(AscendGraphRunnerTest, CanRunHybridCacheMismatch) {
    // hybrid cache group 数不匹配返回 false
}
TEST(AscendGraphRunnerTest, CanRunSpeculativeTargetVerify) {
    // is_target_verify 场景验证
}
TEST(AscendGraphRunnerTest, CanRunLowerBoundSelect) {
    // batch_size=10 → 选择 16 的 bucket
}
```

### 7.2 集成测试

1. **单步 decode 验证**: capture → replay → 结果与 eager 模式一致
2. **多步 decode 验证**: 连续 1000 步 replay，结果与 eager 一致
3. **动态 batch 验证**: batch 在 1..max_bs 之间变化，验证 bucket 匹配正确
4. **Fallback 验证**: batch_size 超出范围时自动 fallback 到 eager
5. **Hybrid KV cache 验证**: 多 group KV cache 场景下 graph 模式正确
6. **Speculative decoding 验证**: target-verify 模式下图功能正确
7. **KV cache block 污染测试**: 验证 block table 清零防止了 stale block 污染
8. **Padding 验证**: batch_size=3 走 bs=8 的 graph，padding 位置不产生异常输出

### 7.2 集成测试

1. **单步 decode 验证**: capture → replay → 结果与 eager 模式一致
2. **多步 decode 验证**: 连续 1000 步 replay，结果与 eager 一致
3. **动态 batch 验证**: batch 在 1..max_bs 之间变化，验证 bucket 匹配正确
4. **Fallback 验证**: batch_size 超出范围时自动 fallback 到 eager
5. **Hybrid KV cache 验证**: 多 group KV cache 场景下 graph 模式正确
6. **Speculative decoding 验证**: target-verify 模式下图功能正确
7. **KV cache block 污染测试**: 验证 block table 清零防止了 stale block 污染
8. **Padding 验证**: batch_size=3 走 bs=8 的 graph，padding 位置不产生异常输出

### 7.3 精度对比

```python
# 对比 graph 模式与 eager 模式每步 hidden_states 的差异
max_diff = torch.max(torch.abs(graph_output - eager_output))
assert max_diff < 1e-3, f"Graph mode output differs: {max_diff}"
```

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| `c10_npu::NPUGraph` 存在未预期限制 | 图模式不可用 | 咨询华为 torch_npu 团队；备选：使用 CANN aclmdl 原生接口 |
| ATB 依赖 | tiling 不工作 | 如果 Attention 使用 aclnn 原生算子而非 ATB，可跳过 tiling plan |
| Python forward 中混有同步操作 | capture 被打断 | DeviceCaptureLock + THREAD_LOCAL mode 缓解 |
| 图捕获内存占用过大 | OOM | 限制 bucket 数量 + 从大到小捕获 |
| Ascend 上 pinned memory 行为不同 | 拷贝性能差 | 使用 aclrtMallocHost 分配 host 内存 |
