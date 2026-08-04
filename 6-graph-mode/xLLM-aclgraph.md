# 图模式算子Tiling信息更新

> **Workspace:** c:\WorkingSpace\Code\xllm

---

## User
图模式下算子tiling信息是如何更新的

## Assistant
现在我已经有了完整的理解。下面为你详细解释图模式下算子 tiling 信息是如何更新的。

---

## 图模式下算子 Tiling 信息的更新流程

整个流程分为 **4 个阶段**：初始化 → 计划（Plan）→ 捕获（Capture）→ 重放（Replay）。下面逐一说明。

---

### 1. 初始化阶段：分配 Tiling 缓冲区

在 [acl_graph_executor_impl.cpp](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp) 的 `initialize_paged_attention_plan_context` 中（L362-380）：

```cpp
constexpr int64_t tiling_buffer_size = 1024 * 256;
tiling_data_ = torch::zeros({tiling_buffer_size},
                            torch::dtype(torch::kInt32).device(device));
```

- 在 Device（NPU）上预分配一块 **固定大小的 Int32 tensor**（`tiling_data_`），作为持久化的 tiling 缓冲区。
- 同时创建 ATB Plan Context 和专属 stream，设置 `GRAPH_LAUNCH_MODE`。
- 创建 `CustomPagedAttention` 算子实例（`custom_pa_op_for_plan_`），配置好 head_num、kv_head_num、qk_scale 等静态参数。

---

### 2. 计划阶段：计算并拷贝 Tiling 数据

核心方法是 [plan_paged_attention_tiling](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L604-L698)。流程如下：

**① 构造 ATB 张量**（L612-645）

将 PyTorch tensor 转换为 ATB tensor，包括 k_cache、v_cache、block_tables、context_lens（kv_seq_lens）以及 `tiling_data_` 本身。同时构造一个只有描述信息（无实际数据）的 query tensor 用于 shape 推导。

**② 调用 Setup 获取 Tiling**（L670-678）

```cpp
atb::Status status = custom_pa_op_for_plan_->Setup(
    custom_variantPack, custom_workspace_size, context_for_plan_);
```

将所有输入 tensor（含 tiling_data）作为 VariantPack 传入 ATB 的 `Setup` 接口。ATB 内部根据**当前实际的 batch_size、seq_len、head_dim** 等维度信息计算出 tiling 参数。

**③ 提取 Host 端 Tiling Buffer**（L680-685）

```cpp
atb::customize::TilingBufferInfo tiling_buffer_info =
    atb::customize::GetHostTilingBufferFromCustomPagedAttentionOperation(
        custom_pa_op_for_plan_);
```

从算子实例中获取 ATB 在 Host 端生成的 tiling buffer（包含所有分块策略参数：batch、numHeads、headDim、blockSize 等，见 [L424-L565](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L424-L565) 的 `parse_pa_host_tiling_buffer`）。

**④ 异步拷贝到 Device**（L690-698）

```cpp
aclError acl_status =
    aclrtMemcpyAsync(tiling_data_.data_ptr(),
                     tiling_data_.numel() * sizeof(uint32_t),
                     tiling_buffer_info.tilingBuffer,
                     tiling_buffer_info.tilingBufferSize,
                     ACL_MEMCPY_HOST_TO_DEVICE,
                     stream);
```

将 Host 端计算好的 tiling 数据 **异步拷贝** 到 Device 上的 `tiling_data_` tensor。

---

### 3. 捕获阶段（Capture）：绑定 Tiling 到图输入

在 [update](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L295-L344) 方法中，**capture 之前**会调用 tiling 计划：

```cpp
if (tiling_data_.numel() > 0) {
    if (need_update_attention_plan_ && k_cache.defined() && ...) {
      plan_paged_attention_tiling(
          tokens, k_cache, v_cache, persistent_block_tables_, params, stream);
    }
}
```

然后将已更新好的 `tiling_data_` 绑定到捕获用的参数中：

```cpp
params_for_capture->graph_buffer.tiling_data = tiling_data();
```

这样在 **图捕获（`torch_npu::NPUGraph::capture`）** 时，`tiling_data` 作为图的一个固定输入地址被记录下来。

---

### 4. 重放阶段（Replay）：每次推理前更新 Tiling

在 [replay](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L935-L960) 中，每次执行图重放前：

```cpp
persistent_param_.update(tokens, k_cache, v_cache, positions, params,
                         num_tokens_, /*return_capture_params=*/false);
```

`update` 方法会再次调用 `plan_paged_attention_tiling`，根据**当前实际输入的 batch 信息**重新计算 tiling 参数，并通过 `aclrtMemcpyAsync` 写入同一个 Device 地址。由于图捕获时绑定的就是这个地址，重放时自动使用最新的 tiling 数据。

---

### 5. Tiling 数据的消费路径

最终在 attention 层执行时：

```
ModelInputParams.graph_buffer.tiling_data
        ↓ (attention_metadata_builder.cpp:90)
AttentionMetadata.paged_attention_tiling_data
        ↓ (npu_torch/attention.cpp:137)
kernel::npu::batch_decode_acl_graph(... tiling_data ...)
        ↓ (npu/attention.cpp:86)
atb::npu_custom_paged_attention(... tiling_data ...)
```

ATB 的 `npu_custom_paged_attention` 算子直接读取 Device 上的 tiling tensor，按照其中记录的分块策略执行 Paged Attention 计算。

---

### 整体流程图

```
┌──────────────────────────────────────────────────────────┐
│  初始化                                                   │
│  tiling_data_ = zeros(1024*256, int32, device)           │
│  创建 custom_pa_op_for_plan_ + context_for_plan_         │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  update() [Capture前 / 每次Replay前]                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │ plan_paged_attention_tiling():                     │  │
│  │   1. 构造 ATB VariantPack (query, k/v_cache, ...)  │  │
│  │   2. Setup() → ATB计算tiling参数                    │  │
│  │   3. GetHostTilingBuffer() → 获取host端tiling       │  │
│  │   4. aclrtMemcpyAsync → H2D拷贝到tiling_data_      │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  Capture阶段                                              │
│  graph_buffer.tiling_data = tiling_data()                │
│  → 图捕获时将tiling_data地址作为固定输入记录               │
└────────────────────────┬─────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  Replay阶段 (每次推理)                                    │
│  update() → 重新计算tiling → 写入同一Device地址           │
│  graph_.replay() → 图执行，自动使用最新tiling数据         │
└──────────────────────────────────────────────────────────┘
```

**关键设计点**：tiling_data 是一块**持久化的 Device 内存**，地址在 capture 和 replay 之间保持不变。每次 replay 前，ATB 的 Setup 机制根据当前实际输入 shape 重新计算 tiling 参数并写入这块内存，从而实现了**图模式下 tiling 信息的动态更新**。

## User
图模式原理、流程介绍

## Assistant
下面以 NPU 的 **ACL Graph**（[acl_graph_executor_impl.cpp](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp)）为主线，介绍 xLLM 图模式的原理与完整流程。CUDA / MLU 路径结构类似，仅底层图 API 不同。

---

## 一、图模式的原理

### 1.1 为什么需要图模式

Decode 阶段（每步只生成 1 个 token）有以下性能瓶颈：

- **算子启动开销大**：每次 forward 包含上百个小 kernel launch，CPU 提交 → NPU 执行的延迟占比高
- **内存分配抖动**：每步都会 alloc/free 大量中间 tensor，分配器开销大
- **重复的 tiling/plan 计算**：每次都重新做 attention 的 setup/tiling

图模式通过**捕获一次完整的计算图并重复重放**解决上述问题：

```
传统 eager 模式:  CPU提交算子 → NPU执行 → CPU提交算子 → NPU执行 ...  (串行)
图模式:          CPU提交1次捕获 → 后续每次只调用 graph.replay()  (跳过CPU提交)
```

### 1.2 三个核心机制

**① 持久化参数（GraphPersistentParam）**

把所有"会变化"的输入（tokens、positions、kv_seq_lens、block_tables、tiling_data、mask 等）放在**固定地址的 Device tensor** 中。捕获时记录的是这些 tensor 的**地址**，重放前只需更新地址里的数据。

**② 内存池（NPUGraph mempool）**

捕获期间所有临时 tensor 都从 NPUGraph 专属 mempool 分配。重放时这些临时内存被**原地复用**，避免反复 alloc/free。每个 bucket 拥有独立的 mempool。

**③ 桶（Bucket）机制**

不同 batch_size 走不同图。例如 `bucket = 1, 2, 4, 8, 16, 32, 48, ...`，首次遇到某 bucket 时捕获，后续相同 bucket 直接重放。

---

## 二、关键数据结构

### [GraphPersistentParam](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.h#L60-L244)

```cpp
class GraphPersistentParam {
  // 持久化输入 tensor（固定地址）
  torch::Tensor persistent_tokens_;          // [max_tokens]
  torch::Tensor persistent_positions_;       // [max_tokens]
  torch::Tensor persistent_block_tables_;    // [max_batch, max_blocks]
  torch::Tensor persistent_new_cache_slots_; // [max_tokens]
  torch::Tensor persistent_mask_;            // [max_tokens, max_seq_len]
  torch::Tensor q_seq_lens_, kv_seq_lens_;   // [max_batch]
  torch::Tensor persistent_embedding_;       // for MTP / multimodal
  torch::Tensor tiling_data_;                // paged attention tiling

  // 持久化输出 tensor
  torch::Tensor hidden_states_;
  torch::Tensor aux_hidden_states_;

  // ATB context + 算子（用于 tiling 计算）
  atb::Context* context_for_plan_;
  atb::Operation* custom_pa_op_for_plan_;
  aclrtStream stream_for_plan_;

  // update(): 更新持久化tensor + 计算tiling + 返回捕获参数
  // plan_paged_attention_tiling(): 通过ATB Setup计算tiling并H2D拷贝
};
```

### [AclGraph](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.h#L254-L303)

```cpp
class AclGraph {
  c10_npu::NPUGraph graph_;                  // 底层NPUGraph句柄
  uint32_t num_tokens_;                       // 该图对应的bucket大小
  GraphPersistentParam& persistent_param_;   // 共享的持久化参数
  std::optional<c10_npu::NPUStream> capture_stream_; // 捕获专用stream
};
```

### [AclGraphExecutorImpl](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.h#L307-L339)

```cpp
class AclGraphExecutorImpl {
  absl::flat_hash_map<uint32_t, std::unique_ptr<AclGraph>> graphs_;  // bucket→图
  std::unique_ptr<GraphPersistentParam> persistent_param_;           // 全局唯一
};
```

---

## 三、完整执行流程

### 3.1 入口：[AclGraphExecutorImpl::run()](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L985-L1113)

```
run(tokens, positions, kv_caches, params)
   │
   ├─ 非decode阶段 / 单层模型 → eager: model_->forward()   ❶
   │
   ├─ kv_max_seq_len > max_position_embeddings → eager     ❷
   │
   ├─ bucket = get_bucket_num_tokens(n_tokens)
   │
   ├─ graphs_.find(bucket) 命中？
   │     ├─ 命中 → graph.replay()                          ❸ 重放
   │     └─ 未命中 → graph.capture() (lazy 捕获)            ❹ 捕获
   │
   └─ 返回 hidden_states（含 aux_hidden_states）
```

### 3.2 Bucket 对齐：[get_bucket_num_tokens()](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L1112-L1130)

```cpp
// buckets: [1, 2, 4, 8, 16, 32, 48, 64, ...max_seqs_per_batch]
// 实际tokens会被padding到bucket大小
```

例如实际 13 个 token → bucket = 16，多出的 3 个用 padding 填充（保证固定 shape 的图能复用）。

### 3.3 捕获阶段：[AclGraph::capture()](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L820-L918)

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: persistent_param_.update(return_capture=true)   │
│   - 拷贝tokens/positions/block_tables到持久化tensor      │
│   - 调用 plan_paged_attention_tiling() 计算tiling       │
│   - 返回用于捕获的 ModelInputParams（引用持久化tensor）  │
├─────────────────────────────────────────────────────────┤
│ Step 2: 同步stream, 确保数据拷贝完成                     │
├─────────────────────────────────────────────────────────┤
│ Step 3: 加 capture_lock（防止其它线程同步操作干扰）     │
├─────────────────────────────────────────────────────────┤
│ Step 4: 切换到 capture_stream_                          │
├─────────────────────────────────────────────────────────┤
│ Step 5: graph_.capture_begin({0,0}, THREAD_LOCAL)       │
│         ↑ 开启NPUGraph捕获模式，后续所有算子都被记录     │
├─────────────────────────────────────────────────────────┤
│ Step 6: model->forward(persistent_tokens,                │
│                        persistent_positions,            │
│                        kv_cache,                         │
│                        graph_params)                     │
│   ↑ 这次forward的所有kernel被"录制"进图                  │
│   ↑ 临时tensor从mempool分配                             │
├─────────────────────────────────────────────────────────┤
│ Step 7: persistent_param_.set_hidden_states(result)     │
│   ↑ 把输出tensor地址绑定到持久化buffer                   │
├─────────────────────────────────────────────────────────┤
│ Step 8: graph_.capture_end()  结束捕获                  │
├─────────────────────────────────────────────────────────┤
│ Step 9: graph_.replay()  验证捕获成功                   │
└─────────────────────────────────────────────────────────┘
```

关键点：
- `ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL`：允许其它线程做同步操作
- **capture_lock**：全局互斥锁，避免数据更新流的同步 H2D 打断捕获（见 [worker_impl.cpp:449](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/worker_impl.cpp#L449)）
- 捕获完成后图被存入 `graphs_[bucket]`，下次相同 bucket 直接复用

### 3.4 重放阶段：[AclGraph::replay()](file:///c:/WorkingSpace/Code/xllm/xllm/core/runtime/acl_graph_executor_impl.cpp#L935-L968)

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: persistent_param_.update(return_capture=false)  │
│   - 把新的tokens/positions拷贝到持久化tensor（原地更新） │
│   - 重新计算tiling（基于当前batch实际shape）             │
│   - H2D拷贝到固定tiling_data_地址                        │
├─────────────────────────────────────────────────────────┤
│ Step 2: graph_.replay()                                 │
│   - 重放捕获的算子序列（跳过CPU提交）                    │
│   - 临时tensor从mempool复用（地址不变）                  │
│   - tiling_data因为是固定地址，自动使用最新值            │
├─────────────────────────────────────────────────────────┤
│ Step 3: 返回 hidden_states[:actual_num_tokens]          │
└─────────────────────────────────────────────────────────┘
```

---

## 四、整体流程图

```
                      ┌──────────────┐
   请求到达 ─────────► │ decode phase?│
                      └──────┬───────┘
                          否 │              是
                 ┌────────────┴───────────────┐
                 ▼                            ▼
        ┌───────────────┐          ┌──────────────────┐
        │ eager forward │          │ bucket = align(n)│
        └───────────────┘          └────────┬─────────┘
                                            │
                                  ┌─────────▼──────────┐
                                  │ graphs_[bucket]?   │
                                  └─────┬────────┬─────┘
                                    未命中│      │命中
                       ┌──────────────────▼      ▼─────────────┐
                       ▼                                         │
            ┌──────────────────────┐         ┌──────────────────▼─────┐
            │ AclGraph::capture()  │         │ AclGraph::replay()      │
            │  - update(t→持久化)  │         │  - update(t→持久化)     │
            │  - plan tiling       │         │  - plan tiling (H2D)    │
            │  - capture_begin     │         │  - graph_.replay()      │
            │  - model.forward()   │         │  - 返回 hidden[:actual] │
            │  - capture_end       │         └─────────────────────────┘
            │  - 存入graphs_[b]    │
            └──────────────────────┘
```

---

## 五、图模式的核心设计要点

| 设计点 | 实现方式 | 作用 |
|--------|----------|------|
| **输入固定地址** | `GraphPersistentParam` 持有持久化 tensor | 捕获时记录地址，重放时原地更新数据 |
| **临时内存复用** | NPUGraph mempool | 避免反复 alloc/free，降低分配开销 |
| **Tiling 动态更新** | `plan_paged_attention_tiling` + H2D 拷贝 | 解决"图固定但 batch 变化"的矛盾（见上轮回答） |
| **Bucket 对齐** | `get_bucket_num_tokens` | 限制图的数量，平衡内存与命中率 |
| **Capture Lock** | `DeviceCaptureLock` | 防止其它线程同步操作打断捕获 |
| **专用 capture stream** | `getStreamFromPool(true, ...)` | NPUGraph 要求非默认 stream 捕获 |
| **Lazy 捕获** | 首次遇到 bucket 才捕获 | 避免预捕获无用 bucket，节省内存 |

---

## 六、与 CUDA Graph / MLU Graph 的对比

三者架构完全一致，仅底层 API 不同：

| 维度 | ACL Graph (NPU) | CUDA Graph (GPU) | MLU Graph (Cambricon) |
|------|-----------------|------------------|-----------------------|
| 底层句柄 | `c10_npu::NPUGraph` | `at::cuda::CUDAGraph` | `torch_mlu::MLUGraph` |
| 捕获模式 | `ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL` | `cudaStreamCaptureModeThreadLocal` | `cnrtQueueCaptureModeRelaxed` |
| Tiling/Plan | ATB `Setup` + H2D | FlashInfer `plan` | - |
| Prefill 支持 | ❌ 仅 decode | ✅ piecewise capture | ❌ 仅 decode |
| 注册名 | `"npu"` | `"cuda"` | `"mlu"` |

---

**总结**：xLLM 图模式的核心思想是 **"地址固定 + 数据更新 + 图重放"**——把所有变化的输入放到固定地址的持久化 buffer 中，捕获一次计算图后，后续每次推理只需更新 buffer 内容（含 tiling）并调用 `replay()`，从而大幅降低 CPU 开销和内存分配开销。

