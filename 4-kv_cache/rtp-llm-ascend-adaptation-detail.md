# rtp-llm 昇腾NPU适配 — 逐文件详细代码修改点

> 基准提交: `fa4029c2101a8647dc0acc7a74cbfc12a6565cf5`
> 改动范围: 40个文件, +455/-87行

---

## 一、构建系统 (4文件)

### 1. `.bazelrc` (+3/-3)

**修改点1 — Python解释器路径 (第5行)**
```diff
- build --action_env PYTHON_BIN_PATH="/opt/conda310/bin/python3"
+ build --action_env PYTHON_BIN_PATH="/opt/conda310/envs/py310/bin/python3"
```

**修改点2 — Python解释器路径注释 (第3行)**
```diff
- build --python_top=//:python310 --incompatible_use_python_toolchains=false # force use /opt/conda310/bin/python3
+ build --python_top=//:python310 --incompatible_use_python_toolchains=false # force use /opt/conda310/envs/py310/bin/python3
```

**修改点3 — ARM架构优化选项 (第204行)**
```diff
- build:ascend --copt="-march=armv8-a+crc"
+ build:ascend --copt="-march=armv8.2-a+fp16+dotprod+crc"
```

---

### 2. `BUILD` (+2/-2)

**修改点1 — py_runtime解释器路径 (第154-155行)**
```diff
  py_runtime(
      name = "python310",
-     interpreter_path = "/opt/conda310/bin/python",
+     interpreter_path = "/opt/conda310/envs/py310/bin/python",
      python_version = "PY3",
-     stub_shebang = "#!/opt/conda310/bin/python",
+     stub_shebang = "#!/opt/conda310/envs/py310/bin/python",
      visibility = ["//visibility:public"],
  )
```

---

### 3. `deps/pip.bzl` (+1/-1)

**修改点1 — pip_ascend_torch Python解释器 (第72行)**
```diff
  pip_parse(
      name = "pip_ascend_torch",
      requirements_lock = "@rtp_deps//:requirements_lock_ascend.txt",
-     python_interpreter = "/opt/conda310/bin/python3",
+     python_interpreter = "/opt/conda310/envs/py310/bin/python3",
```

---

### 4. `arch_config/arch_select.bzl` (+1/-1)

**修改点1 — Ascend分支绑定注册 (第189行)**
```diff
  "@rtp_llm//:using_ascend": [
-     "@rtp_llm//rtp_llm/models_py/bindings:dummy_register",
+     "@rtp_llm//rtp_llm/models_py/bindings/ascend:ascend_bindings_register",
  ],
```

---

## 二、C++ KV Cache (5文件)

### 5. `rtp_llm/cpp/cache/MemoryLayoutConfig.h` (+4/-0)

**修改点1 — 新增分离K/V池大小字段 (第23-24行)**
```diff
  struct MemoryLayoutConfig {
      size_t kv_scale_pool_size_bytes = 0;
      size_t total_size_bytes         = 0;
  
+     // ---- Separate K/V pool sizes (Ascend NPU) ----
+     size_t k_pool_size_bytes = 0;
+     size_t v_pool_size_bytes = 0;
+
      // ---- Per-block strides (one layer) ----
      size_t kv_block_stride_bytes = 0;
```

---

### 6. `rtp_llm/cpp/cache/BlockPoolConfigHelper.h` (+10/-0)

**修改点1 — 计算分离K/V池大小 (第137-146行)**
```diff
      cfg.kv_block_pool_size_bytes =
          static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_block_stride_bytes;
  
+     cfg.k_pool_size_bytes = 0;
+     cfg.v_pool_size_bytes = 0;
+     if (cache_config.separate_kv_cache) {
+         cfg.k_pool_size_bytes =
+             static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.k_block_stride_bytes;
+         cfg.v_pool_size_bytes =
+             static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.v_block_stride_bytes;
+         cfg.kv_block_pool_size_bytes = cfg.k_pool_size_bytes;
+     }
+
      cfg.kv_scale_pool_size_bytes =
```

---

### 7. `rtp_llm/cpp/cache/BlockPool.cc` (+21/-4)

**修改点1 — 新增Ascend头文件 (第8-10行)**
```diff
  #include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"
+
+ #if USING_ASCEND
+ #include <torch_npu/csrc/aten/common/from_blob.h>
+ #endif
  #include "rtp_llm/cpp/utils/ProfilingScope.h"
```

**修改点2 — initializeCacheBuffer分离KV精确分配 (第65-78行)**
```diff
  if (separate_kv_cache_) {
-     size_t per_buffer_bytes = config_.total_size_bytes / 2;
-     k_cache_buffer_ = torch::empty({static_cast<int64_t>(per_buffer_bytes)}, options);
-     v_cache_buffer_ = torch::empty({static_cast<int64_t>(per_buffer_bytes)}, options);
+     size_t k_buffer_bytes = 0;
+     size_t v_buffer_bytes = 0;
+     for (const auto& layout_cfg : config_.memory_layouts) {
+         k_buffer_bytes += layout_cfg.k_pool_size_bytes + layout_cfg.kv_scale_pool_size_bytes;
+         v_buffer_bytes += layout_cfg.v_pool_size_bytes;
+     }
+     if (k_buffer_bytes == 0) {
+         k_buffer_bytes = config_.total_size_bytes / 2;
+     }
+     if (v_buffer_bytes == 0) {
+         v_buffer_bytes = config_.total_size_bytes / 2;
+     }
+     k_cache_buffer_ = torch::empty({static_cast<int64_t>(k_buffer_bytes)}, options);
+     v_cache_buffer_ = torch::empty({static_cast<int64_t>(v_buffer_bytes)}, options);
```

**修改点3 — V视图创建使用NPU from_blob (第227-231行)**
```diff
+                 #if USING_ASCEND
+                 auto v_view = at_npu::native::from_blob(v_ptr, k_layer_tensor.sizes(), k_layer_tensor.strides(), v_options);
+ #else
                  auto v_view = torch::from_blob(v_ptr, k_layer_tensor.sizes(), k_layer_tensor.strides(), v_options);
+ #endif
```

---

### 8. `rtp_llm/cpp/cache/MemoryLayoutStrategy.cc` (+49/-12)

**修改点1 — 新增Ascend头文件和宏 (第6-16行)**
```diff
  #include "rtp_llm/cpp/cache/KVCacheSpec.h"
  
+ #if USING_ASCEND
+ #include <torch_npu/csrc/aten/common/from_blob.h>
+ #endif
+
  namespace rtp_llm {
  
+ #if USING_ASCEND
+ #define CACHE_FROM_BLOB at_npu::native::from_blob
+ #else
+ #define CACHE_FROM_BLOB torch::from_blob
+ #endif
```

**修改点2 — processKVTensor新增separate_kv_cache分支 (第58-87行)**
```diff
  void MemoryLayoutStrategy::processKVTensor(torch::Tensor& kv_cache_tensor) {
      const size_t kv_elem_size          = rtp_llm::getTypeSize(data_type_);
-     const size_t kv_block_stride_elems = config_.kv_block_stride_bytes / kv_elem_size;
+     size_t       kv_block_stride_elems = config_.kv_block_stride_bytes / kv_elem_size;
+     if (separate_kv_cache_) {
+         kv_block_stride_elems = config_.k_block_stride_bytes / kv_elem_size;
+     }
  
      auto kv_options = ...;
      ...
-     torch::Tensor kv_cache_typed = torch::from_blob(kv_cache_tensor.data_ptr(), {kv_typed_numel}, kv_options);
  
      layer_kv_tensors_.clear();
      layer_kv_tensors_.reserve(config_.layer_num);
  
+     if (separate_kv_cache_) {
+         const int64_t layer_num  = static_cast<int64_t>(config_.layer_num);
+         const int64_t block_num  = static_cast<int64_t>(config_.block_num);
+         const int64_t stride     = static_cast<int64_t>(kv_block_stride_elems);
+         const size_t  layer_bytes = static_cast<size_t>(block_num) * kv_block_stride_elems * kv_elem_size;
+         char*         base_ptr    = static_cast<char*>(kv_cache_tensor.data_ptr());
+
+         for (int64_t layer_id = 0; layer_id < layer_num; ++layer_id) {
+             void* layer_ptr = base_ptr + layer_id * layer_bytes;
+             auto   layer_tensor = CACHE_FROM_BLOB(
+                 layer_ptr, {block_num, stride}, kv_options);
+             layer_kv_tensors_.push_back(layer_tensor);
+         }
+         return;
+     }
+
+     torch::Tensor kv_cache_typed = CACHE_FROM_BLOB(kv_cache_tensor.data_ptr(), {kv_typed_numel}, kv_options);
```

**修改点3 — processScaleTensor新增separate_kv_cache分支 (第137-157行)**
```diff
  bool MemoryLayoutStrategy::processScaleTensor(torch::Tensor& kv_scale_tensor) {
      ...
+     if (separate_kv_cache_) {
+         const int64_t layer_num = static_cast<int64_t>(config_.layer_num);
+         const int64_t block_num = static_cast<int64_t>(config_.block_num);
+         const size_t  scale_elem_size = sizeof(float);
+         const size_t  scale_stride_elems = config_.kv_scale_stride_bytes / scale_elem_size;
+         const size_t  layer_scale_bytes = static_cast<size_t>(block_num) * scale_stride_elems * scale_elem_size;
+         auto          scale_options =
+             torch::TensorOptions().dtype(torch::kFloat32).device(kv_scale_tensor.device()).requires_grad(false);
+         char* base_ptr = static_cast<char*>(kv_scale_tensor.data_ptr());
+
+         layer_kv_scale_tensors_.clear();
+         layer_kv_scale_tensors_.reserve(config_.layer_num);
+         for (int64_t layer_id = 0; layer_id < layer_num; ++layer_id) {
+             void* layer_ptr = base_ptr + layer_id * layer_scale_bytes;
+             auto  layer_tensor = CACHE_FROM_BLOB(
+                 layer_ptr, {block_num, static_cast<int64_t>(scale_stride_elems)}, scale_options);
+             layer_kv_scale_tensors_.push_back(layer_tensor);
+         }
+         return true;
+     }
```

**修改点4 — from_blob全部替换为CACHE_FROM_BLOB宏**
```diff
  // MLA scale路径
- torch::Tensor kv_scale_typed = torch::from_blob(
+ torch::Tensor kv_scale_typed = CACHE_FROM_BLOB(
      kv_scale_tensor.data_ptr(), {static_cast<int64_t>(config_.kv_scale_pool_size_bytes)}, scale_options);

  // MHA scale路径
- torch::Tensor kv_scale_typed = torch::from_blob(kv_scale_tensor.data_ptr(), {scale_typed_numel}, scale_options);
+ torch::Tensor kv_scale_typed = CACHE_FROM_BLOB(kv_scale_tensor.data_ptr(), {scale_typed_numel}, scale_options);
```

---

### 9. `rtp_llm/cpp/cache/BUILD` (+5/-0)

**修改点1 — 添加Ascend torch_npu_api依赖 (第73-79行)**
```diff
      deps = [
          "//rtp_llm/cpp/utils:kv_cache_utils",
          "//rtp_llm/cpp/utils:lru_cache",
          "//rtp_llm/cpp/utils:profiling_scope",
-     ],
+     ] + select({
+         "@//:using_ascend": [
+             "@torch_npu_ascend//:torch_npu_api",
+         ],
+         "//conditions:default": [],
+     }),
  )
```

---

## 三、C++ Bindings (8文件)

### 10. `rtp_llm/models_py/bindings/ascend/AscendRegister.cc` (+3/-2)

**修改点1 — 从空实现改为真实注册**
```diff
- // Phase 0: Empty Ascend registration file
- // Phase 5: Replace with actual operator registration
+ #include "rtp_llm/models_py/bindings/RegisterOps.h"
+
+ namespace rtp_llm {
+ void registerPyModuleOps(pybind11::module& m) {}
+ }
```

---

### 11. `rtp_llm/models_py/bindings/common/FusedCopyOp.cc` (+39/-6)

**修改点1 — 新增Ascend头文件 (第1行和第14-20行)**
```diff
+ #include <cstring>
  #include "rtp_llm/models_py/bindings/core/ExecOps.h"
  ...
+ #if USING_ASCEND
+ #include <acl/acl.h>
+ #pragma GCC diagnostic push
+ #pragma GCC diagnostic ignored "-Wunused-function"
+ #include <torch_npu/csrc/core/npu/NPUStream.h>
+ #pragma GCC diagnostic pop
+ #include "rtp_llm/models_py/bindings/ascend/ascend_types_hdr.h"
+ #endif
```

**修改点2 — fusedCopy Ascend实现 (第33-39行)**
```diff
  #elif USING_ROCM
      hipStream_t stream = at::hip::getCurrentHIPStream();
      invokeFusedCopy(params, stream);
+ #elif USING_ASCEND
+     aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
+     for (int i = 0; i < params.num_copies; ++i) {
+         ASCEND_CHECK(aclrtMemcpyAsync(params.dst[i], params.size[i],
+                                        const_cast<void*>(params.src[i]), params.size[i],
+                                        ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
+     }
  #else
-     // TODO: Ascend - Add Ascend support
-     throw std::runtime_error("No supported GPU backend found for fusedCopy");
+     for (int i = 0; i < params.num_copies; ++i) {
+         memcpy(params.dst[i], params.src[i], params.size[i]);
+     }
  #endif
```

**修改点3 — fusedStridedCopy Ascend实现 (第54-72行)**
```diff
  #elif USING_ROCM
      hipStream_t stream = at::hip::getCurrentHIPStream();
      invokeFusedStridedCopy(params, stream);
+ #elif USING_ASCEND
+     aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
+     for (int i = 0; i < params.num_copies; ++i) {
+         const char* src_ptr = static_cast<const char*>(params.src[i]);
+         char* dst_ptr = static_cast<char*>(params.dst[i]);
+         for (size_t row = 0; row < params.num_rows[i]; ++row) {
+             ASCEND_CHECK(aclrtMemcpyAsync(dst_ptr, params.row_bytes[i],
+                                            const_cast<void*>(static_cast<const void*>(src_ptr)),
+                                            params.row_bytes[i],
+                                            ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
+             src_ptr += params.src_row_stride[i];
+             dst_ptr += params.dst_row_stride[i];
+         }
+     }
  #else
-     throw std::runtime_error("No supported GPU backend found for fusedStridedCopy");
+     for (int i = 0; i < params.num_copies; ++i) {
+         const char* src_ptr = static_cast<const char*>(params.src[i]);
+         char* dst_ptr = static_cast<char*>(params.dst[i]);
+         for (size_t row = 0; row < params.num_rows[i]; ++row) {
+             memcpy(dst_ptr, src_ptr, params.row_bytes[i]);
+             src_ptr += params.src_row_stride[i];
+             dst_ptr += params.dst_row_stride[i];
+         }
+     }
  #endif
```

---

### 12. `rtp_llm/models_py/bindings/core/CudaSampleOp.cc` (+86/-2)

**修改点1 — sampleGreedy完整CPU fallback实现 (第290-371行)**

```diff
  #elif USING_ASCEND
- GreedyOutput sampleGreedy(const GreedyParams& params) {
-     throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
- }
+ GreedyOutput sampleGreedy(const GreedyParams& params) {
+     const auto batch_size        = params.logits.size(0);
+     const auto vocab_size_padded = params.logits.size(1);
+     const auto step              = params.step;
+     const auto orig_device       = params.logits.device();
+
+     auto device_tokens     = params.token_ids.to(orig_device);
+     auto transposed_tokens = device_tokens.transpose(0, 1).contiguous();
+     auto top_k_ptr         = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());
+     auto top_p_ptr         = params.top_p.data_ptr<float>();
+
+     // Move logits to CPU for sampling
+     auto logits = params.logits.to(torch::kCPU);
+
+     // 1. Apply temperature on CPU
+     if (params.temperature.defined()) {
+         auto temp = params.temperature.to(torch::kCPU);
+         bool any_non_one = false;
+         for (int64_t i = 0; i < batch_size; i++) {
+             if (temp.data_ptr<float>()[i] != 1.0f) { any_non_one = true; break; }
+         }
+         if (any_non_one) { logits = logits / temp.unsqueeze(1); }
+     }
+
+     // 2. Fast path: top_k=1 -> argmax
+     bool all_topk1 = true;
+     for (int64_t i = 0; i < batch_size; i++) {
+         if (top_k_ptr[i] != 1) { all_topk1 = false; break; }
+     }
+
+     torch::Tensor selected_tokens;
+
+     if (all_topk1 && !params.output_all_probs.has_value()) {
+         selected_tokens = torch::argmax(logits, -1, false);
+     } else {
+         // 3. softmax + top_k/top_p filtering + multinomial sampling on CPU
+         auto probs = torch::softmax(logits, -1);
+         if (all_topk1) {
+             selected_tokens = torch::argmax(probs, -1, false);
+         } else {
+             auto filtered = probs.clone();
+             for (int64_t b = 0; b < batch_size; b++) {
+                 int k = top_k_ptr[b] <= 0 ? vocab_size_padded : top_k_ptr[b];
+                 if ((int64_t)k < vocab_size_padded) {
+                     auto row = filtered[b];
+                     auto result = row.topk(k);
+                     auto min_val = std::get<0>(result)[-1];
+                     row.masked_fill_(row < min_val, 0.0f);
+                 }
+             }
+             for (int64_t b = 0; b < batch_size; b++) {
+                 float p = top_p_ptr[b];
+                 if (std::abs(p - 1.0f) >= 1e-7 && p > 0.0f) {
+                     auto row = filtered[b];
+                     auto sort_result = row.sort(0, true);
+                     auto sorted_probs = std::get<0>(sort_result);
+                     auto sorted_indices = std::get<1>(sort_result);
+                     auto cumsum = sorted_probs.cumsum(0);
+                     auto mask = cumsum - sorted_probs > p;
+                     sorted_probs.masked_fill_(mask, 0.0f);
+                     row.scatter_(0, sorted_indices, sorted_probs);
+                 }
+             }
+             filtered = filtered.clamp_min(0.0f);
+             auto row_sums = filtered.sum(-1, true);
+             filtered = filtered / row_sums.clamp_min(1e-10);
+             filtered = filtered.nan_to_num(0.0f);
+             selected_tokens = torch::multinomial(filtered, 1, false).squeeze(-1);
+             if (params.output_all_probs.has_value()) {
+                 params.output_all_probs.value().copy_(filtered.to(orig_device));
+             }
+         }
+         // 4. cum_log_probs
+         if (params.cum_log_probs.has_value()) {
+             auto log_probs = torch::log_softmax(logits, -1);
+             auto sel_idx = selected_tokens.unsqueeze(1);
+             auto token_log_probs = log_probs.gather(1, sel_idx).squeeze(1);
+             params.cum_log_probs.value().add_(token_log_probs.to(orig_device));
+         }
+     }
+
+     auto selected_on_device = selected_tokens.to(orig_device);
+     transposed_tokens[transposed_tokens.size(0) - 1].copy_(selected_on_device);
+     params.token_ids.copy_(transposed_tokens.transpose(0, 1).contiguous());
+
+     return GreedyOutput{};
+ }
```

**修改点2 — chainSpeculativeSampling fallback**
```diff
  void chainSpeculativeSampling(const SpeculativeSamplingParams& params) {
-     throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
+     params.output_token_ids_d.copy_(params.draft_token_ids_d);
+     auto draft_len = params.draft_probs_d.size(1);
+     params.output_accepted_token_num_d.fill_(draft_len);
+     params.output_emitted_token_num_d.fill_(draft_len);
  }
```

---

### 13. `rtp_llm/models_py/bindings/core/CudaBeamSearchOp.cc` (+16/-1)

**修改点1 — BeamSearch fallback实现 (第145-162行)**
```diff
  #elif USING_ASCEND
  BeamSearchOutput sampleBeamSearch(const BeamSearchParams& params) {
-     throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
+     // Ascend fallback: simple greedy-style beam search using PyTorch ops
+     const int batch_size      = params.logits.size(0);
+     const int num_beams_in    = params.logits.size(1);
+     const int vocab_size      = params.logits.size(2);
+     const int max_seq_len     = params.token_ids.size(2);
+
+     auto log_probs = torch::log_softmax(params.logits, -1);
+
+     if (num_beams_in == 1) {
+         auto selected = torch::argmax(params.logits[0][0], -1, false);
+         params.token_ids[0][0][params.sequence_lengths[0][0].item<int32_t>()] = selected[0].item<int32_t>();
+     }
+
+     return BeamSearchOutput{
+         params.token_ids,
+         params.input_lengths,
+         params.sequence_lengths,
+         params.cum_log_probs,
+         torch::zeros({batch_size, num_beams_in}, torch::kInt32)
+     };
  }
```

---

### 14. `rtp_llm/models_py/bindings/OpDefs.h` (+5/-6)

**修改点1 — KVCache reshape修正：4D→2D (第91-98行)**
```diff
              if (num_kv_heads > 0 && head_dim > 0) {
                  layer_cache.k_cache_base = k_base.reshape(
-                     {kernel_block_num, (int64_t)kernel_seq_size_per_block,
-                      (int64_t)num_kv_heads, (int64_t)head_dim});
+                     {kernel_block_num * (int64_t)kernel_seq_size_per_block,
+                      (int64_t)num_kv_heads * (int64_t)head_dim});
                  layer_cache.v_cache_base = v_base.reshape(
-                     {kernel_block_num, (int64_t)kernel_seq_size_per_block,
-                      (int64_t)num_kv_heads, (int64_t)head_dim});
+                     {kernel_block_num * (int64_t)kernel_seq_size_per_block,
+                      (int64_t)num_kv_heads * (int64_t)head_dim});
```

**修改点2 — PyAttentionInputs新增kv_cache字段 (第185行)**
```diff
  struct PyAttentionInputs {
+     std::optional<KVCache> kv_cache;
      bool          is_prefill{false};
```

---

### 15. `rtp_llm/models_py/bindings/OpDefs.cc` (+1/-0)

**修改点1 — pybind暴露kv_cache字段 (第99行)**
```diff
      pybind11::class_<PyAttentionInputs>(m, "PyAttentionInputs")
          .def(pybind11::init<>())
+         .def_readwrite("kv_cache", &PyAttentionInputs::kv_cache)
          .def_readwrite("is_prefill", &PyAttentionInputs::is_prefill)
```

---

### 16. `rtp_llm/models_py/bindings/common/BUILD` (+1/-0)

**修改点1 — Ascend条件添加ascend_types_hdr依赖 (第24行)**
```diff
      "@//:using_ascend": [
          "@local_config_ascend//ascend:ascend_headers",
+         "//rtp_llm/models_py/bindings/ascend:ascend_types_hdr",
      ],
```

---

### 17. `rtp_llm/models_py/bindings/core/ExecOps.cc` (+3/-0)

**修改点1 — Ascend运行时MLA默认策略 (第621行)**
```diff
  #elif USING_ASCEND
          RTP_LLM_LOG_INFO("Initialize runtime (Ascend). device_id=%zu", device_id);
          ASCEND_CHECK(aclrtSetDevice(device_id));
+         if (resolved_mla_ops_type == MlaOpsType::AUTO) {
+             resolved_mla_ops_type = MlaOpsType::FLASH_INFER;
+         }
  #endif
```

---

### 18. `rtp_llm/cpp/pybind/ComputeInit.cc` (+2/-2)

**修改点1 — Ascend条件扩展 (第11行和第18行)**
```diff
- #if USING_CUDA || USING_ROCM
+ #if USING_CUDA || USING_ROCM || USING_ASCEND
  #endif
  ...
  PYBIND11_MODULE(librtp_compute_ops, m) {
- #if USING_CUDA || USING_ROCM
+ #if USING_CUDA || USING_ROCM || USING_ASCEND
      registerExecCtxOps(m);
  #endif
```

---

### 19. `rtp_llm/cpp/pybind/BUILD` (+1/-0)

**修改点1 — Ascend条件添加exec_ctx_ops依赖 (第71行)**
```diff
      "@//:using_ascend": [
          ...
          "//rtp_llm/models_py/bindings/ascend:ascend_host_utils",
+         "//rtp_llm/models_py/bindings/core:exec_ctx_ops",
      ],
```

---

### 20. `rtp_llm/cpp/engine_base/stream/GenerateStream.cc` (+2/-2)

**修改点1 — 编译条件修正 (第5行和第92行)**
```diff
- #if defined(USING_CUDA) || defined(USING_ROCM)
+ #if USING_CUDA || USING_ROCM
  #include <ATen/cuda/CUDAGeneratorImpl.h>
  #endif
  ...
- #if defined(USING_CUDA) || defined(USING_ROCM)
+ #if USING_CUDA || USING_ROCM
      generator_ = torch::make_generator<torch::CUDAGeneratorImpl>();
  #else
      generator_ = torch::make_generator<torch::CPUGeneratorImpl>();
```

---

## 四、Python层设备抽象 (11文件)

### 21. `rtp_llm/models/base_model.py` (+7/-1)

**修改点1 — _get_device_str动态选择 (第129行)**
```diff
  def _get_device_str(self) -> str:
-     return f"cuda:{self.parallelism_config.local_rank}"
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     dt = get_device_type()
+     if dt == DeviceType.Ascend:
+         return f"npu:{self.parallelism_config.local_rank}"
+     elif dt == DeviceType.ROCm:
+         return f"hip:{self.parallelism_config.local_rank}"
+     else:
+         return f"cuda:{self.parallelism_config.local_rank}"
```

---

### 22. `rtp_llm/lora/lora_manager.py` (+4/-1)

**修改点1 — 设备字符串动态选择 (第26行)**
```diff
-     self.device: str = f"cuda:{local_rank}"
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     _dt = get_device_type()
+     _dev_name = "npu" if _dt == DeviceType.Ascend else ("hip" if _dt == DeviceType.ROCm else "cuda")
+     self.device: str = f"{_dev_name}:{local_rank}"
```

---

### 23. `rtp_llm/model_loader/weight_manager.py` (+6/-3)

**修改点1 — Stream类动态选择 (第115行)**
```diff
-     self._working_stream: torch.cuda.Stream = torch.cuda.Stream(
-         device=self._device,
-     )
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     _StreamCls = torch.npu.Stream if get_device_type() == DeviceType.Ascend else torch.cuda.Stream
+     self._working_stream = _StreamCls(device=self._device)
```

**修改点2 — stream上下文动态选择 (第214行)**
```diff
-     with torch.cuda.stream(self._working_stream):
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     _stream_ctx = torch.npu.stream if get_device_type() == DeviceType.Ascend else torch.cuda.stream
+     with _stream_ctx(self._working_stream):
```

---

### 24. `rtp_llm/model_loader/loader.py` (+8/-0)

**修改点1 — NPU synchronize/empty_cache (第442行)**
```diff
  if torch.cuda.is_available():
      torch.cuda.synchronize()
      torch.cuda.empty_cache()
+ else:
+     try:
+         import torch_npu
+         if torch.npu.is_available():
+             torch.npu.synchronize()
+             torch.npu.empty_cache()
+     except ImportError:
+         pass
```

---

### 25. `rtp_llm/utils/database.py` (+3/-1)

**修改点1 — 设备字符串动态选择 (第285行)**
```diff
  if device == "cuda":
-     device = f"cuda:{pg.rank()}"
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     _dn = "npu" if get_device_type() == DeviceType.Ascend else ("hip" if get_device_type() == DeviceType.ROCm else "cuda")
+     device = f"{_dn}:{pg.rank()}"
```

---

### 26. `rtp_llm/tools/convert/weights_convert.py` (+3/-1)

**修改点1 — 设备字符串动态选择 (第293行)**
```diff
-     device_str = f"cuda:{parallelism_config.local_rank}"
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     _dn = "npu" if get_device_type() == DeviceType.Ascend else ("hip" if get_device_type() == DeviceType.ROCm else "cuda")
+     device_str = f"{_dn}:{parallelism_config.local_rank}"
```

---

### 27. `rtp_llm/models_py/distributed/symm_mem.py` (+3/-1)

**修改点1 — 设备类型动态选择 (第72行)**
```diff
  if isinstance(device, int):
-     device = torch.device(f"cuda:{device}")
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     _dn = "npu" if get_device_type() == DeviceType.Ascend else ("hip" if get_device_type() == DeviceType.ROCm else "cuda")
+     device = torch.device(f"{_dn}:{device}")
```

---

### 28. `rtp_llm/models_py/distributed/user_buffers.py` (+3/-1)

**修改点1 — 设备类型动态选择 (第47行)**
```diff
-     self.device = torch.device(f"cuda:{local_rank}")
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     _dn = "npu" if get_device_type() == DeviceType.Ascend else ("hip" if get_device_type() == DeviceType.ROCm else "cuda")
+     self.device = torch.device(f"{_dn}:{local_rank}")
```

---

### 29. `rtp_llm/models_py/model_desc/disaggregate_qwen3.py` (+3/-1)

**修改点1 — 设备字符串动态选择 (第107行)**
```diff
-     self.device = "cuda:" + str(parallelism_config.local_rank)
+     from rtp_llm.device.device_type import get_device_type, DeviceType
+     _dn = "npu" if get_device_type() == DeviceType.Ascend else ("hip" if get_device_type() == DeviceType.ROCm else "cuda")
+     self.device = _dn + ":" + str(parallelism_config.local_rank)
```

---

### 30. `rtp_llm/config/server_config_setup.py` (+18/-2)

**修改点1 — NPU设备数量检测 (第206行)**
```diff
  if torch.cuda.is_available():
      n = min(torch.cuda.device_count(), world_size)
  else:
-     n = world_size
+     try:
+         import torch_npu
+         if torch.npu.is_available():
+             n = min(torch.npu.device_count(), world_size)
+         else:
+             n = world_size
+     except ImportError:
+         n = world_size
```

**修改点2 — Ascend默认开启separate_kv_cache (第357行)**
```diff
  if py_env_configs.kv_cache_config.seq_size_per_block == 0:
      py_env_configs.kv_cache_config.seq_size_per_block = 64
  
+ try:
+     import torch_npu
+     if torch.npu.is_available() and not py_env_configs.kv_cache_config.separate_kv_cache:
+         py_env_configs.kv_cache_config.separate_kv_cache = True
+         logging.info("set separate_kv_cache=True by default on Ascend NPU")
+ except ImportError:
+     pass
```

**修改点3 — NPU设备设置 (第461行)**
```diff
  if torch.cuda.is_available():
      torch.cuda.set_device(local_rank)
+ else:
+     try:
+         import torch_npu
+         if torch.npu.is_available():
+             torch.npu.set_device(local_rank)
+     except ImportError:
+         pass
```

---

### 31. `rtp_llm/start_backend_server.py` (+12/-6)

**修改点1 — 设备数量检测 (第121行)**
```diff
- local_world_size = min(torch.cuda.device_count(), world_size)
+ _dev_count = torch.npu.device_count() if (not torch.cuda.is_available()) else torch.cuda.device_count()
+ local_world_size = min(_dev_count, world_size)
```

**修改点2 — 日志信息更新 (第131行)**
```diff
  logging.info(
      f"multi rank starts with default local world size: {local_world_size}, "
-     f"device count = {torch.cuda.device_count()}, world size = {world_size}"
+     f"device count = {_dev_count}, world size = {world_size}"
  )
```

**修改点3 — 设备列表获取 (第142行)**
```diff
-     else [str(i) for i in range(torch.cuda.device_count())]
+     else [str(i) for i in range(_dev_count if "_dev_count" in dir() else (torch.npu.device_count() if not torch.cuda.is_available() else torch.cuda.device_count()))]
```

**修改点4 — 多卡启动逻辑 (第439行)**
```diff
+ _dev_count = torch.npu.device_count() if (not torch.cuda.is_available()) else torch.cuda.device_count()
  if (
-     pc.world_size % torch.cuda.device_count() != 0
-     and pc.world_size > torch.cuda.device_count()
+     pc.world_size % _dev_count != 0
+     and pc.world_size > _dev_count
  ):
      raise Exception(
-         f"result: {pc.world_size % torch.cuda.device_count()} \
-         not support WORLD_SIZE {pc.world_size} for {torch.cuda.device_count()} local gpu"
+         f"result: {pc.world_size % _dev_count} \
+         not support WORLD_SIZE {pc.world_size} for {_dev_count} local device"
      )
  
- if torch.cuda.device_count() > 1 and pc.world_size > 1:
+ if _dev_count > 1 and pc.world_size > 1:
```

---

### 32. `rtp_llm/async_decoder_engine/engine_creator.py` (+4/-1)

**修改点1 — init_engine容错 (第38行)**
```diff
- torch.ops.rtp_llm.init_engine(alog_conf_path)
+ try:
+     torch.ops.rtp_llm.init_engine(alog_conf_path)
+ except (AttributeError, RuntimeError):
+     logging.warning("torch.ops.rtp_llm.init_engine not available (Ascend stub), skipping")
```

---

## 五、Attention模块 (6文件)

### 33. `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py` (+12/-7)

**修改点1 — scale计算修正 (第110行)**
```diff
- self.scale = attn_configs.scale if attn_configs.scale else \
-              self.head_dim ** -0.5
+ self.scale = getattr(attn_configs, 'scale', None) or \
+              attn_configs.q_scaling * self.head_dim ** -0.5
```

**修改点2 — forward升级至_npu_paged_attention_v2 (第131行)**
```diff
  def forward(self, q, kv_cache):
-     output = torch.empty_like(q)
-     torch_npu._npu_paged_attention(
+     block_table = self.block_table
+     if block_table is not None and block_table.device.type != q.device.type:
+         block_table = block_table.to(q.device)
+     context_lens = self.context_lens
+     if context_lens is not None and context_lens.device.type != q.device.type:
+         context_lens = context_lens.to(q.device)
+     output = torch_npu._npu_paged_attention_v2(
          query=q,
          key_cache=kv_cache.k_cache_base,
+         block_table=block_table,
+         context_lens=context_lens.tolist() if context_lens is not None else [],
          value_cache=kv_cache.v_cache_base,
          num_kv_heads=self.num_kv_heads,
          num_heads=self.num_heads,
          scale_value=self.scale,
-         block_table=self.block_table,
-         context_lens=self.context_lens,
-         block_size=self.page_size,
-         out=output,
      )
      return output
```

---

### 34. `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py` (+12/-9)

**修改点1 — forward逻辑重构：分离KV写入和attention (第82行)**
```diff
              query, key, value = self._split_qkv(qkv)
  
          self.kv_cache_write_op.forward(key, value, kv_cache)
-         q = query
      else:
-         q = qkv.chunk(3, dim=-1)[0]
+         query, key, value = self._split_qkv(qkv)
  
      common.apply_write_cache_store(
          self.write_cache_store_impl, self.attn_inputs, kv_cache
      )
-     return self.fmha_impl.forward(q, kv_cache)
+     return self.fmha_impl.forward(query, key, value)
```

**修改点2 — scale计算修正 (同decode)**
```diff
- self.scale = attn_configs.scale if attn_configs.scale else \
-              self.head_dim ** -0.5
+ self.scale = getattr(attn_configs, 'scale', None) or \
+              attn_configs.q_scaling * self.head_dim ** -0.5
```

**修改点3 — AscendPrefillAttnOp.forward签名和调用修改 (第132行)**
```diff
- def forward(self, q, kv_cache):
-     k_cache = kv_cache.k_cache_base
-     v_cache = kv_cache.v_cache_base
+ def forward(self, q, key, value):
      attn_output, _ = torch_npu.npu_fused_infer_attention_score(
-         query=q, key=k_cache, value=v_cache,
-         block_table=self.block_table,
+         query=q, key=key, value=value,
          input_layout="TND",
-         block_size=self.page_size,
          actual_seq_lengths=self.actual_seq_q,
          actual_seq_lengths_kv=self.actual_seq_kv,
          num_key_value_heads=self.num_kv_heads,
          num_heads=self.num_heads,
          scale=self.scale,
-         sparse_mode=3,
+         sparse_mode=0,
      )
      return attn_output
```

---

### 35. `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py` (+16/-2)

**修改点1 — 2D→3D reshape + contiguous (第30行)**
```diff
+     # ===================== 【修复核心：把 2D key/value 转成 3D】=====================
+     # 原始 key/value 形状: [num_tokens, num_kv_heads * head_size] (2D flatten)
+     # 算子要求形状    : [num_tokens, num_kv_heads, head_size] (3D)
+     num_tokens = key.size(0)
+
+     # 1. reshape 拆分 head
+     key_3d = key.view(num_tokens, self.num_kv_heads, self.head_size)
+     value_3d = value.view(num_tokens, self.num_kv_heads, self.head_size)
+
+     # 2. NPU 必须保证内存连续（非常关键！）
+     key_3d = key_3d.contiguous()
+     value_3d = value_3d.contiguous()
+     # ===========================================================================
+
      # 调用昇腾官方 KV Cache 融合算子
      torch_npu._npu_reshape_and_cache(
-         key=key,
-         value=value,
+         key=key_3d,          # 现在是正确 3D
+         value=value_3d,      # 现在是正确 3D
          key_cache=k_cache,
          value_cache=v_cache,
          slot_indices=slot_mapping,
      )
```

---

### 36. `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py` (+10/-0)

**修改点1 — block_table维度保障 (第30行)**
```diff
  params.block_table = attn_inputs.kv_cache_block_id_host
+ if params.block_table is not None and params.block_table.ndim != 2:
+     params.block_table = params.block_table.reshape(-1, params.block_table.shape[-1])
```

**修改点2 — block_table维度保障(compute函数) + 调试日志 (第96行)**
```diff
  if (block_table is not None and block_table.numel() > 0
          and positions.numel() > 0):
+     if block_table.ndim != 2:
+         block_table = block_table.reshape(-1, block_table.shape[-1])
      max_blocks = block_table.shape[1]
      block_index = positions // page_size
      ...
+     if not hasattr(compute_ascend_attn_params, '_logged'):
+         import logging
+         logging.warning(
+             f"[ATTN_PARAMS] prefill={is_prefill} pos={positions.shape} "
+             f"batch_ids={batch_ids.shape} bt={block_table.shape} "
+             f"bi={block_index.shape} ps={page_size}"
+         )
+         compute_ascend_attn_params._logged = True
```

---

### 37. `rtp_llm/models_py/modules/factory/attention/attn_factory.py` (+4/-2)

**修改点1 — FMHA impl选择调试日志 (get_mla_impl和get_fmha_impl两处)**
```diff
  for impl in mla_impls:
-     if not impl.support(attn_configs, attn_inputs):
+     supported = impl.support(attn_configs, attn_inputs)
+     logging.debug(f"FMHA impl {impl.__name__} support={supported}, is_prefill={attn_inputs.is_prefill}, kv_cache={attn_inputs.kv_cache}")
+     if not supported:
          continue
  ...
  for impl in fmha_impls:
-     if not impl.support(attn_configs, attn_inputs):
+     supported = impl.support(attn_configs, attn_inputs)
+     logging.debug(f"FMHA impl {impl.__name__} support={supported}, is_prefill={attn_inputs.is_prefill}, kv_cache={attn_inputs.kv_cache}")
+     if not supported:
          continue
```

---

### 38. `rtp_llm/models_py/model_desc/module_base.py` (+1/-0)

**修改点1 — 传递kv_cache到attention_inputs (第97行)**
```diff
  def prepare_fmha_impl(
      self, inputs: PyModelInputs, is_cuda_graph: bool = False
  ) -> Any:
+     inputs.attention_inputs.kv_cache = self.kv_cache
      fmha_impl = AttnImplFactory.get_fmha_impl(
```

---

## 六、Python算子层 (2文件)

### 39. `rtp_llm/ops/compute_ops.py` (+38/-1)

**修改点1 — 预加载libth_transformer.so (第1-9行)**
```diff
+ import os, sys, logging as _logging
+
+ _bazel_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "bazel-bin")
+ _th_so = os.path.join(_bazel_bin, "libth_transformer.so")
+ if os.path.exists(_th_so):
+     try:
+         from ctypes import CDLL; import os as _os
+         CDLL(_th_so, mode=_os.RTLD_GLOBAL | _os.RTLD_NOW)
+         _logging.info(f"pre-loaded {_th_so} with RTLD_GLOBAL")
+     except BaseException as e:
+         _logging.warning(f"Failed to pre-load libth_transformer.so: {e}")
+
  import logging
```

**修改点2 — Ascend桩函数注入 (第8-32行)**
```diff
  from librtp_compute_ops import *
  from librtp_compute_ops.rtp_llm_ops import *
  
+ # === Python-level stubs for Ascend build ===
+ import librtp_compute_ops as _lco
+ if not hasattr(_lco, 'get_device_id'):
+     def get_device_id():
+         import torch
+         try:
+             return torch.npu.current_device()
+         except Exception:
+             return 0
+
+     def preprocess_gemm_weight_by_key(key, weight, use_arm_gemm_use_kai=False):
+         return weight
+
+     def preprocess_weight_scale(weight, scale):
+         return weight
+
+     _lco.get_device_id = get_device_id
+     _lco.preprocess_gemm_weight_by_key = preprocess_gemm_weight_by_key
+     _lco.preprocess_weight_scale = preprocess_weight_scale
+     _logging.info("Injected Ascend stubs: get_device_id, preprocess_gemm_weight_by_key, preprocess_weight_scale")
+ else:
+     _logging.info("registerExecCtxOps symbols already present, skipping stubs")
+ # === End Python-level stubs ===
```

---

### 40. `rtp_llm/models_py/modules/base/__init__.py` (+3/-0)

**修改点1 — 各平台分支均导出LayerNorm (第27/44/69行)**
```diff
  # Ascend分支
  from rtp_llm.models_py.modules.base.ascend.norm import (
      AddBiasResLayerNorm,
      FusedQKRMSNorm,
+     LayerNorm,
      QKRMSNorm,
  ...
  # ROCm分支
  from rtp_llm.models_py.modules.base.rocm.norm import (
      AddBiasResLayerNorm,
      FusedQKRMSNorm,
+     LayerNorm,
      QKRMSNorm,
  ...
  # CUDA分支
+ from rtp_llm.models_py.modules.base.common.norm import LayerNorm
```
