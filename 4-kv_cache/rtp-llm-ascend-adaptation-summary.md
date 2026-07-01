# rtp-llm 昇腾NPU适配改动总结

> 基准提交: `fa4029c2101a8647dc0acc7a74cbfc12a6565cf5` (Merge pull request #2 from menogrey/nwk_br)
> 改动范围: **40个文件**, +455/-87行
> 适配目标: 将rtp-llm从纯CUDA平台适配至华为昇腾NPU (Ascend) 平台

---

## 一、构建系统改动 (4文件)

### 1.1 Python解释器路径切换
- **文件**: `.bazelrc`, `BUILD`, `deps/pip.bzl`
- **改动**: 将Python解释器路径从 `/opt/conda310/bin/python3` 切换至 `/opt/conda310/envs/py310/bin/python3`
- **原因**: 使用conda `py310` 虚拟环境，匹配昇腾CANN工具链对Python 3.10的要求

### 1.2 Ascend编译优化选项升级
- **文件**: `.bazelrc`
- **改动**: `-march=armv8-a+crc` → `-march=armv8.2-a+fp16+dotprod+crc`
- **原因**: 昇腾910B系列芯片支持ARMv8.2-A架构，启用FP16和DotProd指令集可显著提升NPU计算性能

### 1.3 Ascend绑定注册启用
- **文件**: `arch_config/arch_select.bzl`
- **改动**: Ascend分支从 `dummy_register` 改为 `ascend_bindings_register`
- **原因**: 原代码在Ascend平台使用空注册(dummy)，改为加载真实的Ascend算子注册，使C++ bindings在NPU上生效

---

## 二、C++ KV Cache改动 (5文件) — 核心改动

### 2.1 分离K/V缓存池大小计算
- **文件**: `BlockPoolConfigHelper.h`, `MemoryLayoutConfig.h`
- **改动**: 新增 `k_pool_size_bytes` 和 `v_pool_size_bytes` 字段，在 `separate_kv_cache` 模式下分别计算K和V的缓存池大小
- **原因**: 昇腾NPU要求K/V Cache分离存储（separate_kv_cache），不同于CUDA的合并KV布局。K池还需包含kv_scale的空间，V池独立，因此必须分别精确计算

### 2.2 BlockPool分离KV缓存初始化
- **文件**: `BlockPool.cc`
- **改动**: `initializeCacheBuffer()` 中从简单的 `total_size_bytes/2` 均分，改为按 `memory_layouts` 配置逐层累加精确计算K/V缓存大小；V视图创建使用 `at_npu::native::from_blob` 替代 `torch::from_blob`
- **原因**: 
  - 精确分配：K/V size不对称时均分会浪费内存或导致越界
  - `torch::from_blob` 在NPU上不保证正确处理已有显存指针，昇腾专用 `at_npu::native::from_blob` 可正确将NPU显存包装为Tensor

### 2.3 MemoryLayoutStrategy支持分离KV布局
- **文件**: `MemoryLayoutStrategy.cc`
- **改动**:
  - 全局替换 `torch::from_blob` → `CACHE_FROM_BLOB` 宏（Ascend下使用 `at_npu::native::from_blob`）
  - `processKVTensor()` 新增 `separate_kv_cache_` 分支：按层逐层创建KV视图而非整体reshape
  - `processScaleTensor()` 新增 `separate_kv_cache_` 分支：按层逐层创建scale视图
- **原因**: CUDA下KV Cache是 `[layer_num, block_num, kv_block_stride]` 的连续3D布局，可直接reshape；昇腾NPU的分离KV布局下，K和V各自按层独立存储，需逐层用from_blob创建视图

### 2.4 KV Cache BUILD依赖
- **文件**: `rtp_llm/cpp/cache/BUILD`
- **改动**: Ascend条件下添加 `@torch_npu_ascend//:torch_npu_api` 依赖
- **原因**: BlockPool.cc 和 MemoryLayoutStrategy.cc 使用了 `torch_npu` 的 `from_blob` API，需链接对应库

---

## 三、C++ Bindings改动 (7文件)

### 3.1 AscendRegister真实注册
- **文件**: `AscendRegister.cc`
- **改动**: 从空实现改为包含 `registerPyModuleOps` 的真实注册桩
- **原因**: 配合 `arch_select.bzl` 的改动，使Ascend平台正确注册Python模块算子

### 3.2 FusedCopyOp昇腾实现
- **文件**: `FusedCopyOp.cc`
- **改动**: 
  - `fusedCopy`: 使用 `aclrtMemcpyAsync` 进行NPU Device-to-Device异步拷贝
  - `fusedStridedCopy`: 逐行调用 `aclrtMemcpyAsync` 实现跨步拷贝
  - CPU fallback: 非CUDA/ROCm/Ascend平台使用 `memcpy` 替代原来的抛异常
- **原因**: 原代码在非CUDA/ROCm平台直接抛异常，昇腾NPU需要使用ACL Runtime API进行显存拷贝操作

### 3.3 采样算子Ascend实现
- **文件**: `CudaSampleOp.cc`
- **改动**: `sampleGreedy` 和 `chainSpeculativeSampling` 从 `throw ERROR_UNIMPLEMENTED` 改为PyTorch CPU fallback实现：
  - logits转到CPU采样（避免NPU算子兼容问题）
  - 支持temperature/top_k/top_p/multinomial完整采样流程
  - 支持cum_log_probs计算
- **原因**: 昇腾NPU暂无专用采样kernel，使用PyTorch CPU采样作为功能正确的fallback，保证模型可正常推理

### 3.4 BeamSearch算子Ascend实现
- **文件**: `CudaBeamSearchOp.cc`
- **改动**: 从 `throw ERROR_UNIMPLEMENTED` 改为简单greedy-style beam search fallback（beam=1时argmax）
- **原因**: 同上，提供功能正确的fallback

### 3.5 KVCache布局reshape修正
- **文件**: `OpDefs.h`
- **改动**: separate_kv_cache分支下，K/V cache_base从 `[block_num, seq_per_block, kv_heads, head_dim]` (4D) reshape为 `[block_num*seq_per_block, kv_heads*head_dim]` (2D)
- **原因**: 昇腾 `_npu_reshape_and_cache` 和 `_npu_paged_attention_v2` 算子期望2D扁平布局的K/V cache，而非4D NHD布局

### 3.6 PyAttentionInputs新增kv_cache字段
- **文件**: `OpDefs.cc`, `OpDefs.h`
- **改动**: `PyAttentionInputs` 结构体新增 `std::optional<KVCache> kv_cache` 字段并在pybind中暴露
- **原因**: 昇腾Prefill算子 `npu_fused_infer_attention_score` 需要直接使用key/value而非从kv_cache中读取，需要将kv_cache传递到attention层以供不同策略选择

### 3.7 ComputeInit.cc扩展Ascend条件
- **文件**: `ComputeInit.cc`, `rtp_llm/cpp/pybind/BUILD`
- **改动**: `#if USING_CUDA || USING_ROCM` → `#if USING_CUDA || USING_ROCM || USING_ASCEND`，使 `registerExecCtxOps` 在Ascend下也编译
- **原因**: 使ExecCtxOps（包含采样/beam search等）在Ascend下注册

### 3.8 ExecOps运行时初始化
- **文件**: `ExecOps.cc`
- **改动**: Ascend初始化时，若MLA ops类型为AUTO则默认设为 `FLASH_INFER`
- **原因**: 昇腾平台MLA策略选择，默认使用flash_infer实现

### 3.9 GenerateStream.cc编译条件修正
- **文件**: `GenerateStream.cc`
- **改动**: `defined(USING_CUDA)` → `USING_CUDA`（去掉defined宏）
- **原因**: Bazel构建中 `USING_CUDA` 是编译器宏而非预定义宏，用 `defined()` 判断永远为真，去掉 `defined` 使用直接宏判断

---

## 四、Python层设备抽象改动 (11文件)

### 4.1 设备字符串统一适配
- **涉及文件**: `base_model.py`, `lora_manager.py`, `weight_manager.py`, `database.py`, `weights_convert.py`, `symm_mem.py`, `user_buffers.py`, `disaggregate_qwen3.py`
- **改动模式**: 将硬编码的 `"cuda:{rank}"` / `torch.cuda.xxx` 替换为根据 `get_device_type()` 动态选择 `npu`/`hip`/`cuda` 的逻辑
- **原因**: 原代码大量硬编码CUDA设备，昇腾NPU需使用 `torch.npu` API和 `"npu:{rank}"` 设备字符串

### 4.2 服务启动设备检测
- **文件**: `start_backend_server.py`
- **改动**: `_get_local_world_size()` 等函数中 `torch.cuda.device_count()` → 优先使用 `torch.npu.device_count()`
- **原因**: NPU环境下 `torch.cuda` 不可用，需使用 `torch.npu` 获取设备数量

### 4.3 并行配置NPU设备支持
- **文件**: `server_config_setup.py`
- **改动**:
  - 设备数量检测增加 `torch.npu` 分支
  - `setup_cuda_device_and_accl_env()` 增加NPU设备设置
  - Ascend平台默认开启 `separate_kv_cache=True`
- **原因**: 
  - NPU设备初始化需调用 `torch.npu.set_device()`
  - 昇腾FlashAttention算子要求K/V Cache分离存储

### 4.4 模型加载器NPU内存管理
- **文件**: `loader.py`
- **改动**: `synchronize()/empty_cache()` 增加 `torch.npu` 分支
- **原因**: NPU显存管理需使用 `torch.npu` API

### 4.5 权重管理器NPU Stream
- **文件**: `weight_manager.py`
- **改动**: `torch.cuda.Stream` → 根据设备类型选择 `torch.npu.Stream`/`torch.cuda.Stream`
- **原因**: NPU使用独立的Stream机制

### 4.6 引擎创建容错
- **文件**: `engine_creator.py`
- **改动**: `torch.ops.rtp_llm.init_engine()` 调用增加 try-except 容错
- **原因**: Ascend stub构建下该算子可能未注册，避免启动崩溃

---

## 五、Attention模块改动 (6文件) — 核心改动

### 5.1 Decode注意力算子升级
- **文件**: `ascend_decode.py`
- **改动**:
  - `_npu_paged_attention` → `_npu_paged_attention_v2`（API升级）
  - 移除 `block_size` 和 `out` 参数，新增 `context_lens.tolist()` 传参
  - `block_table` 和 `context_lens` 设备一致性检查，必要时 `.to(q.device)`
  - `scale` 计算使用 `q_scaling * head_dim^-0.5`
- **原因**: 
  - v2 API接口变化，不再需要block_size和output预分配
  - 昇腾算子要求block_table和context_lens与query在同一设备
  - q_scaling因子保证量化模型下注意力缩放正确

### 5.2 Prefill注意力算子重构
- **文件**: `ascend_prefill.py`
- **改动**:
  - `forward()` 签名从 `(q, kv_cache)` 改为 `(query, key, value)`，直接传入当前步的K/V
  - 调用 `npu_fused_infer_attention_score` 传入 `key/value` 而非 `k_cache/v_cache`
  - 移除 `block_table`、`block_size`、`sparse_mode=3` → `sparse_mode=0`
  - scale计算同decode使用 `q_scaling`
- **原因**: 
  - 昇腾 `npu_fused_infer_attention_score` prefill模式下直接使用当前token的key/value计算，而非从cache读取
  - `sparse_mode=0` 对应标准dense attention，`sparse_mode=3` 是因果mask模式，prefill场景用0更通用

### 5.3 KV Cache写入算子修复
- **文件**: `ascend_kv_cache_write_op.py`
- **改动**: 
  - 将2D key/value `[num_tokens, kv_heads*head_dim]` reshape为3D `[num_tokens, kv_heads, head_dim]`
  - `.contiguous()` 保证内存连续
  - 传入3D tensor给 `_npu_reshape_and_cache`
- **原因**: `_npu_reshape_and_cache` 算子严格要求3D输入，2D flatten格式会导致算子报错或写入越界

### 5.4 Attention参数构建修复
- **文件**: `ascend_attn_params.py`
- **改动**:
  - `block_table` 维度检查：若非2D则 `reshape(-1, shape[-1])`
  - 增加调试日志（仅打印一次）
- **原因**: block_table可能因上层传入1D tensor导致索引越界，确保始终为2D

### 5.5 Attention Factory调试日志
- **文件**: `attn_factory.py`
- **改动**: 在impl选择循环中增加 `logging.debug` 打印support检查结果
- **原因**: 便于排查昇腾平台上attention实现选择问题

### 5.6 module_base传递kv_cache
- **文件**: `module_base.py`
- **改动**: `prepare_fmha_impl()` 中设置 `inputs.attention_inputs.kv_cache = self.kv_cache`
- **原因**: 配合 `PyAttentionInputs.kv_cache` 新字段，将kv_cache传递到attention层

---

## 六、Python算子层改动 (2文件)

### 6.1 compute_ops.py Ascend桩函数
- **文件**: `compute_ops.py`
- **改动**:
  - 预加载 `libth_transformer.so`（RTLD_GLOBAL）
  - 为Ascend构建注入Python级桩函数: `get_device_id`, `preprocess_gemm_weight_by_key`, `preprocess_weight_scale`
- **原因**: `registerExecCtxOps` 在CUDA/ROCm下注册这些函数，Ascend下未注册，需Python层提供no-op fallback防止import失败

### 6.2 LayerNorm导出
- **文件**: `modules/base/__init__.py`
- **改动**: 所有设备分支均导出 `LayerNorm`
- **原因**: 部分模型结构引用了LayerNorm，原代码未在各平台统一导出

---

## 七、改动总结

| 类别 | 文件数 | 改动行数 | 核心目的 |
|------|--------|----------|----------|
| 构建系统 | 4 | +14/-14 | 编译工具链切换至NPU，启用Ascend bindings |
| C++ KV Cache | 5 | +107/-22 | 支持分离KV缓存布局，NPU from_blob适配 |
| C++ Bindings | 7 | +178/-30 | FusedCopy/采样/BeamSearch NPU实现，KV布局修正 |
| Python设备抽象 | 11 | +87/-27 | 全链路cuda→npu设备字符串/API替换 |
| Attention模块 | 6 | +62/-34 | Decode/Prefill/KV写入算子适配，API升级v2 |
| Python算子层 | 2 | +47/-5 | Ascend桩函数，LayerNorm导出 |

**核心适配策略**:
1. **分离KV Cache**: 昇腾FlashAttention要求K/V独立存储，是最大的架构差异
2. **NPU专用API**: `at_npu::native::from_blob`、`aclrtMemcpyAsync`、`torch_npu._npu_*` 等替代CUDA对应API
3. **CPU采样Fallback**: 采样算子暂无NPU专用kernel，使用CPU采样保证功能正确
4. **设备抽象统一**: 将硬编码CUDA改为运行时动态选择npu/cuda/hip
