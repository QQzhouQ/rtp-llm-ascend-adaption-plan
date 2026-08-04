# 图模式原理与流程介绍

> **Workspace:** c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm

---

## User
图模式原理、流程介绍

## Assistant
我来搜索一下 rtp-llm 项目中关于图模式(Graph Mode)的相关实现。

找到了关键的图模式相关文件,让我详细查看这些核心文件。

让我查看实现文件和工具类。

让我查看其余的实现和工具类。

让我查看 runner.cc 中被截断的中间部分,以及 prefill 实现和 device_shims。

让我查看 capture 和 replay 的关键实现部分。

让我查看 captureDecode 和 capturePrefill 的具体实现,以及 prefill.cc。

让我查看 decode 的具体实现。

下面基于代码详细介绍 rtp-llm 中**图模式(CUDA Graph Mode)**的原理与流程。

## 一、核心原理

图模式本质是利用 **CUDA Graph** 技术:把一次完整 forward 涉及的所有 GPU kernel 一次性"录制"成一张计算图,之后每次推理直接整体 `replay()`,从而消除逐个 kernel 的 **CPU launch 开销**,并允许驱动做全局优化(融合、流水线)。

这对 decode 阶段尤其关键——decode 每 step 的计算量小,kernel 数量多,CPU launch 开销占比极高,图模式收益显著。

核心抽象见 [cuda_graph_base.h](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_base.h):
- `GraphBase`:抽象基类,定义 `initCapture / forward / canRun` 接口
- `CudaGraphRunner`:核心实现类,位于 [cuda_graph_runner.h](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_runner.h)

## 二、两种模式与配置表

代码里通过 `is_prefill_cuda_graph_mode_` + `num_tokens_per_bs_` 区分场景,见 [cuda_graph_runner.cc](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_runner.cc#L9-L26) 顶部配置表:

| 模型类型 | is_prefill_cuda_graph_mode_ | num_tokens_per_bs_ | 用途 |
|---|---|---|---|
| Draft Model(prefill) | true | gen_num_per_cycle+1 | 投机采样草稿模型 |
| Target Model(score/prefill) | false | gen_num_per_cycle+1 | 投机采样目标校验 |
| Draft Model(decode) | false | 1 | 草稿模型解码 |
| Embedding Model(prefill) | true | max_seq_len | embedding 模型 |
| Normal Model(decode) | false | 1 | 普通模型解码 |

## 三、完整流程

整体分为 **Capture(捕获)** 与 **Replay(重放)** 两大阶段。

### 阶段 1:Capture(初始化时一次性录制)

入口 `initCapture()`(L583 起)流程:

1. **确定捕获范围 `capture_range_`**
   - Decode:默认捕获 `[1, 8, 16, 24, 32, 48, 64, ...]` 这些 batch size,见 [cuda_graph_decode.cc](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_decode.cc#L6-L35) 中 `getDecodeBatchSizesToCapture`
   - Prefill:按 `num_tokens_per_bs_` 倍数或 Python 传入的 `prefill_capture_seq_lens_` 捕获,见 [cuda_graph_prefill.cc](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_prefill.cc#L94-L116)

2. **预分配固定 buffer**(`CaptureMemoryHold`):为 `max_bs_` 下所有输入张量(input_ids、attention_inputs、kv_cache_block_id 等)预分配,捕获后地址固定不变,见 [cuda_graph_utils.h](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_utils.h#L22-L68) 中的 `CaptureMemoryHold`

3. **逐个 key 捕获图实例**(`captureOneGraphInstance`,L661 起):
   - WarmUp 两次(触发 lazy 初始化、选好 kernel)
   - 切换到 capture stream(`CudaGraphStreamLife`)
   - `graphCaptureBegin` → 执行 `py_forward_method_` → `graph.capture_end()`,把整次 forward 录制成 `at::cuda::CUDAGraph`,存入 `graph_instances_[key].graph_`
   - replay 验证一次(`replayAndSyncCheck`)确保正确

每个 key 对应一个 `GraphInstance`(at::cuda::CUDAGraph + CaptureMemoryHold),用 `std::unordered_map<int, GraphInstance>` 管理。

### 阶段 2:Replay(每次推理时重放)

入口 `forward()`(L313 起):

1. **`canRun` 判断**(L388 起):用 `std::lower_bound` 在 `capture_range_` 中找 ≥ 当前 batch/seq_len 的最小已捕获 key。若超出最大捕获值则 fallback 到普通执行。

2. **`prepareInputs`**(L56 起):**关键设计**——图模式不能改变输入张量地址,所以必须把真实输入**拷贝到捕获时固定的 buffer 里**:
   - D2D(设备→设备):`input_ids`、`input_lengths_d`、`cu_seqlens`、`kv_cache_kernel_block_id_device` 等,用融合 kernel `fusedCopy` 一次启动
   - H2H(主机→主机):`cu_seqlens_host`、`sequence_lengths` 等 pinned memory 拷贝
   - 对 hybrid kv cache 的 per-group block table 也分别拷贝
   - 调用 Python 侧 `attn_pyobj.prepare_cuda_graph()` 让 attention impl 准备好图内需要的参数

3. **`replayGraph(key)`**:单行 `graph_instances_[key].graph_.replay()` 整体重放整张图。

4. **输出切片**:`hidden_states` 从固定输出 buffer 中 `slice(0, 0, seq_len_sum)` 取实际部分。

5. **`forward_event_.record()`**:记录完成事件,供下一次 prepare 前 synchronize 保证流水正确。

## 四、关键数据结构

| 结构 | 位置 | 作用 |
|---|---|---|
| `GraphParams` | [cuda_graph_base.h#L16-L30](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_base.h#L16-L30) | 启动配置(是否启用、模式、max_seq_len、捕获范围等) |
| `CudaGraphState` | [cuda_graph_base.h#L10-L15](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_base.h#L10-L15) | 运行时当前 batch/seq_len 与选中的 graph key |
| `CaptureMemoryHold` | [cuda_graph_utils.h#L22](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_utils.h#L22) | 捕获时固定的输入/输出 buffer 持有者 |
| `GraphInstance` | [cuda_graph_utils.h#L71](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_utils.h#L71) | 一个 key 对应的 `CUDAGraph + CaptureMemoryHold` |
| `CudaGraphCaptureGuard` | [cuda_graph_utils.h#L110](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_utils.h#L110) | RAII 进入/退出捕获上下文 |

## 五、设计要点总结

1. **静态地址 + 动态拷贝**:图要求张量地址不变,所以预分配最大 buffer,每次 replay 前把真实数据拷贝进去(这是图模式的核心约束)。
2. **batch size 向上对齐**:实际 batch=20 会 replay batch=24 的图,多出的部分通过 fill_(0) 屏蔽,保证正确性。
3. **多后端兼容**:通过 [cuda_graph_device_shims.h](file:///c:\WorkingSpace\Code\rtp-llm-nwk\rtp-llm\rtp_llm\cpp\cuda_graph\cuda_graph_device_shims.h) 的宏抽象,同一套逻辑兼容 CUDA / ROCm / Ascend。
4. **Debug 模式**:`enable_cuda_graph_debug_mode_` 时启用 `graph.enable_debug_mode()` 并 dump `.dot` 可视化文件。

如果你想深入某个具体环节(如 prepareInputs 的融合拷贝、hybrid kv cache 处理、投机采样下的 target verify 路径),可以告诉我。

