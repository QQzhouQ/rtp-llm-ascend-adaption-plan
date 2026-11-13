# RTP-LLM Ascend Profiling 分析文档集

本目录为「rtp-llm 是否可复用 vllm-ascend / xllm 的 torch_npu profiling 采集方案」的完整分析。

## 阅读顺序

| 文档 | 内容 |
|------|------|
| `01_vllm_ascend_profiling_analysis.md` | vllm-ascend 基于 torch_npu 的 profiling 方案剖析（配方 + 调用链 + 耦合点） |
| `02_rtp_llm_current_profiling_state.md` | rtp-llm 现状：架构、torch_npu 依赖、已有 StepWindowProfiler 框架、Ascend 缺口 |
| `03_xllm_profiling_analysis.md` | xllm 方案剖析（结论：非 torch_npu 路径，死代码，不可复用） |
| `04_reuse_assessment.md` | 复用性评估总表 + 最终判断 |
| `05_rtp_llm_profiling_scheme.md` | **最终推荐方案**（vllm-ascend 配方 + rtp-llm 已有 C++ 框架，含代码草图与改动清单） |
| `06_validation_record.md` | **调测记录与验证报告**（编译/启动/触发/产物校验全过程 + 避坑清单 + 可复现命令） |
| `07_fia_contiguous_analysis.md` | **aclnnContiguous 问题分析**（FIA 算子引入实锤 + 隔离单测方法 + trace 证据 + 优化方向） |

## 一句话结论

> **vllm-ascend 的 `torch_npu.profiler.profile(...)` 调用配方可直接复用；套到 rtp-llm 已有的 C++ `StepWindowProfiler`（窗口控制）+ gRPC `StartProfile` + per-req `gen_timeline`（触发链）上。** 改动集中在 2 个文件（1 新增 Python 模块 + 1 修改 `TorchProfiler.{h,cc}`），CUDA 路径零影响、零新依赖、多卡与异步落盘全部复用。xllm 方案无需借鉴（rtp-llm 已有更完整的等价框架，且 xllm 的 Ascend 采集非 torch_npu 路径且为死代码）。
