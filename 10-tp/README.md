# rtp-llm TP 特性调研与实现文档集

围绕「rtp-llm Ascend TP 特性开发」的完整调研、排障与落地记录。

## 阅读顺序

| 文档 | 内容 |
|------|------|
| `01_vllm_ascend_dense_tp_analysis.md` | vllm-ascend dense TP 剖析：默认=标准 Megatron TP + HCCL；融合优化全部 opt-in |
| `02_rtp_llm_tp_status_and_gaps.md` | rtp-llm 现状：TP 架构已完整，Ascend 上仅工程阻塞点 |
| `03_rtp_llm_tp_implementation_plan.md` | **最终定稿**：方案、代码文件清单、排障记录、验证结果 |

> `04` 文档已合并入 `03` 最终定稿，不再单独维护。

## 一句话结论

> **rtp-llm TP = 已有 Megatron 架构 + GPU 式设备管理(ASCEND_RT_VISIBLE_DEVICES) + HCCL 后端，总改动 5 文件 ~70 行。** 排障过程发现两起环境故障（驱动缺拓扑文件、CANN 环境变量残留）和一行代码 bug（base_model 设备号），均已修复并验证。

## TP 特性涉及的全部代码文件

| # | 文件 | 状态 |
|---|------|------|
| 1 | `rtp_llm/start_backend_server.py` | 已提交(Phase1) |
| 2 | `rtp_llm/server/backend_manager.py` | 已提交(Phase1) |
| 3 | `rtp_llm/models_py/distributed/collective_torch.py` | 已提交(Phase1) |
| 4 | `rtp_llm/cpp/models/ModelTypes.cc` | 已提交(Phase1) |
| 5 | `rtp_llm/models/base_model.py` | 暂存待提交 |
