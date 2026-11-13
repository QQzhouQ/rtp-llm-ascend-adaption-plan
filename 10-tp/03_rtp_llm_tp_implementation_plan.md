# RTP-LLM TP 特性实现方案（最终定稿）

> ✅ **已验证通过**：2026-09-03，Qwen3-0.6B，TP=2 HCCL，950PR + CANN 9.2.0
> 基于 `01`~`04` 分析结论，经两轮环境故障排障后收敛为最小改动方案

---

## 0. 最终架构（一句话）

> **rtp-llm 已有的 Megatron 式 TP（切分数学+通信时序+引擎锁步）+ 标准 GPU 式设备管理（ASCEND_RT_VISIBLE_DEVICES）+ HCCL 后端**。
> 总代码改动：**5 个文件，~70 行**（4 个已提交 + 1 个待提交）。

## 1. TP 特性涉及的代码文件（完整清单）

### 已提交（Phase1，commit `0f9a3e787`）

| # | 文件 | 改动 | 作用 |
|---|------|------|------|
| 1 | `rtp_llm/start_backend_server.py` | +12/-4 | 按 `get_device_type()` 选择拉起路径，解锁 NPU 上 `multi_rank_start` |
| 2 | `rtp_llm/server/backend_manager.py` | +5/-1 | 按 `get_device_type()` 选 hccl/nccl 后端 |
| 3 | `rtp_llm/models_py/distributed/collective_torch.py` | +64/-20 | ① backend 断言放开 hccl；② C++ 通信回调(cpp_broadcast/allreduce/allgather)设备无关化(CPU→当前加速器搬运)；③ `_ensure_accel`/`_restore` 助手 |
| 4 | `rtp_llm/cpp/models/ModelTypes.cc` | +12/-4 | pin_memory 防御回退(回调先搬 NPU 再 collective，pin 仅为 host 拷贝优化) |

### 暂存待提交

| # | 文件 | 改动 | 作用 |
|---|------|------|------|
| 5 | `rtp_llm/models/base_model.py` | +8/-1 | `_get_device_str()` 改用 `torch.npu.current_device()`（而非 `local_rank`），确保权重加载设备与 HCCL communicator 设备一致 |

### 未改动但依赖的现有代码（TP 架构资产，Phase1 之前已存在）

| 模块 | 位置 | 说明 |
|------|------|------|
| 权重切分 | `model_loader/model_weight_info.py` + `utils/model_weight.py` | `sp_head`(QKV按头+GQA复制) / `sp_0`(o_proj行切) / `ffn_sp_neg1`+`ffn_sp_0`(FFN列+行) / `sp_0_pad8`(lm_head vocab切) |
| 注意力 TP | `cpp/config/ModelConfig.cc:155-179` | `getAttentionConfigs(tp_size)`: head_num /= tp, kv_head_num /= tp |
| 每层通信 | `modules/hybrid/causal_attention.py:95` + `dense_mlp.py:87` | 每 layer 2 次 all_reduce(TP) + 层首 1 次 embedding all_gather |
| 引擎锁步 | `cpp/models/ModelTypes.cc` (tpSyncModelInputs) + `PyWrappedModel.cc` (logits聚合) | C++ 输入广播 / C++ logits allgather / rank0 采样 |
| 回调桥 | `bindings/core/ExecOps.cc:491-547` | `execBroadcast/execAllReduce/execAllGather` → GIL → Python 回调 |

## 2. 标准启动方式（GPU 式，与 CUDA 完全一致）

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 python -m rtp_llm.start_server \
    --checkpoint_path=/mnt/docker/weights/Qwen3-0.6B \
    --model_type=qwen_3 --tp_size=2 --world_size=2 \
    --start_port=9000 --enable_cuda_graph=false \
    --kv_cache_mem_mb=2048
```

**与 GPU 唯一的区别**：用 `ASCEND_RT_VISIBLE_DEVICES` 替代 `CUDA_VISIBLE_DEVICES`。

## 3. 环境前置条件（排障结论）

| # | 条件 | 详情 |
|---|------|------|
| 1 | CANN ≥ 9.2.0（正式版） | 950PR 上 9.1-beta 的 HCCL 有缺陷（error 4 静默失败） |
| 2 | 驱动含 `topo/` 目录 | `/usr/local/Ascend/driver/topo/950/atlas_350_3.json` 必须存在，缺失时 HCCL 报 EI0016。本机从 Docker overlay 层拷入解决 |
| 3 | `ASCEND_AICPU_PATH` 指向当前 CANN | 升级 CANN 后需更新，否则 HCCL 加载旧版 AICPU 内核（报 `HcclDpuTaskexpShmemRestore` not found, error 15） |
| 4 | `unset` 旧的 `RTP_LLM_DIST_BACKEND` | （已废弃）gloo 过渡方案的环境变量，不再需要 |

## 4. 排障记录（两次环境故障 + 一次代码 bug）

### 故障 1: HCCL RootInfoDetect 失败（error 4）
- **现象**: `hcclCommInitRootInfoConfig` 失败，CANN 9.1-beta 上静默挂死
- **根因**: 容器缺 `/usr/local/Ascend/driver/topo/` 目录（驱动裁剪安装）
- **修复**: 从 Docker overlay 层拷入拓扑 JSON 文件
- **误诊**: 曾归因于 hostname 解析/fork-spawn/IFNAME/device_id——均排除

### 故障 2: HCCL kernel not found（error 15）
- **现象**: 拓扑文件修复后，HCCL communicator 建到一半报 `HcclDpuTaskexpShmemRestore` 找不到
- **根因**: 环境变量 `ASCEND_AICPU_PATH` 指向旧版 CANN 9.1-beta，runtime 按它加载 AICPU 内核，旧版没有这个新内核
- **修复**: `export ASCEND_AICPU_PATH=/usr/local/Ascend/cann` 并 source 新版 set_env.sh

### Bug 3: 设备号不一致（HcclAllGather 崩）
- **现象**: HCCL communicator 建在物理设备 4 上（set_device(4)），但权重加载到 `npu:0`（base_model 用 local_rank 构造设备号）
- **根因**: `base_model.py:_get_device_str()` 用 `local_rank` 直接当设备号，未走 `torch.npu.current_device()`
- **修复**: 改用 `torch.npu.current_device()`（方案无关，VISIBLE_DEVICES 和 offset 模式都对）

### 已废弃的方案

| 方案 | 状态 | 原因 |
|------|------|------|
| gloo 后端过渡 | ❌ 已删除 | HCCL 打通后不需要；环境变量 `RTP_LLM_DIST_BACKEND` 同步移除 |
| `RTP_LLM_NPU_DEV_OFFSET` 物理卡号偏移 | ❌ 已回退 | VISIBLE_DEVICES 方案验证通过后不需要；架构上与 GPU 不一致 |
| RankTable 绕行（HcclCommInitClusterInfo） | ❌ 未实施 | 环境修复后 RootInfoDetect 正常，无需绕行 |

## 5. 验证结果（2026-09-01）

```
环境: 950PR × 2, CANN 9.2.0, torch_npu 2.9.0.post3
配置: TP=2, world_size=2, ASCEND_RT_VISIBLE_DEVICES=4,5
模型: Qwen3-0.6B

✅ 双 rank 启动: "Rank 0/1 started successfully"
✅ HCCL 后端: "initialize process_group ... backend: hccl"
✅ 设备绑定: "use device: npu:0" (逻辑0→物理4)
✅ KVCache allgather: "block_num is N after tp sync"
✅ 引擎就绪: "engine created successfully"
✅ 推理输出: 正常生成（与 TP=1 一致）
```

## 6. 后续优化方向（Phase 2，可选）

| 优化 | 预期收益 | 参考 |
|------|---------|------|
| HCCL 环境调优 (`HCCL_INTRA_PCIE_ENABLE=1` 等) | 通信性能 | vllm-ascend `utils.py` |
| 融合 GEMM+AllReduce (`npu_mm_all_reduce_base`) | 消除独立 AR kernel | vllm-ascend `linear_op.py:392` |
| 序列并行 FlashComm1 (prefill 长序列) | prefill 通信量减半 | vllm-ascend `linear_op.py:500` |
| 融合 AR+add+RMSNorm AscendC kernel | 每层省一次内存往返 | vllm-ascend `csrc/mc2/` |

## 7. 与原方案(`03`文档)的差异总结

| 原方案 P1 计划 | 实际落地 |
|---------------|---------|
| 修 4 个阻塞点 | ✅ 全部完成（其中 1 个是 base_model 设备号 bug，原方案未预见到） |
| ~3 天，5 文件 ~200 行 | 实际 5 文件 ~70 行（更少，因为 Phase1 时部分已有代码已兼容） |
| gloo 过渡 | ✅ 曾实施并验证 TP 正确性 → HCCL 打通后删除 |
| offset 物理卡号 | 曾实施 → VISIBLE_DEVICES 验证通过后回退 |
| **最终结论** | **GPU 式设备管理 + 标准 HCCL，仅需一行设备号修复（base_model.py）** |
