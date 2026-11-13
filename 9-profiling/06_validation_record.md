# RTP-LLM Ascend Profiling 调测记录与验证报告

> 本文档记录 `/home/d30033799/prof/05_rtp_llm_profiling_scheme.md` 推荐方案的**端到端落地调测过程**，包括编译验证、启动踩坑、触发方式、产物校验，供后续复现与运维参考。

---

## 0. 总结论：方案已端到端验证通过

| 环节 | 状态 |
|------|------|
| 代码实现 | ✅ 6 个文件改动（见 `05`） |
| 编译 | ✅ `bazel build //rtp_llm:rtp_llm --config=ascend` 通过，`libth_transformer.so` 链接成功，wheel 产出 |
| 触发 | ✅ 经 OpenAI 接口 `extra_configs.gen_timeline` 触发 `StepWindowProfiler` → `TorchProfile` Ascend 分支 → `torch_npu.profiler` |
| 采集 | ✅ host（aten/engine scope）+ device（aclnn kernel）两侧均采到 |
| 产物 | ✅ 标准 `ascend_pt` dump，`trace_view.json`（12 万事件）+ 全套 CSV 自动生成 |

---

## 1. 编译验证

### 命令
```bash
bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend \
    --test_output=errors --test_env="LOG_LEVEL=INFO"
```

### 结果
```
[11,202 / 11,328] Compiling rtp_llm/cpp/engine_base/TorchProfiler.cc
[11,273 / 11,328] Compiling rtp_llm/models_py/bindings/core/ExecOps.cc
[11,324 / 11,328] Linking libth_transformer.so
INFO: Build completed successfully, 64 total actions
```
我改动的两个 C++ 文件（`TorchProfiler.cc`、`ExecOps.cc`）在 `--config=ascend`（`USING_ASCEND=1`）下均编译通过，CUDA 路径零影响。

### 编译期独立验证（针对 USING_ASCEND 分支）
因单独构建 `//rtp_llm/cpp/engine_base:profiler` 时会被预先存在的 `3rdparty/aclnn_custom_ops` genrule 缺头文件（`profiling/prof_api.h`）阻塞（环境问题，与本改动无关，已用未改动的 `worker_status_info` 目标复现同一错误），额外做了：
- Ascend 路径（`-DUSING_ASCEND=1` + pybind11 + GIL + `unique_ptr` 不完整类型）：**独立 `-fsyntax-only` 通过**（exit 0）
- 非 Ascend 路径（空 stub）：**通过**（exit 0）
- Python 模块语法 + 非 ascend 可导入（惰性 import torch_npu）：**通过**

---

## 2. 服务启动与 OOM 踩坑

### 最终可用启动命令
```bash
python -m rtp_llm.start_server \
    --checkpoint_path=/mnt/docker/weights/Qwen3-0.6B \
    --model_type=qwen_3 \
    --start_port=9000 --enable_cuda_graph=false \
    --torch_cuda_profiler_dir=/home/d30033799/prof \
    --kv_cache_mem_mb=2048
```

### 踩坑 1：NPU OOM（显存不足）

**现象**：不带 `--kv_cache_mem_mb` 启动后，发请求即 OOM：
```
NPU out of memory. Tried to allocate 8.07 GiB (NPU 0; 123.02 GiB total capacity;
118.26 GiB already allocated; 3.85 GiB free)
  ascend_decode.py(247): _ensure_workspace
  ascend_decode.py(327): _forward_fia
```

**根因**（关键，非直觉）：
1. `MemoryEvaluationHelper::getKVCacheMemorySize`（`rtp_llm/cpp/cache/MemoryEvaluationHelper.cc:156`）默认把 KV cache 算到几乎占满整张卡（123 GiB 卡，KV pool 吃了 ~113 GiB）
2. 而 `ascend_decode.py:246` 的 `_ensure_workspace` 大小是 `max_ws + 2 * blocks * page_size * HD`，**`blocks` 就是 KV pool 的总块数** —— 所以 **KV cache 越大、FIA workspace 需求越大**（正反馈）
3. KV pool 撑满后，FIA workspace 要 8 GiB 时只剩 3.85 GiB free → OOM

**解决**：`--kv_cache_mem_mb=2048` 显式限小 KV cache（0.6B 测试模型 2 GiB 足够）。KV pool 缩小 → `blocks` 减少 → workspace 随之变小 → 留足 headroom。**调测 profiling 时务必带上此参数，避免无关 OOM 干扰。**

### 踩坑 2：误以为「走到了图模式」

**现象**：用户疑惑「没开图模式（`enable_cuda_graph=false`），怎么走到了图模式里」。

**澄清**：**没有走图模式**。错误栈是 `_forward_fia`（eager 路径，`ascend_decode.py:327`），不是 `_forward_fia_graph`。图模式对应的调用是 `graph_runner_->forward`，而此处走的是 `py_model_.attr("forward")` 正常前向。`--enable_cuda_graph=false` 完全生效。

> 经验：看到 `ascend_decode.py` 不要误判为图模式，它是正常的 eager decode attention 实现。

---

## 3. 触发方式（关键：profiling 不会自动启动）

`--torch_cuda_profiler_dir` **只设置输出目录，不启动采集**。必须显式触发。三种方式：

### 方式 A（推荐，调测最方便）：OpenAI 接口 + `extra_configs.gen_timeline`

OpenAI `/v1/chat/completions` 通过请求体的 `extra_configs` 字段透传 `GenerateConfig`（`OpenaiEndpoint.cc:48` 读 `req.extra_configs`，其中含 `gen_timeline`/`profile_step`/`profile_trace_name`，经 `QueryConverter.cc:41` 入流）：

```bash
curl -s http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen_3",
        "messages": [{"role": "user", "content": "东莞有啥特色饮食？"}],
        "max_tokens": 10,
        "extra_configs": {
            "gen_timeline": true,
            "profile_step": 3,
            "profile_trace_name": "test_npu_prof"
        }
    }'
```
- `gen_timeline: true` → 触发 `NormalEngine.cc:471` 的 `StepWindowProfiler`
- `profile_step: 3` → 采集 3 个 decode step（`max_tokens` 需 ≥ `profile_step`）
- `profile_trace_name` → trace 文件名前缀

### 方式 B：gRPC `StartProfile` RPC
```
StartProfile{trace_name, start_step, num_steps, enable_all_rank}
```
handler 在 `LocalRpcServer.cc:451`，支持 TP-group 广播。需 grpcurl / python gRPC 客户端。

### 方式 C：`--gen_timeline_sync`
服务级开关，每个请求都同步 profile（适合批量调测，开销大）。

### 触发成功的日志标志
```
ascend torch_npu profiler started: dir=/home/d30033799/prof prefix=test_npu_prof_wr0_...
```

---

## 4. 产物校验（实测 dump）

### dump 目录结构（标准 torch_npu `ascend_pt`）
```
test_npu_prof_wr0__<pid>_<ts>_ascend_pt/
├── ASCEND_PROFILER_OUTPUT/      # 已自动 analyse（analyse.done 标记存在）
│   ├── trace_view.json          # 22 MB，Chrome Tracing，主 timeline
│   ├── kernel_details.csv       # 1.0 MB，逐 kernel 详情
│   ├── operator_details.csv     # 1.3 MB，逐算子详情
│   ├── op_statistic.csv         # 算子统计 + 耗时占比
│   ├── api_statistic.csv
│   ├── step_trace_time.csv      # step 级耗时
│   ├── task_time.csv
│   └── analyse.done
├── PROF_000001_<ts>_<pid>XXX/   # 原始 msprof 数据
│   ├── device_0/                # 设备侧
│   ├── host/                    # host 侧
│   └── msprof_<ts>.db           # SQLite 数据库
├── FRAMEWORK/
│   ├── torch.op_mark            # torch 算子标注
│   └── torch.op_range           # torch 算子区间
├── logs/                        # 各 parser 日志（全部成功）
├── profiler_info.json
└── profiler_metadata.json
```

### 关键产物实测数据
| 检查项 | 实测结果 |
|--------|----------|
| `trace_view.json` | 有效 JSON，**120,103 个事件** |
| 事件 phase 分布 | `X` 完整事件 86,914 / `f`+`s` 流 16,556 / `M` 元数据 35 / `C` 计数 42 |
| FIA 注意力算子 | `FusedInferAttentionScore` **504 事件、84 次执行** ✅ |
| `op_statistic.csv` | MatMulV3(339) / Slice(336) / Mul(1434) / Cast(856) / Add(678) / ReduceMean(339)... |
| `kernel_details.csv` | 真实 aclnn kernel（Embedding/Cast/Square/ReduceMean），含 Task ID/Stream/Duration/Input Shapes/Dtypes |
| `step_trace_time.csv` | Computing 24.2ms / Free 60.3ms / Stage 84.6ms |

### 采集深度（三层都采到）
- **设备侧 kernel**：`aclnnEmbedding_EmbeddingAiCore...`、`aclnnInplaceCopy_CastAiCore...`（AI_CORE / AI_VECTOR_CORE）
- **PyTorch 算子**：`aten::empty`、`aten::zero_`、`aten::empty_strided`
- **rtp-llm 引擎 scope**：`executor.gather_model_input`、`executor.tp_sync_input`、`executor.kv_cache_update`（来自 C++ `RTP_LLM_PROFILE_SCOPE`）

> 自动分析已随 `stop()` 完成（`analyse.done`），**无需再手动跑 `torch_npu.profiler.profiler.analyse()`**。

---

## 5. 产物查看方式

| 工具 | 操作 |
|------|------|
| **MindStudio Insight** | 打开 `ASCEND_PROFILER_OUTPUT/trace_view.json`（Ascend 官方推荐，展示最完整） |
| **chrome://tracing** | 浏览器地址栏输入，加载同一文件 |
| **Perfetto** | https://ui.perfetto.dev ，拖入 trace_view.json |

---

## 6. 调测避坑清单（Checklist）

| # | 坑 | 规避 |
|---|-----|------|
| 1 | KV cache 默认占满显存 → FIA workspace OOM | 启动必带 `--kv_cache_mem_mb=2048`（小模型测试） |
| 2 | 以为 `--torch_cuda_profiler_dir` 会自动采集 | 必须显式触发（`gen_timeline` / gRPC StartProfile） |
| 3 | OpenAI 接口怎么传 `gen_timeline` | 放在请求体 `extra_configs` 字段内 |
| 4 | 看到 `ascend_decode.py` 误判为图模式 | 那是 eager decode，非图模式；图模式是 `graph_runner_->forward` |
| 5 | 单独 `bazel build //rtp_llm/cpp/engine_base:profiler` 失败 | 是预先存在的 `aclnn_custom_ops` 缺 `profiling/prof_api.h` 环境问题；用完整 `//rtp_llm:rtp_llm` 构建可绕过 |
| 6 | `profile_step` 与 `max_tokens` 关系 | `profile_step` 是采集步数，需 `max_tokens ≥ profile_step` 才能采够 |
| 7 | 多卡 trace 文件区分 | 文件名含 `wr<rank>` 前缀（`StepWindowProfiler` 的 `world_rank_`），每 rank 各一份 |

---

## 7. 完整可复现命令序列

```bash
# 1. 编译
bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend

# 2. 启动服务（带 KV cache 限制 + profiling 输出目录）
python -m rtp_llm.start_server \
    --checkpoint_path=/mnt/docker/weights/Qwen3-0.6B \
    --model_type=qwen_3 \
    --start_port=9000 --enable_cuda_graph=false \
    --torch_cuda_profiler_dir=/home/d30033799/prof \
    --kv_cache_mem_mb=2048

# 3. 另开终端，发带 profiling 触发的推理请求
curl -s http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen_3",
        "messages": [{"role": "user", "content": "东莞有啥特色饮食？"}],
        "max_tokens": 10,
        "extra_configs": {
            "gen_timeline": true,
            "profile_step": 3,
            "profile_trace_name": "test_npu_prof"
        }
    }'

# 4. 查看日志确认：ascend torch_npu profiler started: dir=.../prof prefix=test_npu_prof_...
# 5. 产物在 /home/d30033799/prof/test_npu_prof_wr0_*_ascend_pt/ASCEND_PROFILER_OUTPUT/
# 6. 用 MindStudio Insight / chrome://tracing 打开 trace_view.json
```

---

## 8. 结论

推荐方案（`05`）已在真实 NPU 环境验证通过：
- **改动小**：6 文件，CUDA 零影响，零新依赖
- **触发灵活**：复用已有 gRPC `StartProfile` + per-req `gen_timeline` + `gen_timeline_sync`
- **采集完整**：host（aten + 引擎 scope）+ device（aclnn kernel）全覆盖
- **产物标准**：与 vllm-ascend / Ascend 官方工具链（MindStudio Insight）完全对齐
- **唯一调测注意**：启动务必带 `--kv_cache_mem_mb` 避免 KV pool 撑爆显存
