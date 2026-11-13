# rtp-llm Ascend MTP Phase 1 + Phase 2 代码改动详细总结

> 时间：2026-09-24 ~ 2026-09-28
> 基线：`feature/qwen35-graph-mode`（方案C 图模式 + PR#54 FIA 寻址统一）
> 最终改动：**15 文件修改 + 1 新测试文件，+711/-75 行**（纯功能代码，调试代码已全部清理）
> 最终性能：**11-12ms/token**（无 MTP eager ~70 → 6× 加速；无 MTP 图模式 16 → 1.4× 提升）
> 接受率：2.63（k=2 含 bonus）

---

## 〇、改动总览矩阵

| # | 文件 | Phase | 内容摘要 |
|---|---|---|---|
| 1 | `bindings/core/CudaSampleOp.cc` | P1 | Ascend `chainSpeculativeSampling` 纯 torch 实现（+69） |
| 2 | `bindings/ascend/ops/tests/SamplerPybind.cc` | P1 | `chain_speculative_sampling` 测试绑定（+43） |
| 3 | `bindings/ascend/ops/tests/test_chain_speculative_sampling.py` | P1 | **新增**双层单测（CPU 8 例 + NPU 2 例） |
| 4 | `models/PyWrappedModel.h` | P1+P2 | sp 时 draft eager 降级 / target 创建 verify 图（H2 修正） |
| 5 | `models/PyWrappedModel.cc` | P1+P2 | `graph_runner_` 空指针守卫；物理组表透出（+28） |
| 6 | `bindings/OpDefs.h` / `.cc` | P1 | PyAttentionInputs 新增物理组表字段 + pybind（+12） |
| 7 | `model_desc/block_map.py` | P1 | 每层选组同步切换物理表（+15） |
| 8 | `model_desc/qwen3_next.py` | P1 | 8 处 GDN 调用 kernel→物理表（+16/-8） |
| 9 | `ascend_graph/ascend_graph_runner.cc` | P1+P2 | verify 图管道 / 物理组表 capture+replay / seq+1 修复 / capture 前缀封顶 / 兼容拷贝（+99） |
| 10 | `ascend_graph/ascend_graph_utils.h` | P2 | MemHold 补物理组表拷贝（+7，根因六修复） |
| 11 | `kernels/ascend/paged_state.py` | P2 | 新增 `decode_state_indices_multi`（+38） |
| 12 | `kernels/ascend/recurrent.py` | P1+P2 | T>1 逐步 T==1 launch → 设备元数据路径（+136/-40） |
| 13 | `kernels/ascend/causal_conv1d.py` | P1+P2 | `initial_state_mode` 修复 + 设备多 token 路径（+106/-20） |
| 14 | `ascend_impl/ascend_prefill.py` | P1+P2 | FIA straddle 视图修复 + verify 设备元数据 + dispatch-mode 图（+192/-30） |

---

## 一、Phase 1：MTP eager 端到端跑通（4 个根因修复）

### 1.1 断点解锁：chainSpeculativeSampling 实现

**文件**：`bindings/core/CudaSampleOp.cc`（+69 行）

原为 `throw OpException(ERROR_UNIMPLEMENTED)`——Ascend 上 MTP 的唯一硬断点。替换为纯 torch 向量化实现，语义精确镜像 ROCm HIP 核（`cpp/rocm/speculative_sampling/sampling.cu`）：

```cpp
// 链式接受：cumprod 前缀积 = HIP 核逐 token break
auto accept = (uniform_samples_d.slice(1,0,k) * p_sel) < q_sel;
auto accept_prefix = accept.to(torch::kFloat).cumprod(1).to(torch::kBool);
auto pos = accept_prefix.sum(1);
// 残差重采样：逆 CDF = 首个 cdf > u2 的下标
auto sampled = (cdf <= u2).sum(-1).clamp_max(V-1);
params.output_token_ids_d.copy_(output);
params.output_emitted_token_num_d += (pos + 1).to(torch::kInt32);
```

**测试**：`test_chain_speculative_sampling.py` 新增——CPU 层 8 例（全接受/首拒/中拒/k=1/边界 u*p==q/残差全零/混合 batch/200 fuzz）+ NPU 层 2 例（真实 binding 对拍 V=1519）。

### 1.2 根因一：GDN 状态页号误用 kernel 块 id → 物理表化（6 文件）

**根因**：混合 KV 池的 kernel 表值 = `phys × bpk + half`（值域 [0,2N)），GDN 状态池第一维 = 物理块数 N——用 kernel id 当状态页号越界/错页 → 跨请求全局污染。

**改动**：
| 文件 | 改动 |
|---|---|
| `OpDefs.h` | 新增 `kv_cache_block_id_{host,device}_by_group` 字段 |
| `OpDefs.cc` | pybind 注册两字段 |
| `PyWrappedModel.cc` | `setupKVCacheForAttentionInputs` 填充物理组表（3D hybrid / 2D fallback） |
| `block_map.py` | `select_block_map_for_layer` 同步切换 `kv_cache_block_id_device/host` |
| `qwen3_next.py` | **8 处** GDN 调用 `kv_cache_kernel_block_id_device` → `kv_cache_block_id_device`（decode conv/fla、prefill conv/load/store、CP ×3） |
| `ascend_graph_runner.cc` | initCapture 建物理组表 buffer + prepareInputs 刷新 |

### 1.3 根因二：fla-npu 多 token 快照中间页错误 → 逐步 T==1 launch

**文件**：`kernels/ascend/recurrent.py`

fla-npu `npu_recurrent_gated_delta_rule` 单次多 token launch 的中间页 BF16 快照写错（kernel pending flush 时序 bug）。改为逐步复用已验证的 T==1 语义：每 token 先 seed write[t] from 前一状态页，再 T==1 in-place 更新。

### 1.4 根因三：conv 多 token 不读初态 → `initial_state_mode`

**文件**：`kernels/ascend/causal_conv1d.py`

老接口 `npu_causal_conv1d` 不传 `initial_state_mode` 时 conv_states 被当纯输出（窗口历史从零重启）。修复：`initial_state_mode=[1]*batch` + `x[B,1,D]` rank-3 + `run_mode=0`（UPDATE 与 host 初态互斥）。

### 1.5 根因四：prefill FIA split 视图 → straddle 视图

**文件**：`ascend_impl/ascend_prefill.py`（`AscendPrefillAttnOp.forward`）

**根因**：池布局是 kernel 交错（每 kernel 块 [K512|V512]），旧 `split_kv_kernel_blocks` 按物理 [K1024|V1024] 重建视图 → **V 被映射到 K 后半**，读 stale（新池零值 → FIA 隐性失效；请求跨 512 token 后变真垃圾 → 全局累积污染）。

```python
# 修复：镜像已验证的 decode 路径（AscendDecodeAttnOp._split_kv）
base = kv_cache.kv_cache_base
k_cache = base[:, 0].reshape(base.shape[0], base.shape[2], -1)
v_cache = base[:, 1].reshape(base.shape[0], base.shape[2], -1)
```

### 1.6 保护性改动：图模式互斥

**文件**：`PyWrappedModel.h` + `.cc`

- sp 开启时 draft 保持 eager（`else if (sp && model_id)` 分支 + warning）
- `graph_runner_ != nullptr` 空指针守卫

---

## 二、Phase 2：MTP 图模式（target-verify 入图）

### 2.1 Runner 管道（`ascend_graph_runner.cc` + `PyWrappedModel.h`）

| 改动 | 位置 | 目的 |
|---|---|---|
| H2 修正 | `PyWrappedModel.h` | `num_tokens_per_bs` 按 model_id 守卫对齐 CUDA 决策表：target verify = gen+1，draft/normal = 1 |
| verify 图 capture | `ascend_graph_runner.cc` | 5 档位 bs（1/8/16/24/32）× 3 token = [3/24/48/72/96] capture |
| 物理组表 per-bucket slice | `prepareCaptureInputs` | by_group 物理表按档位 slice（与 kernel 表同款） |
| capture 前缀封顶 | `initCaptureAttentionInputs` | `min(max_seq_len - T, 2048 - T - 8)`——避免 2048×2048 mask 溢出 warmup |
| **seq+1 verify 修复**（根因五） | `prepareInputs` | verify 的 `sequence_lengths_plus_1_d` 从 `prefix_lengths+1` 计算（原从空的 sequence_lengths 计算 → 保持捕获封顶值 → GDN 页号全错） |
| layer_to_group 兼容 | `prepareInputs` | engine 42 层 vs 捕获 40 层——双向 min 截断 |
| sequence_lengths 空守卫 | `prepareInputs` | verify 清空了 sequence_lengths → numel>0 才拷贝 |
| **MemHold 补物理组表**（根因六） | `ascend_graph_utils.h` | MemHold 构造函数逐字段拷贝漏了物理 by_group → 捕获 buffer 空 → 图内 GDN 页号全 0（pad 槽）→ 所有 GDN 层错 |

### 2.2 GDN T>1 设备元数据路径（`paged_state.py` + `recurrent.py` + `causal_conv1d.py`）

将 eager-only 的 host 元数据多 token 路径替换为全 device 算子（aclgraph 可捕获）：

```python
# paged_state.py — 新增
def decode_state_indices_multi(block_map, seq_lens_plus_1, block_size, token_count):
    """read col = (len-2)//block; write col[t] = (len-1)//block + t — 全 device gather"""
    read_idx = block_map.gather(1, read_col...)
    write_idxs = [block_map.gather(1, (write_base+t)...) for t in range(T)]

# recurrent.py — 多 token 分发到设备路径
def _fused_recurrent_multi_device(...):
    for t in range(T):
        seed_state_segment(state, src, write_idxs[t])     # device 页迁移
        npu_recurrent_gated_delta_rule(
            q[:, t]..., asl=cached_常量, ssm_state_indices=write_idxs[t]...  # device 索引
        )

# causal_conv1d.py — 同款
def _causal_conv1d_update_multi_device(...):
    for t in range(T):
        seed_state_segment(conv_state, src, write_idxs[t])
        npu_causal_conv1d_update(x[:,:,t].contiguous(), ..., conv_state_indices=write_idxs[t]...)
```

### 2.3 FIA verify 设备元数据 + dispatch-mode 图（`ascend_prefill.py`）

**设备元数据路径**（`_update_verify_params_device`）——替代 host 计算，图内 gather/cumsum 重算链自动从刷新 buffer 重取值：

```python
def _update_verify_params_device(self, kv_cache, kernel_page):
    positions = prefix_d.repeat_interleave(T) + ar.repeat(batch)  # device
    slot = gather(kernel_block_table, block_index) * kernel_page + offset  # device
    self.params.positions_d = positions       # rope 用
    self.params.slot_mapping = slot           # KV 写用
    self.fmha_impl.actual_seq_q = cumsum(input_d)   # FIA 用（device）
    self.fmha_impl.actual_seq_kv = prefix_d + input_d
    self.fmha_impl.block_table = kernel_block_table
```

**dispatch-mode FIA 图**（`_forward_fia_graph`）——使用 torch_npu 官方 `_GraphDispatchMode`（`graph.update` 底层，内置 IFA v2 handler）：

```python
mode = AscendPrefillAttnOp._dispatch_modes.setdefault(bs, _GraphDispatchMode())
with mode:
    result = torch_npu.npu_fused_infer_attention_score_v2(...)  # .default 调用
    return result[0] if isinstance(result, (list, tuple)) else result
```

- 捕获时：dispatch mode 拦截调用 → 自动预分配 workspace/out → ExternalEvent 等待点 + graph task group → 记录调用
- 每次 replay 前（`prepare_cuda_graph` → `update_graph_fia_verify`）：`mode.update_capture_record([{"actual_seq_kvlen": kvlen}])` 在 update stream 上重 launch 刷新 kvlen

**`prepare_cuda_graph`**（replay 前回调）：
```python
def prepare_cuda_graph(self, attn_inputs, graph_bs=None):
    # 从 prefix_lengths 算 kvlen（= prefix + T），pad 到档位 bs
    kvlen = (prefix[:valid] + tps).tolist()
    kvlen = kvlen[:bucket] + [tps] * max(0, bucket - len(kvlen))
    self.fmha_impl.update_graph_fia_verify(kvlen, bucket)
```

**`signal_graph_events`**（capture 后 sanity replay 释放 ExternalEvent 等待点防死锁）。

---

## 三、6 个根因的因果链总结

| # | 根因 | 症状 | 修复 |
|---|---|---|---|
| 1 | GDN 状态页号用 kernel id | 跨请求全局污染 | 物理表化（6 文件） |
| 2 | fla-npu 多 token 快照中间页错 | 请求内 accept 崩塌 | 逐步 T==1 launch |
| 3 | conv 不读初态 | 轻度累积劣化 | `initial_state_mode=[1]*batch` |
| 4 | FIA split 视图 V 错映射 | 隐性基线退化 + 全局累积污染 | straddle 零拷贝视图 |
| 5 | verify 图 seq+1 从空 sequence_lengths 算 | 图 replay GDN 页号保持封顶值 | 从 prefix_lengths+1 计算 |
| 6 | MemHold 漏物理组表拷贝 | 图 replay GDN 块表全 0 | MemHold 补两行拷贝 |

根因 4 是 Phase 1 最终根因（隐性基线退化从首个请求就存在）；根因 5/6 是 Phase 2 图 replay 的两个独立 bug。

---

## 四、最终验证数据

| 指标 | 数值 |
|---|---|
| 性能 | **11.3ms/token**（512t: 5.8s） |
| 接受率 | **2.63**（k=2 含 bonus） |
| 正确性 | greedy 恢复真实模型行为（中文，与图模式验证文档一致）；700t 长生成无退化；多请求 ×3 稳定；高温采样正常 |
| 性能对比 | vs 无 MTP eager ~70ms → **6.2×**；vs 无 MTP 图模式 16ms → **1.4×**；vs MTP eager 35ms → **3.1×** |

---

## 五、启动命令

```bash
cd /home/d30033799/weishan/rtp-llm
export ASCEND_CUSTOM_OPP_PATH=$PWD/bazel-bin/3rdparty/aclnn_custom_ops/opp
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/bazel-bin/3rdparty/aclnn_custom_ops/lib"
python3 -m rtp_llm.start_server \
    --checkpoint_path=/mnt/docker/weights/Qwen3.5-35B-A3B \
    --model_type=qwen35_moe --start_port=9000 \
    --enable_cuda_graph=true \
    --kv_cache_mem_mb=8192 \
    --seq_size_per_block=1024 --kernel_seq_size_per_block=512 \
    --sp_type=mtp --sp_model_type=qwen35_moe_mtp --gen_num_per_cycle=2 \
    --sp_checkpoint_path=/mnt/docker/weights/Qwen3.5-35B-A3B \
    --enable_detail_log=true
```

注意：
- `LD_LIBRARY_PATH` 必须含 `libcust_opapi.so` 路径（top-p 采样必需）
- `gen_num_per_cycle ≤ 7`（GDN 算子 8 token 上限），推荐 2~3
- 不带图模式时去掉 `--enable_cuda_graph`（自动降级 eager MTP，~35ms/token）
