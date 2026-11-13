# SGLang Decode 阶段 GDN（线性注意力）实现调研

> 调研时间：2026-09-14 ~ 2026-09-15
> 调研仓库（本地）：
> - `/home/d30033799/sglang`（SGLang 主仓，Python 部分）
> - `/home/d30033799/sgl-kernel-npu`（华为 NPU 算子仓，csrc/ 下有 AscendC 算子，python/ 下有 triton 实现）
> 调研动机：rtp-llm 图模式（ACL Graph）叠加 Qwen3.5 时，GDN decode 算子（fla_npu 的 AscendC 实现）的元数据是 host 侧 int 数组，capture 中做 D2H 同步被 ACL 禁止，**"SGLang 里 GDN decode 没有入图"的说法需要核实**，并寻找可借鉴的解法。

---

## 一、一句话结论

**SGLang 的 GDN decode 是直接入图的（整图 capture），不存在"GDN 层切到图外执行"的机制。**它能入图依赖三层设计：

1. **算子层**：GDN decode 的 conv / recurrent 内核（sgl-kernel-npu 的 triton-ascend 版和 AscendC 版）全部把 `cache_indices`、`cu_seqlens/query_start_loc`、`ssm_state_indices/h0_indices` 实现为 **device tensor**，kernel 内通过 `tl.load` / `GetValue` 做 device 侧间接寻址，**没有任何 host 指针依赖**；
2. **Backend 层**：`MambaAttnBackendBase` 在 `init_cuda_graph_state` 为每个捕获 batch size 预分配**固定地址**的 device 元数据 buffer（`state_indices_list` / `query_start_loc_list`），replay 前在图外用 device `copy_` 原地刷新内容（padding 逻辑在 CPU 上图外算完再拷入），padding 槽位用 `-1`（`pad_slot_id`）标记、kernel 内直接跳过；
3. **Graph Runner 层**：`NPUGraphRunner` 整图 capture（`torch.npu.graph(..., auto_dispatch_capture=True)`）。对仍然需要 host 输入的 full-attention Ascend 算子（`npu_fused_infer_attention_score` 的 `actual_seq_lengths_kv` 是 host list），用 torch_npu 的 **`graph.update(cpu_update_input=...)`** 机制在 replay 时更新，并用独立线程与 `graph.replay()` 并行以隐藏开销。**GDN 部分不依赖该机制。**

---

## 二、背景：Qwen3.5 / Qwen3-Next 的混合注意力结构

Qwen3.5-35B-A3B（及 Qwen3-Next 80B）是混合注意力模型：40 层里 30 层是**线性注意力（Gated DeltaNet，GDN）**、10 层是全注意力（full attention）。decode 阶段：

- **全注意力层**：读 KV cache，走 `npu_fused_infer_attention_score`（FIA）算子；
- **GDN 层**：没有传统 KV cache，靠两个随序列演进的**状态**：
  - `conv_state`：深度卷积的滑动窗口状态 `[slots, dim, state_len]`（state_len = conv_kernel - 1）
  - `ssm_state`（temporal）：递归状态 `[slots, num_v_heads, head_k_dim, head_v_dim]`
  - 每步 decode = 读状态 → 单步更新（conv 窗口滚动 + delta rule 状态更新）→ 写回状态

SGLang 中两者统一由 `HybridLinearAttnBackend` 调度：full-attention 用 `AscendAttnBackend`，线性注意力用 `GDNAttnBackend`。

**GDN 能否入图的关键**：状态是"按序列寻址"的——每个请求对应状态池的一个 slot（`cache_indices`），这一映射在连续批处理下每个 step 都会变化。如果 kernel 只接受 host 侧 int 数组来描述这个映射，capture 就无法重放（replay 时映射变了但 host 参数已固化）——这正是 rtp-llm 用 fla_npu AscendC 算子时遇到的问题。

---

## 三、SGLang 的 GDN decode 调用链

### 3.1 模型层：Qwen3GatedDeltaNet

`/home/d30033799/sglang/python/sglang/srt/models/qwen3_next.py`

Qwen3.5 与 Qwen3-Next 共用这个模型文件（`model_runner.py` 依据 `Qwen3NextConfig` 判定 hybrid_gdn 架构）。decode 路径：

```
Qwen3GatedDeltaNet.forward (qwen3_next.py:391-473)
  ├─ is_cuda_graph=True 时用 triton fused kernel 拆 qkvz/ba（:403-411）
  ├─ 把 conv 权重、a/b/A_log/dt_bias 等打包进 kwargs
  └─ forward_batch.attn_backend.forward(...) (:448-455)
        ↓
HybridLinearAttnBackend.forward_dispatch → GDNAttnBackend.forward_decode
```

### 3.2 Backend 层：GDNAttnBackend.forward_decode

`/home/d30033799/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:524-596`

```python
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
    mixed_qkv = kwargs["mixed_qkv"]; conv_weights = kwargs["conv_weights"]; ...
    layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
    conv_states = layer_cache.conv[0]           # [slots, dim, state_len] 状态池
    ssm_states = layer_cache.temporal           # [slots, HV, K, V] 状态池
    query_start_loc = self.forward_metadata.query_start_loc    # device int32
    cache_indices = self.forward_metadata.mamba_cache_indices  # device int32 ←关键

    mixed_qkv = causal_conv1d_update(
        mixed_qkv, conv_states, conv_weights, bias, activation,
        conv_state_indices=cache_indices,          # ← device tensor
    )
    ...
    core_attn_out = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log, dt_bias=dt_bias, q=query, k=key, v=value, a=a, b=b,
        initial_state_source=ssm_states,
        initial_state_indices=cache_indices,       # ← device tensor
        cu_seqlens=query_start_loc,                # ← device tensor
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0, softplus_threshold=20.0,
    )
    return core_attn_out
```

**注意 `cache_indices` 是"每序列一个状态槽"的单索引语义**（不是块表）：来自 `req_to_token_pool.get_mamba_indices(req_pool_indices)`（`memory_pool.py:402-403`，是 device 索引操作 `self.req_index_to_mamba_index_mapping[req_indices]`）。SGLang 的 GDN 状态池**不分页**（每个请求固定一个槽，状态大小与序列长度无关），比 rtp-llm 的"状态进 KV cache 分页块、每 1024 token 换页"简单得多。

### 3.3 算子层：NPU 上的 import 绑定

`hybrid_linear_attn_backend.py:38-57`：

```python
if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn as ...
elif is_npu():
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu)
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu, causal_conv1d_update_npu)

    chunk_gated_delta_rule = chunk_gated_delta_rule_npu            # prefill
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu  # decode
    causal_conv1d_fn = causal_conv1d_fn_npu                        # prefill
    causal_conv1d_update = causal_conv1d_update_npu                # decode
```

即 NPU 上 GDN 的 **prefill（chunk）/ decode（recurrent）/ conv 全部换成 sgl_kernel_npu 包的实现**（triton-ascend 内核 + torch 组合），不用 fla_npu。

---

## 四、算子层详解：device 元数据是入图的根本

### 4.1 recurrent：`fused_sigmoid_gating_delta_rule_update_npu`（triton kernel）

`/home/d30033799/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_sigmoid_gating_recurrent.py`

签名（:170）：

```python
def fused_sigmoid_gating_delta_rule_update_npu(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b,
    initial_state_source: torch.Tensor,    # ssm 状态池
    initial_state_indices: torch.Tensor,   # device int32: 每序列的状态槽号
    scale=None, use_qk_l2norm_in_kernel=False,
    cu_seqlens=None, is_kda=False,
)
```

kernel 内部（:95-168）——**所有元数据都在 kernel 里用 `tl.load` 读 device 内存**：

```python
if IS_VARLEN:
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1)...
...
if USE_INITIAL_STATE:
    idx = tl.load(h0_indices + i_n)        # ← device 侧读状态槽号
    if idx >= 0:                            # ← pad_slot_id=-1 直接跳过
        p_h0 = (h0_source + idx*HV*K*V + i_hv*K*V + o_k[:, None]*V + o_v[None, :])
        b_h = tl.load(p_h0, mask=mask_h).to(tl.float32)

# gating 公式（与 rtp-llm 的 fused_gdn_gating 完全一致）:
#   g = -exp(A_log) * softplus(a + dt_bias),  beta = sigmoid(b)
...
# GDN delta rule 单步:
#   h *= exp(g); v -= (h·k); v *= beta; h += k⊗v; o = h·q
...
if USE_INITIAL_STATE:
    idx = tl.load(h0_indices + i_n)
    if idx >= 0:
        tl.store(p_h0, b_h.to(...), mask=mask_h)   # ← in-place 写回同一槽
```

要点：
- **读和写回同一个 `idx` 槽**（`tl.load(p_h0)` → `tl.store(p_h0, ...)`）；
- **`idx < 0`（padding 槽）时整段跳过**，这就是 padding 请求在图内安全的原因；
- gating 融合在 kernel 里（不用先算 g/beta 再传入），减少 kernel 间同步；
- q/k 布局 `[B*T, H, K]` 扁平、a/b 布局 `[B*T, HV]`、k 头与 v 头的映射是 `i_h = i_hv // (HV//H)`（等价 k.repeat_interleave）。

### 4.2 conv：`causal_conv1d_update_npu`（常规 decode 走纯 torch 路径）

`/home/d30033799/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/mamba/causal_conv1d.py:1276`

接口与 vLLM 兼容（`conv_state_indices` 是 **device tensor**）。内部按场景分流（:1375-1400）：

```python
if cache_seqlens is None and num_accepted_tokens is None:
    # 常规 decode（单 token）: 纯 torch 路径——gather → 卷积 → scatter 回写
    conv_state_update = conv_state[conv_state_indices]          # device 索引 gather
    out, conv_state[conv_state_indices] = torch_causal_conv1d_update_npu(
        x, conv_state_update, weight, bias=bias, activation=activation)
else:
    # spec/MTP（多 token）: triton kernel（conv_state_indices_ptr 在 kernel 内 tl.load）
    _causal_conv1d_update_kernel[grid](...)
```

`torch_causal_conv1d_update_npu`（:1242-1273）就是 `cat([conv_state, x])` → 逐元素乘权重求和 → silu → 取窗口尾部为新状态，**全 device 算子，天然可入图**。

### 4.3 AscendC 原生算子（并行路线，接口同样 device 化）

sgl-kernel-npu 的 csrc/ 下还有一整套 AscendC 原生算子（注册为 `torch.ops.sgl_kernel_npu::*`），metadata 同样是 device Tensor：

| 算子 | 文件 | 元数据 |
|---|---|---|
| `causal_conv1d_update` | `csrc/causal_conv1d_update/op_host/causal_conv1d_update.cpp` | `conv_state_indices` / `query_start_loc` / `num_accepted_tokens` 全 device Tensor（:206-218） |
| `recurrent_gated_delta_rule` | `csrc/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule.cpp:33-39,177-180` | `actual_seq_lengths`、`ssm_state_indices`、`cache_indices` 全 device Tensor |
| `mega_chunk_gdn`（prefill） | `csrc/mega_chunk_gdn/` | `cu_seqlens` device Tensor |

AscendC kernel 侧确认 device 读取：`recurrent_gated_delta_rule_kernel.cpp:94-110` 把 `cuSeqlens`、`ssmStateIndices`、`cacheIndices` 声明为 `(__gm__ int32_t*)` 全局缓冲，核内 `GetValue` 读取（`:226 cacheIndicesGm_.GetValue(batchIdx)`）——**纯 device 侧间接寻址，无 host 指针**。

**特别值得注意的 AscendC 图模式专项适配**（`causal_conv1d_update.cpp:19-24, 145-198`）：AscendC 需要 host 算好的 tiling 结构体，实现是"**graph mode tiling cache**"——tiling 按参数 hash 一次 H2D 拷入预分配的持久 device buffer `globalTilingBuffer`（`MAX_CAPTURE_NUM=1024`），capture/replay 时直接复用固定 device 地址；且注释明确说明用 torch_npu dispatch 而非裸 `aclrtMemcpy`，否则"break stream ordering in graph mode"。conv_state 预移位也用 `index_select/index_copy_`（device 索引）。

### 4.4 小结：算子层设计原则

> **凡是"每 step 会变"的量（状态槽号、序列长度、偏移），一律做成 device tensor，由 kernel 在执行时读取；凡是 host 才有的量（tiling），capture 前算好放进固定地址的 device buffer。**

---

## 五、Backend 层详解：capture 固定地址，replay 改内容

`/home/d30033799/sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`

### 5.1 capture 前预分配（init_cuda_graph_state，:156-194）

```python
def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
    for i in range(max_bs):
        self.state_indices_list.append(
            torch.full((i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device))
        self.query_start_loc_list.append(
            torch.empty((i + 2,), dtype=torch.int32, device=self.device))
        ...
```

每个捕获 bs 档位一组：`state_indices_list[bs-1]`（状态槽号）、`query_start_loc_list[bs-1]`（varlen 偏移）。**这些 buffer 的地址在 capture 后不变**——图内的 triton kernel 录下的就是对这些地址的读取。

### 5.2 replay 前刷新（_replay_metadata，:234-298）

```python
def _replay_metadata(self, bs, req_pool_indices, forward_mode, spec_info, seq_lens_cpu):
    num_padding = torch.count_nonzero(seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value())
    req_pool_indices[bs - num_padding:] = 0
    mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
    mamba_indices[bs - num_padding:] = -1                    # padding 槽 → -1
    self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)   # ← device copy_
    if forward_mode.is_decode_or_idle():
        # query_start_loc 用预计算好的 arange 表拷入（decode 是每序列 1 token 的固定布局）
        self.query_start_loc_list[bs - 1].copy_(
            self.cached_cuda_graph_decode_query_start_loc[: bs + 1])
    ...
    return ForwardMetadata(
        query_start_loc=self.query_start_loc_list[bs - 1],
        mamba_cache_indices=self.state_indices_list[bs - 1])
```

要点：
- **padding 计数在 CPU tensor 上图外完成**（`seq_lens_cpu`），避免图内 host 同步；
- **内容更新用 device `copy_`**（图外、replay 前），kernel 下次读到的就是新槽号；
- decode 的 `query_start_loc` 是固定模式（`arange(0, bs+1)`），直接整表拷贝。

这就是标准的 **"capture 固定地址、replay 改内容"** 的 device 化元数据方案。

---

## 六、Graph Runner 层详解：整图 capture + graph.update

`/home/d30033799/sglang/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`（继承通用 `CudaGraphRunner`，`model_runner.py:2441-2448` 按 device 选择）

### 6.1 整图 capture（:67-78）

```python
def _capture_graph(self, graph, pool, stream, run_once_fn):
    with torch.npu.graph(graph, pool=pool, stream=stream,
                         auto_dispatch_capture=True):   # ← torch_npu 的 aclgraph 封装
        out = run_once_fn()
```

`run_once_fn` 跑**整个模型 forward**（含所有 GDN 层），**没有按层/按算子跳过或图外 hook 的分支**——GDN 层的 triton kernel 直接被录进 aclgraph。

模型侧仅有的 capture 相关处理（qwen3_next.py）：
- `:356-360` NPU 上 `DUAL_STREAM_TOKEN_THRESHOLD=0`（禁用 qkvz/ba 双流投影，NPU 无 alt_stream，避免多流 capture 问题）；
- `:694` full-attn 层 qk-norm 在 capture 模式用 alt_stream 重叠（CUDA only）。

### 6.2 replay：host 输入的动态更新（:129-158）

```python
def replay(self, forward_batch, ...):
    ...
    if forward_batch.forward_mode.is_target_verify():
        seq_lens_cpu = forward_batch.seq_lens.cpu() + self.num_tokens_per_bs
        seq_lens = seq_lens_cpu.tolist() + [0] * (self.bs - self.raw_bs)
    else:
        seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (self.bs - self.raw_bs)
    thread = threading.Thread(target=self._update_inputs, args=(seq_lens,))
    thread.start()                       # ← 后台线程更新 host 输入
    self.graphs[self.bs].replay()        # ← 与图重放并行，隐藏更新开销
    thread.join()

def _update_inputs(self, seq_lens):
    if isinstance(self.update_attr_type, torch.Tensor):
        seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
    self.graphs[self.bs].update(
        cpu_update_input=[{self.update_attr_name: seq_lens}])   # ← torch_npu 的图输入更新机制
```

这一段是为了 **full-attention 的 FIA 算子**：`npu_fused_infer_attention_score` 的 `actual_seq_lengths_kv` 在图内是 **host list**（`ascend_backend.py:1002-1042`），用 torch_npu 的 `graph.update(cpu_update_input=...)` 在 replay 时改写，独立线程与 replay 并行。**GDN 不依赖这个机制**（它的元数据已全 device 化，靠第五节的 copy_ 刷新）。

### 6.3 extend/prefill 路径的补充说明

prefill（extend）路径允许 host 同步（`fla/mega_chunk_gdn.py:41` 有 `.cpu().tolist()`），因为它跑在 decode 图之外。若开 piecewise cuda graph，`gdn_with_output`（qwen3_next.py:1053-1086 注册的自定义算子）是 `compilation/backend.py:29-32` 里 `SPLIT_OPS` 之一（与 `unified_attention_with_output` 并列），作为**分段编译的图边界**——注意力/GDN 段落在 piecewise 图边界外执行。但这只影响 extend 路径，**decode 的整图 capture 不经过此机制**。

---

## 七、与 rtp-llm 的对照（我们借鉴了什么）

rtp-llm 移植 fla_npu AscendC 算子时遇到的死结：`npu_causal_conv1d` / `npu_recurrent_gated_delta_rule` 的兼容接口参数名就叫 `cache_indices_cpu`，aclnn 调用把它转成 **host int 数组**（`_aclnn_ctypes.py` 的 `ctx.int_array(cache_indices_cpu)`），Python 侧还必须 `.cpu().tolist()` 读 block_map——**capture 中 D2H 直接被 ACL 拒绝**（`Stream during the capture stage is not supported`）。这与"SGLang GDN 没入图"的印象相反，核实后确认 SGLang 是靠换算子（sgl_kernel_npu 的 device 元数据版本）解决的。

本次 rtp-llm 的最终做法（详见《图模式叠加Qwen3.5验证总结.md》）：

| SGLang 的做法 | rtp-llm 的借鉴 | rtp-llm 的差异 |
|---|---|---|
| decode 用 `causal_conv1d_update_npu`（纯 torch gather/scatter） | 拷贝其纯 Python 部分（`sgl/` 目录）+ 自写 `causal_conv1d_update_graph`（同为 gather→窗口卷积→scatter） | rtp 的 conv_state 在 KV cache 分页块里，多一步"按 block_map 算 read/write 页"（device gather 实现） |
| decode 用 `fused_sigmoid_gating_delta_rule_update_npu`（triton kernel，device `initial_state_indices`） | 直接拷贝该 kernel（`sgl/fused_sigmoid_gating_recurrent.py`）+ `graph_compat.py` 适配 | rtp 的 ssm 池布局是 `[pages,HV,V,K]`，kernel 要 `[pages,HV,K,V]`，需转置适配（曾因此引入 133% 误差的 bug）；rtp 的状态分页（每 1024 token 换页）需要 `_seed_write_pages`（跨块搬运，device index_copy_） |
| 元数据 buffer 固定地址 + replay 前 `copy_` 刷新 | 未采用（架构不同） | rtp 的 C++ AscendGraphRunner 自己维护 capture buffer（block_map/seq_lens），replay 时由 C++ 拷贝更新；Python 图内用 device gather 从 block_map 现算状态页索引，**语义等价**（capture 的地址固定 + replay 改内容） |
| eager 与图模式用同一套 sgl 算子 | eager 保留 fla_npu AscendC（性能），仅 capture 时切图实现 | rtp 用 `torch.npu.is_current_stream_capturing()` 分流（另有 is_cuda_graph 属性因 C++ 值拷贝丢失的问题，见总结文档 C5） |
| GDN 状态不分页（每请求一个固定槽） | — | rtp 复用 KV cache 分配器给 linear 状态分页（支持 prefix 复用），换来更复杂的页索引逻辑 |

**一句话**：SGLang 证明了"GDN 直接入图"在 NPU 上是可行的，前提是**算子的元数据必须 device 化**；rtp-llm 通过借用 sgl-kernel-npu 的 triton kernel + 自写 device 侧页索引适配，达到了同样的效果（图模式与 eager 输出逐 token 一致）。

---

## 八、关键文件索引

| 内容 | 位置 |
|---|---|
| GDN decode 入口 / NPU 算子绑定 | `sglang/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py:38-57`（import）、`:524-596`（forward_decode） |
| 元数据预分配 / replay 刷新 | 同文件 `:156-194`（init_cuda_graph_state）、`:234-298`（_replay_metadata） |
| mamba 索引来源 | `sglang/python/sglang/srt/mem_cache/memory_pool.py:402-403` |
| 模型层（qwen3_next/qwen3.5 共用） | `sglang/python/sglang/srt/models/qwen3_next.py:391-473` |
| NPU 图 runner（整图 capture + graph.update） | `sglang/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py:67-78, 129-158` |
| FIA 的 host list 输入（graph.update 的服务对象） | `sglang/python/sglang/srt/layers/attention/ascend_backend.py:1002-1042` |
| piecewise split op（仅 extend） | `sglang/python/sglang/srt/compilation/backend.py:29-32` |
| recurrent triton kernel（decode 核心） | `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/fused_sigmoid_gating_recurrent.py:18-168（kernel）、170+（包装）` |
| conv update（纯 torch 路径 + triton 路径） | `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/mamba/causal_conv1d.py:1242-1273（torch 参考）、1276+（入口）、1375-1400（分流）` |
| AscendC 版 recurrent（device 元数据 + 图模式 tiling cache） | `sgl-kernel-npu/csrc/recurrent_gated_delta_rule/`、`csrc/causal_conv1d_update/` |
| Qwen3.5 投影布局差异（in_proj_qkv/z 分离） | `sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/fla/utils.py:465-473` |
