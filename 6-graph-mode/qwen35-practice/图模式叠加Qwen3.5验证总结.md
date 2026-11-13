# 图模式（ACL Graph）叠加 Qwen3.5-35B-A3B 验证总结

> 验证时间：2026-09-14 ~ 2026-09-15
> 验证人：rtp-llm 图模式 + Qwen3.5 移植联调
> 代码分支：`feature/tp-ascend`（rtp-llm 仓，基于图模式分支），Qwen3.5 算子代码移植自 `rtp-llm-npu` 仓 `qwen35-ascend-ops` 分支

---

## 一、任务背景

rtp-llm 的 ACL Graph（华为 NPU 图模式，对标 CUDA Graph）分支此前只在 **Qwen3-0.6B**（纯 MHA 小模型）上跑通过。本次任务是把 **Qwen3.5-35B-A3B**（混合注意力 + MoE 大模型）的算子代码从 `rtp-llm-npu` 仓库移植过来，验证"**图模式叠加 Qwen3.5**"是否存在新问题，并逐一解决。

### 为什么 Qwen3.5 对图模式是全新挑战

Qwen3.5-35B-A3B 与 Qwen3-0.6B 的架构差异极大，每一条都与图模式冲突：

| 架构特性 | Qwen3-0.6B | Qwen3.5-35B-A3B | 对图模式的影响 |
|---|---|---|---|
| 注意力 | 纯 MHA | 混合注意力：40 层中 30 层线性注意力（GDN）+ 10 层全注意力 | GDN 算子能否入图是最大难点 |
| FFN | 普通 FFN | MoE（256 专家 top-8 + 共享专家） | MoE 路由/专家算子需可被图捕获 |
| KV Cache | 单一分组 | 混合分组（full 组 + linear 组），物理块 1024 / kernel 块 512 分裂 | 图内 KV 读写、块表切换全部复杂化 |
| 权重格式 | 单文件合并 | 分离的 q/k/v、in_proj_qkv/z/b/a、stacked MoE | 权重映射、q/gate 拆分 |

### 环境信息

- 机器：Ascend950PR 单卡（131GB HBM），CANN 9.2.0，torch 2.9.0+cpu，torch_npu 2.9.0.post3
- 依赖：triton-ascend 3.2.2（`pip install triton-ascend==3.2.2 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi`）、flash-linear-attention-npu（源码构建，SoC 绑定）
- 编译：`bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend`

### 最终启动命令（验证通过的参数）

```bash
python3 -m rtp_llm.start_server \
    --checkpoint_path=/mnt/docker/weights/Qwen3.5-35B-A3B \
    --model_type=qwen35_moe \
    --start_port=9000 \
    --enable_cuda_graph=true \
    --kv_cache_mem_mb=8192 \
    --seq_size_per_block=1024 \
    --kernel_seq_size_per_block=512
```

> 两个块大小参数是必须的，原因见问题 C2、C3。

---

## 二、最终结论

1. **图模式 + Qwen3.5-35B 端到端跑通**：5 个 batch 档位（1/8/16/24/32）图捕获全部成功，replay 正常，生成速度约 120ms/token（35B-A3B bf16 单卡）。
2. **图模式与 eager 输出完全一致**（greedy 逐 token 相同，GDN/conv 单测误差 0.25%/0.29%，为 bf16 正常水平）——**图模式叠加没有引入新的精度问题**，这是本次验证的核心目标。
3. 过程中共发现并修复 **15 个问题**（详见下表），其中 8 个是图模式叠加 Qwen3.5 引入的新问题。
4. 遗留：rtp-llm 的 Qwen3.5 **eager 路径本身与 HF 存在一个上游精度偏差**（来自 rtp-llm-npu 仓库移植的 GDN prefill 算子，非本次图模式引入，详见第六节"遗留问题"）。

---

## 三、问题全景

| # | 类别 | 问题 | 根因 | 修复文件 |
|---|---|---|---|---|
| A1 | 移植 | Qwen3.5 算子代码缺失 | 图模式分支没有 kernels/ascend 等 | 从 rtp-llm-npu 移植 + 手工合并 ascend_decode.py |
| A2 | 移植 | `import rtp_llm.models` 崩溃 | transformers 5.x 删除了 gpt2 fast tokenizer 子模块路径 | models/starcoder*.py |
| A3 | 移植 | MoE 策略选择失败 | CANN MoE 策略显式拒绝图模式；且 config_dtype 缺失导致被判非 bf16 | cann.py / fused_moe/__init__.py / qwen3_next.py |
| A4 | 移植 | hybrid KV 配置断言失败 | linear 组块跨度 > full 组（1.05MB > 256KB） | 启动参数 seq_size_per_block=1024 |
| B1 | 乱码 | 输出完全乱码（中英混杂乱码） | **config.json 的 dtype 未解析 → 引擎默认 FP16 → 精度崩坏** | models/qwen3_next/qwen3_next.py |
| B2 | 乱码 | greedy（temperature=0）输出"!!!!!" | NPU sampler 对 logits `div_(0)` → NaN → argmax 错乱 | CudaSampleOp.cc |
| B3 | 乱码 | 混合 batch 采样语义错误（上游 PR#1349 review P1/P2） | do_sample 未按行 mask；cum_log_probs 用被覆盖的 logits 算；top_k 哨兵被无符号误读 | CudaSampleOp.cc |
| C1 | 图模式 | GDN decode 算子 capture 时崩溃 `ACL stream synchronize failed` | **AscendC 算子的元数据（cache_indices 等）是 host 侧 int 数组，capture 中做 D2H 同步被禁止**（SGLang 同款问题） | 新增 graph_compat.py + sgl/ 目录 |
| C2 | 图模式 | capture 表 page 0 非法 | C++ 用 `zeros` 初始化块表，而 GDN 算子要求 page>0 | ascend_graph_runner.cc |
| C3 | 图模式 | FIA 报 block_size 1024 超限 | FIA 算子要求 block_size ∈ [16,512] | 启动参数 kernel_seq_size_per_block=512 |
| C4 | 图模式 | replay 时 3D 分组表拷贝崩溃 | hybrid 下 engine 侧块表是 [group,batch,cols]，C++ 拷贝函数按 2D 处理 | ascend_graph_runner.cc |
| C5 | 图模式 | GDN 层不走图分支（is_cuda_graph 丢失） | C++ capture 时值拷贝 PyModelInputs，pybind 动态属性丢失 | qwen3_next.py（capturing 兜底） |
| C6 | 图模式 | 图模式输出"模型看到的输入是别的"（上下文错位） | **update_graph_fia 的 ctx 个数用 max_bs（32）而不是图档位 bs** → FIA 的 ctx_lens 更新失败 → 读到 capture 时的固定 kvlen | ascend_decode.py + ascend_graph_runner.cc |
| C7 | 图模式 | 图模式输出重复退化（"给你看一个给你看一个..."） | **blocks_per_phys 在 capture 时被推断成 1**（capture 块表同宽），真实是 2（1024/512）→ 图内 KV slot 映射、K/V 切分全错位 | ascend_attn_params.py / ascend_decode.py / ascend_prefill.py |
| C8 | 图模式 | GDN 图实现输出错（单测 err 133%） | **sgl triton kernel 的 state 池布局是 [pages,HV,K,V]，rtp 池是 [pages,HV,V,K]**，差一个转置 | graph_compat.py |

---

## 四、问题详解

### A. 移植与环境类

#### A1. Qwen3.5 算子代码移植

**现象**：图模式分支没有 `qwen35_moe` 模型注册所需的 Ascend 算子（线性注意力、CANN MoE 等）。

**做法**：从 rtp-llm-npu 的 `qwen35-ascend-ops` 分支移植 9 个提交的内容：
- 新增 `rtp_llm/models_py/kernels/ascend/`（GDN 的 conv/recurrent/chunk/block 算子封装，调 fla_npu）
- 新增 CANN MoE pipeline（`fused_moe/impl/ascend/executors/`、`routers/`、`strategy/cann.py`）
- `bindings/ascend/AscendRegister.cc` 注册 `write_cache_store` 绑定
- 修改 `model_desc/qwen3_next.py`：按设备条件导入 Ascend 算子、`layer_idx` 传递（hybrid 分组按层选块表）

其中 `ascend_impl/ascend_decode.py` 两边各有改动（本仓是图模式版本、npu 仓是混合 KV 版本），**手工合并**：保留本仓的 `graph_task_group`/`prepare_cuda_graph` 图机制，合入 npu 仓的 `infer_blocks_per_phys`/`split_kv_kernel_blocks`/`layer_idx` 混合 KV 支持。

#### A2. transformers 5.x 兼容

```python
# models/starcoder.py / starcoder2.py
# 旧（transformers 4.x 路径，5.x 已删除）：
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
# 新（顶层导入，新旧版本通用）：
from transformers import GPT2TokenizerFast
```

#### A3. MoE 策略选择失败（两层原因）

1. `Qwen3NextBase._parse_basic_config` 不读 config.json 的 dtype → `config_dtype=None` → `init_precision_config` 默认 FP16 → `AscendCannStrategy` 的 `is_bf16` 检查失败 → 只剩纯 PyTorch fallback（又依赖 NPU 上不可用的 triton kernel）→ `No suitable MOE strategy found`。修复：

```python
# models/qwen3_next/qwen3_next.py  (_parse_basic_config)
# Qwen3.5 ships "dtype" (bf16) inside text_config; without config_dtype
# the engine defaults to FP16 and the BF16-only Ascend MoE strategy rejects.
config.config_dtype = config_json.get("torch_dtype") or config_json.get("dtype")
```

2. `AscendCannStrategy` 显式 `checker.check(not config.enable_cuda_graph)` 拒绝图模式。由于 CANN 的 MoE 算子（`npu_moe_gating_top_k` 等）是 aclnn 调用、可被图捕获，移除该限制并双注册兜底策略：

```python
# fused_moe/__init__.py (Ascend 注册段)
registry = StrategyRegistry()
registry.register(AscendCannStrategy())       # eager 首选（priority 52）
registry.register(AscendBf16FallbackStrategy())  # 兜底（图模式/特殊配置）
```

#### A4. hybrid KV 块跨度约束

`HybridConfigCreator.cc` 要求 `full_kv_block_stride_bytes >= linear_kv_block_stride_bytes`（Qwen3.5 的 linear 组 GDN 状态每块约 1.05MB，默认 128-token 的 full 组 KV 每块只有 256KB），启动报 `not support full attention with padding now`。用 `--seq_size_per_block=1024`（full 组 2MB ≥ linear 1.05MB）解决。这也是 rtp-llm-npu 仓库 benchmark 使用 qwen35_moe 时的同款参数。

---

### B. 输出乱码三根因

乱码问题经历了三轮定位，最终用"**逐层 hidden 与 HF 对比**"（40 层 prefill 中间输出 vs HF transformers 同输入 hidden states）锁定，三个根因相互独立。

#### B1. 根因一：引擎跑在 FP16 上（乱码的最大根因）

**现象**：所有请求输出乱码（`'Thinkingschau perjaninspaceتی一項imbang Gà'`），但逐层对比发现 **40 层 hidden 全部与 HF 一致（cos=1.0）**——模型计算没问题，问题在别处。

**定位**：hidden 全对但输出乱 → 怀疑采样；换 greedy（temperature=0）输出"!!!!!" → 确定采样路径；同时发现 MoE 策略日志里模型被判为 FP16。

**根因**：A3-1 的 config_dtype 缺失 → 引擎 data_type=FP16。35B 的 GDN/MoE 权重以 FP16 计算会数值溢出（GDN 的指数衰减门控、MoE 路由 softmax 在 FP16 下动态范围不足），产生乱码。**修复后（bf16）输出立即正常**。

#### B2. 根因二：greedy 除零（"!!!!!"）

**现象**：bf16 修复后，默认采样输出正常，但 `temperature=0` 的 greedy 请求输出全是"!!!!!"。

**根因**：NPU 版 `sampleGreedy`（`CudaSampleOp.cc`）里 temperature=0 的请求 `do_sample` 仍为 True，执行 `logits.div_(0)` → 全 inf/NaN → argmax 返回错误位置 → 固定 token"!"。修复：

```cpp
// models_py/bindings/core/CudaSampleOp.cc (NPU sampleGreedy)
// temperature == 0 requests greedy decoding but do_sample stays true:
// dividing by zero yields inf/NaN and breaks argmax. Treat
// non-positive temperatures as 1.0 (no scaling, pure greedy).
auto temperature_npu = params.temperature.to(device_type);
temperature_npu.masked_fill_(temperature_npu <= 0.0f, 1.0f);
params.logits.div_(temperature_npu.reshape({(int64_t)batch_size, 1}));
```

#### B3. 根因三：sampler 语义缺陷（上游 PR#1349 review 指出）

上游 alibaba/rtp-llm#1349 的 review 明确指出 Ascend sampler 的三个问题（检查其 PR head 分支，**这些问题上游实际未修复**），一并修复：

```cpp
// 1) 混合 batch（部分 do_sample=false）时，非 top_k=1 分支对所有行 multinomial。
//    按行 mask 合并 argmax：
auto selected = torch::multinomial(probs_t, 1, false, gen).squeeze(-1);
if (has_not_do_sample) {
    auto greedy_sel    = torch::argmax(probs_t, -1, false);
    auto do_sample_npu = params.do_sample.value().to(device_type);
    selected           = torch::where(do_sample_npu, selected, greedy_sel);
}

// 2) cum_log_probs 在 logits 被 softmax 概率覆盖后再 log_softmax（双重 softmax）。
//    改为直接取所选 token 的概率再取对数：
auto token_probs     = probs_t.gather(-1, samples_t.reshape({batch, 1})).squeeze(-1);
auto token_log_probs = token_probs.clamp_min(1e-10f).log();
cum_log_probs_t.add_(token_log_probs.to(cum_log_probs_t.device()));

// 3) top_k 哨兵值 -1（不做 top-k 截断）被按 uint32 读取，`t <= 0` 恒等于 `t == 0`。
auto top_k_ptr = params.top_k.data_ptr<int32_t>();  // signed read
```

---

### C. 图模式叠加 Qwen3.5 的新问题（核心章节）

#### C1. GDN decode 算子无法入图（本次最大的架构冲突）

**现象**：图 capture 时崩溃：
```
RuntimeError: ACL stream synchronize failed, error code:107027
  Reason: Stream (stream_id=24) during the capture stage is not supported.
At: causal_conv1d.py(48): _as_int_list
```

**根因**：GDN decode 用的 fla_npu **AscendC 算子**（`npu_causal_conv1d` / `npu_recurrent_gated_delta_rule`）的元数据是 **host 侧 int 数组**（`ctx.int_array(cache_indices_cpu)`），Python 侧需要 `.cpu().tolist()` 读 block_map 计算 cache_indices——**capture 中做 D2H 同步被 ACL 禁止**。这印证了用户提示的"**SGLang 里 GDN decode 没有入图**"，也是 rtp-llm-npu 文档说"NPU graph 不在本阶段验收范围内"的原因。

**解决方案（借鉴 sgl-kernel-npu）**：调研发现 SGLang 的做法是"GDN **直接入图**"，靠三层设计：
1. **算子层**：sgl-kernel-npu 的 conv/recurrent 内核全部用 **device tensor 元数据**（kernel 内 `tl.load(h0_indices + i_n)` 读索引），无 host 指针；
2. **Backend 层**：capture 固定地址、replay 改内容；
3. **Runner 层**：整图 capture。

于是把 sgl-kernel-npu 的**纯 Python/triton 部分**（不依赖其 C++ so）拷入 `rtp_llm/models_py/kernels/ascend/sgl/`（triton kernel + 工具），并新写 `graph_compat.py` 做适配：

```python
# rtp_llm/models_py/kernels/ascend/graph_compat.py（新文件，核心逻辑）
def _decode_state_indices(block_map, sequence_lengths_plus_1, seq_size_per_block):
    """device 侧从 block_map 取 read/write 页索引（与 fla_npu 的
    _resolve_state_pages 的 host 语义一致）"""
    length = sequence_lengths_plus_1.reshape(-1).to(torch.int64)
    read_col  = (length - 2).clamp_min(0) // seq_size_per_block
    write_col = (length - 1).clamp_min(0) // seq_size_per_block
    read_idx  = block_map.gather(1, read_col.view(-1, 1)).squeeze(1).long()
    write_idx = block_map.gather(1, write_col.view(-1, 1)).squeeze(1).long()
    return read_idx, write_idx

def _seed_write_pages(state, read_idx, write_idx):
    """跨块时把 read 页状态搬到 write 页（index_select/index_copy_ 可入图）"""
    ...

def causal_conv1d_update_graph(x, conv_states, weight, block_map, ...):
    """单 token conv 更新：纯 torch gather/窗口卷积/scatter，全 device op"""
    ...

def fused_recurrent_gated_delta_rule_graph(alog, dt_bias, q, k, v, a, b, ssm_states, ...):
    """GDN 单步更新：sgl 的 triton kernel（device 索引），gating 融合在 kernel 内
    （公式与 fused_gdn_gating 完全一致：g=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b)）"""
    ...
```

模型侧在 decode 路径按 capture 状态分流（见 C5）。

**注意**：q/k/v 的 `use_qk_l2norm_in_kernel=True`、ssm 池布局（见 C8）、write 页 in-place 语义都需精确对齐 fla_npu 参考实现，最终单测对拍误差 0.25%（bf16 正常水平）。

#### C2. capture 块表的 page 0 非法

**现象**：图 capture 的 warmup forward 报 `non-positive decode state pages are not supported on Ascend yet`。

**根因**：C++ `initCaptureAttentionInputs` 用 `torch::zeros` 初始化块表（page 0 是 pad/无效槽位），GDN 算子要求 page>0。修复为 `torch::ones`（page 1 恒有效）：

```cpp
// cpp/ascend_graph/ascend_graph_runner.cc
// Block tables are seeded with page 1 (an always-valid physical page).
// Page 0 is the pad/invalid slot: the Qwen3.5 linear-attention decode ops
// reject non-positive state pages during warmup/capture.
inputs.attention_inputs.kv_cache_kernel_block_id_device =
    torch::ones({int(max_bs_), max_blocks}, options_npu_int32_);
// ... host 表、phys 表、by_group 表同理全部 ones
```

#### C3. FIA 的 block_size 上限

**现象**：`Parameter block_size of FusedInferAttentionScore has incorrect value 1024. It should be a multiple of 16 in range of [16, 512].`

**解决**：A4 把物理块设为 1024 后，FIA 的 kernel 块也变成 1024 超限。加 `--kernel_seq_size_per_block=512`，即 **kernel 块 512、物理块 1024（分裂比 bpk=2）**——这正好走到移植代码里 kernel/phys 分裂的路径（`split_kv_kernel_blocks` 等），是混合 KV 的完整验证。

#### C4. replay 时 3D 分组块表拷贝崩溃

**现象**：首个 decode 请求 replay 时 `output with shape [1, 1] doesn't match the broadcast shape [1, 1, 1]`。

**根因**：hybrid 模型下 engine 侧块表是 3D `[group=2, batch, cols]`，`copyTensorSlice` 的 squeeze 条件（`size(0)==1`）不成立，3D 拷到 2D 报错。而图内实际引用的是 per-group 的 capture buffer（by_group），flat 表只需 group 0（全注意力组）：

```cpp
// cpp/ascend_graph/ascend_graph_runner.cc (copyTensorSlice)
// Hybrid models stack the engine-side block tables as [group, batch, cols].
// The captured graph references the per-group buffers (by_group), so the
// flat capture tensor only needs the full-attention group (group 0).
if (s.dim() > dst.dim() && s.size(0) > 1) {
    s = s[0];
}
```

#### C5. GDN 层拿不到 is_cuda_graph 标志

**现象**：C1 的 graph 分支没生效，capture 仍走 eager 算子崩溃。

**根因**（实证过程很曲折）：加打印发现 `attn_inputs.is_cuda_graph=False`。`prepare_fmha_impl(inputs, true)` 确实把该动态属性设在了 inputs 上，但 C++ 侧 `auto inputs = mem_hold_.py_model_inputs_` 是**值拷贝**，pybind 动态属性跟对象实例走——forward 拿到的新实例上没有该属性。（fmha 不受影响是因为 impl 在构造时缓存了原对象。）

**修复**：分支判断增加 capture 状态兜底（warmup 走 eager 合法——在图外；capture 走图实现；replay 不进 Python）：

```python
# models_py/model_desc/qwen3_next.py (GDN decode 的 _conv1d/_fla)
use_graph_path = getattr(attn_inputs, "is_cuda_graph", False) or (
    hasattr(torch, "npu") and torch.npu.is_current_stream_capturing()
)
```

#### C6. 图模式上下文错位（"模型看到的输入是别的"）

**现象**：C1-C5 修完后图模式能出连贯文本，但模型思维链里声称"输入是'你好，世界'"（实际输入"你是谁"/"量子计算"）——**KV 上下文错乱**。eager 正常。

**根因**（两层）：

第一层：`prepare_cuda_graph`（每次 replay 前由 C++ 调用）里：
```python
batch_size = attn_inputs.sequence_lengths.size(0)   # capture buffer 是 [max_bs]=32！
ctx_list = (seq_lens.to(torch.int32) + 1).tolist()   # padding 行=0+1=1
self.fmha_impl.update_graph_fia(ctx_list, batch_size)
```
用 32 个 ctx 去 update 图档位 bs=1 的 FIA graph task → **参数 shape 与 capture 时不匹配 → ctx_lens 更新被丢弃 → FIA 保持 capture 时的固定 kvlen → 读到垃圾上下文**。

第二层：`_graph_handles` 是 flat list，5 个档位 capture 时**跨档位累积**（每层 5 组 handle），replay bs=1 时把 bs=32 的 task 也错误更新。

**修复**：
```python
# ascend_decode.py: C++ 把图档位 bs 传进来，padding 到档位 bs
def prepare_cuda_graph(self, attn_inputs, graph_bs=None):
    ...
    seq = attn_inputs.sequence_lengths
    valid = int((seq != 0).sum().item())      # 有效行数
    if graph_bs is None:
        graph_bs = valid
    ctx_list = (seq[:valid].to(torch.int32) + 1).tolist()
    ctx_list = ctx_list[:graph_bs] + [1] * max(0, graph_bs - len(ctx_list))
    self.fmha_impl.update_graph_fia(ctx_list, graph_bs)

# handles/refs 按 capture 的 bs 分组，replay 只更新当前档位：
self._graph_handles = {}   # bs -> [task handles]
self._graph_refs    = {}   # bs -> [(q, k, v, table, mask, out, lse, page)]
```
```cpp
// ascend_graph_runner.cc: 把图档位 bs 传给 Python
attn_pyobj.attr("prepare_cuda_graph")(py_model_inputs.attention_inputs,
                                      py::int_(state.current_real_graph_bs));
```

修复后模型正确锚定真实输入（"用户问'你是谁'"）。

#### C7. 图模式输出重复退化（bpk 推断错误）

**现象**：C6 修完后输入锚定正确，但输出几十个 token 后重复退化（"我来给你看一个给你看一个..."）。

**根因**：混合 KV 的 kernel/phys 分裂比（bpk=1024/512=2）原来用 `infer_blocks_per_phys`（从 kernel 表/phys 表**宽度比**推断）。**capture 时 C++ 给两种表分配了相同宽度**，推断返回 1——而真实是 2。图内所有用 bpk 的计算（KV 写入的 slot_mapping：`phys_id = kernel_id // bpk`、FIA 读的 K/V 切分 `split_kv_kernel_blocks`）全部错位 → full-attn 层读到错乱 KV → 上下文逐步劣化 → 重复。

**修复**：bpk 优先从**真实缓存配置**取（`attn_configs.tokens_per_block / kernel_tokens_per_block`，服务路径下会正确设置为 1024/512），表宽度推断只作 fallback。decode 和 prefill 都修：

```python
# ascend_impl/ascend_attn_params.py（helper 放公共处，decode/prefill 共用）
def _blocks_per_phys_from_config(attn_configs, attn_inputs) -> int:
    """Prefer the real cache configuration (tokens_per_block / kernel tokens
    per block): during aclgraph capture the C++ runner allocates the kernel
    and physical block tables with identical widths, so inferring the ratio
    from the table shapes returns 1 and silently breaks the KV slot mapping
    for subdivided blocks (bpk > 1)."""
    tpb  = getattr(attn_configs, "tokens_per_block", 0) or 0
    ktpb = getattr(attn_configs, "kernel_tokens_per_block", 0) or 0
    if tpb > 0 and ktpb > 0 and tpb % ktpb == 0:
        return tpb // ktpb
    return infer_blocks_per_phys(attn_inputs)
```
`AscendDecodeImpl`/`AscendDecodeAttnOp`/`AscendPrefillImpl`/`AscendPrefillAttnOp` 的 bpk 全部改走该函数（注意 `prepare()` 等方法里要用 `self.attn_configs` 而不是构造参数名）。

修复后输出与 eager 完全一致。

#### C8. GDN 图实现的 state 池布局不匹配

**现象**：C1 的单测对拍：输出 err 133%，state err 40%+。

**根因**：sgl triton kernel 按指针算术 `idx*HV*K*V + i_hv*K*V + o_k*V + o_v` 寻址，即 state 池布局是 **[pages, HV, K, V]**；而 rtp-llm 的 ssm 池是 **[pages, HV, V, K]**（`get_ssm_state_tensor` 定义），差一个末两维转置。修复（图内两次转置拷贝都是普通 device op，可捕获）：

```python
# graph_compat.py: fused_recurrent_gated_delta_rule_graph
ssm_k_major = ssm_states.transpose(-1, -2).contiguous()   # -> [pages, HV, K, V]
o = fused_sigmoid_gating_delta_rule_update_npu(
    ..., initial_state_source=ssm_k_major,
    initial_state_indices=write_idx.to(torch.int32), ...)
ssm_states.copy_(ssm_k_major.transpose(-1, -2))            # 写回原布局
```

修复后单测 err 0.25%。（conv 的图实现单测直接通过，无需改。）

---

## 五、验证方法与结果

### 5.1 算子级单测（NPU 真机，eager 方式对拍 fla_npu AscendC 参考实现）

| 测试 | 内容 | 结果 |
|---|---|---|
| `test_gdn_graph.py` | `fused_recurrent_gated_delta_rule_graph` vs `fused_recurrent_gated_delta_rule`（同输入/同状态池/同 block_map，含 read/write 页选择） | 输出 err 0.25%，state err 0.13%（bf16 正常水平）；修复前 133% |
| `test_conv_graph.py` | `causal_conv1d_update_graph` vs `causal_conv1d_update` | 输出 err 0.29%，状态页 err 0 |

### 5.2 模型级精度对比（eager vs HF transformers）

- HF 侧：transformers 5.2.0 加载同一权重（GDN 用纯 torch 参考实现），`output_hidden_states` 取 40 层输出。
- rtp 侧：临时在层循环加 env 开关的 hidden dump（验证后已删除）。
- **结论**：40 层、全部 token 位置的 hidden 与 HF 的对比中，**位置 0 完全一致（cos=1.0）**；非 0 位置存在系统性偏差（见第六节遗留问题，该偏差 eager 一直存在、非图模式引入）。

### 5.3 端到端验证（最终）

| 场景 | eager | 图模式 |
|---|---|---|
| "你是谁" greedy 30 token | `嗯，用户问"你是谁"...我是 Qwen3.5...` | **逐 token 完全一致** |
| "你是谁" 默认采样 | 正常中文 | 正常中文（质量一致） |
| "写一篇500字介绍量子计算的文章" 1024 token | 正常（完整思维链+文章，**总 1043 token 跨过 1024 物理块边界**） | 同左 |
| 速度 | ~120ms/token | ~120ms/token |
| 图捕获 | - | 5 个档位（1/8/16/24/32）全部成功，replay check 通过 |

---

## 六、遗留问题（如实记录）

### 6.1 rtp-llm 的 Qwen3.5 eager 路径与 HF 存在精度偏差（上游遗留，非图模式问题）

**现象**：greedy 对比 HF，首 token 分叉——HF 选"Thinking"（p=0.905），rtp（eager 与图模式一致）选"嗯"（HF 分布第二名，p=0.040）。两者输出语义都合理（中英两种思考风格），但 token 序列不同。

**定位**：全 token 逐层 hidden 对比显示，**第 0 层（GDN）的非 0 位置就开始有偏差**（如位置 3"你好" err≈0.96），并逐层放大。位置 0 全层一致。

**归属**：该偏差在 eager 路径上一直存在（与图模式无关，本次所有修复之前就有）。怀疑点在移植自 rtp-llm-npu 的 **GDN prefill 链路**（`chunk_gated_delta_rule` 的 varlen/cu_seqlens 处理与 HF 全序列处理的差异，或 fla_npu 算子版本与 rtp-llm-npu 验证时的版本不一致——该仓库文档明确说"端到端精度验证未完成、FLA-NPU 不同 commit 契约仍在演进"）。**建议**：后续由熟悉 GDN 算子的同学做 GDN prefill 的逐算子对拍（npu 仓库的 `example/ascendc_npu` 下有单算子 golden 测试脚本可复用）。

### 6.2 其他已知限制

- `logprobs` API 在 NPU sampler 下报 `all_probs is None`（NPU 版 sampleGreedy 未实现 output_all_probs），待上游补齐。
- MTP/speculative decoding 在 Ascend 上 fail-fast（上游限制，非本次范围）。
- 多请求并发（bs>1 padding 到档位）逻辑已实现（C6 的 padding 修复），但只做过小规模测试，建议压测。
- 采样修复（B2/B3）建议合入时同步给上游 rtp-llm-npu / alibaba PR#1349。

---

## 七、本次修改文件清单

### 新增
| 文件 | 说明 |
|---|---|
| `rtp_llm/models_py/kernels/ascend/`（graph_compat.py、sgl/ 除外） | GDN Ascend 算子封装（自 rtp-llm-npu 移植） |
| `rtp_llm/models_py/kernels/ascend/graph_compat.py` | **GDN/conv 的图模式实现**（device 元数据方案，借鉴 sgl-kernel-npu） |
| `rtp_llm/models_py/kernels/ascend/sgl/` | sgl-kernel-npu 的纯 Python/triton kernel（fused_sigmoid_gating_recurrent、fla_utils） |
| `rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/executors/`、`routers/`、`strategy/cann.py` | CANN MoE pipeline（自 rtp-llm-npu 移植） |
| `docs/references/qwen35_ascend_ops.md` 等 | 算子接入文档（自 rtp-llm-npu 移植） |

### 修改（功能，均已清理调试代码）
| 文件 | 修改 |
|---|---|
| `rtp_llm/models/qwen3_next/qwen3_next.py` | config_dtype 解析（B1） |
| `rtp_llm/models_py/model_desc/qwen3_next.py` | Ascend 算子条件导入、layer_idx 传递（A1）、GDN decode 图分支 + capturing 兜底（C1/C5） |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_decode.py` | 图机制与混合 KV 手工合并；prepare_cuda_graph(graph_bs)（C6）；handles 按 bs 分组（C6）；bpk 从配置取（C7） |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_prefill.py` | bpk 从配置取（C7，注意 self.attn_configs 作用域） |
| `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_attn_params.py` | `_blocks_per_phys_from_config`（C7）、layer_idx 分组表选择、split_kv 等混合 KV 支持（自 npu 仓合入） |
| `rtp_llm/models_py/modules/factory/fused_moe/__init__.py` | Ascend 双策略注册（A3） |
| `rtp_llm/models_py/modules/factory/fused_moe/impl/ascend/strategy/cann.py`、`executors/npu_fused_experts.py` | 移除图模式拒绝（A3） |
| `rtp_llm/models_py/bindings/core/CudaSampleOp.cc` | sampler 三处修复（B2/B3） |
| `rtp_llm/cpp/ascend_graph/ascend_graph_runner.cc` | capture 表 ones 初始化（C2）；copyTensorSlice 支持 3D 分组表（C4）；graph_bs 传给 Python（C6） |
| `rtp_llm/models_py/modules/base/ascend/select_topk.py` 等 | npu 仓移植的算子适配 |
| `rtp_llm/models/starcoder.py`、`starcoder2.py` | transformers 5.x 兼容（A2） |

> 验证期间临时加的调试代码（hidden dump、attn dump、`[DUMP_DBG]`/`[FMHA_DBG]` 打印）在完成精度对比后已全部删除，`causal_attention.py`/`module_base.py`/`attn_factory.py` 已还原到无调试状态（`git diff` 确认干净）。

---

## 八、快速复现/回归建议

1. **启动**：用第二节的命令（注意两个块大小参数）。
2. **基础回归**：
   ```bash
   # greedy 一致性（图模式应与 eager 逐 token 一致）
   curl -s http://localhost:9000/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model":"qwen_3","messages":[{"role":"user","content":"你是谁"}],"max_tokens":30,"temperature":0}'
   # 长生成（跨 1024 块边界，覆盖 GDN 状态页迁移）
   curl -s http://localhost:9000/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model":"qwen_3","messages":[{"role":"user","content":"写一篇500字介绍量子计算的文章"}],"max_tokens":1024}'
   ```
3. **算子单测**：GDN/conv 图实现 vs fla_npu 参考的 NPU 对拍脚本（见 5.1，可固化为 CI 用例）。
4. **观察日志**：`Ascend graph capture success for batch size: N` × 5 档位 + `initCapture done, captured 5 graph instances`。
