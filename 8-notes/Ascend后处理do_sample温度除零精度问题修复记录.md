# Ascend 后处理（Sampler）do_sample / temperature=0 精度问题修复记录

> 仓库：`/home/d30033799/rtp-llm`（`feature/tp-ascend` 分支）
> 修复文件：`rtp_llm/models_py/bindings/core/CudaSampleOp.cc`（`#elif USING_ASCEND` 的 `sampleGreedy` NPU 实现，约 285 行起的 Ascend 段）
> 关联上游：alibaba/rtp-llm PR #1349 review 的 P1/P2 问题（部分已修，本仓补齐并额外修复了 temperature=0 除零）
> 时间：2026-09-14 ~ 2026-09-15

---

## 一、问题现象

图模式叠加 Qwen3.5-35B 端到端跑通后，采样行为出现异常：

| 请求参数 | 实际输出 | 期望 |
|---|---|---|
| `temperature=0`（greedy） | **`'!!!!!!!!!!!!!!!!!!'`**（全是感叹号，固定 token） | 正常中文回答 |
| 不带 temperature（默认采样，temp≈0.7~1.0） | 正常中文 | 正常 |

且**乱码与图模式无关**（eager 模式下 greedy 同样输出"!!!!!"），说明问题在后处理采样器，不在 GDN/图路径。

## 二、根因分析

### 2.1 调用链

```
OpenAI 请求 (temperature=0)
  → GenerateConfig: do_sample 保持默认 True, temperature=0.0
  → C++ Sampler.cc: 判定走 execSampleGreedy
  → NPU sampleGreedy (CudaSampleOp.cc, USING_ASCEND 段)
```

### 2.2 根因：temperature=0 时 logits 除以 0

NPU 版 `sampleGreedy` 的温度处理（修复前）：

```cpp
if (need_do_sample) {
    if (std::any_of(..., [](auto t) { return t != 1.0f; })) {
        auto temperature_npu = params.temperature.to(device_type)
                                   .reshape({batch_size, 1});
        params.logits.div_(temperature_npu);   // ← temperature=0 时除零！
    }
}
```

三个条件叠加导致 bug：

1. **OpenAI endpoint 不会因 `temperature=0` 关闭 `do_sample`**——`GenerateConfig.do_sample` 默认 `True`（`generate_config.py:58`），temperature=0 只被当作"温度很低"，`do_sample` 仍为 True；
2. NPU 实现里 `need_do_sample == true`，于是执行 `logits.div_(0.0)`；
3. **除零后 logits 全部变为 ±inf/NaN**——`argmax` 在 NaN 上的行为未定义（NPU 上返回固定的错误位置，如 token id 对应 `'!'`），于是 greedy 每步都采样出同一个错误 token → 输出一串 `'!!!!!'`。

为什么 CUDA 版没这个问题：温度=0 的请求在 CUDA 版走的是另一条保护分支（且 OpenAI 规范里 temperature=0 等价 greedy），NPU 移植时漏掉了这个语义。

### 2.3 验证方式（当时用来实锤的二分）

- `temperature=0` → `'!!!!!'`；`temperature=0.1` → 正常中文；
- 0.1 正常、0.0 崩 → 符合"除零"特征（0.1 时 `div_(0.1)` 只是放大 logits，不产生 NaN）；
- HF transformers 同输入 greedy 正常（首 token `<think>`，p=0.9953）→ 证明模型前向正确，问题仅在采样。

## 三、修复方案

### 3.1 主修复：非正温度按 1.0 处理（不缩放 = 纯 greedy）

```cpp
// CudaSampleOp.cc (NPU sampleGreedy) — 温度处理
if (need_do_sample) {
    if (std::any_of(..., [](auto t) { return t != 1.0f; })) {
        // temperature == 0 requests greedy decoding but do_sample stays true:
        // dividing by zero yields inf/NaN and breaks argmax. Treat
        // non-positive temperatures as 1.0 (no scaling, pure greedy).
        auto temperature_npu = params.temperature.to(device_type);
        temperature_npu.masked_fill_(temperature_npu <= 0.0f, 1.0f);
        params.logits.div_(temperature_npu.reshape({(int64_t)batch_size, 1}));
    }
}
```

要点：
- 用 `masked_fill_(temperature <= 0.0f, 1.0f)` **只把非正温度置 1**，正温度（如 0.7）不受影响，保留逐 batch 温度语义；
- 除以 1.0 等于不缩放 → argmax 结果与"不除"完全一致 → **greedy 语义精确保持**；
- 避免了 `clamp_min(1e-6)` 这类做法（会把 logits 放大 10^6 倍，虽然 argmax 不变但可能引起 fp 溢出）。

### 3.2 顺带修复（上游 PR #1349 review 指出的同族问题）

排查中发现上游 review 已指出 NPU sampler 的另外两个问题（检查上游 PR head 分支发现**实际未修复**），一并修齐：

**（1）混合 batch 的 do_sample 语义（P1）**

修复前：只要 batch 中有一个请求 `do_sample=true`，非 top_k=1 分支就对**所有**行执行 `multinomial`——`do_sample=false`（greedy）的行也被随机采样。

修复后（按行 mask 合并 argmax 与采样）：

```cpp
auto selected = torch::multinomial(probs_t, 1, /*replacement=*/false, gen).squeeze(-1);
// Upstream review P1: a mixed batch must honour per-request do_sample.
// Rows with do_sample=false are greedy: overwrite their multinomial
// draw with the argmax result instead of sampling every request.
if (has_not_do_sample) {
    auto greedy_sel   = torch::argmax(probs_t, -1, /*keepdim=*/false);
    auto do_sample_npu = params.do_sample.value().to(device_type);
    selected          = torch::where(do_sample_npu, selected, greedy_sel);
}
samples_t.copy_(selected);
```

**（2）cum_log_probs 双重 softmax（P1）**

修复前：第 4 步已把 `params.logits` 原地覆盖为 softmax 概率，第 7 步又对这份"概率"做 `log_softmax`——等于对概率分布再 softmax 一次，累计对数概率全错。

修复后（直接从概率取所选 token 再取对）：

```cpp
// Upstream review P1: params.logits was overwritten with softmax probs
// in step 4, so log_softmax(logits) double-softmaxes and yields wrong
// values. Take the probability of the selected token directly from
// probs_t and take its log instead.
auto token_probs     = probs_t.gather(-1, samples_t.reshape({batch_size, 1})).squeeze(-1);
auto token_log_probs = token_probs.clamp_min(1e-10f).log();
cum_log_probs_t.add_(token_log_probs.to(cum_log_probs_t.device()));
```

（`clamp_min(1e-10)` 防止 log(0)。）

**（3）top_k 哨兵值无符号比较（P2）**

修复前：`auto top_k_ptr = reinterpret_cast<uint32_t*>(params.top_k.data_ptr<int32_t>());` 把 int32 按 uint32 读，`t <= 0` 对无符号数恒等于 `t == 0`，`top_k = -1`（"不做 top-k 截断"哨兵）被误判为"配置了 top_k"。

修复后：

```cpp
auto top_k_ptr = params.top_k.data_ptr<int32_t>();  // signed read
```

## 四、修复后验证

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `temperature=0`（greedy） | `'!!!!!'` 固定 token | 正常中文思维链（与 HF 风格一致） |
| `temperature=0.1/0.7/1.0` | 正常 | 正常（未回归） |
| 图模式 vs eager（同 greedy 请求） | — | 逐 token 完全一致（顺带验证了图模式正确性） |
| 端到端 | — | 50 / 1024 token（跨块）长生成全部正常 |

## 五、经验总结

1. **`temperature=0` 是 OpenAI 协议里"greedy"的表达**，后处理实现必须显式处理（要么在 config 层把 `do_sample` 置 false，要么在算子里防除零）——两道防线至少要有其一；
2. NPU 移植 CUDA 代码时，**采样/精度类分支要逐条对照原平台的保护逻辑**，本次的除零就是移植遗漏；
3. 排查此类问题的快速路径：**同参数换 temperature 值二分**（0.0 崩 / 0.1 正常 → 除零特征）+ **HF 同输入对照**（证明模型前向无辜，锁定后处理）;
4. 修复后建议把 greedy 与默认采样各跑一轮回归（本次作为图/eager 一致性验证的副产品完成）。
