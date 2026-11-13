# Pre-existing 并发 Prefill 乱码问题调查记录

> 时间：2026-09-28
> 发现于：MTP 图模式多 batch 精度测试（ceval CI + 并发脚本）
> 结论：**PR#54 基线（无 MTP）即存在，与 MTP/图 replay 无关**
> 状态：未修复，待单独排查

---

## 一、问题描述

当 ≥3 个请求同时到达服务时，部分请求的首 token（prefill 输出）为乱码（随机文字、`<|endoftext|>`、`<|box_end|>`、符号拼接等）。乱码请求的后续生成也可能异常。

**关键特征**：
- 串行请求：**始终正确**
- 并发 ×2：**始终正确**（多轮测试全对）
- 并发 ×3：**间歇性错误**（round1 全对、round2 错 1 个、round3 错 1 个——不同的请求出错）
- 并发 ×4：**随机错误**（每轮 1-2 个错，且是不同的请求）
- **禁用 MTP（`force_disable_sp_run: true`）后问题依旧** → 不是 MTP 引入的
- **基线分支 `perf/fia-kv-kernel-addressing`（PR#54，无任何 MTP 代码）同样复现** → pre-existing

## 二、复现环境

| 项 | 值 |
|---|---|
| 基线分支 | `perf/fia-kv-kernel-addressing`（PR#54，`79d7554a7`） |
| MTP 分支 | `feature/mtp-ascend`（`e8fa39278`，Phase 1+2 完整代码） |
| 模型 | Qwen3.5-35B-A3B（bf16，单卡 Ascend950PR） |
| 启动参数 | `--enable_cuda_graph=true --seq_size_per_block=1024 --kernel_seq_size_per_block=512` |
| 测试方法 | Python `ThreadPoolExecutor` 并发发 OpenAI 请求 |

## 三、复现命令

```bash
python3 - << 'EOF'
import concurrent.futures, requests, time

URL = "http://localhost:9000/v1/chat/completions"

def ask(prompt):
    r = requests.post(URL, json={
        "model": "qwen35_moe",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0,
        "extra_configs": {"chat_template_kwargs": {"enable_thinking": False}},
    }, timeout=60)
    return r.json()["choices"][0]["message"]["content"].strip()

prompts4 = ["What is 1+1? Answer:", "What is the capital of China? Answer:",
            "What is H2O? Answer:", "What is the largest planet? Answer:"]

# 串行基线（全对）
print("=== 串行 ===")
for p in prompts4:
    print(f"  {ask(p)[:35]!r}")

# 并发 ×4（随机出错）
for round_num in range(3):
    time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(ask, prompts4))
    statuses = ["OK" if any(w in r for w in ["2","Beijing","H₂O","largest"]) else "BAD"
                for r in results]
    print(f"round{round_num+1}: {' '.join(statuses)}")
    for i, (s, r) in enumerate(zip(statuses, results)):
        if s == "BAD":
            print(f"  [{i}] {r[:40]!r}")
EOF
```

## 四、已排除的嫌疑

| 排除项 | 依据 |
|---|---|
| MTP / verify 图 replay | `force_disable_sp_run=true` 后仍复现 |
| MTP 代码改动 | 基线分支（无 MTP 代码）同样复现 |
| chainSpeculativeSampling | 禁 MTP 后不走该路径 |
| GDN/conv 多 token 路径 | 禁 MTP 后不走多 token 路径 |
| 采样问题 | temperature=0 greedy，首 token 就错（prefill 输出） |

## 五、未排除的嫌疑（下一轮排查方向）

| # | 嫌疑 | 理由 | 排查方法 |
|---|---|---|---|
| 1 | **多序列 prefill 时 FIA 层的 block_table / KV 写 slot 错位** | 首个 FIA 层（L3）的 KV 写/读在多序列 varlen 场景下可能寻址错 | 逐层 hidden 对比（serial vs 并发同输入），定位首个分叉层 |
| 2 | **调度器合批时块表填充竞态** | 服务日志显示 input_lengths=[2]/[3]（确实合批），间歇性错误暗示竞态 | 在 `NormalModelInputGatherer.cc` 的块表填充处加日志，对比正确/错误请求的块表值 |
| 3 | **GDN prefill 的 varlen cu_seqlens 处理** | `chunk_gated_delta_rule` 的多序列 varlen 路径在多序列时 cu_seqlens 可能错 | 单测：多序列 GDN prefill vs 逐序列独立 prefill 对拍 |
| 4 | **GDN 状态页 store 的多序列写冲突** | `store_ssm_state_to_block_map` 按序列独立处理，但若两个序列的 `last_block_positions` 恰好指向同一物理块（新分配的块），可能写冲突 | 检查多序列时各序列的块分配是否互斥 |
| 5 | **图模式 decode 的 padding 行污染** | 并发时 decode 需 pad 到图档位（bs=8），padding 行的 KV/状态写可能污染有效行 | 检查 `prepareInputs` 的 padding 清零逻辑是否覆盖所有相关 buffer |

## 六、为什么并发 ×2 正常、×3 以上出错

推测：当只有 2 个请求时，调度器可能**不合批**（逐个 prefill），两个请求各自走单序列路径（已验证正确）。当 ≥3 个请求同时到达时，调度器开始**合批**（多序列 prefill forward），多序列路径有 bug。

从服务日志确认：并发时 `input_lengths` shape 为 `[2]` 或 `[3]`——确实有多序列 prefill。

但也可能不是精确的 ≥3 阈值——而是**到达时间窗口**内积累的请求数决定的（并发 ×2 时两个请求可能不在同一调度窗口内）。

## 七、建议排查步骤

1. **切到基线分支**（排除 MTP 干扰）：
   ```bash
   git checkout perf/fia-kv-kernel-addressing
   # 修 .bazelrc ARM march 后重编
   ```

2. **逐层 hidden 定位**：在 `qwen3_next.py` forward 层循环加逐层 `hidden.sum()` 日志，串行 vs 并发对比，找首个分叉层
   - 分叉在 GDN 层（L0-L2）→ 嫌疑 3/4（GDN prefill varlen / 状态写冲突）
   - 分叉在 FIA 层（L3 起）→ 嫌疑 1（FIA 多序列 KV 寻址）
   - 分叉在 MoE/FFN → 新嫌疑

3. **块表 dump**：并发 prefill 时打印每个序列的 `kv_cache_block_id_host` 和 `kv_cache_kernel_block_id_host` 的前几列值，检查是否有页号冲突或错位

4. **GDN 状态页冲突检查**：多序列 prefill 后 dump 各序列的 `store_ssm_state_to_block_map` 写入的物理页号，检查互斥性

5. **修复后回归**：用第四节的复现脚本跑 ≥10 轮并发 ×4 全对后确认修复

## 八、关联信息

- MTP 分支 `feature/mtp-ascend`（`e8fa39278`）不受此 bug 影响（单流全场景正确）
- MTP 相关文档：`/home/d30033799/weishan/MTP/rtp-llm-Ascend-MTP-Phase1-代码改动详细总结.md`
- Profiling 数据：`/home/d30033799/prof/mtp_phase1_eager_*`
