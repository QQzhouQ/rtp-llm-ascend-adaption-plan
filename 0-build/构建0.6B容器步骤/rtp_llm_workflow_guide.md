# rtp-llm (Ascend NPU) 完整 Workflow 操作指南

> 基于 `start_docker.sh`、`start_compile.sh`、`build_cmd.sh`、`Inference_server.txt`、
> `rtp-llm推理框架拉起方式分析.txt`、`rtp_llm_framework_compile_record.md`、
> `rtp_llm_inference_test.md`、`rtp_llm_version_matrix.md` 汇总整理。
>
> **目标环境**：x86_64, Ascend NPU (950PR / 9579), openEuler, Python 3.10
>
> **工作目录**：所有操作在容器内 `/workspace/work` 下进行。

---

## 1. 创建容器

使用 Docker 镜像 `quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86`（预装 CANN 9.0.0 和 conda 环境）。

> **执行目录**：宿主机任意路径

```bash
CONTAINER_NAME="rtp-llm-test"
IMAGE="quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86"

docker run \
    --name "$CONTAINER_NAME" \
    --privileged \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /mnt:/workspace \
    -it "$IMAGE" bash
```

> ⚠️ **映射目录说明**（根据实际情况修改）：
> - `-v /mnt:/workspace`：将宿主机 `/mnt` 目录映射为容器内 `/workspace`。模型权重、源码仓库、下载文件均通过此映射共享。
> - 不映射 `/home` 目录。
> - 以下所有章节均在**容器内**操作。

---

## 2. 进入容器

```bash
docker exec -it rtp-llm-test bash
```

---

## 3. 创建工作目录 & 下载 rtp-llm 仓库

> **执行目录**：`/workspace/`

```bash
mkdir -p /workspace/work && cd /workspace/work
git clone https://github.com/skywang2/rtp-llm-npu.git
cd rtp-llm-npu
```

---

## 4. 下载安装 CANN Toolkit 和 ops 包

> **执行目录**：容器内 `/workspace/work/`

```bash
cd /workspace/work

# 下载 CANN Toolkit
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T6/Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run

# 下载 CANN ops 包
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T6/Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run
```

> 容器镜像预装 CANN 9.0.0。本文档使用 **CANN 9.1.0-beta.3**。
>
> 📦 本地副本：`/workspace/work/Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run`
> 及 `/workspace/work/Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run`

### 4.1 安装 CANN Toolkit

```bash
chmod +x Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run
./Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run --install --quiet
```

### 4.2 安装 CANN ops 包

```bash
chmod +x Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run
./Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run --install
```

> CANN `.run` 安装包执行时自动将 `/usr/local/Ascend/ascend-toolkit/latest` 和 `/usr/local/Ascend/cann` 指向新安装的版本，无需手动创建软链接。
>
> 如安装时选择"不设为默认"或链接未生效，手动执行：
> ```bash
> ln -sfn /usr/local/Ascend/cann-9.1.0-beta.3 /usr/local/Ascend/ascend-toolkit/latest
> ln -sfn /usr/local/Ascend/cann-9.1.0-beta.3 /usr/local/Ascend/cann
> ```

### 4.3 创建 profiling header 软链接

> CANN 9.1.0-beta.3 将 profiling 头文件（`prof_api.h` / `devprof_pub.h` / `aprof_pub.h`）安装在
> `x86_64-linux/pkg_inc/profiling/` 而非传统的 `include/profiling/`。
> `aclnn_custom_ops` 通过 CMake 目标 `profapi` 引用 profiling 头文件，编译时依赖
> `include/profiling/` 路径。不创建软链接会报头文件找不到的编译错误。

```bash
ln -sf /usr/local/Ascend/cann-9.1.0-beta.3/x86_64-linux/pkg_inc/profiling \
       /usr/local/Ascend/cann-9.1.0-beta.3/include/profiling
```

### 4.4 安装 CANN TBE 编译器 Python 依赖（安装conda后执行）

| 包 | 命令 | 原因 |
|---|---|---|
| `scipy` | `pip install scipy` | bisheng relay/analysis 依赖 |
| `attrs` | `pip install attrs` | tvm relay transform 依赖 |
| `psutil` | `pip install psutil` | toolchain 依赖 |
| `decorator` | `pip install decorator` | toolchain 依赖 |

---

## 5. 安装独立算子包 (ops-transformer)

> ops-transformer 是 CANN 独立算子包，为 CANN 提供的补充算子集合。
> rtp-llm Ascend 推理依赖其中的 `scatter_pa_kv_cache` 等自定义算子。

```bash
# 使用预先编译好的算子包直接安装
./cann-950-ops-transformer_9.1.0-beta.3_linux-x86_64.run --full --quiet
```

> 📦 本地副本：`/workspace/work/cann-950-ops-transformer_9.0.0_linux-x86_64.run`

> **算子包生成方式**（仅供了解，非必要步骤）：
>
> `scatter_pa_kv_cache` 算子原本只存在于 ops-transformer 的 `master` 分支中，
> CANN 9.1.0-beta.3 对应的 release 分支默认不包含。因此需要将 master 分支中的
> 该算子源码拷贝到 9.1.0-beta.3 分支后进行全量编译生成 `.run` 安装包。

---

## 6. 下载安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3
```

> 📦 本地副本：`/workspace/work/Miniconda3-latest-Linux-x86_64.sh`

---

## 7. 创建 Python 3.10 环境

> **执行目录**：容器内任意路径

```bash
# 创建 conda 环境
conda create -n rtp-env python=3.10 -y

# 激活环境
conda activate rtp-env
source activate rtp-env
```

### 7.1 安装 PyTorch (可选)

> ⚠️ 此步骤可选。后续 `bazel build` 阶段会自动下载并解压 PyTorch 和 torch_npu 的 wheel 包用于编译链接，不需要手动预装。如果习惯提前安装以便在环境中调试 Python 代码，可执行以下命令：

```bash
pip install torch==2.9.0+cpu \
  -f https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl

# torch_npu 从 gitcode 下载
pip install torch_npu==2.9.0.post3 \
  -f https://gitcode.com/ascend/pytorch/-/releases/v2.9.0.post3/downloads/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl
```

### 7.2 确保 libpython3.10.so 符号链接存在

> **执行目录**：`/root/miniconda3/envs/rtp-env/lib/`

```bash
export PYTHON_BIN_PATH=/root/miniconda3/envs/rtp-env/bin/python3
cd /root/miniconda3/envs/rtp-env/lib
ln -sf libpython3.10.so.1.0 libpython3.10.so
```

---

## 8. 升级 Bazel (5.3.0 → 6.4.0)

```bash
curl -fsSL "https://mirrors.huaweicloud.com/bazel/6.4.0/bazel-6.4.0-linux-x86_64" -o /usr/bin/bazel
chmod +x /usr/bin/bazel
```

### 8.1 配置 Bazel 缓存路径

> Bazel 默认将编译缓存和外部依赖存放在 `~/.cache/bazel/` 下。
> 为避免占用系统盘空间，可将 `output_user_root` 指向数据盘路径（如 `/workspace/work/`）。

**方式一：修改全局 `.bazelrc`（推荐，持久生效）**

```bash
# 写入 /root/.bazelrc
cat >> /root/.bazelrc << 'EOF'
startup --output_user_root=/workspace/work/bazel_cache
EOF
```

> `output_user_root` 是 Bazel 所有 cache 的根目录，不同 workspace 在
> 该目录下按 `<hash>` 子目录自动隔离。修改后所有 Bazel 命令生效，无需重启。

**方式二：环境变量（当前 shell 会话生效）**

```bash
export TEST_TMPDIR=/workspace/work/bazel_cache
```

> 仅在当前终端 session 生效，退出后失效。适合临时切换。

**方式三：命令行参数（单次 build 生效）**

```bash
bazel --output_user_root=/workspace/work/bazel_cache build //rtp_llm:rtp_llm --config=ascend
```

> 每次执行需携带该参数，适合一次性场景。

**关键路径说明**：

| 配置 | 作用 | 默认值 |
|---|---|---|
| `startup --output_user_root=<dir>` | 所有 workspace 缓存根目录 | `~/.cache/bazel/_bazel_<user>/` |
| `build --disk_cache=<dir>` | 分布式持久编译缓存（可选，需额外配置） | 未设置则不使用 |

> ⚠️ `output_base` = `{output_user_root}/{workspace_hash}` —— 无需手动管理子目录，Bazel 自动为每个 workspace 分目录。

---

## 9. 对框架包管理的修改

> 以下修改确保 Bazel 构建系统使用正确的 Python 环境和 x86_64 wheel。
> **所有路径以远程包为首选**（aliyun / pythonhosted / GitHub），仅在不
> 可达时使用本地文件作为 fallback。
>
> **执行目录**：`/workspace/work/rtp-llm-npu/`

### 9.1 `deps/pip.bzl` — Python 解释器路径 + pip 缓存目录

```diff
-python_interpreter = "/opt/conda310/bin/python3"
+python_interpreter = "/root/miniconda3/envs/rtp-env/bin/python3"
```
（替换所有 8 处 `pip_parse` 中的值）

```diff
-"--cache-dir=~/.cache/pip",
+"--cache-dir=/tmp/pip-cache",
```

> 📦 torch / torch_npu wheel 本地副本：
> - `/workspace/work/torch-2.9.0+cpu-cp310-cp310-manylinux_2_28_x86_64.whl`
> - `/workspace/work/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl`

### 9.2 `deps/http.bzl` — torch / torch_npu wheel URL aarch64 → x86_64

> torch_npu 2.9.0.post3 的 x86_64 wheel 仅发布在 gitcode，pythonhosted 上无 x86_64 版本（aarch64→x86_64 替换会导致哈希路径不匹配 404）。torch 继续使用 aliyun 远程 URL。

```diff
   http_archive(
       name = "torch_cpu_ascend",
       urls = [
-         "https://mirrors.aliyun.com/...torch-2.9.0%2Bcpu...manylinux_2_28_aarch64.whl",
+         "https://mirrors.aliyun.com/...torch-2.9.0%2Bcpu...manylinux_2_28_x86_64.whl",
       ],
   )

   http_archive(
       name = "torch_npu_ascend",
       urls = [
-         "https://files.pythonhosted.org/.../torch_npu-2.9.0-cp310-cp310-manylinux_2_28_aarch64.whl",
+         "https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.9.0/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl",
       ],
   )
```

> gitcode 的 `releases/download/<tag>/<filename>` 格式支持直接 HTTP 下载（非 SPA 路由），Bazel `http_archive` 可直接使用。

### 9.3 `deps/requirements_ascend.txt` — 改为 x86_64 URL

```diff
-https://mirrors.aliyun.com/...torch-2.9.0%2Bcpu...aarch64.whl
+https://mirrors.aliyun.com/...torch-2.9.0%2Bcpu...x86_64.whl
-https://files.pythonhosted.org/...torch_npu-2.9.0...aarch64.whl
+https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.9.0/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl
```

### 9.4 `deps/requirements_lock_ascend.txt` — 改为 x86_64 URL

```diff
-torch @ https://mirrors.aliyun.com/...aarch64.whl
+torch @ https://mirrors.aliyun.com/...x86_64.whl
-torch-npu @ https://files.pythonhosted.org/...aarch64.whl
+torch-npu @ https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.9.0/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl
```

### 9.5 `deps/git.bzl` — GitHub 不可达时的本地替代（可选）

> ⚠️ 仅在 GitHub 网络不可达时需要。**网络正常时完全跳过此步**。

将所有 `git_repository` / `new_git_repository` 的 `remote` 从 `https://github.com/...` 改为本地 `file://` 路径。具体操作参考下方代码。

```bash
# 准备本地仓库
LOCAL="/workspace/work/.local_repos"
rm -rf $LOCAL && mkdir -p $LOCAL

# 复制已缓存的完整仓库（如从其他构建环境复用）
SRC=/path/to/bazel-cache/external
for repo in rules_cc rules_python com_google_googletest com_google_absl \
            rapidjson grpc boringssl havenask; do
  cp -r $SRC/$repo $LOCAL/
  cd $LOCAL/$repo && git init -q && git add -A && git commit -q -m "init"
done

# 创建 CUDA 专用空 stub（ascend 编译不依赖）
for repo in cutlass cutlass_fa cutlass_h_moe cutlass3.6 cutlass4.0 \
            flashinfer_cpp flashmla nacos_sdk_cpp KleidiAI; do
  mkdir -p $LOCAL/$repo && cd $LOCAL/$repo && touch .gitkeep
  git init -q && git add -A && git commit -q -m "init"
done
```

然后修改 `deps/git.bzl` 中的 `remote` 为 `file://` 路径，`commit` 改为 `git rev-parse HEAD` 的值。

---

## 10. 对源码的修改

> **执行目录**：`/workspace/work/rtp-llm-npu/`

### 10.1 `.bazelrc` — 移除 x86_64 不支持的 ARM 编译选项 + 修改 PYTHON_BIN_PATH

```diff
-build:ascend --copt="-march=armv8.2-a+fp16+dotprod+crc"
+#build:ascend --copt="-march=armv8.2-a+fp16+dotprod+crc" # x86_64 not applicable
```

```diff
-build --action_env PYTHON_BIN_PATH="/opt/conda310/bin/python3"
+build --action_env PYTHON_BIN_PATH="/root/miniconda3/envs/rtp-env/bin/python3"
```

### 10.2 `BUILD` — py_runtime 路径

```diff
 py_runtime(
     name = "python310",
-    interpreter_path = "/opt/conda310/bin/python",
-    stub_shebang = "#!/opt/conda310/bin/python",
+    interpreter_path = "/root/miniconda3/envs/rtp-env/bin/python",
+    stub_shebang = "#!/root/miniconda3/envs/rtp-env/bin/python",
 )
```

### 10.3 `3rdparty/py/BUILD.tpl` — libpython3.10.so 路径

```diff
-cp -f "/opt/conda310/lib/libpython3.10.so" "$(@D)/libpython3.10.so"
+cp -f "/root/miniconda3/envs/rtp-env/lib/libpython3.10.so" "$(@D)/libpython3.10.so"
```

### 10.4 `3rdparty/aclnn_custom_ops/build_for_bazel.sh` — conda 环境路径

```diff
-if [ -d "/root/miniconda3/envs/py310/bin" ]; then
-    export PATH="/root/miniconda3/envs/py310/bin:${PATH}"
-    HOST_PYTHON="/root/miniconda3/envs/py310/bin/python3"
+if [ -d "/root/miniconda3/envs/rtp-env/bin" ]; then
+    export PATH="/root/miniconda3/envs/rtp-env/bin:${PATH}"
+    HOST_PYTHON="/root/miniconda3/envs/rtp-env/bin/python3"
```

### 10.5 `rtp_llm/models_py/modules/factory/attention/ascend_impl/ascend_kv_cache_write_op.py` — 添加 `cache_mode="Norm"`

> `aclnnScatterPaKvCache` 算子对 Qwen3-0.6B 的 `head_dim=128`（bf16 = 256 bytes）报 `EZ0026 Invalid_Argument`。

```diff
         torch_npu.npu_scatter_pa_kv_cache(
-            key, value, k_view, v_view, slot_mapping,
+            key, value, k_view, v_view, slot_mapping, cache_mode="Norm",
         )
```

---

## 11. 编译 rtp-llm 框架

> **执行目录**：`/workspace/work/rtp-llm-npu/`

```bash
source /usr/local/Ascend/cann/set_env.sh
export PYTHON_BIN_PATH=/root/miniconda3/envs/rtp-env/bin/python3
cd /workspace/work/rtp-llm-npu

bazel build //rtp_llm:rtp_llm --verbose_failures --config=ascend
```

编译输出：`bazel-bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl`

---

## 12. 链接 & 安装 rtp-llm 框架

### 12.1 软连接 proto 文件

> **执行目录**：`/workspace/work/rtp-llm-npu/rtp_llm/cpp/model_rpc/proto/`

```bash
cd /workspace/work/rtp-llm-npu/rtp_llm/cpp/model_rpc/proto

ln -sf ../../../../bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py .
ln -sf ../../../../bazel-out/k8-opt/bin/rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py .
```

> ⚠️ 必须在 proto 目录下执行：链接 `src` 使用 `../../../../` 上溯 4 级到仓库根目录。

### 12.2 安装 wheel

> **执行目录**：`/workspace/work/rtp-llm-npu/`

```bash
cd /workspace/work/rtp-llm-npu

pip install --force-reinstall --no-deps \
  bazel-out/k8-opt/bin/rtp_llm/rtp_llm-0.2.0-cp310-cp310-manylinux1_x86_64.whl
```

> ⚠️ 不能用 `*.whl` 通配——`py3-none-any` 和 `cp310` 两个 wheel 同时存在会导致 pip 冲突。

验证安装：

```bash
python -c "import rtp_llm; print('OK')"
ls -lh $(python -c "import rtp_llm; print(rtp_llm.__path__[0])")/../libs/libth_transformer*.so
```

---

## 13. 启动推理服务

### 13.1 清理残留进程（可选）

```bash
pkill -9 -f "rtp_llm" 2>/dev/null
sleep 3
```

### 13.2 启动服务

> **执行目录**：`/workspace/work/rtp-llm-npu/`

```bash
source /usr/local/Ascend/cann/set_env.sh
conda activate rtp-env

python -m rtp_llm.start_server \
  --checkpoint_path=/workspace/weights/Qwen3-0.6B \
  --model_type=qwen_3 \
  --start_port=9000
```

预期输出（关键标志）：

```
initLogger log_file_path: .../rtp_llm/config/alog.conf
```

> 也可通过环境变量方式启动：
> ```bash
> export CHECKPOINT_PATH=/workspace/weights/Qwen3-0.6B
> export MODEL_TYPE=qwen_3
> export START_PORT=9000
> python -m rtp_llm.start_server
> ```

### 13.3 健康检查

```bash
curl -s http://127.0.0.1:9000/health
# 预期: ok
```

---

## 14. 进行推理测试验证

### 14.1 模型列表

```bash
curl -s http://127.0.0.1:9000/v1/models
```

### 14.2 OpenAI 格式推理请求

```bash
# 耗时约30s
curl -s http://127.0.0.1:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"你是什么模型"}]}'
```

预期响应：

```json
{
  "id": "chat-",
  "object": "chat.completion",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "我是一个基于大型语言模型（LLM）开发的智能助手..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 220,
    "total_tokens": 231
  },
  "aux_info": {
    "cost_time": 44195.476,
    "first_token_cost_time": 2207.724
  }
}
```

---

## 15. 常见问题

### Q1: `ModuleNotFoundError: No module named 'rtp_llm'`

**原因**：wheel 未安装或 Python 环境不正确。

**解决**：确认 conda rtp-env 已激活，重新执行步骤 12.2。

### Q2: `ImportError: libth_transformer_config.so`

**原因**：`.so` 文件未正确安装到 libs 目录。

**解决**：检查步骤 12.2 的安装输出，确认 `libth_transformer*.so` 已释放。

### Q3: `npu_scatter_pa_kv_cache: 561103 Invalid_Argument`

**原因**：CANN `aclnnScatterPaKvCache` 算子对 Qwen3-0.6B 的 `head_dim=128` 不兼容。

**解决**：确认步骤 10.5 的 `cache_mode="Norm"` 修改已应用并重新编译安装。

### Q4: `ModuleNotFoundError: rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2`

**原因**：proto 文件软链接未创建或路径错误。

**解决**：重新执行步骤 12.1，确认在 proto 目录下执行。

### Q5: 端口冲突 / `Server start fail`

**原因**：残留进程占用端口。

**解决**：执行 `pkill -9 -f "rtp_llm"; sleep 3` 后再启动。

### Q6: 启动日志 WARNING

| WARNING | 说明 | 是否需要处理 |
|---|---|---|
| `Failed to load C++ FusedRopeKVCacheOp` | Ascend NPU 通过 Python 独立实现 | ❌ 不需要 |
| `fuse is not valid` | 集群模式 fuse 服务发现，单机部署不适用 | ❌ 不需要 |
| 日志重复出现多份 | 多进程架构 (1 backend + N frontend) | ❌ 不需要 |
| `Please install pyav to use video processing functions` | 视频处理功能依赖，LLM 推理不涉及 | ❌ 不需要 |

---

## 修改文件清单

| # | 文件 | 修改要点 |
|---|---|---|
| 1 | `.bazelrc` | 注释 ARM `-march`；修改 `PYTHON_BIN_PATH` |
| 2 | `BUILD` | `py_runtime` 路径改为 rtp-env |
| 3 | `3rdparty/py/BUILD.tpl` | `libpython3.10.so` 路径改为 rtp-env |
| 4 | `deps/http.bzl` | aarch64→x86_64 URL；远程包首选 |
| 5 | `deps/requirements_ascend.txt` | aarch64→x86_64 URL |
| 6 | `deps/requirements_lock_ascend.txt` | aarch64→x86_64 URL |
| 7 | `deps/pip.bzl` | `python_interpreter`→rtp-env；`cache-dir`→`/tmp/pip-cache` |
| 8 | `3rdparty/aclnn_custom_ops/build_for_bazel.sh` | conda 路径 py310→rtp-env |
| 9 | `deps/git.bzl` | GitHub remote→本地 `file://`（可选，网络正常时跳过） |
| 10 | `rtp_llm/models_py/.../ascend_kv_cache_write_op.py` | 添加 `cache_mode="Norm"` |
