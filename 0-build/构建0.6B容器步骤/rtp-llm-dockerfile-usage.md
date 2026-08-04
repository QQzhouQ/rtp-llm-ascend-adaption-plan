# rtp-llm Dockerfile 使用说明

## 概述

基于 `quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86` 基础镜像，
构建包含 CANN 9.1.0-beta.3 + Python 3.10 rtp-env 的 Ascend NPU 推理容器镜像。

## 目录结构

构建前确保以下目录结构：

```
Dockerfile所在目录/
├── Dockerfile
└── local_pkgs/                                    # ※ 本地包目录（必需）
    └── cann-950-ops-transformer_9.0.0_linux-x86_64.run   # 独立算子包
```

## 本地包目录说明

| 目录 | 用途 | 说明 |
|------|------|------|
| `local_pkgs/` | 存放无法在线下载的包 | **仅需放入 ops-transformer 独立算子包** |

所有其他安装包均从远程下载，无需放入本地目录。

## 安装包获取方式一览

| 包名 | 获取方式 | 远程地址 |
|------|----------|----------|
| CANN Toolkit 9.1.0-beta.3 | **远程下载** | `ascend-repo.obs.cn-east-2.myhuaweicloud.com` |
| CANN 950 Ops 9.1.0-beta.3 | **远程下载** | `ascend-repo.obs.cn-east-2.myhuaweicloud.com` |
| ops-transformer 9.0.0 | **本地 `local_pkgs/`** | 需自行编译或从已有环境拷贝 |
| Miniconda | **远程下载** | `repo.anaconda.com` |
| torch 2.9.0+cpu | **远程下载 (pip)** | `mirrors.aliyun.com` |
| torch_npu 2.9.0.post3 | **远程下载 (pip)** | `gitcode.com` |
| torchvision 0.24.0+cpu | **远程下载 (pip)** | `mirrors.aliyun.com` |
| Bazel 6.4.0 | **远程下载** | `mirrors.huaweicloud.com` |

## 构建步骤

### 1. 准备 local_pkgs

在 Dockerfile 同级目录创建 `local_pkgs/` 并放入 ops-transformer 安装包：

```bash
mkdir -p local_pkgs
cp /path/to/cann-950-ops-transformer_9.0.0_linux-x86_64.run local_pkgs/
```

### 2. 构建镜像

```bash
docker build -t rtp-llm-a5-image:cann-9.1.0-beta3-x86_64 .
```

> 构建预计耗时 15-30 分钟（视网络状况），CANN 安装包共约 4.7GB。
> pip 已配置阿里云镜像源加速，conda 自动同意 Terms of Service。

### 3. 创建容器

```bash
docker run -itd --name rtp-llm \
  --privileged --network host --security-opt label=disable --shm-size=1g \
  --hostname localhost.localdomain -w /workspace \
  -v /home:/home \
  -v /mnt:/workspace \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  rtp-llm-a5-image:cann-9.1.0-beta3-x86_64
```

### 4. 进入容器

```bash
docker exec -it rtp-llm bash
```

### 5. 激活 conda 环境

```bash
conda activate rtp-env
```

## 镜像内关键路径

| 路径 | 说明 |
|------|------|
| `/usr/local/Ascend/cann-9.1.0-beta.3/` | CANN 安装目录 |
| `/usr/local/Ascend/cann` | CANN 默认版本软链接 |
| `/root/miniconda3/envs/rtp-env/` | Python 3.10 conda 环境 |
| `/workspace/` | 工作目录 |

## 核心版本信息

| 组件 | 版本 |
|------|------|
| CANN | 9.1.0-beta.3 |
| Python | 3.10 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post3 |
| transformers | 4.51.2 |
| numpy | 2.2.6 |
| Bazel | 6.4.0 |
