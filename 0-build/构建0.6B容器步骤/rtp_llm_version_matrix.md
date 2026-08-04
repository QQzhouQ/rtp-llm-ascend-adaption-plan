# rtp-llm (Ascend NPU) 版本配套表

> 生成时间：2026-07-27
> 验证状态：✅ 编译通过，推理正常（Qwen3-0.6B, 首 token ~2.2s, 220 tokens）

## 一、硬件平台

| 项目 | 规格 |
|---|---|
| CPU 架构 | x86_64 |
| NPU 型号 | Ascend 950PR (9579) |
| NPU 数量 | 8 × davinci |
| 显存 | 128 GB HBM /卡 |

## 二、系统软件

| 组件 | 版本 | 来源 |
|---|---|---|
| 操作系统 | openEuler 24.03 (LTS-SP2) | 宿主机 |
| 内核 | 6.6.0-132.0.0.111.oe2403sp3 | — |
| glibc | 2.38 | 系统自带 |
| GCC | 12.3.1 | 系统自带 |
| CMake | 4.3.2 | `pip install cmake` |

## 三、CANN / Ascend 工具链

| 组件 | 版本 | 来源 / 获取方式 |
|---|---|---|
| CANN Toolkit | **9.1.0-beta.3** | `wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T6/Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run` |
| CANN ops (Kernel) | **9.1.0-beta.3** | `wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T6/Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run` |
| Ascend Driver | 25.7.rc1 | 宿主机自带 |
| ops-transformer | 独立算子包 | 已编译的本地包，build_container/local_pkgs/cann-950-ops-transformer_9.0.0_linux-x86_64.run |
| aclnn_custom_ops | 源码自动编译 | 仓库内 `3rdparty/aclnn_custom_ops/`，Bazel自动编译 |
| TBE Python 依赖 | scipy, attrs, psutil, decorator | `pip install scipy attrs psutil decorator` |

## 四、编译工具链

| 组件 | 版本 | 来源 / 获取方式 |
|---|---|---|
| Bazel | 6.4.0 | `curl -fsSL https://mirrors.huaweicloud.com/bazel/6.4.0/bazel-6.4.0-linux-x86_64 -o /usr/bin/bazel` |
| Python (编译用) | 3.10.20 | conda虚拟环境 |
| conda | 26.5.3 | `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh` |

## 五、PyTorch / torch_npu

| 组件 | 版本 | 来源 / 获取方式 |
|---|---|---|
| PyTorch | 2.9.0+cpu (x86_64) | Bazel `http_archive` 自动下载：`https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl` |
| torch_npu | 2.9.0.post3 (x86_64) | gitcode 远程下载：`https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.9.0/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl` |
| torchvision | 0.24.0+cpu | pip（wheel 依赖链自动安装） |

## 六、Docker 镜像信息

| 项目 | 值 |
|---|---|
| 基础镜像 | quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86 |
| 预装 CANN | 9.0.0（后续升级为 9.1.0-beta.3） |
