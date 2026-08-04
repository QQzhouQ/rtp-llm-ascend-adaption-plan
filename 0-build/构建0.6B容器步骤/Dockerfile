# =============================================================================
# rtp-llm Ascend NPU 推理容器镜像
# 构建命令：
#   docker build -t rtp-llm-a5-image:cann-9.1.0-beta3-x86_64 .
# =============================================================================

FROM quay.io/wjunlu/vllm-ascend-daily:main-a5-openEuler-x86

# -------------------------- 环境变量 --------------------------
ENV CANN_VERSION=9.1.0-beta.3 \
    CANN_HOME=/usr/local/Ascend/cann-9.1.0-beta.3 \
    ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest \
    ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1 \
    SOC_VERSION=ascend950dt_9582 \
    TASK_QUEUE_ENABLE=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

ENV ASCEND_TOOLKIT_HOME=${CANN_HOME} \
    ASCEND_AICPU_PATH=${CANN_HOME} \
    ASCEND_OPP_PATH=${CANN_HOME}/opp \
    TOOLCHAIN_HOME=${CANN_HOME}/toolkit \
    ASCEND_HOME_PATH=${CANN_HOME} \
    CMAKE_PREFIX_PATH=${CANN_HOME}/toolkit/tools/tikicpulib/lib/cmake:${CANN_HOME}/lib64/cmake

ENV PATH=${CANN_HOME}/bin:${CANN_HOME}/tools/ccec_compiler/bin:${CANN_HOME}/tools/profiler/bin:${CANN_HOME}/tools/msobjdump:${CANN_HOME}/tools/show_kernel_debug_data:${CANN_HOME}/tools/ascend_system_advisor/asys:${ASCEND_TOOLKIT_LATEST_HOME}/bin:${ASCEND_TOOLKIT_LATEST_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_LATEST_HOME}/tools/ccec_compiler/bin:${ATB_HOME_PATH}/bin:${PATH}

ENV LD_LIBRARY_PATH=${CANN_HOME}/lib64:${CANN_HOME}/lib64/plugin/opskernel:${CANN_HOME}/lib64/plugin/nnengine:${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${ATB_HOME_PATH}/lib:${ATB_HOME_PATH}/examples:${ATB_HOME_PATH}/tests/atbopstest:${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64/plugin:${ASCEND_TOOLKIT_LATEST_HOME}/lib64:${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

ENV PYTHONPATH=${ASCEND_TOOLKIT_LATEST_HOME}/python/site-packages:${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe:${CANN_HOME}/python/site-packages:${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe:

ENV ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0 \
    ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0 \
    ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0 \
    ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1 \
    ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5 \
    ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1 \
    ATB_COMPARE_TILING_EVERY_KERNEL=0 \
    ATB_MATMUL_SHUFFLE_K_ENABLE=1 \
    LCCL_DETERMINISTIC=0 \
    LCCL_PARALLEL=0

WORKDIR /workspace

# -------------------------- 阶段一：CANN Toolkit --------------------------
# 所有安装包均从远程下载，仅 ops-transformer 使用本地包
ARG BASE_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.T6
ARG CANN_RUN_DIR=/tmp/cann-install

RUN mkdir -p ${CANN_RUN_DIR}

# 下载并安装 CANN Toolkit 9.1.0-beta.3
RUN wget -q -P ${CANN_RUN_DIR} \
        "${BASE_URL}/Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run" && \
    chmod +x ${CANN_RUN_DIR}/Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run && \
    ${CANN_RUN_DIR}/Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run --install --quiet && \
    rm -f ${CANN_RUN_DIR}/Ascend-cann-toolkit_9.1.0-beta.3_linux-x86_64.run

# 下载并安装 CANN 950 Ops
RUN wget -q -P ${CANN_RUN_DIR} \
        "${BASE_URL}/Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run" && \
    chmod +x ${CANN_RUN_DIR}/Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run && \
    ${CANN_RUN_DIR}/Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run --install --quiet && \
    rm -f ${CANN_RUN_DIR}/Ascend-cann-950-ops_9.1.0-beta.3_linux-x86_64.run

# 安装 ops-transformer (独立算子包 — 使用本地文件)
# 需将 cann-950-ops-transformer_9.0.0_linux-x86_64.run 放入 build context 的 local_pkgs/ 目录
COPY local_pkgs/ ${CANN_RUN_DIR}/
RUN chmod +x ${CANN_RUN_DIR}/cann-950-ops-transformer_9.0.0_linux-x86_64.run && \
    ${CANN_RUN_DIR}/cann-950-ops-transformer_9.0.0_linux-x86_64.run --full --quiet && \
    rm -f ${CANN_RUN_DIR}/cann-950-ops-transformer_9.0.0_linux-x86_64.run

# profiling 头文件软链接 (CANN 9.1.0-beta.3 兼容修复)
# CANN 9.1.0-beta.3 将 profiling 头文件装在 x86_64-linux/pkg_inc/profiling/ 下
# aclnn_custom_ops 编译时依赖 include/profiling/ 路径
RUN ln -sf /usr/local/Ascend/cann-9.1.0-beta.3/x86_64-linux/pkg_inc/profiling \
            /usr/local/Ascend/cann-9.1.0-beta.3/include/profiling && \
    mkdir -p /usr/local/Ascend/cann-9.1.0-beta.3/include

# 确保 cann 和 ascend-toolkit/latest 指向正确版本
RUN ln -sfn /usr/local/Ascend/cann-9.1.0-beta.3 /usr/local/Ascend/cann && \
    ln -sfn /usr/local/Ascend/cann-9.1.0-beta.3 /usr/local/Ascend/ascend-toolkit/latest

# -------------------------- 阶段二：Miniconda + Python 环境 --------------------------
# 下载并安装 Miniconda
RUN wget -q -P /tmp \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 && \
    rm -f /tmp/Miniconda3-latest-Linux-x86_64.sh && \
    /root/miniconda3/bin/conda init bash

SHELL ["/bin/bash", "-c"]

# 创建 rtp-env (Python 3.10)
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n rtp-env python=3.10 -y && \
    conda clean -a -y

# -------------------------- 阶段三：Python 依赖 --------------------------
# 配置 pip 国内镜像源加速下载
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com

# 安装 torch / torch_npu
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate rtp-env && \
    pip install \
        torch==2.9.0+cpu \
        -f https://mirrors.aliyun.com/pytorch-wheels/cpu/ && \
    pip install \
        https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.9.0/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl && \
    pip install \
        torchvision==0.24.0+cpu \
        -f https://mirrors.aliyun.com/pytorch-wheels/cpu/

# 安装 CANN TBE 编译器 Python 依赖 + 推理框架核心依赖
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate rtp-env && \
    pip install \
        attrs \
        decorator \
        psutil \
        scipy

# 安装推理运行时依赖
RUN source /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate rtp-env && \
    pip install \
        numpy==2.2.6 \
        transformers==4.51.2 \
        sentence-transformers==5.6.0 \
        fastapi==0.139.2 \
        uvicorn==0.30.0 \
        starlette==1.3.1 \
        grpcio==1.82.1 \
        protobuf==4.25.0 \
        sentencepiece==0.2.2 \
        tiktoken==0.13.0 \
        safetensors==0.8.0 \
        pybind11-stubgen==2.5.5 \
        einops==0.8.2 \
        Jinja2==3.1.6 \
        PyYAML==6.0.3 \
        packaging==26.0 \
        requests==2.34.2 \
        regex==2026.7.19 \
        rich==15.0.0 \
        openai==2.46.0 \
        httpx==0.28.1 \
        aiohttp==3.14.2 \
        dashscope==1.26.4 \
        cryptography==49.0.0 \
        jmespath==0.10.0 \
        prettytable==3.18.0 \
        bitsandbytes==0.49.2 \
        timm==0.9.12 \
        librosa==0.11.0 \
        setproctitle==1.3.7 \
        concurrent-log-handler==0.9.29 \
        hf-xet==1.5.2 \
        lru-dict==1.4.1 \
        oss2==2.19.1 \
        pillow_heif==1.5.0 \
        pillow-avif-plugin==1.6.0 \
        thrift==0.24.0

# 确保 libpython3.10.so 符号链接存在
RUN ln -sf /root/miniconda3/envs/rtp-env/lib/libpython3.10.so.1.0 \
           /root/miniconda3/envs/rtp-env/lib/libpython3.10.so && \
    rm -rf ${CANN_RUN_DIR}

# -------------------------- 阶段四：Bazel 升级 --------------------------
RUN curl -fsSL "https://mirrors.huaweicloud.com/bazel/6.4.0/bazel-6.4.0-linux-x86_64" \
         -o /usr/bin/bazel && \
    chmod +x /usr/bin/bazel

# -------------------------- 阶段五：运行时配置 --------------------------
# 更新 .bashrc 中的 CANN 路径
RUN sed -i 's|cann-9\.0\.0|cann-9.1.0-beta.3|g' /root/.bashrc

# 容器入口：source CANN 环境 -> 保持运行
ENTRYPOINT ["/bin/bash", "-c", \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
     source /usr/local/Ascend/cann-9.1.0-beta.3/share/info/ascendnpu-ir/bin/set_env.sh && \
     source /usr/local/Ascend/nnal/atb/set_env.sh && \
     exec \"$@\"", "--"]
CMD ["bash"]

# 元数据
LABEL maintainer="rtp-llm" \
      cann.version="9.1.0-beta.3" \
      torch.version="2.9.0+cpu" \
      torch_npu.version="2.9.0.post3" \
      python.version="3.10" \
      soc="ascend950dt_9582"
