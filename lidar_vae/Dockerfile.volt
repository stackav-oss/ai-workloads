# Dockerfile for lidar_vae on Volt (ARM64 / Blackwell GPUs)
FROM ghcr.io/prefix-dev/pixi:noble-cuda-13.0.0

USER root

# Install system libraries needed for OpenCV, rendering, and spconv JIT compilation.
# Use CUDA 12.9 nvcc because CUDA 13 CCCL requires C++17 but spconv uses C++14.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    g++ \
    gcc \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    make \
    ninja-build \
    cuda-nvcc-12-9 \
    cuda-cudart-dev-12-9 \
    libcurand-dev-12-9 \
    libcublas-dev-12-9 \
    libcusparse-dev-12-9 \
    libcusolver-dev-12-9 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lidar-vae-pixi

COPY pixi.volt.toml ./pixi.toml

RUN CONDA_OVERRIDE_CUDA=13.0 pixi install

RUN pixi run pip install --no-build-isolation torch-scatter && \
    (pixi run pip install --no-build-isolation "mmcv>=2.0.0rc4,<2.2.0" || \
     pixi run pip install --no-build-isolation git+https://github.com/open-mmlab/mmcv.git@v2.1.0) && \
    pixi run pip install --no-deps "mmdet3d>=1.1.0,<1.5.0"

ENV CUMM_CUDA_ARCH_LIST="10.0"
RUN pixi run pip install --index-url https://pypi.org/simple/ pccm && \
    git clone --filter=blob:none --branch v0.8.2 https://github.com/FindDefinition/cumm.git /opt/cumm && \
    pixi run pip install --no-build-isolation -e /opt/cumm && \
    git clone --filter=blob:none --branch v2.3.8 https://github.com/traveller59/spconv.git /opt/spconv && \
    pixi run pip install --no-build-isolation --no-deps -e /opt/spconv && \
    pixi run pip uninstall -y lance && \
    pixi run pip install --force-reinstall --no-deps pylance==4.0.1

ENV PIXI_ENV=/opt/lidar-vae-pixi/.pixi/envs/default
ENV CUDA_HOME=/usr/local/cuda-12.9
ENV CUDA_LIB=${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cu13/lib
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_LIB}:${PIXI_ENV}/lib:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_LIB}
ENV CPLUS_INCLUDE_PATH=${CUDA_HOME}/include
ENV PATH="${CUDA_HOME}/bin:${PIXI_ENV}/bin:${PATH}"
ENV PYTHONPATH=/workspace/ai-workloads/lidar_vae/autoencoder:/workspace/ai-workloads/lidar_vae/jormungand:/workspace/ai-workloads/lidar_vae/third_party/OpenSceneFlow
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0"
ENV CUMM_CUDA_VERSION="12.9"

RUN ln -sf /usr/local/cuda-12.9 /usr/local/cuda && \
    ln -sf ${CUDA_LIB}/libnvrtc.so.13 ${CUDA_HOME}/lib64/libnvrtc.so && \
    ln -sf ${CUDA_LIB}/libnvrtc-builtins.so.13.0 ${CUDA_HOME}/lib64/libnvrtc-builtins.so && \
    ln -sf ${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cu13/include/nvrtc.h ${CUDA_HOME}/include/nvrtc.h

RUN ln -sf /usr/local/cuda-12.9/bin/nvcc ${PIXI_ENV}/bin/nvcc && \
    ln -sf /usr/local/cuda-12.9/bin/ptxas ${PIXI_ENV}/bin/ptxas && \
    ln -sf /usr/local/cuda-12.9/bin/cicc ${PIXI_ENV}/bin/cicc && \
    ln -sf /usr/local/cuda-12.9/bin/fatbinary ${PIXI_ENV}/bin/fatbinary

RUN echo 'eval "$(cd /opt/lidar-vae-pixi && pixi shell-hook)"' >> ~/.bashrc \
    && echo 'export PATH="/usr/local/cuda-12.9/bin:$PATH"' >> ~/.bashrc \
    && echo 'export CUDA_HOME=/usr/local/cuda-12.9' >> ~/.bashrc \
    && echo 'export CUMM_CUDA_ARCH_LIST="10.0"' >> ~/.bashrc \
    && echo 'export CUMM_CUDA_VERSION="12.9"' >> ~/.bashrc \
    && echo 'export CPLUS_INCLUDE_PATH=/usr/local/cuda-12.9/include' >> ~/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc \
    && cp ~/.bashrc ~/.profile

WORKDIR /workspace/ai-workloads/lidar_vae
COPY autoencoder/ autoencoder/
COPY jormungand/ jormungand/
COPY third_party/ third_party/
COPY pyproject.toml pixi.toml pixi.volt.toml ./

WORKDIR /workspace/ai-workloads/lidar_vae/autoencoder

CMD ["/bin/bash", "-l"]
