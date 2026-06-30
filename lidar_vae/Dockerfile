# Single-image Dockerfile for lidar_vae.
# Build with: ./build_pig.sh
FROM ghcr.io/prefix-dev/pixi:noble-cuda-13.0.0

SHELL ["/bin/bash", "-c"]
USER root

ARG UID=1000
ARG GID=1000
ARG USERNAME=developer

ENV DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime \
    && echo "/usr/share/zoneinfo/America/New_York" > /etc/timezone

RUN apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        btop \
        ca-certificates \
        curl \
        g++ \
        gcc \
        git \
        gpg \
        jq \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libx11-dev \
        libxext6 \
        libxrender-dev \
        make \
        ninja-build \
        nvtop \
        software-properties-common \
        tmux \
        unzip \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

RUN (groupadd -g "${GID}" "${USERNAME}" || true) \
    && (useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}" || true) \
    && mkdir -p /project \
    && chown -R "${UID}:${GID}" /project

WORKDIR /opt/lidar-vae-pixi

COPY pixi.toml .

RUN pixi install
RUN pixi run pip install --no-build-isolation torch-scatter \
    && (pixi run pip install --no-build-isolation "mmcv>=2.0.0rc4,<2.2.0" \
        || pixi run pip install --no-build-isolation git+https://github.com/open-mmlab/mmcv.git@v2.1.0) \
    && pixi run pip install --no-deps "mmdet3d>=1.1.0,<1.5.0"

ENV PIXI_ENV=/opt/lidar-vae-pixi/.pixi/envs/default
ENV LD_LIBRARY_PATH=${PIXI_ENV}/lib:${LD_LIBRARY_PATH}
ENV PATH=${PIXI_ENV}/bin:${PATH}
ENV PYTHONPATH=/project/ai-workloads/lidar_vae/autoencoder:/project/ai-workloads/lidar_vae/jormungand:/project/ai-workloads/lidar_vae/third_party/OpenSceneFlow
ENV PYTHONUNBUFFERED=1
ENV VOLT_CONFIG_DIR=/home/${USERNAME}/.volt/config
ENV AWS_PROFILE=magic

RUN echo 'eval "$(cd /opt/lidar-vae-pixi && pixi shell-hook)"' >> /root/.bashrc \
    && echo 'eval "$(cd /opt/lidar-vae-pixi && pixi shell-hook)"' >> /root/.profile \
    && if [[ -d "/home/${USERNAME}" ]]; then \
        echo 'eval "$(cd /opt/lidar-vae-pixi && pixi shell-hook)"' >> "/home/${USERNAME}/.bashrc"; \
        echo 'eval "$(cd /opt/lidar-vae-pixi && pixi shell-hook)"' >> "/home/${USERNAME}/.profile"; \
        chown "${UID}:${GID}" "/home/${USERNAME}/.bashrc" "/home/${USERNAME}/.profile"; \
    fi

WORKDIR /project/ai-workloads/lidar_vae/autoencoder

CMD ["/bin/bash", "-l"]
