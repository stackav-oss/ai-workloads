# COSMOS Physical AI Benchmark

This repository contains scripts for running COSMOS inference and benchmark evaluation using the Physical AI Benchmark dataset.

## 🚀 Quick Start

### Environment Setup
Make sure to use `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` image and set `HF_TOKEN` for your container. You need to have `git` installed to clone this directory:

### Example K8S container definition

```bash
containers:
- name: ai-workloads
  image: nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
  command: ["/bin/bash"]
  args: ["-c", "sleep infinity"]
  resources:
      limits:
      nvidia.com/gpu: 4
      requests:
      nvidia.com/gpu: 4
  env:
  - name: HF_TOKEN
    value: "your_hf_token"
```

### Example docker cmd

```bash
docker run --gpus=all -it --rm -e HF_TOKEN="$HF_TOKEN" nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
```

### Install git
```bash
apt update
apt install -y git
```

### Clone ai-workloads repository

```bash
cd /
git clone https://github.com/stackav-oss/ai-workloads.git
cd /ai-workloads/
git checkout ugur/cosmos-transfer
cd /ai-workloads/cosmos/transfer
```

### Text-to-World Pipeline

```bash
./step1_setup.sh
./step2_download_data.sh
./step3_inference.sh
```
