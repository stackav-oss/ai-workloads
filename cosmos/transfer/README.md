# COSMOS Physical AI Benchmark

This repository contains scripts for running COSMOS inference and benchmark evaluation using the Physical AI Benchmark dataset.

## 🚀 Quick Start

### Environment Setup
Make sure to use `ubuntu:24.04` image and set `HF_TOKEN` for your container. You need to have `git` installed to clone this directory:

### Example K8S container definition

```bash
containers:
- name: ai-workloads
  image: ubuntu:24.04
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
docker run --gpus=all -it --rm -e HF_TOKEN="$HF_TOKEN" ubuntu:24.04
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
cd /ai-workloads/cosmos/transfer
```


### Run steps

```bash
./step1_setup.sh
./step2_download_data.sh
./step3_inference.sh   {control_type}   # 'vis', 'edge', 'depth', 'seg', 'all'
./step4_evaluation.sh  {control_type}   # 'vis', 'edge', 'depth', 'seg', 'all'
```


## Evaluation Results

```bash
Model: Cosmos-Transfer2.5-2B
```

### Reference Paper Results
https://arxiv.org/pdf/2511.00062 Table 12

|    Model                                |  Blur SSIM ↑ |   Edge F1  ↑  | Depth si-RMSE ↓ |  Mask mIoU  ↑ | Quality Score ↑ |
|-----------------------------------------|--------------|---------------|-----------------|---------------|-----------------|
| Cosmos-Transfer2.5-2B [Blur]            |     0.90     |      0.26     |       0.59      |      0.75     |       9.75      |
| Cosmos-Transfer2.5-2B [Edge]            |     0.79     |      0.49     |       0.76      |      0.75     |       8.73      |
| Cosmos-Transfer2.5-2B [Depth]           |     0.71     |      0.19     |       0.70      |      0.73     |       8.85      |
| Cosmos-Transfer2.5-2B [Seg]             |     0.68     |      0.14     |       1.02      |      0.71     |       8.81      |
| Cosmos-Transfer2.5-2B [Uniform]         |     0.87     |      0.41     |       0.67      |      0.76     |       9.31      |


### Benchmark Results

|    Model                                |  Blur SSIM ↑ |   Edge F1  ↑  | Depth si-RMSE ↓ |  Mask mIoU  ↑ | Quality Score ↑ |
|-----------------------------------------|--------------|---------------|-----------------|---------------|-----------------|
| Cosmos-Transfer2.5-2B [Blur]    (GB200) |     0.88     |      0.17     |       0.86      |      0.71     |       7.44      |
| Cosmos-Transfer2.5-2B [Edge]    (GB200) |     0.76     |      0.39     |       0.71      |      0.74     |       8.31      |
| Cosmos-Transfer2.5-2B [Edge]    (H100)  |     0.76     |      0.39     |       0.71      |      0.74     |       8.43      |
| Cosmos-Transfer2.5-2B [Depth]   (GB200) |     0.69     |      0.18     |       0.86      |      0.72     |       8.04      |
| Cosmos-Transfer2.5-2B [Seg]     (GB200) |     0.65     |      0.14     |       1.07      |      0.71     |       8.16      |
| Cosmos-Transfer2.5-2B [Uniform] (GB200) |     0.46     |      0.18     |       1.05      |      0.68     |       6.78      |

