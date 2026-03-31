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
./step5_throughput.sh
```


## Evaluation Results

```bash
Model: Cosmos-Transfer2.5-2B
```

### Reference Quality Scores
https://arxiv.org/pdf/2511.00062 Table 12

|    Model                                |  Blur SSIM ↑ |   Edge F1  ↑  | Depth si-RMSE ↓ |  Mask mIoU  ↑ | Quality Score ↑ |
|-----------------------------------------|--------------|---------------|-----------------|---------------|-----------------|
| Cosmos-Transfer2.5-2B [Blur]            |     0.90     |      0.26     |       0.59      |      0.75     |       9.75      |
| Cosmos-Transfer2.5-2B [Edge]            |     0.79     |      0.49     |       0.76      |      0.75     |       8.73      |
| Cosmos-Transfer2.5-2B [Depth]           |     0.71     |      0.19     |       0.70      |      0.73     |       8.85      |
| Cosmos-Transfer2.5-2B [Seg]             |     0.68     |      0.14     |       1.02      |      0.71     |       8.81      |
| Cosmos-Transfer2.5-2B [Uniform]         |     0.87     |      0.41     |       0.67      |      0.76     |       9.31      |


### Quality Scores with 1 prompt variation

|    Model                                |  Blur SSIM ↑ |   Edge F1  ↑  | Depth si-RMSE ↓ |  Mask mIoU  ↑ | Quality Score ↑ |
|-----------------------------------------|--------------|---------------|-----------------|---------------|-----------------|
| Cosmos-Transfer2.5-2B [Blur]    (GB200) |     0.88     |      0.17     |       0.86      |      0.71     |       7.44      |
| Cosmos-Transfer2.5-2B [Edge]    (GB200) |     0.76     |      0.39     |       0.71      |      0.74     |       8.31      |
| Cosmos-Transfer2.5-2B [Depth]   (GB200) |     0.69     |      0.18     |       0.86      |      0.72     |       8.04      |
| Cosmos-Transfer2.5-2B [Seg]     (GB200) |     0.65     |      0.14     |       1.07      |      0.71     |       8.16      |
| Cosmos-Transfer2.5-2B [Uniform] (GB200) |     0.46     |      0.18     |       1.05      |      0.68     |       6.78      |


### Quality Scores with 6 prompt variation

|    Model                                |  Blur SSIM ↑ |   Edge F1  ↑  | Depth si-RMSE ↓ |  Mask mIoU  ↑ | Quality Score ↑ |
|-----------------------------------------|--------------|---------------|-----------------|---------------|-----------------|
| Cosmos-Transfer2.5-2B [Blur]    (GB200) |     0.88     |      0.17     |       0.88      |      0.72     |       8.14      |
| Cosmos-Transfer2.5-2B [Edge]    (GB200) |     0.73     |      0.40     |       0.81      |      0.73     |       8.89      |
| Cosmos-Transfer2.5-2B [Depth]   (GB200) |     0.66     |      0.17     |       0.89      |      0.72     |       8.67      |
| Cosmos-Transfer2.5-2B [Seg]     (GB200) |     0.62     |      0.13     |       1.19      |      0.71     |       8.25      |
| Cosmos-Transfer2.5-2B [Uniform] (GB200) |     N/A      |      N/A      |       N/A       |      N/A      |       N/A       |


### Runtime Metrics For Single Video Generation

```bash
Output video dimensions: 640x480
Output video frame count: 121
Output video frame rate: 30
Output video length: 5 seconds
```


#### H100

| Control Type | GPU  | GPU # | Seconds|
|--------------|------|-------|--------|
| Blur         | B200 |   1   |    180 |
| Blur         | B200 |   2   |    120 |
| Blur         | B200 |   4   |    77  |
| Edge         | B200 |   1   |    180 |
| Edge         | B200 |   2   |    120 |
| Edge         | B200 |   4   |    77  |
| Depth        | B200 |   1   |    180 |
| Depth        | B200 |   2   |    120 |
| Depth        | B200 |   4   |    78  |
| Seg          | B200 |   1   |    180 |
| Seg          | B200 |   2   |    120 |
| Seg          | B200 |   4   |    77  |
| Uniform      | B200 |   1   |    264 |
| Uniform      | B200 |   2   |    181 |
| Uniform      | B200 |   4   |    125 |


#### B200

| Control Type | GPU  | GPU # | Seconds|
|--------------|------|-------|--------|
| Blur         | H100 |   1   |    350 |
| Blur         | H100 |   2   |    201 |
| Blur         | H100 |   4   |    115 |
| Blur         | H100 |   8   |    72  |
| Edge         | H100 |   1   |    351 |
| Edge         | H100 |   2   |    201 |
| Edge         | H100 |   4   |    116 |
| Edge         | H100 |   8   |    73  |
| Depth        | H100 |   1   |    351 |
| Depth        | H100 |   2   |    201 |
| Depth        | H100 |   4   |    114 |
| Depth        | H100 |   8   |    73  |
| Seg          | H100 |   1   |    351 |
| Seg          | H100 |   2   |    201 |
| Seg          | H100 |   4   |    114 |
| Seg          | H100 |   8   |    72  |
| Uniform      | H100 |   1   |    487 |
| Uniform      | H100 |   2   |    283 |
| Uniform      | H100 |   4   |    165 |
| Uniform      | H100 |   8   |    105 |
