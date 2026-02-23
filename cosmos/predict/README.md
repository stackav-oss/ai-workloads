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
cd /ai-workloads/cosmos/predict
```

### Text-to-World Pipeline

```bash
./step1_setup.sh
./step2_download_data.sh
./step3_inference.sh text2world
./step4_evaluation.sh text2world
./step5_throughput.sh text2world
```

### Image-to-World Pipeline

```bash
./step1_setup.sh
./step2_download_data.sh
./step3_inference.sh image2world
./step4_evaluation.sh image2world
./step5_throughput.sh image2world
```

## 📁 File Structure

- **`step1_setup.sh`** - System setup and repository cloning
- **`step2_download_data.sh`** - Dataset download from Hugging Face
- **`step3_inference.sh`** - Run inference (text2world/image2world)
- **`step4_evaluation.sh`** - Run evaluation benchmarks (domain + quality scores)
- **`step5_throughput.sh`** - Run throughput testing with multiple GPU configurations
- **`inference.py`** - Main inference script
- **`throughput.py`** - Throughput evaluation script  
- **`aggregate_pai_bench_results.py`** - PAI-Bench results aggregation and final scoring
- **`generate_throughput_summary.py`** - Throughput results compilation

## Reference Paper
https://arxiv.org/pdf/2511.00062 Table 10 and Table 11

## Evaluation Results

### text2world evaluation scores:
|            | Domain Score | Quality Score | Overall Score |
|------------|--------------|---------------|---------------|
| Paper      | 0.804        | 0.732         | 0.768         |
| H100       | 0.807        | 0.732         | 0.770         |
| B200       | 0.802        | 0.732         | 0.767         |

### image2world evaluation scores:
|            | Domain Score | Quality Score | Overall Score |
|------------|--------------|---------------|---------------|
| Paper      | 0.840        | 0.779         | 0.810         |
| H100       | 0.836.       | 0.780         | 0.808         |
| B200       | 0.39.        | 0.780         | 0.10          |


## Throughput Results

### text2world

| **TEXT2WORLD** | **H100 (seconds per video)** | **B200 (seconds per video)** |
|----------------|-------------------------------|-------------------------------|
| **1 GPU**      | 249                           | 125                           |
| **2 GPU**      | 138                           | 75                            |
| **4 GPU**      | 74                            | 45                            |
| **8 GPU**      | 42                            | N/A                           |

### image2world

| **IMAGE2WORLD** | **H100 (seconds per video)** | **B200 (seconds per video)** |
|-----------------|-------------------------------|-------------------------------|
| **1 GPU**       | 249                           | 125                           |
| **2 GPU**       | 139                           | 77                            |
| **4 GPU**       | 76                            | 46                            |
| **8 GPU**       | 44                            | N/A                           |


## H100 Detailed Evaluation Results

### text2world
```bash
Overall Accuracy: 0.8070
Total Videos Evaluated: 1036

CATEGORY-SPECIFIC SCORES:
----------------------------------------
AV             : 0.6399
COMMON_SENSE   : 0.8559
HUMAN          : 0.7930
INDUSTRY       : 0.8417
MISC           : 0.8912
PHYSICS        : 0.9264
ROBOT          : 0.7636

Domain  score 0.807
Quality score 0.732
Overall score 0.770
```

### image2world
```bash
Overall Accuracy: 0.8359
Total Videos Evaluated: 1036

CATEGORY-SPECIFIC SCORES:
----------------------------------------
AV             : 0.6600
COMMON_SENSE   : 0.9291
HUMAN          : 0.8102
INDUSTRY       : 0.8558
MISC           : 0.9263
PHYSICS        : 0.9331
ROBOT          : 0.7998

Domain  score 0.836
Quality score 0.780
Overall score 0.808
```

## H100 Detailed Throughput Results

### text2world
```bash
1 GPU:
Average Time: 249.1129 seconds
Std Deviation: 0.4098 seconds

2 GPUs:
Average Time: 138.3463 seconds
Std Deviation: 0.5924 seconds

4 GPUs:
Average Time: 74.9248 seconds
Std Deviation: 0.1208 seconds

8 GPUs:
Average Time: 42.8290 seconds
Std Deviation: 0.4511 seconds
```

### image2world
```bash
1 GPU:
Average Time: 249.1734 seconds
Std Deviation: 0.4824 seconds

2 GPUs:
Average Time: 139.3447 seconds
Std Deviation: 0.1777 seconds

4 GPUs:
Average Time: 76.3702 seconds
Std Deviation: 0.4741 seconds

8 GPUs:
Average Time: 44.0013 seconds
Std Deviation: 0.3202 seconds
```

## B200 Detailed Evaluation Results

### text2world
```bash
Overall Accuracy: 0.8016
Total Videos Evaluated: 1036

CATEGORY-SPECIFIC SCORES:
----------------------------------------
AV             : 0.6558
COMMON_SENSE   : 0.8221
HUMAN          : 0.7881
INDUSTRY       : 0.8435
MISC           : 0.8875
PHYSICS        : 0.9193
ROBOT          : 0.7627

Domain  score 0.802
Quality score 0.732
Overall score 0.767
```

### image2world
```bash
Overall Accuracy: 0.8389
Total Videos Evaluated: 1036
Model: Qwen/Qwen2.5-VL-72B-Instruct

CATEGORY-SPECIFIC SCORES:
----------------------------------------
AV             : 0.6548
COMMON_SENSE   : 0.9257
HUMAN          : 0.8318
INDUSTRY       : 0.8659
MISC           : 0.9216
PHYSICS        : 0.9291
ROBOT          : 0.7866

Domain  score 0.839
Quality score 0.780
Overall score 0.810
```

## B200 Detailed Throughput Results

### text2world
```bash
1 GPU:
Average Time: 125.1838 seconds
Std Deviation: 0.1794 seconds

2 GPUs:
Average Time: 75.5894 seconds
Std Deviation: 0.3107 seconds

4 GPUs:
Average Time: 45.9414 seconds
Std Deviation: 0.0681 seconds
```

### image2world
```bash
1 GPU:
Average Time: 125.9848 seconds
Std Deviation: 0.2278 seconds

2 GPUs:
Average Time: 77.4188 seconds
Std Deviation: 0.1846 seconds

4 GPUs:
Average Time: 46.8531 seconds
Std Deviation: 0.1111 seconds
```
