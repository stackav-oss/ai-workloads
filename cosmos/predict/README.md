# COSMOS Physical AI Benchmark

This repository contains scripts for running COSMOS inference and benchmark evaluation using the Physical AI Benchmark dataset.

## 🚀 Quick Start

### Environment Setup
Make sure to use `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` image and set `HF_TOKEN`. 

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


## B200 Evaluation Results

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

## B200 Throughput Results

### text2world
```bash
__ 1 GPU:
   Average Time: 125.1838 seconds
   Std Deviation: 0.1794 seconds

__ 2 GPUs:
   Average Time: 75.5894 seconds
   Std Deviation: 0.3107 seconds

__ 4 GPUs:
   Average Time: 45.9414 seconds
   Std Deviation: 0.0681 seconds
```

### image2world
```bash
__ 1 GPU:
   Average Time: 125.9848 seconds
   Std Deviation: 0.2278 seconds

__ 2 GPUs:
   Average Time: 77.4188 seconds
   Std Deviation: 0.1846 seconds

__ 4 GPUs:
   Average Time: 46.8531 seconds
   Std Deviation: 0.1111 seconds
```
