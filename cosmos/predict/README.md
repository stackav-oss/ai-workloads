# COSMOS Physical AI Benchmark

This repository contains scripts for running COSMOS inference and benchmark evaluation using the Physical AI Benchmark dataset.

## 🚀 Quick Start

### Docker Environment Setup

```bash
docker run --gpus=all -v ./:/cosmos -it --rm -e HF_TOKEN="$HF_TOKEN" nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
```

### Text-to-World Pipeline

```bash
cd /cosmos/predict
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

## 🛠 Requirements

- **Hardware**: Minimum 4 NVIDIA GPUs
- **Software**: Docker with GPU support
- **Authentication**: Hugging Face token for dataset access
- **Storage**: Sufficient space for datasets and results
