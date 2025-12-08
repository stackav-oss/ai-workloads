# CLIP Evaluation with vLLM

This directory contains a CLIP evaluation script for image classification tasks using vLLM for efficient inference.

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for optimal performance)
- **CUDA Toolkit 12.6 with nvcc compiler** installed on the machine
- Access to ImageNet dataset (for evaluation), please look at imagenet installation guide if imagenet validation set is not already downloaded.

### Installation Steps

#### 1. Create a Python Virtual Environment

```bash
# Navigate to the clip directory
cd ai-workloads/clip

# Create virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

#### 2. Upgrade pip

```bash
pip install --upgrade pip
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Verify Installation

```bash
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### GPU Setup

If you have a CUDA-capable GPU, verify it's detected:

```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

## Usage

### Imagenet Download Guide

```bash
mkdir imagenet
cd imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
tar -xvf ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_devkit_t12.tar.gz
```

### Demo code

We have added a folder `data/images`, where we have 6 images of speed limit signs, and we intend to classify them. To run this demo for clip models, please launch the environment and run:

```bash
python src/main.py \
  --imagenet-root ./data/images \
  --model openai/clip-vit-large-patch14 \
  --batch-size 256 \
  --demo
```

### Scripts

```bash
# Use ensemble templates for better accuracy
python src/main.py \
  --imagenet-root /path/to/imagenet \
  --model openai/clip-vit-large-patch14 \
  --batch-size 256 \
  --ensemble-templates

# Custom prompt template
python src/main.py \
  --imagenet-root /path/to/imagenet \
  --prompt-template "a photo of a {}" \
  --batch-size 256
```

To run the script with a different attention backend, prefix it with the attention backend you need. Example:

````bash
# xformers attention
VLLM_ATTENTION_BACKEND=XFORMERS python src/main.py \
  --imagenet-root /path/to/imagenet \
  --model openai/clip-vit-large-patch14 \
  --batch-size 256 \
  --ensemble-templates

# Flex Attention attention
VLLM_ATTENTION_BACKEND=FLEX_ATTENTION python src/main.py \
  --imagenet-root /path/to/imagenet \
  --model openai/clip-vit-large-patch14 \
  --batch-size 256 \
  --ensemble-templates

# Scaled Dot Product Attention attention
VLLM_ATTENTION_BACKEND=TORCH_SPDA python src/main.py \
  --imagenet-root /path/to/imagenet \
  --model openai/clip-vit-large-patch14 \
  --batch-size 256 \
  --ensemble-templates
```

### Command-Line Arguments

| Argument               | Type | Default                        | Description                   |
| ---------------------- | ---- | ------------------------------ | ----------------------------- |
| `--imagenet-root`      | str  | **required**                   | Path to ImageNet directory    |
| `--model`              | str  | `openai/clip-vit-base-patch32` | CLIP model name               |
| `--prompt-template`    | str  | `a photo of a {}`              | Text prompt template          |
| `--batch-size`         | int  | `512`                          | Batch size for processing     |
| `--ensemble-templates` | flag | `False`                        | Use multiple prompt templates |

## Deactivating the Environment

When you're done:

```bash
deactivate
````

## Files in This Directory

- **`src/imagenet_eval.py`** - Classes for Imagenet evaluation script, embedding generation and results
- **`src/imagenet_classes.py`** - ImageNet class names and prompt templates
- **`src/benchmarker.py`** - Benchmarks the image embed and text embed calls to vllm for throughput calculation
- **`src/main.py`** - Main file
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This file

## Performance Tips

1. **Use ensemble templates** (`--ensemble-templates`) for 2-5% accuracy improvement

## Results

### A10 single GPU

| Model                             | Accuracy | Embed Text Throughput | Embed Image Throughput | Attention Backend |
| --------------------------------- | -------- | --------------------- | ---------------------- | ----------------- |
| openai/clip-vit-base-patch16      | 67.11    | 1074.16               | 79.38                  | XFORMERS          |
| openai/clip-vit-base-patch32      | 61.16    | 1137.11               | 84.26                  | XFORMERS          |
| openai/clip-vit-large-patch14-336 | 75.82    | 1059.80               | 74.93                  | XFORMERS          |
| openai/clip-vit-large-patch14     | 74.51    | 1105.33               | 84.05                  | XFORMERS          |
| google/siglip-base-patch16-256    | 76.59    | 827.24                | 89.32                  | Flex Attention    |
| google/siglip-large-patch16-384   | 82.25    | 722.86                | 78.77                  | Flex Attention    |
| google/siglip-so400m-patch14-384  | 83.24    | 653.16                | 67.90                  | Flex Attention    |
| google/siglip2-base-patch16-224   | 78.38    | 982.69                | 102.85                 | Flex Attention    |
| google/siglip2-large-patch16-384  | 83.33    | 624.54                | 56.27                  | Flex Attention    |
| google/siglip2-so400m-patch14-224 | 83.38    | 479.79                | 83.05                  | Flex Attention    |

### H100 single GPU

| Model                             | Accuracy | Embed Text Throughput | Embed Image Throughput | Attention Backend |
| --------------------------------- | -------- | --------------------- | ---------------------- | ----------------- |
| openai/clip-vit-base-patch16      | 67.11    | 3482.79               | 166.51                 | XFORMERS          |
| openai/clip-vit-base-patch32      | 61.15    | 3440.67               | 180.80                 | XFORMERS          |
| openai/clip-vit-large-patch14-336 | 75.81    | 3447.81               | 163.10                 | XFORMERS          |
| openai/clip-vit-large-patch14     | 74.51    | 3417.71               | 179.10                 | XFORMERS          |
| google/siglip-base-patch16-256    | 76.60    | 1995.52               | 176.29                 | Flex Attention    |
| google/siglip-large-patch16-384   | 82.26    | 1813.10               | 174.72                 | Flex Attention    |
| google/siglip-so400m-patch14-384  | 83.24    | 1711.31               | 172.03                 | Flex Attention    |
| google/siglip2-base-patch16-224   | 78.38    | 2274.42               | 183.29                 | Flex Attention    |
| google/siglip2-large-patch16-384  | 83.32    | 1540.09               | 161.91                 | Flex Attention    |
| google/siglip2-so400m-patch14-224 | 83.39    | 1525.11               | 187.20                 | Flex Attention    |
