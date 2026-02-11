```bash

docker run --gpus=all -v ./:/cosmos -it --rm -e HF_TOKEN="$HF_TOKEN" nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

cd /cosmos
./step1_setup.sh
./step2_download_data.sh
./step3_inference.sh text2world
./step4_benchmark.sh text2world

./step1_setup.sh
./step2_download_data.sh
./step3_inference.sh image2world
./step4_benchmark.sh image2world
```