# Whisper Throughput and Accuracy Calculations

## Requirements

FFMPEG needs to be available in the system and can be installed with the following command.

```bash
sudo apt install ffmpeg
```

## Create virtual environment

In the `whisper/offline` folder run the following commands for setting up the virtual environment.

```bash
cd whisper/offline

# create a virtual env and activate
python -m venv .venv
source .venv/bin/activate

# upgrade pip
python -m pip install --upgrade pip

# Install pip packages
pip install -r requirements.txt
```

## Example cli commands

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS

# demo with sample audio files
python3 src/main.py --model large-v3-turbo --demo

# benchmark with librispeech
python3 src/main.py --model large-v3 --kv-cache-type auto
python3 src/main.py --model large-v3-turbo --kv-cache-type auto
python3 src/main.py --model large-v3 --kv-cache-type fp8
python3 src/main.py --model large-v3-turbo --kv-cache-type fp8
python3 src/main.py --model large-v3 --kv-cache-type fp8 --max-num-seqs 256
python3 src/main.py --model large-v3-turbo --kv-cache-type fp8 --max-num-seqs 256
```

## Results

### A10 single GPU

```python
INFO:__main__:  WER: 2.31   |   throughput: 9.47 reqs/sec   |   model: large-v3         |   kv_cache_type: auto  |  max_num_seqs: 128
INFO:__main__:  WER: 2.33   |   throughput: 14.49 reqs/sec  |   model: large-v3-turbo   |   kv_cache_type: auto  |  max_num_seqs: 128
INFO:__main__:  WER: 2.27   |   throughput: 10.33 reqs/sec  |   model: large-v3         |   kv_cache_type: fp8   |  max_num_seqs: 128
INFO:__main__:  WER: 2.41   |   throughput: 15.37 reqs/sec  |   model: large-v3-turbo   |   kv_cache_type: fp8   |  max_num_seqs: 128
INFO:__main__:  WER: 2.26   |   throughput: 9.61 reqs/sec   |   model: large-v3         |   kv_cache_type: fp8   |  max_num_seqs: 256
INFO:__main__:  WER: 2.40   |   throughput: 14.25 reqs/sec  |   model: large-v3-turbo   |   kv_cache_type: fp8   |  max_num_seqs: 256
```

### H100 single GPU

```python
INFO:__main__:  WER: 2.31   |   throughput: 35.30 reqs/sec  |   model: large-v3         |   kv_cache_type: auto  |  max_num_seqs: 128
INFO:__main__:  WER: 2.41   |   throughput: 48.05 reqs/sec  |   model: large-v3-turbo   |   kv_cache_type: auto  |  max_num_seqs: 128
INFO:__main__:  WER: 2.27   |   throughput: 35.62 reqs/sec  |   model: large-v3         |   kv_cache_type: fp8   |  max_num_seqs: 128
INFO:__main__:  WER: 2.40   |   throughput: 47.62 reqs/sec  |   model: large-v3-turbo   |   kv_cache_type: fp8   |  max_num_seqs: 128
INFO:__main__:  WER: 2.27   |   throughput: 41.34 reqs/sec  |   model: large-v3         |   kv_cache_type: fp8   |  max_num_seqs: 256
INFO:__main__:  WER: 2.40   |   throughput: 50.23 reqs/sec  |   model: large-v3-turbo   |   kv_cache_type: fp8   |  max_num_seqs: 256
```


### GB200 single GPU
INFO:__main__:  WER: 2.31   |   throughput: 19.18 reqs/sec  |   model: large-v3     |   kv_cache_type: auto  |  max_num_seqs: 128
