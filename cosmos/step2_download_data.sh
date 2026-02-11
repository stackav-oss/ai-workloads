source $HOME/.local/bin/env

# Check if HF_TOKEN is set and not empty
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF_TOKEN is either unset or empty"
fi

mkdir -p /datasets
cd /datasets
uv venv --clear
source .venv/bin/activate
uv pip install huggingface_hub


# Download datasets if not already present
if [ -d "/datasets/physical-ai-bench-generation/vqa" ]; then
  echo "physical-ai-bench-generation dataset exists"
else
  hf download shi-labs/physical-ai-bench-generation --repo-type dataset --local-dir /datasets/physical-ai-bench-generation  > /dev/null
fi

#if [ -d "/datasets/physical-ai-bench-conditional-generation" ]; then
#  echo "physical-ai-bench-conditional-generation dataset exists"
#else
#  hf download shi-labs/physical-ai-bench-conditional-generation --repo-type dataset --local-dir /datasets/physical-ai-bench-conditional-generation  > /dev/null
#fi

deactivate
