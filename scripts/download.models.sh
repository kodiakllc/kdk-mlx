#!/bin/bash

# List of models to download
models=(
  "Llama-3.2-3B-Instruct"
  "Llama-3.2-11B-Vision-Instruct-8bit"
  "Mistral-Nemo-Instruct-2407-4bit"
  "Qwen2.5-7B-Instruct-Uncensored-4bit"
  "Qwen2.5.1-Coder-7B-Instruct-4bit"
  "Qwen2.5-Coder-32B-Instruct-4bit"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../"
# Activate the virtual environment
source ${PROJECT_ROOT}/mlx_env/bin/activate
# Base directory where models are stored
base_dir=${PROJECT_ROOT}/hf_models

for model_name in "${models[@]}"; do
  model_dir="$base_dir/$model_name"
  
  # Check if model directory exists and contains .safetensors files
  if [[ -d "$model_dir" && -n "$(find "$model_dir" -type f -name '*.safetensors*' -print -quit)" ]]; then
    echo "Model '$model_name' already exists with .safetensors files. Skipping download."
  else
    echo "Downloading model '$model_name'..."
    huggingface-cli download --local-dir "$model_dir" mlx-community/"$model_name"
  fi
done
