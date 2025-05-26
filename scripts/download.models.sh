#!/bin/bash

# List of models to download
models=(
#  "Llama-3.2-3B-Instruct"
#  "Llama-3.2-11B-Vision-Instruct-4bit"
#  "Mistral-Nemo-Instruct-2407-4bit"
  ##########################################
  ### Qwen
  "Qwen2.5-7B-Instruct-Uncensored-4bit"
  "Qwen2.5.1-Coder-7B-Instruct-4bit"
  "Qwen2.5-Coder-32B-Instruct-4bit"
  "Qwen3-32B-4bit"
  ##########################################
  ### Llama
  "meta-llama-Llama-4-Scout-17B-16E-4bit"
  ########################################## 
  ### DeepSeek
  "DeepSeek-R1-Distill-Llama-8B-4bit"
  "DeepSeek-R1-Distill-Qwen-14B-4bit"
  "DeepSeek-R1-Distill-Qwen-32B-4bit"
  ##########################################
  ### Gemma 3
  "gemma-3-4b-it-4bit"
  "gemma-3-12b-it-4bit"
  # Quantization Aware Training (QAT)
  "gemma-3-4b-it-qat-4bit"
  "gemma-3-12b-it-qat-4bit"
  # Distilled Weight Quantized (DWQ)
  "gemma-3-4b-it-4bit-DWQ"
  "gemma-3-12b-it-4bit-DWQ"
  ##########################################
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
