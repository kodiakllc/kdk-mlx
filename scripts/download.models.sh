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
  ### DeepSeek + Qwen
  "DeepSeek-R1-0528-Qwen3-8B-4bit"
  "DeepSeek-R1-0528-Qwen3-8B-4bit-DWQ"
  "DeepSeek-R1-0528-Qwen3-8B-8bit"
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
  ### NON-MLX Models
  "osmosis-ai/Osmosis-Structure-0.6B"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../"
# Activate the virtual environment
source ${PROJECT_ROOT}/mlx_env/bin/activate
# Base directory where models are stored
base_dir=${PROJECT_ROOT}/hf_models

# Create non-mlx models tracking file if it doesn't exist
non_mlx_models_file="$base_dir/non_mlx_models.json"
if [[ ! -f "$non_mlx_models_file" ]]; then
  echo "{}" > "$non_mlx_models_file"
fi

for model_name in "${models[@]}"; do
  # Check if this is a non-MLX model (contains "/")
  if [[ "$model_name" == *"/"* ]]; then
    # For non-MLX models, extract just the model name for the directory
    dir_name="${model_name##*/}"
    model_dir="$base_dir/$dir_name"
    repo_path="$model_name"
    is_non_mlx=true
  else
    # For MLX models, use the name as-is
    dir_name="$model_name"
    model_dir="$base_dir/$model_name"
    repo_path="mlx-community/$model_name"
    is_non_mlx=false
  fi
  
  # Check if model directory exists and contains .safetensors files
  if [[ -d "$model_dir" && -n "$(find "$model_dir" -type f -name '*.safetensors*' -print -quit)" ]]; then
    echo "Model '$dir_name' already exists with .safetensors files. Skipping download."
  else
    echo "Downloading model '$model_name' to '$model_dir'..."
    huggingface-cli download --local-dir "$model_dir" "$repo_path"
    
    # Update non-mlx models tracking file if this is a non-MLX model
    if [[ "$is_non_mlx" == true ]]; then
      # Use Python to update the JSON file
      python3 -c "
import json
with open('$non_mlx_models_file', 'r') as f:
    data = json.load(f)
data['$dir_name'] = {
    'repo_path': '$model_name',
    'type': 'non-mlx',
    'downloaded': True
}
with open('$non_mlx_models_file', 'w') as f:
    json.dump(data, f, indent=2)
"
      echo "Added '$dir_name' to non-MLX models tracking file"
    fi
  fi
done
