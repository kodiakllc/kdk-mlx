#!/bin/bash

# prompt for the model name
echo "Enter the model name: "
read model_name

# if it's empty, use the default model name
if [ -z "$model_name" ]; then
  model_name="Mistral-Nemo-Instruct-2407-4bit"
fi

huggingface-cli download --local-dir ~/Documents/kdk-mlx/hf_models/$model_name mlx-community/$model_name
