#!/usr/bin/env python3
"""
Example usage of the custom MLX converter
"""

from custom_mlx_convert import custom_convert
from pathlib import Path

# Example 1: Convert a local model directory
def convert_local_model():
    """Convert a model that's already downloaded locally"""
    custom_convert(
        hf_path="/path/to/local/model",  # Local path to model
        mlx_path="output/mlx_model",
        quantize=True,
        q_bits=4,
        use_custom_path=True  # This will use the custom path handling
    )

# Example 2: Convert from Hugging Face with standard behavior
def convert_from_hf():
    """Convert a model from Hugging Face hub"""
    custom_convert(
        hf_path="meta-llama/Llama-2-7b-hf",  # HF model ID
        mlx_path="output/llama2_mlx",
        quantize=True,
        q_bits=4,
        use_custom_path=False  # This will use standard get_model_path
    )

# Example 3: Convert with custom path resolution
def convert_with_custom_logic():
    """Convert with custom path resolution logic"""
    # You can modify custom_get_model_path in custom_mlx_convert.py
    # to implement your own path resolution logic
    
    custom_convert(
        hf_path="./models/my_model",  # This will be handled by custom logic
        mlx_path="output/my_model_mlx",
        quantize=True,
        q_group_size=64,
        q_bits=4,
        dtype="float16",
        use_custom_path=True
    )

# Example 4: Command line usage
"""
# With custom path handling (default):
python custom_mlx_convert.py --hf-path ./local/model --mlx-path output/model_mlx -q --q-bits 4

# With standard HF path handling:
python custom_mlx_convert.py --hf-path meta-llama/Llama-2-7b-hf --mlx-path output/llama_mlx -q --use-standard-path

# With mixed quantization:
python custom_mlx_convert.py --hf-path ./model --mlx-path output/model_mlx -q --quant-predicate mixed_4_6
"""

if __name__ == "__main__":
    print("Custom MLX Converter Examples")
    print("Choose an example to run:")
    print("1. Convert local model")
    print("2. Convert from Hugging Face")
    print("3. Convert with custom logic")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        # You'll need to update the path
        print("Update the path in convert_local_model() first!")
        # convert_local_model()
    elif choice == "2":
        print("This will download from HF...")
        # convert_from_hf()
    elif choice == "3":
        print("Update paths in convert_with_custom_logic() first!")
        # convert_with_custom_logic()
    else:
        print("Invalid choice")