import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from typing import List, Dict, Union
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")
CURRENT_T_MODEL = "Llama-3.2-11B-Vision-Instruct-4bit"

model = None
processor = None
config = None

def create_llama_prompt(system_prompt: str, conversation_history: List[dict], user_message: str, image_paths: List[str]) -> List[dict]:
    messages = []
    
    # Add system message if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history
    messages.extend(conversation_history)
    
    # Format user message with images
    if image_paths:
        content = []
        for img in image_paths:
            content.append({"type": "image", "text": img})
        content.append({"type": "text", "text": user_message})
        messages.append({
            "role": "user",
            "content": content
        })
    else:
        messages.append({
            "role": "user",
            "content": user_message
        })
    
    return messages

def load_model():
    global model, processor, config
    if model is None or processor is None or config is None:
        model_path = MODELS_PATH + CURRENT_T_MODEL
        model, processor = load(model_path)
        config = load_config(model_path)


def generate_content(
    messages: List[dict],
    image_paths: List[str] = []
) -> str:
    # Validate image paths
    valid_images = []
    for path in image_paths:
        if os.path.exists(path):
            valid_images.append(path)
        else:
            print(f"Warning: Image not found: {path}")
    
    # Apply chat template
    formatted_prompt = apply_chat_template(
        processor,
        config,
        messages,
        num_images=len(valid_images)
    )
    
    # Generate output
    output = generate(
        model,
        processor,
        valid_images if valid_images else None,
        formatted_prompt,
        verbose=False
    )
    
    return output

if __name__ == "__main__":
    system_prompt = None
    conversation_history = []

    # Load the model
    load_model()
    
    print("Enter image paths (comma-separated) or press Enter for text-only chat:")
    image_input = input().strip()
    image_paths = [p.strip() for p in image_input.split(",")] if image_input else []
    
    while True:
        user_message = input("You: ")
        if user_message.lower() in ['exit', 'quit']:
            break
        
        messages = create_llama_prompt(
            system_prompt,
            conversation_history,
            user_message,
            image_paths
        )
        
        print("Assistant: ", end='')
        response = generate_content(messages=messages, image_paths=image_paths)
        print(response)
        
        # Update conversation history
        conversation_history.append({
            "role": "user",
            "content": [{"type": "text", "text": user_message}]
        })
        conversation_history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
