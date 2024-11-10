import os
from mlx_lm import load, generate

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models path relative to the script directory
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")

CURRENT_T_MODEL = "Qwen2.5.1-Coder-7B-Instruct-4bit"

def create_llama_prompt(system_prompt: str, conversation_history: list, user_message: str) -> list:
    messages = [
        {"role": "system", "content": system_prompt}
    ] + conversation_history + [
        {"role": "user", "content": user_message}
    ]
    return messages

generation_args = {
    "temp": 0.7,  # Increase temperature for more randomness
    "repetition_penalty": 1.5,  # Increase repetition penalty to reduce repetition
    "repetition_context_size": 500,  # Decrease context size to reduce over-penalization
    "top_p": 0.9,  # Decrease top_p to consider a larger set of tokens
}

def generate_content(
    messages: list,
    mlx_model: str = CURRENT_T_MODEL,
    adapter: str | None = None,
) -> str:
    model, tokenizer = load(MODELS_PATH + mlx_model)
    
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = messages[-1]["content"]
    
    response = generate(model, tokenizer, prompt=prompt, verbose=True, **generation_args)
    return response

def stream_content(messages: list):
    response = generate_content(messages=messages)
    print(response, end='', flush=True)
    return response

if __name__ == "__main__":
    system_prompt = "You are a helpful AI that responds in markdown format."
    conversation_history = []
    while True:
        user_message = input("You: ")
        messages = create_llama_prompt(system_prompt, conversation_history, user_message)
        print("Assistant: ", end='')
        response = stream_content(messages=messages)
        print()  # For a new line after the response
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": response})
