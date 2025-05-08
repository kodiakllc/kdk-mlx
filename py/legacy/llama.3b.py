import os
from mlx_lm import load, stream_generate
from collections.abc import Generator

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models path relative to the script directory
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")

# Set the current model
CURRENT_T_MODEL = "Llama-3.2-3B-Instruct"

def create_llama_prompt(system_prompt: str, conversation_history: str, user_message: str) -> str:
    prompt = f'''
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|>

    {conversation_history}

    <|start_header_id|>user<|end_header_id|>
    {user_message}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    '''
    return prompt

generation_args = {
    "temp": 0.1,
    "repetition_penalty": 1.2,
    "repetition_context_size": 20,
    "top_p": 0.95,
}

def generate_content(
    prompt: str,
    mlx_model: str = CURRENT_T_MODEL,
    adapter: str | None = None,
) -> Generator[str, None, None]:
    model, tokenizer = load(MODELS_PATH + mlx_model)
    for t in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=128000, **generation_args
    ):
        yield from t

def stream_content(prompt: str):
    for chunk in generate_content(prompt=prompt):
        print(chunk, end='', flush=True)

if __name__ == "__main__":
    system_prompt = "You are a helpful AI that responds in markdown format."
    conversation_history = ""
    while True:
        user_message = input("You: ")
        llama_prompt = create_llama_prompt(system_prompt, conversation_history, user_message)
        print("Assistant: ", end='')
        response = ""
        for chunk in generate_content(prompt=llama_prompt):
            print(chunk, end='', flush=True)
            response += chunk
        print()  # For a new line after the response
        conversation_history += f"<|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>\n"
