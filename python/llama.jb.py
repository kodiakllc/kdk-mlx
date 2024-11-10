import os
from mlx_lm import load, generate, stream_generate
from collections.abc import Generator

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models path relative to the script directory
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")

# Set the current model
CURRENT_T_MODEL = "Qwen2.5-7B-Instruct-Uncensored-4bit"

# Load the model and tokenizer
model, tokenizer = load(MODELS_PATH + CURRENT_T_MODEL)

# Initialize conversation
system_prompt = "You are a helpful AI that responds in markdown format."
messages = [{"role": "system", "content": system_prompt}]

# generation_args = {
#     "temp": 0.1,
#     "repetition_penalty": 1.2,
#     "repetition_context_size": 20,
#     "top_p": 0.95,
# }

generation_args = {
    "temp": 0.7,
    "repetition_penalty": 1.2,
    "repetition_context_size": 45,
    "top_p": 0.9
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

def stream_content(prompt: str) -> str:
    response = "Assistant: "
    print(response, end='', flush=True)
    for chunk in generate_content(prompt=prompt):
        print(chunk, end='', flush=True)
        response += chunk
    response += "\n\n"
    print("\n\n", end='', flush=True)
    return response

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages.append({"role": "user", "content": user_input})

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # quit
        break

    # Stream response and capture it
    response = stream_content(prompt)
    messages.append({"role": "assistant", "content": response})
