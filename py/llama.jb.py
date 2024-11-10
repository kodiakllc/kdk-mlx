import os
from mlx_lm import load, generate, stream_generate
from collections.abc import Generator

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models path relative to the script directory
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")

# Set the current model
# CURRENT_T_MODEL = "Qwen2.5-7B-Instruct-Uncensored-4bit"
CURRENT_T_MODEL = "Qwen2.5.1-Coder-7B-Instruct-4bit"

# Init the model and tokenizer
model = None
tokenizer = None

# Initialize conversation
system_prompt = "You are a helpful AI that responds in markdown format."
messages = [{"role": "system", "content": system_prompt}]

generation_args = {
    "temp": 0.7,
    "repetition_penalty": 1.2,
    "repetition_context_size": 20,
    "top_p": 0.95,
}

def generate_content(
    prompt: str,
    max_tokens: int = 1024,
    adapter: str | None = None,
) -> Generator[str, None, None]:
    global model, tokenizer
    for t in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, **generation_args
    ):
        yield from t

def stream_content(prompt: str, max_tokens: int) -> str:
    response = "Assistant: "
    print(response, end='', flush=True)
    for chunk in generate_content(prompt=prompt, max_tokens=max_tokens):
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

    # Check if we have loaded the model
    if model is None or tokenizer is None:
        model, tokenizer = load(MODELS_PATH + CURRENT_T_MODEL)

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # quit
        break

    # Stream response and capture it
    response = stream_content(prompt, 1024)
    messages.append({"role": "assistant", "content": response})
