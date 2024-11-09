import os
from mlx_lm import load, stream_generate
from collections.abc import Generator

MODELS_PATH = model_dir = os.path.expanduser("~/Documents/kdk-mlx/hf_models/")
CURRENT_T_MODEL = "Llama-3.2-3B-Instruct"

def create_llama_prompt(system_prompt: str, user_message: str) -> str:
    prompt = f'''
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|>

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
        model, tokenizer, prompt=prompt, max_tokens=1200, **generation_args
    ):
        yield from t

def get_content(prompt)->str:
    response = "".join(
        chunk
        for chunk in generate_content(
            prompt=prompt,
        )
    )
    return response

if __name__ == "__main__":
    system_prompt = "You are a helpful AI that responds in markdown format."
    user_message = "Tell me the population of Japan."
    llama_prompt = create_llama_prompt(system_prompt, user_message)
    response = get_content(prompt=llama_prompt)
    print(response)
