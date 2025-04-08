---
library_name: transformers
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
tags:
- mlx
---

# mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit

The Model [mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit](https://huggingface.co/mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit) was
converted to MLX format from [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
using mlx-lm version **0.20.2**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit")

prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
