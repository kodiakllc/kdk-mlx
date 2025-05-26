---
license: gemma
library_name: mlx
pipeline_tag: text-generation
extra_gated_heading: Access Gemma on Hugging Face
extra_gated_prompt: To access Gemma on Hugging Face, you’re required to review and
  agree to Google’s usage license. To do this, please ensure you’re logged in to Hugging
  Face and click below. Requests are processed immediately.
extra_gated_button_content: Acknowledge license
base_model: google/gemma-3-12b-it
tags:
- mlx
---

# mlx-community/gemma3-12b-it-4bit-DWQ

This model [mlx-community/gemma3-12b-it-4bit-DWQ](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit-DWQ) was
converted to MLX format from [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)
using mlx-lm version **0.24.0**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-3-12b-it-4bit-DWQ")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
