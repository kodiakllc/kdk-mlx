---
language:
- en
- fr
- de
- es
- it
- pt
- ru
- zh
- ja
license: apache-2.0
tags:
- mlx
---

# mlx-community/Mistral-Nemo-Instruct-2407-4bit

The Model [mlx-community/Mistral-Nemo-Instruct-2407-4bit](https://huggingface.co/mlx-community/Mistral-Nemo-Instruct-2407-4bit) was converted to MLX format from [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) using mlx-lm version **0.16.0**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-Nemo-Instruct-2407-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)
```
