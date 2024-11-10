---
base_model: meta-llama/Llama-3.2-11B-Vision-Instruct
language:
- en
library_name: transformers
license: llama3.2
tags:
- llama-3
- llama
- meta
- facebook
- unsloth
- transformers
- mlx
---

# mlx-community/Llama-3.2-11B-Vision-Instruct-8bit
This model was converted to MLX format from [`unsloth/Llama-3.2-11B-Vision-Instruct`]() using mlx-vlm version **0.1.0**.
Refer to the [original model card](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct) for more details on the model.
## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model mlx-community/Llama-3.2-11B-Vision-Instruct-8bit --max-tokens 100 --temp 0.0
```
