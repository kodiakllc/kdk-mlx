---
license: apache-2.0
library_name: transformers
tags:
- code
- granite
- mlx
base_model: ibm-granite/granite-34b-code-base
datasets:
- bigcode/commitpackft
- TIGER-Lab/MathInstruct
- meta-math/MetaMathQA
- glaiveai/glaive-code-assistant-v3
- glaive-function-calling-v2
- bugdaryan/sql-create-context-instruction
- garage-bAInd/Open-Platypus
- nvidia/HelpSteer
metrics:
- code_eval
pipeline_tag: text-generation
inference: true
model-index:
- name: granite-34b-code-instruct
  results:
  - task:
      type: text-generation
    dataset:
      name: HumanEvalSynthesis(Python)
      type: bigcode/humanevalpack
    metrics:
    - type: pass@1
      value: 62.2
      name: pass@1
    - type: pass@1
      value: 56.7
      name: pass@1
    - type: pass@1
      value: 62.8
      name: pass@1
    - type: pass@1
      value: 47.6
      name: pass@1
    - type: pass@1
      value: 57.9
      name: pass@1
    - type: pass@1
      value: 41.5
      name: pass@1
    - type: pass@1
      value: 53.0
      name: pass@1
    - type: pass@1
      value: 45.1
      name: pass@1
    - type: pass@1
      value: 50.6
      name: pass@1
    - type: pass@1
      value: 36.0
      name: pass@1
    - type: pass@1
      value: 42.7
      name: pass@1
    - type: pass@1
      value: 23.8
      name: pass@1
    - type: pass@1
      value: 54.9
      name: pass@1
    - type: pass@1
      value: 47.6
      name: pass@1
    - type: pass@1
      value: 55.5
      name: pass@1
    - type: pass@1
      value: 51.2
      name: pass@1
    - type: pass@1
      value: 47.0
      name: pass@1
    - type: pass@1
      value: 45.1
      name: pass@1
---

# mlx-community/granite-34b-code-instruct-4bit

The Model [mlx-community/granite-34b-code-instruct-4bit](https://huggingface.co/mlx-community/granite-34b-code-instruct-4bit) was converted to MLX format from [ibm-granite/granite-34b-code-instruct](https://huggingface.co/ibm-granite/granite-34b-code-instruct) using mlx-lm version **0.13.0**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/granite-34b-code-instruct-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)
```
