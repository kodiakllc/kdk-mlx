---
language:
- zh
- en
license: gpl-3.0
tags:
- qwen
- uncensored
- mlx
base_model: Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
datasets:
- NobodyExistsOnTheInternet/ToxicQAFinal
- anthracite-org/kalo-opus-instruct-22k-no-refusal
- Orion-zhen/dpo-toxic-zh
- unalignment/toxic-dpo-v0.2
- Crystalcareai/Intel-DPO-Pairs-Norefusals
pipeline_tag: text-generation
model-index:
- name: Qwen2.5-7B-Instruct-Uncensored
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: IFEval (0-Shot)
      type: HuggingFaceH4/ifeval
      args:
        num_few_shot: 0
    metrics:
    - type: inst_level_strict_acc and prompt_level_strict_acc
      value: 72.04
      name: strict accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: BBH (3-Shot)
      type: BBH
      args:
        num_few_shot: 3
    metrics:
    - type: acc_norm
      value: 35.83
      name: normalized accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MATH Lvl 5 (4-Shot)
      type: hendrycks/competition_math
      args:
        num_few_shot: 4
    metrics:
    - type: exact_match
      value: 1.36
      name: exact match
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: GPQA (0-shot)
      type: Idavidrein/gpqa
      args:
        num_few_shot: 0
    metrics:
    - type: acc_norm
      value: 7.05
      name: acc_norm
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MuSR (0-shot)
      type: TAUR-Lab/MuSR
      args:
        num_few_shot: 0
    metrics:
    - type: acc_norm
      value: 13.58
      name: acc_norm
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: MMLU-PRO (5-shot)
      type: TIGER-Lab/MMLU-Pro
      config: main
      split: test
      args:
        num_few_shot: 5
    metrics:
    - type: acc
      value: 38.07
      name: accuracy
    source:
      url: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard?query=Orion-zhen/Qwen2.5-7B-Instruct-Uncensored
      name: Open LLM Leaderboard
---

# mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit

The Model [mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit) was converted to MLX format from [Orion-zhen/Qwen2.5-7B-Instruct-Uncensored](https://huggingface.co/Orion-zhen/Qwen2.5-7B-Instruct-Uncensored) using mlx-lm version **0.19.1**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen2.5-7B-Instruct-Uncensored-4bit")

prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
