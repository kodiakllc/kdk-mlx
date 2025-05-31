---
license: apache-2.0
---

# `Osmosis-Structure-0.6B`: Small Language Model for Structured Outputs

<div align="center">

</div>

`Osmosis-Structure-0.6B` is a specialized small language model (SLM) designed to excel at structured output generation. Despite its compact 0.6B parameter size, this model demonstrates remarkable performance on extracting structured information when paired with supported frameworks.

Our approach leverages structured output during training, forcing our model to only focus on the value for each key declared by the inference engine, which significantly improves the accuracy of the model's ability to produce well-formatted, structured responses across various domains, particularly in mathematical reasoning and problem-solving tasks.

<div align="center">
  
![Osmosis Structure Demo](output.gif)

</div>

## Results

We evaluate the effectiveness of osmosis-enhanced structured generation on challenging mathematical reasoning benchmarks. The following results demonstrate the dramatic performance improvements achieved through structured outputs with osmosis enhancement across different model families - the same technique that powers `Osmosis-Structure-0.6B`.

### Math DAPO 17K Dataset

<div align="center">

| Model | Structured Output | Structured w/ Osmosis | Performance Gain |
|-------|:-------------:|:-------------:|:-------------:|
| Claude 4 Sonnet | 15.52% | **69.40%** | +347% |
| Claude 4 Opus | 15.28% | **69.91%** | +357% |
| GPT-4.1 | 10.53% | **70.03%** | +565% |
| OpenAI o3 | 91.14% | **94.05%** | +2.9% |

<em>Table 1: Performance on Math DAPO 17K.</em>

</div>

### AIME 1983-2024 Dataset

<div align="center">

| Model | Structured Output | Structured w/ Osmosis | Performance Gain |
|-------|:-------------:|:-------------:|:-------------:|
| Claude 4 Sonnet | 16.29% | **62.59%** | +284% |
| Claude 4 Opus | 22.94% | **65.06%** | +184% |
| GPT-4.1 | 2.79% | **39.66%** | +1322% |
| OpenAI o3 | 92.05% | **93.24%** | +1.3% |

<em>Table 2: Performance on AIME 1983-2024.</em>

</div>

> **Key Insight**: These results demonstrate that by allowing models to think freely and leverage test time compute, we are able to increase performance and still maintain the structured guarantee after the fact with a SLM. `Osmosis-Structure-0.6B` is specifically designed and optimized to maximize these benefits in a compact 0.6B parameter model.

## Model Training

`Osmosis-Structure-0.6B` is built on top of `Qwen3-0.6B`. We first established a baseline format using 10 samples of randomly generated text and their JSON interpretations. We then applied reinforcement learning to approximately 500,000 examples of JSON-to-natural language pairs, consisting of either reasoning traces with their final outputs, or natural language reports with their expected structured formats.

We used [verl](https://github.com/volcengine/verl) as the framework to train our model and [SGLang](https://github.com/sgl-project/sglang) as the rollout backend. To enable structured training, we modified parts of the verl codebase to allow for *per sample schema* to be passed into the training data.

## Usage


### SGLang

We recommend an engine like SGLang to be used to serve the model, to serve, run the following:

`python3 -m sglang.launch_server --model-path Osmosis/Osmosis-Structure-0.6B --host 0.0.0.0 --api-key osmosis`

And to use the endpoint:

```python
import json
from openai import OpenAI

api_key = "osmosis"
api_base_url = "http://0.0.0.0:30000/v1"
client = OpenAI(
    api_key=api_key,
    base_url=api_base_url,
)

# Schema for extracting structured output from reasoning traces
json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"]
    }
)

# You can also dump pydantic models to json schema as well

# Example reasoning trace input
reasoning_trace = """
Problem: Solve for x in the equation 2x + 5 = 13

Let me work through this step by step:

First, I need to isolate the term with x. I'll subtract 5 from both sides:
2x + 5 - 5 = 13 - 5
2x = 8

Next, I'll divide both sides by 2 to solve for x:
2x ÷ 2 = 8 ÷ 2
x = 4

Let me verify this answer by substituting back into the original equation:
2(4) + 5 = 8 + 5 = 13 ✓

Ok, which means I got the correct answer, and I'm confident about my answer.
"""
response = client.chat.completions.create(
    model="Osmosis/Osmosis-Structure-0.6B",
    messages=[
        {
            "role": "system",
            "content": f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {json_schema}"
        },
        {
            "role": "user", 
            "content": reasoning_trace,
        },
    ],
    temperature=0,
    max_tokens=512,
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "reasoning_extraction", "schema": json.loads(json_schema)},
    },
)

print(json.dumps(response.choices[0].message.content, indent=2))
```


### Ollama

You can also use Ollama as an inference provider on local machines, here is a sample code of the setup:

```python
from ollama import chat
from pydantic import BaseModel

class Answer(BaseModel):
  answer: int

reasoning_trace = """
Problem: Solve for x in the equation 2x + 5 = 13

Let me work through this step by step:

First, I need to isolate the term with x. I'll subtract 5 from both sides:
2x + 5 - 5 = 13 - 5
2x = 8

Next, I'll divide both sides by 2 to solve for x:
2x ÷ 2 = 8 ÷ 2
x = 4

Let me verify this answer by substituting back into the original equation:
2(4) + 5 = 8 + 5 = 13 ✓

Ok, which means I got the correct answer, and I'm confident about my answer.
"""

response = chat(
  messages=[
    {
        "role": "system",
        "content": f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {Answer.model_json_schema()}"
    },
    {
      'role': 'user',
      'content': reasoning_trace,
    }
  ],
  model='Osmosis/Osmosis-Structure-0.6B',
  format=Answer.model_json_schema(),
)

answer = Answer.model_validate_json(response.message.content)
print(answer)
```
