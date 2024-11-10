from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Union, Optional
import os
from mlx_lm import load, stream_generate, generate
from collections.abc import Generator
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogRequestBodyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        logger.info(f"Request body: {body.decode('utf-8')}")
        response = await call_next(request)
        return response


app = FastAPI()
app.add_middleware(LogRequestBodyMiddleware)

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models path relative to the script directory
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")

# Set the current model
DEFAULT_T_MODEL = "Qwen2.5-7B-Instruct-Uncensored-4bit"

model = None
tokenizer = None
current_model_id = None

def load_model(model_id: str):
    global model, tokenizer, current_model_id
    if model_id == None:
        logger.info(f"### 🔃 Loading default model: {DEFAULT_T_MODEL}")
        model, tokenizer = load(MODELS_PATH + DEFAULT_T_MODEL)
        current_model_id = DEFAULT_T_MODEL
    else:
        logger.info(f"### 🔃 Loading model: {model_id}")
        model, tokenizer = load(MODELS_PATH + model_id)
        current_model_id = model_id

class Content(BaseModel):
    type: str
    text: str

class Message(BaseModel):
    role: str
    content: Union[str, List[Content]]

class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[Union[dict, None]]
    finish_reason: str

class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[Choice]
    usage: Optional[Usage] = None

class RequestBody(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    stream: Optional[bool] = True
    stream_options: Optional[StreamOptions] = None

class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    object: str
    data: List[Model]

generation_args = {
    "temp": 0.7,
    "repetition_penalty": 1.2,
    "repetition_context_size": 45,
    "top_p": 0.9
}

openai_models = {
    "object": "list",
    "data": [
        {"id": "gpt-3.5-turbo", "object": "model", "created": 1715367049, "owned_by": "system"},
        {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "system"},
        {"id": "local", "object": "model", "created": 1715367049, "owned_by": "system"}
    ]
}

local_models = {
    "object": "list",
    "data": [
        {"id": "Qwen2.5-7B-Instruct-Uncensored-4bit", "object": "model", "created": 1715367049, "owned_by": "system"},
        {"id": "granite-34b-code-instruct-4bit", "object": "model", "created": 1715367049, "owned_by": "system"}
    ]
}

def get_logit_bias_for_frequency_penalty(transformed_messages: List[dict], frequency_penalty: float) -> dict:
    biases = {}
    if frequency_penalty != 0:
        for msg in transformed_messages:
            if msg["role"] == "assistant":
                tokens = tokenizer.encode(msg["content"])
                for token in tokens:
                    biases[token] = biases.get(token, 0) - frequency_penalty
        logger.info(f"### 📊 Changed logit_bias for frequency penalty ✅")
    else:
        biases = None
        logger.info(f"### 📊 Removed logit_bias for frequency penalty ✔️")
    return biases

def get_logit_bias_for_presence_penalty(transformed_messages: List[dict], presence_penalty: float) -> dict:
    biases = {}
    if presence_penalty != 0:
        seen_tokens = set()
        for msg in transformed_messages:
            tokens = tokenizer.encode(msg["content"])
            for token in tokens:
                if token in seen_tokens:
                    biases[token] = biases.get(token, 0) - presence_penalty
                seen_tokens.add(token)
        logger.info(f"### 📊 Changed logit_bias for presence penalty ✅")
    else:
        biases = None
        logger.info(f"### 📊 Removed logit_bias for presence penalty ✔️")
    return biases

def combine_logit_biases(bias1: dict, bias2: dict) -> dict:
    combined_bias = bias1.copy()
    for token, bias in bias2.items():
        if token in combined_bias:
            combined_bias[token] += bias
        else:
            combined_bias[token] = bias
    return combined_bias

def generate_content(prompt: str, max_tokens: int, stream: bool = True) -> Generator[str, None, None]:
    if stream:
        for t in stream_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, **generation_args
        ):
            yield from t
    else:
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, **generation_args)
        yield response

def is_model_supported(model_id: str):
    is_openai_model = any(model['id'] == model_id for model in openai_models['data'])
    is_local_model = any(model['id'] == model_id for model in local_models['data'])
    if not is_openai_model and not is_local_model:
        return False
    return True

def is_local_model(model_id: str):
    is_local_model = any(model['id'] == model_id for model in local_models['data'])
    return is_local_model

@app.post("/v1/chat/completions")
async def completions(request: Request, body: RequestBody):
    if not is_model_supported(body.model):
        raise HTTPException(status_code=400, detail="Model not supported")
    
    if model is None or tokenizer is None:
        if not is_local_model(body.model):
            load_model(None)
        else:
            load_model(body.model)
    elif current_model_id != body.model and is_local_model(body.model):
        load_model(body.model)

    messages = body.messages

    # Transform the messages
    transformed_messages = []
    for message in messages:
        if isinstance(message.content, list):
            user_input = " ".join([content.text for content in message.content])
            transformed_messages.append({"role": message.role, "content": user_input})
        else:
            transformed_messages.append({"role": message.role, "content": message.content})

    # Handle frequency_penalty
    frequency_logit_bias = {}
    if body.frequency_penalty:
        frequency_logit_bias = get_logit_bias_for_frequency_penalty(transformed_messages, body.frequency_penalty)

    # Handle presence_penalty
    presence_logit_bias = {}
    if body.presence_penalty:
        presence_logit_bias = get_logit_bias_for_presence_penalty(transformed_messages, body.presence_penalty)

    # Combine logit biases
    combined_logit_bias = combine_logit_biases(frequency_logit_bias, presence_logit_bias)
    if combined_logit_bias:
        generation_args["logit_bias"] = combined_logit_bias
    else:
        generation_args.pop("logit_bias", None)

    # Handle temperature
    if body.temperature is not None:
        generation_args["temp"] = body.temperature
    else:
        generation_args["temp"] = 0.7
    
    # Handle top_p
    if body.top_p is not None:
        generation_args["top_p"] = body.top_p
    else:
        generation_args["top_p"] = 0.9

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(transformed_messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in transformed_messages])

    if body.stream:
        async def event_stream():
            is_stream_open = True
            try:
                for chunk in generate_content(prompt, body.max_tokens):
                    # if the chunk is [DONE], close the stream
                    if chunk.strip() == "[DONE]":
                        break
                    if not is_stream_open:
                        break
                    # if the chunk ends with done, strip it off and close the stream
                    if chunk.strip().endswith("[DONE]"):
                        chunk = chunk.strip()[:-6]
                        is_stream_open = False
                    # yield the chunk
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}, 'finish_reason': None}], 'usage': None})}\n\n"
                if is_stream_open:
                    #yield f"data: {json.dumps({'choices': [{'delta': {'content': '[DONE]'}, 'finish_reason': 'stop'}]})}\n\n"
                    usage = {
                        "prompt_tokens": len(tokenizer.encode(prompt)),
                        "completion_tokens": 0,
                        "total_tokens": len(tokenizer.encode(prompt)),
                        "completion_tokens_details": {
                            "reasoning_tokens": 0,
                            "accepted_prediction_tokens": 0,
                            "rejected_prediction_tokens": 0
                        }
                    }
                    #yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
                    yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop', 'usage': None}]})}\n\n"
                    is_stream_open = False
            except Exception as e:
                if is_stream_open:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'finish_reason': 'error'}], 'usage': None})}\n\n"
                    is_stream_open = False

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        response_content = "".join(generate_content(prompt, body.max_tokens, stream=False))
        prompt_tokens = len(tokenizer.encode(prompt))
        response_tokens = len(tokenizer.encode(response_content))
        total_tokens = prompt_tokens + response_tokens
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                # calculate the number of tokens in the prompt
                "prompt_tokens": prompt_tokens,
                "completion_tokens": response_tokens,
                "total_tokens": total_tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            }
        }
        logger.info(f"Response (NOT streaming): {response_data}")
        response = ChatCompletionResponse(**response_data)
        return response
    
@app.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    return openai_models

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
