from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Union
import os
from mlx_lm import load, stream_generate
from collections.abc import Generator
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import json

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

MODELS_PATH = os.path.expanduser("~/Documents/kdk-mlx/hf_models/")
CURRENT_T_MODEL = "Qwen2.5-7B-Instruct-Uncensored-4bit"

model, tokenizer = load(MODELS_PATH + CURRENT_T_MODEL)

class Content(BaseModel):
    type: str
    text: str

class Message(BaseModel):
    role: str
    content: Union[str, List[Content]]

class RequestBody(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    stream: bool

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

def generate_content(prompt: str, max_tokens: int) -> Generator[str, None, None]:
    for t in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, **generation_args
    ):
        yield from t

@app.post("/v1/chat/completions")
async def completions(request: Request, body: RequestBody):
    if body.model != 'local':
        raise HTTPException(status_code=400, detail="Model not supported")

    messages = body.messages

    # Transform the messages
    transformed_messages = []
    for message in messages:
        if isinstance(message.content, list):
            user_input = " ".join([content.text for content in message.content])
            transformed_messages.append({"role": message.role, "content": user_input})
        else:
            transformed_messages.append({"role": message.role, "content": message.content})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(transformed_messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in transformed_messages])

    if body.stream:
        async def event_stream():
            is_stream_open = True
            try:
                for chunk in generate_content(prompt, body.max_tokens):
                    if not is_stream_open:
                        break
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                if is_stream_open:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': '[DONE]'}, 'finish_reason': 'stop'}]})}\n\n"
                    is_stream_open = False
            except Exception as e:
                if is_stream_open:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'finish_reason': 'error'}]})}\n\n"
                    is_stream_open = False

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        response = "".join(generate_content(prompt, body.max_tokens))
        return {"choices": [{"message": {"content": response}, "finish_reason": "stop"}]}
    
@app.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    models_data = {
        "object": "list",
        "data": [
            {"id": "gpt-4o", "object": "model", "created": 1715367049, "owned_by": "system"},
            {"id": "local", "object": "model", "created": 1715367049, "owned_by": "system"}
        ]
    }
    return models_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
