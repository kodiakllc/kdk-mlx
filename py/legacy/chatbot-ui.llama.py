from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Union, Optional
import os
from mlx_lm import load, stream_generate, generate
from mlx_lm.sample_utils import make_sampler
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
        if body:
            logger.info(f"Request body: {body.decode('utf-8')}")
        else:
            logger.info("Request body is empty or None")
        response = await call_next(request)
        return response


app = FastAPI()
app.add_middleware(LogRequestBodyMiddleware)

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models path relative to the script directory
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")

# Set the current model
DEFAULT_T_MODEL = "DeepSeek-R1-Distill-Llama-8B-4bit"

model = None
tokenizer = None
current_model_id = None

# Flag to enable/disable tool responses - SET TO FALSE TO FIX LOOPING ISSUE
ENABLE_TOOL_RESPONSES = False

# Define sampler parameters
sampler_params = {
    "temp": 0.1,
    "top_p": 0.9,
    # Optional parameters
    "min_p": 0.05,
    "min_tokens_to_keep": 5,
    "top_k": 50
}

def load_model(model_id: str):
    global model, tokenizer, current_model_id
    if model_id == None:
        logger.info(f"### 🔃 Loading default model: {DEFAULT_T_MODEL}")
        model, tokenizer = load(path_or_hf_repo=MODELS_PATH + DEFAULT_T_MODEL, lazy=True)
        current_model_id = DEFAULT_T_MODEL
    else:
        logger.info(f"### 🔃 Loading model: {model_id}")
        model, tokenizer = load(path_or_hf_repo=MODELS_PATH + model_id, lazy=True)
        current_model_id = model_id

class Content(BaseModel):
    type: str
    text: str

class ToolCall(BaseModel):
    type: str
    function: dict
    id: str

class Message(BaseModel):
    role: str
    content: Optional[Union[str, List[Content]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    
    # Allow arbitrary extra fields for tool_call_id and other fields
    class Config:
        extra = "allow"

class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[Union[dict, None]] = None
    finish_reason: str
    
    # Allow arbitrary extra fields for compatibility
    class Config:
        extra = "allow"

class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails
    
    # Allow arbitrary fields for compatibility
    class Config:
        extra = "allow"

class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False
    
    # Allow arbitrary fields for compatibility
    class Config:
        extra = "allow"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    
    # Allow arbitrary fields for compatibility
    class Config:
        extra = "allow"

class Tool(BaseModel):
    function: dict
    type: Optional[str] = "function"
    
    # Allow arbitrary fields for compatibility
    class Config:
        extra = "allow"

class RequestBody(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 4096
    max_completion_tokens: Optional[int] = None  # Added support for max_completion_tokens
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    stream: Optional[bool] = True
    stream_options: Optional[StreamOptions] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = None
    
    # Allow arbitrary fields for compatibility
    class Config:
        extra = "allow"

class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    object: str
    data: List[Model]

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
        {"id": "Qwen2.5-Coder-32B-Instruct-4bit", "object": "model", "created": 1715367049, "owned_by": "system"},
        {"id": "Qwen2.5.1-Coder-7B-Instruct-4bit", "object": "model", "created": 1715367049, "owned_by": "system"},
        {"id": "DeepSeek-R1-Distill-Llama-8B-4bit", "object": "model", "created": 1715367049, "owned_by": "system"},
        {"id": "DeepSeek-R1-Distill-Qwen-32B-4bit", "object": "model", "created": 1715367049, "owned_by": "system"},
        {"id": "meta-llama-Llama-4-Scout-17B-16E-4bit", "object": "model", "created": 1715367049, "owned_by": "system"}
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

def extract_thinking(text: str):
    """
    Extract thinking content from <think></think> tags.
    Handles cases where opening tag might be missing.
    """
    thinking = ""
    remaining = text
    
    # Case 1: Has both opening and closing tags
    if "<think>" in text and "</think>" in text:
        parts = text.split("<think>", 1)
        pre_thinking = parts[0]
        thinking_and_rest = parts[1]
        
        if "</think>" in thinking_and_rest:
            thinking_parts = thinking_and_rest.split("</think>", 1)
            thinking = thinking_parts[0]
            post_thinking = thinking_parts[1]
            remaining = pre_thinking + post_thinking
        else:
            # Unclosed thinking tag - treat everything after <think> as thinking
            thinking = thinking_and_rest
            remaining = pre_thinking
    
    # Case 2: Only has closing tag (some models might omit the opening tag)
    elif "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0]
        remaining = parts[1]
    
    return thinking.strip(), remaining.strip()

def generate_content(prompt: str, max_tokens: int, stream: bool = True):
    global model, tokenizer
    
    # Create sampler with defined parameters
    sampler = make_sampler(**sampler_params)
    
    if stream:
        response = stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
        for token in response:
            yield token.text
    else:
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
        yield response.text

def create_tool_response(tool_name: str, arguments: dict):
    """
    Creates a simulated tool response.
    This is in a separate function to make it easier to enable/disable
    """
    tool_call = {
        "id": f"call_{int(time.time())}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments)
        }
    }
    return tool_call

def is_model_supported(model_id: str):
    is_openai_model = any(model['id'] == model_id for model in openai_models['data'])
    is_local_model = any(model['id'] == model_id for model in local_models['data'])
    if not is_openai_model and not is_local_model:
        return False
    return True

def is_local_model(model_id: str):
    is_local_model = any(model['id'] == model_id for model in local_models['data'])
    return is_local_model

chat_config = {
    "modify_generation_args": False,
    "allow_model_reload": False
}

@app.post("/v1/chat/completions")
async def completions(request: Request, body: RequestBody):
    if not is_model_supported(body.model):
        raise HTTPException(status_code=400, detail="Model not supported")
    
    # Log full request for debugging
    logger.info(f"Request body: {body.dict()}")
    
    # Handle tool requests - for now, we'll just acknowledge them
    # In a real system, we'd integrate with actual tool implementations
    if body.tools and len(body.tools) > 0:
        logger.info(f"Received tools request with {len(body.tools)} tools")
        for i, tool in enumerate(body.tools):
            logger.info(f"Tool {i+1}: {tool.function.get('name', 'unknown')}")
    
    if model is None or tokenizer is None:
        if not is_local_model(body.model):
            load_model(None)
        else:
            load_model(body.model)
    elif current_model_id != body.model and is_local_model(body.model) and chat_config["allow_model_reload"] is True:
        load_model(body.model)

    messages = body.messages

    # Transform the messages
    transformed_messages = []
    for message in messages:
        # Handle tool results/tool_call_id - convert to format needed for models
        if message.role == "tool" and hasattr(message, "tool_call_id") and hasattr(message, "content"):
            transformed_messages.append({
                "role": "tool", 
                "content": message.content or "",
                "tool_call_id": message.tool_call_id
            })
            continue
            
        # Handle messages with tool_calls but missing content
        if message.role == "assistant" and message.tool_calls and not message.content:
            # Set a default empty content for assistant tool call messages
            transformed_messages.append({
                "role": message.role, 
                "content": "",  # Empty string as default content
                "tool_calls": [tool_call.dict() for tool_call in message.tool_calls]
            })
        elif isinstance(message.content, list):
            user_input = " ".join([content.text for content in message.content])
            transformed_messages.append({"role": message.role, "content": user_input})
        else:
            transformed_messages.append({"role": message.role, "content": message.content or ""})

    if chat_config["modify_generation_args"] is True:
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
            sampler_params["logit_bias"] = combined_logit_bias
        else:
            sampler_params.pop("logit_bias", None)

        # Handle temperature
        if body.temperature is not None:
            sampler_params["temp"] = body.temperature
        else:
            sampler_params["temp"] = 0.7
        
        # Handle top_p
        if body.top_p is not None:
            sampler_params["top_p"] = body.top_p
        else:
            sampler_params["top_p"] = 0.9

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(transformed_messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in transformed_messages])

    if body.stream:
        async def event_stream():
            is_stream_open = True
            accumulated_text = ""
            in_thinking_mode = False
            thinking_content = ""
            had_thinking_tag = False  # Track if we've seen a </think> tag
            
            # Start in thinking mode for DeepSeek models
            if "DeepSeek".casefold() in body.model.casefold():
                in_thinking_mode = True
                
            try:
                # Use max_completion_tokens if provided, otherwise fall back to max_tokens
                max_tokens = body.max_completion_tokens or body.max_tokens
                for chunk in generate_content(prompt, max_tokens):
                    # Check for stream ending markers
                    if chunk.strip() == "[DONE]":
                        break
                    if not is_stream_open:
                        break
                    if chunk.strip().endswith("[DONE]"):
                        chunk = chunk.strip()[:-6]
                        is_stream_open = False
                    
                    # Add to accumulated text for tag detection
                    accumulated_text += chunk
                    
                    # Check for thinking tags - handle both cases
                    if "<think>" in chunk and not in_thinking_mode:
                        in_thinking_mode = True
                        # Split the chunk at <think>
                        parts = chunk.split("<think>", 1)
                        visible_part = parts[0]
                        thinking_part = "<think>" + (parts[1] if len(parts) > 1 else "")
                        
                        # Yield the visible part if any
                        if visible_part:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': visible_part}, 'finish_reason': None}], 'usage': None})}\n\n"
                        
                        # Start accumulating thinking content
                        thinking_content += thinking_part
                        continue
                    
                    # Handle case where first chunk might start with thinking content without an opening tag
                    if not in_thinking_mode and "</think>" in chunk and "<think>" not in accumulated_text:
                        in_thinking_mode = True
                        thinking_content = chunk  # Capture this chunk for processing
                        
                    if "</think>" in chunk and in_thinking_mode:
                        had_thinking_tag = True  # We've seen a </think> tag
                        in_thinking_mode = False
                        # Split at </think>
                        parts = chunk.split("</think>", 1)
                        thinking_part = parts[0]
                        visible_part = parts[1] if len(parts) > 1 else ""
                        
                        # Add to thinking content
                        thinking_content += thinking_part + "</think>"
                        
                        # Extract and process thinking content
                        full_thinking, _ = extract_thinking(thinking_content)
                        
                        # Only emit reasoning if we have actual thinking content
                        if full_thinking:
                            # Use reasoning field for all models (Claude expects 'reasoning')
                            yield f"data: {json.dumps({'choices': [{'delta': {'reasoning': full_thinking}, 'finish_reason': None}], 'usage': None})}\n\n"
                        
                        # Yield the visible part if any
                        if visible_part:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': visible_part}, 'finish_reason': None}], 'usage': None})}\n\n"
                        
                        thinking_content = ""
                        continue
                    
                    # If in thinking mode, accumulate to thinking content
                    if in_thinking_mode:
                        thinking_content += chunk
                    else:
                        # Otherwise yield as normal content
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}, 'finish_reason': None}], 'usage': None})}\n\n"
                
                # Check for unclosed thinking tags at the end
                if thinking_content and is_stream_open:
                    # Process any remaining thinking content
                    full_thinking, remaining = extract_thinking(thinking_content)
                    
                    # Only emit reasoning if we have actual thinking content or saw a </think> tag
                    if full_thinking and (had_thinking_tag or "</think>" in thinking_content):
                        yield f"data: {json.dumps({'choices': [{'delta': {'reasoning': full_thinking}, 'finish_reason': None}], 'usage': None})}\n\n"
                    
                    # Always emit remaining content
                    if remaining:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': remaining}, 'finish_reason': None}], 'usage': None})}\n\n"
                
                # Special handling for tool_calls - only used if ENABLE_TOOL_RESPONSES is True
                if ENABLE_TOOL_RESPONSES:
                    # If any message contains "use tool" or similar, simulate returning a tool_call
                    tool_call_requested = any("use tool" in str(msg.get("content", "")).lower() for msg in transformed_messages)
                    
                    if tool_call_requested:
                        # Simulate a tool call response for demonstration
                        tool_call = create_tool_response(
                            "get_current_weather", 
                            {"location": "San Francisco, CA", "unit": "celsius"}
                        )
                        
                        # Send the tool call in the stream
                        yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [tool_call]}, 'finish_reason': None}], 'usage': None})}\n\n"
                        yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}], 'usage': None})}\n\n"
                        is_stream_open = False
                        yield f"data: [DONE]\n\n"
                # End of tool calls handling
                
                # Close the stream normally if no tool calls
                if is_stream_open:
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
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': None}, 'finish_reason': 'stop', 'usage': None}]})}\n\n"
                    yield f"data: [DONE]\n\n"
                    is_stream_open = False
            except Exception as e:
                if is_stream_open:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': f'Error: {str(e)}'}, 'finish_reason': 'error'}], 'usage': None})}\n\n"
                    is_stream_open = False

        return StreamingResponse(event_stream(), media_type="application/json")
    else:
        # Use max_completion_tokens if provided, otherwise fall back to max_tokens
        max_tokens = body.max_completion_tokens or body.max_tokens
        response_content = "".join(generate_content(prompt, max_tokens, stream=False))
        prompt_tokens = len(tokenizer.encode(prompt))
        
        # Extract thinking content if present
        thinking_content, visible_content = extract_thinking(response_content)
        
        # Only consider thinking content if it contained <think> or </think> tags
        has_thinking_tags = "<think>" in response_content or "</think>" in response_content
        
        # Prepare response message
        response_message = {
            "role": "assistant",
            "content": visible_content if thinking_content and has_thinking_tags else response_content,
        }
        
        # Add reasoning only if thinking content was found AND we had thinking tags
        if thinking_content and has_thinking_tags:
            # All models use reasoning field for Claude Code compatibility
            response_message["reasoning"] = thinking_content
            
        # Special handling for tool requests - only enabled if ENABLE_TOOL_RESPONSES is True
        if ENABLE_TOOL_RESPONSES and body.tools and any("use tool" in str(msg.content).lower() for msg in body.messages if hasattr(msg, "content") and msg.content):
            # Create a simulated tool_call response
            tool_call = create_tool_response(
                "get_current_weather", 
                {"location": "San Francisco, CA", "unit": "celsius"}
            )
            
            # Create a tool call response instead of a text response
            tool_response_message = {
                "role": "assistant",
                "content": None,  # No content for tool calls
                "tool_calls": [tool_call]
            }
            
            response_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [{
                    "index": 0,
                    "message": tool_response_message,
                    "logprobs": None,
                    "finish_reason": "tool_calls"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 20,  # Estimate for tool call tokens
                    "total_tokens": prompt_tokens + 20,
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "accepted_prediction_tokens": 20,
                        "rejected_prediction_tokens": 0
                    }
                }
            }
        else:
            # Regular text response handling
            # Calculate token counts
            response_tokens = len(tokenizer.encode(response_content))
            reasoning_tokens = len(tokenizer.encode(thinking_content)) if thinking_content else 0
            total_tokens = prompt_tokens + response_tokens
            
            response_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [{
                    "index": 0,
                    "message": response_message,
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": response_tokens,
                    "total_tokens": total_tokens,
                    "completion_tokens_details": {
                        "reasoning_tokens": reasoning_tokens,
                        "accepted_prediction_tokens": response_tokens - reasoning_tokens,
                        "rejected_prediction_tokens": 0
                    }
                }
            }
        
        logger.info(f"Response (NOT streaming): {response_data}")
        response = ChatCompletionResponse(**response_data)
        return response
    
@app.post("/v1/engines/copilot-codex/completions")
async def codex_completions(request: Request):
    # Parse the request body manually since it doesn't match our RequestBody model
    body_json = await request.json()
    prompt = body_json.get("prompt", "")
    suffix = body_json.get("suffix", "")
    max_tokens = body_json.get("max_tokens", 4096)
    max_completion_tokens = body_json.get("max_completion_tokens")
    stream = body_json.get("stream", True)
    
    # Use max_completion_tokens if provided, otherwise fall back to max_tokens
    tokens_to_generate = max_completion_tokens or max_tokens
    
    # Combine prompt and suffix
    full_prompt = prompt + suffix

    def create_event_stream():
        is_stream_open = True
        try:
            for chunk in generate_content(full_prompt, tokens_to_generate):
                if chunk.strip() == "[DONE]":
                    break
                if not is_stream_open:
                    break
                if chunk.strip().endswith("[DONE]"):
                    chunk = chunk.strip()[:-6]
                    is_stream_open = False
                yield f"data: {json.dumps({'choices': [{'text': chunk, 'index': 0, 'finish_reason': None, 'logprobs': None, 'p': 'aa'}]})}\n\n"
            if is_stream_open:
                yield f"data: {json.dumps({'choices': [{'text': None, 'index': 0, 'finish_reason': 'stop', 'logprobs': None, 'p': 'aa'}]})}\n\n"
                yield "data: [DONE]\n\n"
                is_stream_open = False
        except Exception as e:
            if is_stream_open:
                yield f"data: {json.dumps({'choices': [{'text': f'Error: {str(e)}', 'index': 0, 'finish_reason': 'error', 'logprobs': None, 'p': 'aa'}]})}\n\n"
                is_stream_open = False

    if stream:
        return StreamingResponse(create_event_stream(), media_type="application/json")
    else:
        response_content = "".join(generate_content(full_prompt, tokens_to_generate, stream=False))
        response_data = {
            "id": "cmpl-AU5fdRbH8AIr5G5pGH7dU6aYZcBOJ",
            "created": int(time.time()),
            "model": body_json.get("model", "gpt-35-turbo"),
            "choices": [{
                "text": response_content,
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "p": "aa"
            }]
        }
        return response_data
    
@app.get("/v1/models", response_model=ModelsResponse)
async def get_models():
    return ModelsResponse(object="list", data=[
        *local_models["data"],
        *openai_models["data"]
    ])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
