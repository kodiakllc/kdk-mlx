import os
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
from collections.abc import Generator

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    GREY = "\033[90m"
    RED = "\033[31m"

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models path relative to the script directory
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")

# Set the current model
CURRENT_T_MODEL = "Qwen3-32B-4bit"

# Init the model and tokenizer
model = None
tokenizer = None

# Initialize conversation
system_prompt = """You are a helpful AI that responds in markdown format.

When responding, please follow this structure:
1. First, share your thinking process within <think>...</think> tags. This helps the user understand your reasoning.
2. After your thoughts, provide your final answer.

You can use the following formatting in your responses:
- For your internal thoughts, always enclose them in <think>...</think> tags.
- Use **bold** for emphasis
- Use _italics_ for secondary emphasis
- Use `code` for inline code
- Use ```language\\ncode\\n``` for code blocks
- Use bullet points and numbered lists when appropriate

Always think through your responses carefully before answering, and make sure to include your thought process."""

messages = [{"role": "system", "content": system_prompt}]

# Define sampler parameters
sampler_params = {
    "temp": 0.7,
    "top_p": 0.95,
    # Optional parameters
    "min_p": 0.05,
    "min_tokens_to_keep": 5,
    "top_k": 50
}

def generate_content(
    prompt: str,
    max_tokens: int = 1024,
    adapter: str | None = None,
) -> Generator[str, None, None]:
    global model, tokenizer
    
    # Create sampler with defined parameters
    sampler = make_sampler(**sampler_params)
    
    response = stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
    )
    for token in response:
        # Handle token.text for GenerationResponse objects from newer mlx-lm versions
        if hasattr(token, 'text'):
            yield token.text
        else:
            yield token

def stream_content(prompt: str, max_tokens: int) -> str:
    response = f"{Colors.BOLD}{Colors.CYAN}Assistant: {Colors.RESET}"
    print(response, end='', flush=True)
    
    full_response = ""
    thoughts_mode = True  # Start assuming everything is thoughts
    current_buffer = ""
    first_chunk = True
    
    # Initial formatting for thoughts
    print(f"{Colors.GREY}{Colors.ITALIC}", end='', flush=True)
    
    for chunk in generate_content(prompt=prompt, max_tokens=max_tokens):
        # Handle first chunk - strip out leading <think> if present
        if first_chunk:
            first_chunk = False
            if chunk.startswith("<think>"):
                chunk = chunk[7:]  # Remove the opening <think> tag
        
        full_response += chunk
        current_buffer += chunk
        
        # Check for thought transitions
        if "</think>" in current_buffer and thoughts_mode:
            # End of thoughts section
            thoughts_mode = False
            parts = current_buffer.split("</think>", 1)  # Split only on first occurrence
            thought_end = parts[0]
            post_thought = parts[1] if len(parts) > 1 else ""
            
            # Print end of thought and reset formatting
            print(thought_end, end='', flush=True)
            print(f"{Colors.RESET}", end='', flush=True)
            
            if post_thought:
                print(post_thought, end='', flush=True)
            
            current_buffer = post_thought
            
        elif "<think>" in current_buffer and not thoughts_mode:
            # Start of a new thoughts section
            thoughts_mode = True
            parts = current_buffer.split("<think>", 1)  # Split only on first occurrence
            pre_thought = parts[0]
            thought_start = parts[1] if len(parts) > 1 else ""
            
            if pre_thought:
                print(pre_thought, end='', flush=True)
            
            # Print beginning of thought in grey italic
            print(f"{Colors.GREY}{Colors.ITALIC}", end='', flush=True)
            if thought_start:
                print(thought_start, end='', flush=True)
            
            current_buffer = thought_start
            
        else:
            # Continue printing with current formatting
            print(chunk, end='', flush=True)
    
    # Ensure we reset colors at the end
    print(f"{Colors.RESET}\n\n", end='', flush=True)
    return full_response

def process_output(output_ids, tokenizer):
    """Process the output_ids to separate thinking content from regular content"""
    try:
        # Try to find the index of </think> tag in the output
        think_end_tag_id = 151668  # ID for </think> token
        # Searching from end to beginning (reversed list)
        reversed_ids = output_ids[::-1]
        if think_end_tag_id in reversed_ids:
            index = len(output_ids) - reversed_ids.index(think_end_tag_id)
            thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return thinking_content, content
        else:
            # No thinking tags found
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            return "", content
    except ValueError:
        # No thinking tags found
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return "", content

while True:
    user_input = input(f"{Colors.BOLD}{Colors.GREEN}You: {Colors.RESET}")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages.append({"role": "user", "content": user_input})

    # Check if we have loaded the model
    if model is None or tokenizer is None:
        print(f"{Colors.YELLOW}Loading model...{Colors.RESET}")
        model, tokenizer = load(path_or_hf_repo=MODELS_PATH + CURRENT_T_MODEL, lazy=True)
        print(f"{Colors.GREEN}Model loaded!{Colors.RESET}")

    # Apply chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        # Use apply_chat_template with enable_thinking parameter
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode as in the reference code
        )
    else:
        print(f"{Colors.RED}Chat template not available. Cannot continue.{Colors.RESET}")
        break

    # Stream response and capture it
    response = stream_content(prompt, 32768)  # Using larger max token count for Qwen3-32B
    messages.append({"role": "assistant", "content": response})
