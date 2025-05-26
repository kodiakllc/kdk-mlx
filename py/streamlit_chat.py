import os
import streamlit as st
import time
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
import re

# Set page configuration and custom CSS for styling
st.set_page_config(
    page_title="MLX Chat Interface",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for styling the thinking section with Dracula-inspired theme
st.markdown("""
<style>
.thinking-box {
    background-color: #282a36;
    border-left: 5px solid #bd93f9;
    padding: 16px;
    margin: 14px 0;
    border-radius: 4px;
    font-style: italic;
    color: #f8f8f2;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}
.thinking-header {
    color: #ff79c6;
    font-weight: bold;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.thinking-box code {
    background-color: #44475a;
    color: #50fa7b;
    padding: 2px 4px;
    border-radius: 3px;
}
.thinking-box pre {
    background-color: #44475a;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# Get the directory where the script is running
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the models path relative to the script directory
MODELS_PATH = os.path.join(SCRIPT_DIR, "../hf_models/")

# Available models
AVAILABLE_MODELS = [
    ##########################################
    ### Gemma 3
    "gemma-3-4b-it-4bit",
    "gemma-3-12b-it-4bit",
    # Quantization Aware Training (QAT)
    "gemma-3-4b-it-qat-4bit",
    "gemma-3-12b-it-qat-4bit",
    # Distilled Weight Quantized (DWQ)
    "gemma-3-4b-it-4bit-DWQ",
    "gemma-3-12b-it-4bit-DWQ",
    ##########################################
    ### DeepSeek
    "DeepSeek-R1-Distill-Llama-8B-4bit",
    "DeepSeek-R1-Distill-Qwen-14B-4bit",
    "DeepSeek-R1-Distill-Qwen-32B-4bit",
    ##########################################
    ### Qwen
    "Qwen3-32B-4bit"
]

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI that responds in markdown format.

When responding, please follow this structure:
1. First, share your thinking process within <think>...</think> tags. This helps the user understand your reasoning.
2. After your thoughts, provide your final answer.

You can use the following formatting in your responses:
- For your internal thoughts, always enclose them in <think>...</think> tags.
- Use **bold** for emphasis
- Use _italics_ for secondary emphasis
- Use `code` for inline code
- Use ```language
code
``` for code blocks
- Use bullet points and numbered lists when appropriate

Always think through your responses carefully before answering, and make sure to include your thought process."""

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None
# Performance metrics
if "token_metrics" not in st.session_state:
    st.session_state.token_metrics = {
        "tokens_generated": 0,
        "generation_time": 0,
        "tokens_per_second": 0
    }
# Base sampler parameters (always enabled)
if "base_params" not in st.session_state:
    st.session_state.base_params = {
        "temp": 0.7,
        "top_p": 0.95,
    }
# Optional sampler parameters that can be toggled
if "optional_params" not in st.session_state:
    st.session_state.optional_params = {
        "min_p": {"enabled": False, "value": 0.05},
        "min_tokens_to_keep": {"enabled": False, "value": 5},
        "top_k": {"enabled": False, "value": 50},
    }
# Generation parameters
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 8192

def load_selected_model(model_name):
    """Load the selected model and tokenizer"""
    if st.session_state.current_model == model_name and st.session_state.model is not None:
        st.success(f"Model {model_name} already loaded!")
        return
    
    # Unload the previous model if it exists
    if st.session_state.model is not None:
        with st.spinner(f"Unloading previous model {st.session_state.current_model}..."):
            # Set references to None to allow garbage collection
            st.session_state.model = None
            st.session_state.tokenizer = None
            # Force Python garbage collection to free memory
            import gc
            gc.collect()
            st.info(f"Previous model {st.session_state.current_model} unloaded")
    
    # Reset token metrics when switching models
    st.session_state.token_metrics = {
        "tokens_generated": 0,
        "generation_time": 0,
        "tokens_per_second": 0
    }
    
    with st.spinner(f"Loading model {model_name}..."):
        st.session_state.model, st.session_state.tokenizer = load(
            path_or_hf_repo=MODELS_PATH + model_name, 
            lazy=True
        )
        st.session_state.current_model = model_name
        st.success(f"Model {model_name} loaded!")

def generate_content(prompt, max_tokens=None):
    """Generate content from the model"""
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    # If max_tokens is not provided, use the value from session state
    if max_tokens is None:
        max_tokens = st.session_state.max_tokens
    
    # Build sampler parameters by combining base params with enabled optional params
    sampler_params = st.session_state.base_params.copy()
    
    # Add optional parameters if enabled
    for param_name, param_data in st.session_state.optional_params.items():
        if param_data["enabled"]:
            sampler_params[param_name] = param_data["value"]
    
    # Create sampler with selected parameters
    sampler = make_sampler(**sampler_params)
    
    # Generate tokens
    response = stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler
    )
    
    return response

def extract_thinking(full_text):
    """Extract thinking content and answer from text that contains <think>...</think> tags"""
    # Simple case: If no thinking tags, everything is the answer
    if "<think>" not in full_text and "</think>" not in full_text:
        return "", full_text
    
    # If we have complete thinking blocks
    if "</think>" in full_text:
        # Split at the first </think> tag to get everything before it as thinking
        parts = full_text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return thinking, answer
    
    # If we only have opening <think> but no closing tag yet
    # (we're still in the thinking part of the response)
    if "<think>" in full_text and "</think>" not in full_text:
        thinking = full_text.replace("<think>", "").strip()
        return thinking, ""
    
    # Default fallback case - just return the text as thinking
    return full_text, ""

def clear_conversation():
    """Clear the conversation history"""
    st.session_state.messages = [{"role": "system", "content": st.session_state.messages[0]["content"]}]

def create_message_container():
    """Create styled message containers based on chat history"""
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue  # Skip system messages
            
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                # Check if there's thinking content
                thinking, answer = extract_thinking(message["content"])
                if thinking:
                    # Style the thinking section with custom HTML
                    st.markdown(f"""
                    <div class="thinking-box">
                        <div class="thinking-header">🤔 Thinking Process</div>
                        {thinking}
                    </div>
                    """, unsafe_allow_html=True)
                if answer:
                    st.markdown(answer)

def main():
        # Sidebar for model selection and parameter configuration
    with st.sidebar:
        st.title("MLX Chat Interface")
        
        selected_model = st.selectbox(
            "Select Model", 
            AVAILABLE_MODELS,
            index=0 if st.session_state.current_model is None else AVAILABLE_MODELS.index(st.session_state.current_model)
        )
        
        if st.button("Load Model"):
            load_selected_model(selected_model)

        st.divider()
        st.subheader("Sampling Parameters")
        
        # Basic parameters section
        st.markdown("**Basic Parameters**")
        
        # Temperature slider
        st.session_state.base_params["temp"] = st.slider(
            "Temperature", 0.0, 2.0, st.session_state.base_params["temp"], 0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        # Top-p slider
        st.session_state.base_params["top_p"] = st.slider(
            "Top-p", 0.0, 1.0, st.session_state.base_params["top_p"], 0.05,
            help="Nucleus sampling: consider tokens with top_p cumulative probability"
        )
        
        # Max tokens slider
        st.session_state.max_tokens = st.slider(
            "Max Tokens", 1024, 32768, st.session_state.max_tokens, 1024,
            help="Maximum number of tokens to generate"
        )
        
        # Advanced parameters section (collapsible)
        with st.expander("Advanced Sampling Parameters", expanded=False):
            # Min-p checkbox and slider
            min_p_enabled = st.checkbox(
                "Enable Min-p", 
                value=st.session_state.optional_params["min_p"]["enabled"],
                help="Enable min-p filtering"
            )
            st.session_state.optional_params["min_p"]["enabled"] = min_p_enabled
            
            if min_p_enabled:
                st.session_state.optional_params["min_p"]["value"] = st.slider(
                    "Min-p Value", 0.0, 0.5, st.session_state.optional_params["min_p"]["value"], 0.01,
                    help="Only consider tokens with at least this probability"
                )
            
            # Top-k checkbox and slider
            top_k_enabled = st.checkbox(
                "Enable Top-k", 
                value=st.session_state.optional_params["top_k"]["enabled"],
                help="Enable top-k filtering"
            )
            st.session_state.optional_params["top_k"]["enabled"] = top_k_enabled
            
            if top_k_enabled:
                st.session_state.optional_params["top_k"]["value"] = st.slider(
                    "Top-k Value", 0, 100, st.session_state.optional_params["top_k"]["value"], 1,
                    help="Only consider top k tokens"
                )
            
            # Min tokens to keep checkbox and slider
            min_tokens_enabled = st.checkbox(
                "Enable Min Tokens to Keep", 
                value=st.session_state.optional_params["min_tokens_to_keep"]["enabled"],
                help="Enable minimum tokens to keep"
            )
            st.session_state.optional_params["min_tokens_to_keep"]["enabled"] = min_tokens_enabled
            
            if min_tokens_enabled:
                st.session_state.optional_params["min_tokens_to_keep"]["value"] = st.slider(
                    "Min Tokens Value", 1, 20, st.session_state.optional_params["min_tokens_to_keep"]["value"], 1,
                    help="Minimum number of tokens to keep regardless of other filters"
                )
            
            st.info("Only enabled parameters will be used in the sampling process.")
        
        # System prompt
        st.divider()
        st.subheader("System Prompt")
        system_prompt = st.text_area(
            "Edit System Prompt", 
            st.session_state.messages[0]["content"],
            height=200
        )
        
        if st.button("Update System Prompt"):
            st.session_state.messages[0]["content"] = system_prompt
            st.success("System prompt updated!")
        
        if st.button("Clear Conversation"):
            clear_conversation()
            st.experimental_rerun()
    
    # Main chat interface
    st.title("💬 MLX Chat")
    
    # Display model info
    if st.session_state.current_model:
        st.caption(f"Using model: **{st.session_state.current_model}**")
    else:
        st.warning("Please select and load a model from the sidebar.")
    
    # Create message container
    create_message_container()
    
    # Show previous generation metrics in a small format at the bottom of the chat
    if st.session_state.token_metrics["tokens_per_second"] > 0:
        st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8em; margin-top: 5px; margin-bottom: 5px;'>💨 <span style='font-weight: bold;'>{st.session_state.token_metrics['tokens_per_second']:.1f}</span> tokens/sec | <span style='font-weight: bold;'>{st.session_state.token_metrics['tokens_generated']}</span> tokens in {st.session_state.token_metrics['generation_time']:.1f}s</div>", unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Type your message here"):
        if st.session_state.model is None:
            st.error("Please load a model first!")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Apply chat template if available
            messages = st.session_state.messages
            tokenizer = st.session_state.tokenizer
            
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                prompt_template = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                
                # Track transition from thinking to answer
                in_thinking_mode = True
                thinking_content = ""
                answer_content = ""
                
                # Token generation metrics
                token_counter = 0
                start_time = time.time()
                token_speed_container = st.empty()
                
                # Stream the response
                for token in generate_content(prompt_template):
                    token_text = token.text if hasattr(token, 'text') else token
                    full_response += token_text
                    
                    # Update token counter and display metrics (every 10 tokens to reduce UI overhead)
                    token_counter += 1
                    if token_counter % 10 == 0:
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            tokens_per_second = token_counter / elapsed
                            token_speed_container.markdown(f"""
                            <div style='position: fixed; bottom: 120px; right: 20px; background: rgba(0,0,0,0.15); 
                                 padding: 8px 15px; border-radius: 6px; font-size: 16px; z-index: 1000; 
                                 box-shadow: 0 3px 10px rgba(0,0,0,0.2); border-left: 3px solid #50fa7b;'>
                                <div style='font-weight: bold; margin-bottom: 2px;'>💨 Generation Speed</div>
                                <div><span style='font-weight: bold; font-size: 18px;'>{tokens_per_second:.1f}</span> tokens/sec 
                                <span style='color: #50fa7b; margin-left: 5px;'>↑</span></div>
                                <div style='font-size: 14px; opacity: 0.8;'>{token_counter} tokens in {elapsed:.1f}s</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Check if we've reached the end of a thinking block
                    if in_thinking_mode and "</think>" in full_response:
                        # Split at the </think> tag
                        parts = full_response.split("</think>", 1)
                        
                        # Extract thinking content (remove opening <think> tag if present)
                        thinking_content = parts[0].replace("<think>", "").strip()
                        
                        # Start tracking answer content separately
                        answer_content = parts[1].strip() if len(parts) > 1 else ""
                        in_thinking_mode = False
                    elif in_thinking_mode:
                        # Still in thinking mode, update thinking content
                        thinking_content = full_response.replace("<think>", "").strip()
                    else:
                        # Already in answer mode, update answer content
                        if "<think>" in token_text:
                            # If a new thinking block starts, reset to thinking mode
                            in_thinking_mode = True
                            parts = full_response.split("<think>", 1)
                            answer_content = parts[0].strip()
                            thinking_content = parts[1].replace("</think>", "").strip()
                        else:
                            # Just append to answer
                            answer_content = full_response.split("</think>", 1)[1].strip()
                    
                    # Build display text
                    display_text = ""
                    if thinking_content:
                        # Style thinking section with custom HTML
                        display_text += f"""
                        <div class="thinking-box">
                            <div class="thinking-header">🧠 Thinking...</div>
                            {thinking_content}
                        </div>
                        """
                    
                    # Add answer if available
                    if answer_content:
                        display_text += answer_content
                    
                    message_placeholder.markdown(display_text, unsafe_allow_html=True)
                    time.sleep(0.001)  # Small delay to reduce CPU usage
                
                # Special handling for end of response
                # If we never encountered a </think> tag, assume all content is the answer
                if "</think>" not in full_response:
                    # Clean up any partial thinking tags that might be causing display issues
                    clean_response = re.sub(r"<think>|</think>", "", full_response)
                    message_placeholder.markdown(clean_response, unsafe_allow_html=True)
                
                # Clear the temporary token speed display
                token_speed_container.empty()
                
                # Final token metrics
                final_time = time.time() - start_time
                if final_time > 0 and token_counter > 0:
                    final_tokens_per_second = token_counter / final_time
                    # Store metrics in session state
                    st.session_state.token_metrics = {
                        "tokens_generated": token_counter,
                        "generation_time": final_time,
                        "tokens_per_second": final_tokens_per_second
                    }
                    
                    # Add the final metrics to the displayed response
                    metrics_html = f"""
                    <div style='background: rgba(0,0,0,0.05); padding: 8px; border-radius: 6px; margin-top: 10px; border-left: 3px solid #50fa7b;'>
                        <div style='font-weight: bold; font-size: 14px;'>💨 Generation metrics</div>
                        <div>
                            <span style='font-weight: bold; font-size: 16px;'>{final_tokens_per_second:.1f}</span> tokens/sec
                            <span style='color: #50fa7b; margin-left: 3px;'>↑</span> | 
                            <span style='font-weight: bold;'>{token_counter}</span> tokens generated in {final_time:.1f}s
                        </div>
                    </div>
                    """
                    
                    # Append metrics to the last message that's displayed
                    # This ensures it appears as part of the assistant's response
                    full_display = display_text + metrics_html
                    message_placeholder.markdown(full_display, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error("Chat template not available for this model.")

if __name__ == "__main__":
    main()
