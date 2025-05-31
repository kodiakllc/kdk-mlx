import os
import streamlit as st
import streamlit.components.v1 as components
import time
import json
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
import re
from mlx_quantizer import MLXQuantizer, create_streamlit_quantizer_ui
from hot_model_inference import HotModelInference, FeedbackSignal, create_feedback_ui, create_hmi_control_panel

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

def load_available_models():
    """Load available models from JSON file"""
    models_json_path = os.path.join(SCRIPT_DIR, "../hf_models/models.json")
    try:
        with open(models_json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Models configuration file not found: {models_json_path}")
        return {}
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in models configuration file: {models_json_path}")
        return {}

# Load available models from JSON
AVAILABLE_MODELS = load_available_models()

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI that responds in markdown format.

When responding, please follow this structure:
1. First, share your thinking process within <think> </think> tags. This helps the user understand your reasoning.
2. After your thoughts, provide your final answer.

You can use the following formatting in your responses:
- For your internal thoughts, always enclose them in <think> </think> tags.
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
        "temp": 0.6,
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

# Hot Model Inference system
if "hmi" not in st.session_state:
    st.session_state.hmi = None
if "hmi_enabled" not in st.session_state:
    st.session_state.hmi_enabled = True
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False
if "feedback_expanded" not in st.session_state:
    st.session_state.feedback_expanded = False

# Generation control
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

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
        
        # Initialize HMI if enabled
        if st.session_state.hmi_enabled:
            st.session_state.hmi = HotModelInference(MODELS_PATH + model_name)
            st.info("🚀 Hot Model Inference system initialized!")
        
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
        
        # Group models by base family + version and organize by type
        families = {}
        for model_name, info in AVAILABLE_MODELS.items():
            base_family = info['base_family']
            version = info.get('version', '')
            family_key = f"{base_family}{version}"
            
            if family_key not in families:
                families[family_key] = {
                    'base': [],
                    'distilled': [],
                    'specialized': []
                }
            
            model_type = info.get('type', 'base')
            families[family_key][model_type].append((model_name, info))
        
        # Sort families and models within families by parameters
        def extract_param_number(param_str):
            """Extract numeric value from parameter string like '4B' -> 4"""
            return float(param_str.replace('B', ''))
        
        def create_display_name(model_name, info):
            """Create a display name showing the model details"""
            display_parts = [f"{info['parameters']}"]
            
            # Add model type information FIRST
            if info.get('type') == 'distilled':
                display_parts.append("[ DeepSeek-R1 ]")
            elif info.get('type') == 'specialized':
                display_parts.append(info.get('specialization', 'Special'))
            
            # Add quantization bits if not 4bit (since 4bit is the default)
            quantization = info.get('quantization', '4bit')
            if quantization != '4bit':
                display_parts.append(f"{quantization}")
            
            # Add quantization method (QAT, DWQ, etc.) AFTER model type
            quant_options = info.get('quantization_options', [])
            if 'qat' in quant_options:
                display_parts.append("QAT")
            elif 'dwq' in quant_options:
                display_parts.append("DWQ")
            
            # Add variant if present (but not version since it's in the group header)
            if 'variant' in info:
                display_parts.append(f"({info['variant']})")
            
            return f"  {' - '.join(display_parts)}"
        
        # Create organized model options
        model_options = []
        for base_family in sorted(families.keys()):
            family_data = families[base_family]
            
            # Add family header
            model_options.append((f"── {base_family} ──", None))
            
            # Combine all models in the family and sort by parameter size
            all_family_models = []
            all_family_models.extend(family_data['base'])
            all_family_models.extend(family_data['specialized'])
            all_family_models.extend(family_data['distilled'])
            
            # Sort all models by parameter size
            sorted_models = sorted(all_family_models, key=lambda x: extract_param_number(x[1]['parameters']))
            
            # Add all models in parameter order
            for model_name, info in sorted_models:
                display_name = create_display_name(model_name, info)
                model_options.append((display_name, model_name))
        
        # Find current model index
        current_index = 0
        if st.session_state.current_model:
            for i, (_, model_name) in enumerate(model_options):
                if model_name == st.session_state.current_model:
                    current_index = i
                    break
        
        selected_display = st.selectbox(
            "Select Model", 
            [display for display, _ in model_options],
            index=current_index,
            format_func=lambda x: x if not x.startswith("──") else x
        )
        
        # Get the actual model name from the selection (skip if it's a header)
        selected_model = None
        for display, model in model_options:
            if display == selected_display and model is not None:
                selected_model = model
                break

        # Add Hot Model Inference toggle right after model selection
        hmi_enabled = st.checkbox(
            "🚀 Enable Hot Model Inference (HMI)", 
            value=st.session_state.hmi_enabled,
            help="Enable Hot Model Inference for real-time adaptation and feedback learning"
        )
        st.session_state.hmi_enabled = hmi_enabled

        if st.button("Load Model"):
            if selected_model is not None:
                load_selected_model(selected_model)
            else:
                st.error("Please select a model, not a family header.")

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
            st.rerun()
        
        # Add Hot Model Inference controls (if HMI is enabled and initialized)
        if hmi_enabled and st.session_state.hmi:
            st.divider()
            st.subheader("🚀 Hot Model Inference Controls")
            create_hmi_control_panel(st.session_state.hmi)
            
            # Feedback controls
            st.session_state.show_feedback = st.checkbox(
                "Show Feedback Panel", 
                value=st.session_state.show_feedback
            )
        
        # Add quantization UI
        st.divider()
        quantizer = MLXQuantizer(MODELS_PATH)
        create_streamlit_quantizer_ui(quantizer)
    
    # Main chat interface
    st.title("💬 MLX Chat")
    
    # Display model info
    if st.session_state.current_model:
        model_info = f"Using model: **{st.session_state.current_model}**"
        if st.session_state.hmi_enabled and st.session_state.hmi:
            model_info += " 🚀 **HMI Active** (Hot Model Inference enabled)"
        st.caption(model_info)
    else:
        st.warning("Please select and load a model from the sidebar.")
    
    # Create message container
    create_message_container()
    
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
        
        # Set generation state and minimize feedback panel
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        st.session_state.feedback_expanded = False  # Minimize feedback panel during generation
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Apply chat template if available
            messages = st.session_state.messages
            tokenizer = st.session_state.tokenizer
            
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                # Check if current model is a reasoning model
                current_model_info = AVAILABLE_MODELS.get(st.session_state.current_model, {})
                is_reasoning_model = current_model_info.get('reasoning', False)
                
                prompt_template = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=is_reasoning_model
                )
                
                # Check if current model is a reasoning model
                current_model_info = AVAILABLE_MODELS.get(st.session_state.current_model, {})
                is_reasoning_model = current_model_info.get('reasoning', False)
                
                # Track transition from thinking to answer - only for reasoning models
                in_thinking_mode = is_reasoning_model
                thinking_content = ""
                answer_content = ""
                
                # Token generation metrics
                token_counter = 0
                start_time = time.time()
                token_speed_container = st.empty()
                
                # Stream the response - use HMI if enabled, otherwise use standard generation
                if st.session_state.hmi_enabled and st.session_state.hmi:
                    # Use Hot Model Inference with real-time feedback
                    def real_time_feedback_callback(partial_response):
                        # Simple heuristic feedback during generation
                        if len(partial_response) > 100:
                            # Check for repetition or poor quality indicators
                            words = partial_response.split()
                            if len(words) > 10:
                                unique_words = len(set(words[-10:]))
                                if unique_words < 5:  # High repetition detected
                                    from hot_model_inference import FeedbackSignal
                                    return FeedbackSignal(
                                        response_quality=0.3,
                                        coherence=0.4,
                                        relevance=0.5,
                                        creativity=0.2,
                                        factuality=0.5,
                                        user_satisfaction=0.3,
                                        timestamp=time.time()
                                    )
                        return None
                    
                    # Generate using HMI system with streaming
                    token_stream = st.session_state.hmi.adaptive_generate(
                        prompt_template, 
                        feedback_callback=real_time_feedback_callback
                    )
                else:
                    # Standard generation
                    token_stream = generate_content(prompt_template)
                
                # Process the token stream (works for both HMI and standard generation)
                for token in token_stream:
                    # Check for stop signal
                    if st.session_state.stop_generation:
                        break
                        
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
                    
                    # Only process thinking tags for reasoning models
                    if is_reasoning_model:
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
                                parts = full_response.split("</think>", 1)
                                if len(parts) > 1:
                                    answer_content = parts[1].strip()
                                else:
                                    answer_content = ""
                    else:
                        # For non-reasoning models, everything is answer content
                        answer_content = full_response
                    
                    # Build display text
                    display_text = ""
                    if is_reasoning_model:
                        # For reasoning models, show thinking and answer separately
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
                    else:
                        # For non-reasoning models, show everything as answer
                        clean_response = re.sub(r"<think>|</think>", "", answer_content)
                        display_text = clean_response
                    
                    message_placeholder.markdown(display_text, unsafe_allow_html=True)
                    time.sleep(0.001)  # Small delay to reduce CPU usage
                
                # Special handling for end of response
                # If we never encountered a </think> tag for reasoning models, or if not a reasoning model
                current_model_info = AVAILABLE_MODELS.get(st.session_state.current_model, {})
                is_reasoning_model = current_model_info.get('reasoning', False)
                
                if is_reasoning_model and "</think>" not in full_response:
                    # For reasoning models that didn't complete thinking, show everything in thinking box
                    display_text = f"""
                    <div class="thinking-box">
                        <div class="thinking-header">🧠 Thinking Process</div>
                        {full_response.replace('<think>', '').replace('</think>', '')}
                    </div>
                    """
                    message_placeholder.markdown(display_text, unsafe_allow_html=True)
                elif not is_reasoning_model:
                    # For non-reasoning models, display everything as normal answer
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
                
                # Reset generation state
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
            else:
                st.error("Chat template not available for this model.")
                # Reset generation state even on error
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
    
    # Show stop button during generation (positioned after chat input)
    if st.session_state.get('is_generating', True):
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🛑 Stop Generation", type="secondary", use_container_width=True, key="stop_button"):
                st.session_state.stop_generation = True
                st.rerun()
    
    # Show previous generation metrics in a small format at the bottom
    if st.session_state.token_metrics["tokens_per_second"] > 0:
        st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8em; margin-top: 10px; margin-bottom: 5px;'>💨 <span style='font-weight: bold;'>{st.session_state.token_metrics['tokens_per_second']:.1f}</span> tokens/sec | <span style='font-weight: bold;'>{st.session_state.token_metrics['tokens_generated']}</span> tokens in {st.session_state.token_metrics['generation_time']:.1f}s</div>", unsafe_allow_html=True)
    
    # Feedback panel at the very bottom - only visible when NOT generating
    if (st.session_state.show_feedback and st.session_state.hmi and 
        not st.session_state.get('is_generating', False)):
        
        st.divider()  # Visual separator
        
        with st.expander("🔄 Real-time Feedback", expanded=st.session_state.feedback_expanded):
            feedback = create_feedback_ui()
            if feedback and len(st.session_state.messages) > 1:
                # Get the last assistant response
                last_response = ""
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "assistant":
                        last_response = msg["content"]
                        break
                
                if last_response:
                    st.session_state.hmi.learn_from_feedback(last_response, feedback)
                    st.success("Feedback applied! Model will adapt for future responses.")
                    # Auto-minimize the feedback panel after successful submission
                    st.session_state.feedback_expanded = False
                    time.sleep(3)
                    
                    # JavaScript to close the feedback expander
                    close_expander_script = '''
                    <script>
                        // Find the feedback expander by looking for "Real-time Feedback" text
                        setTimeout(function() {
                            console.log("Looking for feedback expander...");
                            
                            // Get all expanders
                            let allExpanders = parent.document.querySelectorAll("[data-testid='stExpander']");
                            console.log("Found " + allExpanders.length + " expanders");
                            
                            for (let i = 0; i < allExpanders.length; i++) {
                                let expander = allExpanders[i];
                                
                                // Look for "Real-time Feedback" text in the summary
                                let summary = expander.querySelector("summary");
                                if (summary && summary.textContent.includes("Real-time Feedback")) {
                                    console.log("Found Real-time Feedback expander");
                                    
                                    // Find the details element and remove open attribute
                                    let detailsElement = expander.querySelector("details[open]");
                                    if (detailsElement) {
                                        detailsElement.removeAttribute("open");
                                        console.log("Feedback expander closed successfully.");
                                        break;
                                    } else {
                                        console.log("Details element not found or not open.");
                                    }
                                }
                            }
                        }, 100);
                    </script>
                    '''
                    
                    # Create a temporary container to inject the script
                    script_container = st.empty()
                    with script_container:
                        components.html(close_expander_script, height=0)
                    
                    # Remove script after executing
                    time.sleep(0.5)
                    script_container.empty()
                    
                    st.rerun()

if __name__ == "__main__":
    main()
