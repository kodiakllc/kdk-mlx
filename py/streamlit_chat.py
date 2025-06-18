import os
import streamlit as st
import streamlit.components.v1 as components
import time
import json
import io
import tempfile
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
import re
from mlx_quantizer import MLXQuantizer, create_streamlit_quantizer_ui
from hot_model_inference import HotModelInference, FeedbackSignal, create_feedback_ui, create_hmi_control_panel
from PIL import Image

# Try to import VLM support
try:
    from mlx_vlm import load as vlm_load, stream_generate as vlm_stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False

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
    """Load available models from JSON file, excluding non-MLX models"""
    models_json_path = os.path.join(SCRIPT_DIR, "../hf_models/models.json")
    non_mlx_json_path = os.path.join(SCRIPT_DIR, "../hf_models/non_mlx_models.json")
    
    try:
        with open(models_json_path, 'r') as f:
            all_models = json.load(f)
            
        # Load non-MLX models to exclude
        excluded_models = set()
        if os.path.exists(non_mlx_json_path):
            try:
                with open(non_mlx_json_path, 'r') as f:
                    non_mlx_data = json.load(f)
                    excluded_models = set(non_mlx_data.keys())
            except:
                pass
        
        # Filter out non-MLX models
        mlx_models = {
            name: info for name, info in all_models.items() 
            if name not in excluded_models
        }
        
        return mlx_models
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
if "processor" not in st.session_state:
    st.session_state.processor = None
if "is_vlm" not in st.session_state:
    st.session_state.is_vlm = False
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
            st.session_state.processor = None  # For VLMs
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
    
    # Check if this is a vision model (default to 'lm' if model_type not specified)
    model_info = AVAILABLE_MODELS.get(model_name, {})
    is_vlm = model_info.get('model_type', 'lm') == 'vlm'
    
    with st.spinner(f"Loading model {model_name}..."):
        if is_vlm and VLM_AVAILABLE:
            # Load as vision model
            st.session_state.model, st.session_state.processor = vlm_load(
                path_or_hf_repo=MODELS_PATH + model_name, lazy = True
            )
            st.session_state.tokenizer = st.session_state.processor.tokenizer if hasattr(st.session_state.processor, 'tokenizer') else None
            st.info(f"Loaded as Vision-Language Model")
        else:
            # Load as text model
            st.session_state.model, st.session_state.tokenizer = load(
                path_or_hf_repo=MODELS_PATH + model_name, 
                lazy=True
            )
            st.session_state.processor = None
        
        st.session_state.current_model = model_name
        st.session_state.is_vlm = is_vlm
        
        # Initialize HMI if enabled (only for text models currently)
        if st.session_state.hmi_enabled and not is_vlm:
            st.session_state.hmi = HotModelInference(MODELS_PATH + model_name)
            st.info("🚀 Hot Model Inference system initialized!")
        
        st.success(f"Model {model_name} loaded!")

def generate_content(prompt, max_tokens=None, image=None):
    """Generate content from the model"""
    model = st.session_state.model
    
    # If max_tokens is not provided, use the value from session state
    if max_tokens is None:
        max_tokens = st.session_state.max_tokens
    
    # Build sampler parameters by combining base params with enabled optional params
    sampler_params = st.session_state.base_params.copy()
    
    # Add optional parameters if enabled
    for param_name, param_data in st.session_state.optional_params.items():
        if param_data["enabled"]:
            sampler_params[param_name] = param_data["value"]
    
    # Check if this is a VLM (with or without image)
    if st.session_state.is_vlm and VLM_AVAILABLE:
        # Use VLM generation
        processor = st.session_state.processor
        
        # Handle both list (messages) and string (formatted prompt) inputs
        if isinstance(prompt, list):
            # For message list, we can pass it directly to apply_chat_template
            messages = prompt
            # Find the last user message with image
            last_user_image = None
            for msg in reversed(messages):
                if msg["role"] == "user" and "image" in msg:
                    last_user_image = msg["image"]
                    break
        else:
            # For formatted prompt, create a simple message structure
            messages = [{"role": "user", "content": prompt}]
            last_user_image = image
        
        if not messages:
            yield type('Token', (), {'text': "Error: No messages found"})()
            return
        
        # Handle image if present
        temp_image_path = None
        if last_user_image is not None:
            if isinstance(last_user_image, Image.Image):
                pil_image = last_user_image
            else:
                pil_image = Image.open(io.BytesIO(last_user_image))
            
            # mlx_vlm expects image path, not PIL Image, so save temporarily
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                pil_image.save(tmp_file.name)
                temp_image_path = tmp_file.name
        
        try:
            # Apply chat template for VLM which adds the appropriate image tokens
            model_config = model.config
            
            # Format the prompt with the VLM chat template
            # VLM apply_chat_template can handle chat format (like in the example)
            formatted_prompt = apply_chat_template(
                processor, 
                model_config,
                messages,  # Pass the full message structure
                num_images=1 if temp_image_path else 0
            )
            
            # Generate with VLM using stream_generate
            # mlx_vlm stream_generate expects image as a list of paths, not a single path
            image_list = [temp_image_path] if temp_image_path else []
            
            response = vlm_stream_generate(
                model, 
                processor,
                formatted_prompt,
                image_list,  # Pass image as list
                max_tokens=max_tokens,
                temperature=sampler_params.get('temp', 0.7),
                top_p=sampler_params.get('top_p', 0.95),
            )
            
            # mlx_vlm stream_generate yields GenerationResult objects
            # We need to yield the incremental text (last_segment) not the full text
            for result in response:
                # GenerationResult has a 'last_segment' attribute with incremental text
                if hasattr(result, 'last_segment'):
                    # Yield the incremental text segment
                    yield type('Token', (), {'text': result.last_segment})()
                elif hasattr(result, 'text'):
                    # Fallback: if no last_segment, this might be first iteration
                    # Yield the full text on first iteration only
                    yield type('Token', (), {'text': result.text})()
                else:
                    # Final fallback
                    yield type('Token', (), {'text': str(result)})()
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"VLM generation error: {error_details}")
            yield type('Token', (), {'text': f"\nError: {str(e)}\n"})()
        finally:
            # Clean up temporary file
            import os
            if 'temp_image_path' in locals() and temp_image_path is not None and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
    else:
        # Standard text generation
        tokenizer = st.session_state.tokenizer
        
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
    """Create styled message containers with edit/delete functionality"""
    # Add session state for editing
    if "editing_message_idx" not in st.session_state:
        st.session_state.editing_message_idx = None
    
    for idx, message in enumerate(st.session_state.messages):
        if message["role"] == "system":
            continue  # Skip system messages
            
        # Create columns for message and controls
        col1, col2 = st.columns([10, 1])
        
        with col1:
            if message["role"] == "user":
                with st.chat_message("user"):
                    # Check if this message is being edited
                    if st.session_state.editing_message_idx == idx:
                        # Show text area for editing
                        edited_content = st.text_area(
                            "Edit message:",
                            value=message["content"],
                            key=f"edit_{idx}"
                        )
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("Save", key=f"save_{idx}"):
                                st.session_state.messages[idx]["content"] = edited_content
                                st.session_state.editing_message_idx = None
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_{idx}"):
                                st.session_state.editing_message_idx = None
                                st.rerun()
                    else:
                        # Check if message contains image data
                        if "image" in message:
                            st.image(message["image"], caption="Uploaded image", width=300)
                        st.markdown(message["content"])
            else:  # assistant message
                with st.chat_message("assistant"):
                    if st.session_state.editing_message_idx == idx:
                        # Show text area for editing
                        edited_content = st.text_area(
                            "Edit message:",
                            value=message["content"],
                            key=f"edit_{idx}"
                        )
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("Save", key=f"save_{idx}"):
                                st.session_state.messages[idx]["content"] = edited_content
                                st.session_state.editing_message_idx = None
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_{idx}"):
                                st.session_state.editing_message_idx = None
                                st.rerun()
                    else:
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
        
        with col2:
            # Show edit/delete buttons only when not editing
            if st.session_state.editing_message_idx != idx and not st.session_state.get('is_generating', False):
                if st.button("✏️", key=f"edit_btn_{idx}", help="Edit message"):
                    st.session_state.editing_message_idx = idx
                    st.rerun()
                if st.button("🗑️", key=f"delete_btn_{idx}", help="Delete message"):
                    st.session_state.messages.pop(idx)
                    st.rerun()

def main():
        # Sidebar for model selection and parameter configuration
    with st.sidebar:
        st.title("MLX Chat Interface")
        
        # VLM support status
        if VLM_AVAILABLE:
            st.success("✅ Vision-Language Model support available")
        else:
            st.info("ℹ️ Vision models not available (install mlx-vlm)")
        
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
        
        def create_display_name(info):
            """Create a display name showing the model details"""
            display_parts = [f"{info['parameters']}"]
            
            # Add vision model indicator
            if info.get('model_type') == 'vlm':
                display_parts.append("🖼️ VLM")
            
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
                display_name = create_display_name(info)
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
    
    # Initialize image upload expanded state and pending image
    if 'image_upload_expanded' not in st.session_state:
        st.session_state.image_upload_expanded = False
    if 'pending_image' not in st.session_state:
        st.session_state.pending_image = None
    
    # Show stop button during generation
    if st.session_state.get('is_generating', False):
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        _, col2, _ = st.columns([1, 1, 1])
        with col2:
            if st.button("🛑 Stop Generation", type="secondary", use_container_width=True, key="stop_button"):
                st.session_state.stop_generation = True
                st.rerun()
    
    # Show previous generation metrics in a small format
    if st.session_state.token_metrics["tokens_per_second"] > 0:
        st.markdown(f"<div style='text-align: right; color: gray; font-size: 0.8em; margin-top: 10px; margin-bottom: 5px;'>💨 <span style='font-weight: bold;'>{st.session_state.token_metrics['tokens_per_second']:.1f}</span> tokens/sec | <span style='font-weight: bold;'>{st.session_state.token_metrics['tokens_generated']}</span> tokens in {st.session_state.token_metrics['generation_time']:.1f}s</div>", unsafe_allow_html=True)
    
    # Chat input area - positioned here, after generation metrics and before image upload
    # Show pending image indicator above chat input if exists (only when NOT generating)
    # if (st.session_state.is_vlm and st.session_state.pending_image is not None 
    #     and not st.session_state.get('is_generating', False)):
    #     col1, col2, col3 = st.columns([1, 6, 1])
    #     with col2:
    #         st.image(st.session_state.pending_image, caption="Ready to send", width=100)
    #     with col3:
    #         if st.button("❌", help="Remove image", key="remove_pending_image"):
    #             st.session_state.pending_image = None
    #             st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Type your message here"):
        if st.session_state.model is None:
            st.error("Please load a model first!")
            return
    
        # Add user message to chat history with image if available
        user_message = {"role": "user", "content": prompt}
        if st.session_state.is_vlm and st.session_state.pending_image is not None:
            user_message["image"] = st.session_state.pending_image
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            if "image" in user_message:
                st.image(user_message["image"], caption="Attached image", width=300)
            st.markdown(prompt)
        
        # Clear pending image immediately after use
        st.session_state.pending_image = None
        st.session_state.image_upload_expanded = False  # Close image upload expander
        
        # Increment upload counter to reset file uploader if we had an image
        if "image" in user_message:
            if 'upload_counter' not in st.session_state:
                st.session_state.upload_counter = 0
            st.session_state.upload_counter += 1
            
        # JavaScript to close the image upload expander (only if we had an image)
        if "image" in user_message:
            close_image_expander_script = '''
            <script>
                // Find the image upload expander by looking for "Add Image" text
                setTimeout(function() {
                    console.log("Looking for image upload expander...");
                    
                    // Get all expanders
                    let allExpanders = parent.document.querySelectorAll("[data-testid='stExpander']");
                    console.log("Found " + allExpanders.length + " expanders");
                    
                    for (let i = 0; i < allExpanders.length; i++) {
                        let expander = allExpanders[i];
                        
                        // Look for "Add Image" text in the summary
                        let summary = expander.querySelector("summary");
                        if (summary && summary.textContent.includes("Add Image")) {
                            console.log("Found Add Image expander");
                            
                            // Find the details element and remove open attribute
                            let detailsElement = expander.querySelector("details[open]");
                            if (detailsElement) {
                                detailsElement.removeAttribute("open");
                                console.log("Image upload expander closed successfully.");
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
                components.html(close_image_expander_script, height=0)
            
            # Remove script after executing
            time.sleep(0.2)
            script_container.empty()
        
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
            
            # Check if current model is a reasoning model (works for both VLM and text models)
            current_model_info = AVAILABLE_MODELS.get(st.session_state.current_model, {})
            is_reasoning_model = current_model_info.get('reasoning', False)
            
            # Handle VLMs differently since they don't have standard chat templates
            if st.session_state.is_vlm:
                # For VLMs, we'll pass the messages directly and handle formatting in generate_content
                prompt_template = messages
            elif hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                prompt_template = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=is_reasoning_model
                )
            else:
                # No chat template available
                prompt_template = None
                
            # Initialize common variables for all model types
            in_thinking_mode = is_reasoning_model
            thinking_content = ""
            answer_content = ""
            
            # Token generation metrics
            token_counter = 0
            start_time = time.time()
            token_speed_container = st.empty()
            
            # Generate content based on model type and availability
            if prompt_template is not None:
                # Stream the response - use HMI if enabled, otherwise use standard generation
                if st.session_state.hmi_enabled and st.session_state.hmi and not st.session_state.is_vlm:
                    # Use Hot Model Inference with real-time feedback (text models only)
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
                    # Standard generation (for both VLMs and text models without HMI)
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
                    
                    # VLMs - simpler streaming approach that shows content immediately
                    if st.session_state.is_vlm:
                        # For streaming, we need a simpler approach that shows content as it arrives
                        # Clean the full response of VLM delimiters
                        clean_response = full_response
                        clean_response = clean_response.replace('<end_of_turn>', '')
                        clean_response = clean_response.replace('<start_of_turn>', '')
                        clean_response = clean_response.replace('<|endoftext|>', '')
                        clean_response = clean_response.replace('<|end|>', '')
                        clean_response = clean_response.replace('model\n', '')
                        clean_response = clean_response.replace('user\n', '')
                        
                        # Check if we have any thinking tags at all
                        if "<think>" not in clean_response and "</think>" not in clean_response:
                            # No thinking tags - just display cleaned content
                            message_placeholder.markdown(clean_response.strip())
                        else:
                            # We have thinking tags - do simple replacement for streaming
                            # This shows content immediately but may not be perfectly formatted until complete
                            display_parts = []
                            current_text = clean_response
                            
                            # Split by <think> tags and process each part
                            parts = current_text.split("<think>")
                            
                            # First part is always regular content (before any thinking)
                            if parts[0].strip():
                                display_parts.append(parts[0].strip())
                            
                            # Process remaining parts that start with thinking content
                            for i in range(1, len(parts)):
                                part = parts[i]
                                if "</think>" in part:
                                    # This part has a complete thinking block
                                    think_end = part.find("</think>")
                                    thinking_content = part[:think_end].strip()
                                    remaining_content = part[think_end + 8:].strip()
                                    
                                    if thinking_content:
                                        display_parts.append(f"""<div class="thinking-box">
<div class="thinking-header">🧠 VLM Thinking...</div>
{thinking_content}
</div>""")
                                    
                                    if remaining_content:
                                        display_parts.append(remaining_content)
                                else:
                                    # Incomplete thinking block - show what we have so far
                                    if part.strip():
                                        display_parts.append(f"""<div class="thinking-box">
<div class="thinking-header">🧠 VLM Thinking...</div>
{part.strip()}
</div>""")
                            
                            # Join all parts and display
                            display_text = "\n\n".join(display_parts)
                            message_placeholder.markdown(display_text, unsafe_allow_html=True)
                    # Non-reasoning text models also get immediate display
                    elif not is_reasoning_model:
                        message_placeholder.markdown(full_response)
                    # Reasoning models need thinking tag processing
                    elif is_reasoning_model:
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
                            # Try to detect and beautify JSON in answer content
                            try:
                                trimmed = answer_content.strip()
                                if trimmed and (trimmed.startswith('{') or trimmed.startswith('[')):
                                    json_obj = json.loads(trimmed)
                                    beautified_json = json.dumps(json_obj, indent=2, ensure_ascii=False)
                                    display_text += f"\n```json\n{beautified_json}\n```"
                                else:
                                    display_text += answer_content
                            except (json.JSONDecodeError, ValueError):
                                display_text += answer_content
                    else:
                        # For non-reasoning models, show everything as answer
                        clean_response = re.sub(r"<think>|</think>", "", answer_content)
                        
                        # Try to detect and beautify JSON
                        try:
                            # First strip whitespace and check if it looks like JSON
                            trimmed = clean_response.strip()
                            
                            # Look for JSON patterns anywhere in the response
                            json_match = None
                            if '{' in trimmed or '[' in trimmed:
                                # Try to find the start of JSON
                                for i, char in enumerate(trimmed):
                                    if char in '{[':
                                        # Try to parse from this position
                                        potential_json = trimmed[i:]
                                        try:
                                            json.loads(potential_json)
                                            json_match = potential_json
                                            break
                                        except:
                                            continue
                            
                            if json_match:
                                trimmed = json_match
                            elif trimmed and (trimmed.startswith('{') or trimmed.startswith('[')):
                                # Clean up common JSON issues
                                # Remove quotes around the JSON if present
                                if trimmed.startswith('"') and trimmed.endswith('"'):
                                    trimmed = trimmed[1:-1]
                                    # Unescape any escaped quotes
                                    trimmed = trimmed.replace('\\"', '"')
                            else:
                                # No JSON-like content found
                                raise ValueError("No JSON content detected")
                            
                            # Try to fix common malformed JSON patterns
                            # Count braces to see if there's an imbalance
                            open_braces = trimmed.count('{')
                            close_braces = trimmed.count('}')
                            if close_braces > open_braces:
                                # Remove extra closing braces from the end
                                trimmed = trimmed.rstrip('}')
                                trimmed += '}' * open_braces
                            
                            # Try to parse as JSON
                            json_obj = json.loads(trimmed)
                            # If successful, beautify it
                            beautified_json = json.dumps(json_obj, indent=2, ensure_ascii=False)
                            display_text = f"```json\n{beautified_json}\n```"
                        except (json.JSONDecodeError, ValueError) as e:
                            # Not valid JSON, display as is
                            # Add debug info in console
                            print(f"JSON parse error: {e}")
                            print(f"Attempted to parse: {trimmed[:100]}...")
                            display_text = clean_response
                    
                    message_placeholder.markdown(display_text, unsafe_allow_html=True)
                    time.sleep(0.001)  # Small delay to reduce CPU usage
                
                # Special handling for end of response
                # If we never encountered a </think> tag for reasoning models, or if not a reasoning model
                current_model_info = AVAILABLE_MODELS.get(st.session_state.current_model, {})
                is_reasoning_model = current_model_info.get('reasoning', False)
                
                # First handle VLMs specifically
                if st.session_state.is_vlm:
                    # Use the same interleaved thinking logic for final display
                    display_text = ""
                    remaining_text = full_response
                    
                    # Process all thinking blocks in the response
                    while "<think>" in remaining_text or "</think>" in remaining_text:
                        # Find the next <think> tag
                        think_start = remaining_text.find("<think>")
                        think_end = remaining_text.find("</think>")
                        
                        if think_start == -1 and think_end == -1:
                            # No more thinking tags
                            break
                        elif think_start != -1 and (think_end == -1 or think_start < think_end):
                            # We have a <think> tag (start of thinking block)
                            # Add any content before the thinking block
                            before_think = remaining_text[:think_start].strip()
                            if before_think:
                                # Clean VLM delimiters from regular content
                                clean_before = before_think
                                clean_before = clean_before.replace('<end_of_turn>', '')
                                clean_before = clean_before.replace('<start_of_turn>', '')
                                clean_before = clean_before.replace('<|endoftext|>', '')
                                clean_before = clean_before.replace('<|end|>', '')
                                clean_before = clean_before.replace('model\n', '')
                                clean_before = clean_before.replace('user\n', '')
                                clean_before = clean_before.strip()
                                if clean_before:
                                    display_text += clean_before + "\n\n"
                            
                            # Now find the end of this thinking block
                            remaining_text = remaining_text[think_start + 7:]  # Skip past <think>
                            next_end = remaining_text.find("</think>")
                            
                            if next_end != -1:
                                # Complete thinking block
                                thinking_content = remaining_text[:next_end].strip()
                                if thinking_content:
                                    display_text += f"""<div class="thinking-box">
                                        <div class="thinking-header">🧠 VLM Thinking...</div>
                                        {thinking_content}
                                    </div>
                                    """
                                remaining_text = remaining_text[next_end + 8:]  # Skip past </think>
                            else:
                                # Incomplete thinking block - rest of content is thinking
                                thinking_content = remaining_text.strip()
                                if thinking_content:
                                    display_text += f"""<div class="thinking-box">
                                        <div class="thinking-header">🧠 VLM Thinking...</div>
                                        {thinking_content}
                                    </div>
                                    """
                                remaining_text = ""
                                break
                        else:
                            # We have a </think> without a preceding <think> - treat as regular content
                            before_end = remaining_text[:think_end].strip()
                            if before_end:
                                display_text += before_end + "\n\n"
                            remaining_text = remaining_text[think_end + 8:]
                    
                    # Add any remaining content after all thinking blocks
                    if remaining_text:
                        clean_remaining = remaining_text
                        # Remove common VLM conversation delimiters
                        clean_remaining = clean_remaining.replace('<end_of_turn>', '')
                        clean_remaining = clean_remaining.replace('<start_of_turn>', '')
                        clean_remaining = clean_remaining.replace('<|endoftext|>', '')
                        clean_remaining = clean_remaining.replace('<|end|>', '')
                        clean_remaining = clean_remaining.replace('model\n', '')
                        clean_remaining = clean_remaining.replace('user\n', '')
                        clean_remaining = clean_remaining.strip()
                        if clean_remaining:
                            display_text += clean_remaining
                    
                    # Final display update
                    message_placeholder.markdown(display_text, unsafe_allow_html=True)
                elif is_reasoning_model and "</think>" not in full_response:
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
                    
                    # Try to detect and beautify JSON
                    try:
                        # First strip whitespace
                        trimmed = clean_response.strip()
                        
                        # Look for JSON patterns anywhere in the response
                        json_match = None
                        if '{' in trimmed or '[' in trimmed:
                            # Try to find the start of JSON
                            for i, char in enumerate(trimmed):
                                if char in '{[':
                                    # Try to parse from this position
                                    potential_json = trimmed[i:]
                                    try:
                                        json.loads(potential_json)
                                        json_match = potential_json
                                        break
                                    except:
                                        continue
                        
                        if json_match:
                            trimmed = json_match
                        elif trimmed and (trimmed.startswith('{') or trimmed.startswith('[')):
                            # Clean up common JSON issues
                            # Remove quotes around the JSON if present
                            if trimmed.startswith('"') and trimmed.endswith('"'):
                                trimmed = trimmed[1:-1]
                                # Unescape any escaped quotes
                                trimmed = trimmed.replace('\\"', '"')
                        else:
                            # No JSON-like content found
                            raise ValueError("No JSON content detected")
                        
                        # Try to fix common malformed JSON patterns
                        # Count braces to see if there's an imbalance
                        open_braces = trimmed.count('{')
                        close_braces = trimmed.count('}')
                        if close_braces > open_braces:
                            # Remove extra closing braces from the end
                            trimmed = trimmed.rstrip('}')
                            trimmed += '}' * open_braces
                        
                        # Try to parse as JSON
                        json_obj = json.loads(trimmed)
                        # If successful, beautify it
                        beautified_json = json.dumps(json_obj, indent=2, ensure_ascii=False)
                        display_text = f"```json\n{beautified_json}\n```"
                    except (json.JSONDecodeError, ValueError) as e:
                        # Not valid JSON, display as is
                        print(f"Final JSON parse error: {e}")
                        print(f"Attempted to parse: {clean_response[:100]}...")
                        display_text = clean_response
                        
                    message_placeholder.markdown(display_text, unsafe_allow_html=True)
                
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
                    
                    # Display the main content first
                    message_placeholder.markdown(display_text, unsafe_allow_html=True)
                    
                    # Then display metrics in a separate markdown call
                    # This avoids mixing code blocks with HTML
                    st.markdown(metrics_html, unsafe_allow_html=True)
                
                # Add assistant response to chat history
                # For VLMs, clean up the response before saving
                if st.session_state.is_vlm:
                    # Remove common VLM conversation delimiters
                    clean_full_response = full_response
                    clean_full_response = clean_full_response.replace('<end_of_turn>', '')
                    clean_full_response = clean_full_response.replace('<start_of_turn>', '')
                    clean_full_response = clean_full_response.replace('<|endoftext|>', '')
                    clean_full_response = clean_full_response.replace('<|end|>', '')
                    clean_full_response = clean_full_response.replace('model\n', '')
                    clean_full_response = clean_full_response.replace('user\n', '')
                    clean_full_response = clean_full_response.strip()
                    st.session_state.messages.append({"role": "assistant", "content": clean_full_response})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Reset generation state
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                
                # Force a rerun to ensure edit/delete buttons are available
                st.rerun()
            else:
                # Handle models without chat templates
                if st.session_state.is_vlm:
                    # VLMs can still work without standard chat templates
                    prompt_template = st.session_state.messages
                    token_stream = generate_content(prompt_template)
                    
                    # Process the token stream
                    for token in token_stream:
                        if st.session_state.stop_generation:
                            break
                        
                        token_text = token.text if hasattr(token, 'text') else token
                        full_response += token_text
                        
                        # Update display
                        message_placeholder.markdown(full_response)
                        time.sleep(0.001)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("Chat template not available for this model.")
                
                # Reset generation state
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                
                # Force a rerun to ensure edit/delete buttons are available
                st.rerun()
    
    # Image upload interface for VLM models - only show when NOT generating
    if st.session_state.is_vlm and not st.session_state.get('is_generating', False):
        # Add a button to toggle the image upload area
        if not st.session_state.image_upload_expanded and st.session_state.pending_image is None:
            if st.button("📷 Add Image", key="open_image_upload", use_container_width=True):
                st.session_state.image_upload_expanded = True
                st.rerun()
        
        # Show image upload expander (but collapse it if we have a pending image ready)
        if st.session_state.image_upload_expanded or st.session_state.pending_image is not None:
            # If we have a pending image, keep the expander collapsed
            expanded_state = st.session_state.image_upload_expanded and st.session_state.pending_image is None
            with st.expander("📷 Add Image", expanded=expanded_state):
                # Create a unique key for the file uploader that changes when image is cleared
                uploader_key = f"image_uploader_{st.session_state.get('upload_counter', 0)}"
                
                uploaded_file = st.file_uploader(
                    "Choose an image", 
                    type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                    key=uploader_key,
                    help="Upload an image to include with your message"
                )
                
                if uploaded_file is not None and st.session_state.pending_image is None:
                    # Process the uploaded image
                    # Read and display the image
                    image = Image.open(uploaded_file)
                    st.session_state.pending_image = image
                
                # Show the pending image if it exists
                if st.session_state.pending_image is not None:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.image(st.session_state.pending_image, caption="Image ready to send", width=300)
                    with col2:
                        if st.button("Remove", key="remove_upload"):
                            st.session_state.pending_image = None
                            # Increment upload counter to reset file uploader
                            if 'upload_counter' not in st.session_state:
                                st.session_state.upload_counter = 0
                            st.session_state.upload_counter += 1
                            st.rerun()
    
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
