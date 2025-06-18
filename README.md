# KDK-MLX: Advanced MLX-based Chat Interface with Hot Model Inference

🚀 **Revolutionary Hot Model Inference (HMI) System for Apple Silicon**

A cutting-edge MLX-based chat interface featuring **Hot Model Inference (HMI)** - a React-like hot reload system for machine learning models. This system enables real-time parameter adaptation, feedback-driven learning, and resource-efficient model modifications without expensive model reloads.

## 🎯 Key Innovations

### 🔥 Hot Model Inference (HMI)
The HMI system represents a paradigm shift in model inference, bringing hot reload concepts from web development to machine learning:

- **⚡ Real-time Parameter Hot-Swapping**: Modify inference parameters without model reload
- **🧠 Dynamic LoRA Adapter System**: Lightweight model modifications using Low-Rank Adaptation  
- **📊 Feedback-Driven Learning**: Continuous improvement through user feedback loops
- **💾 Session Persistence**: Datetime-based session management with automatic saving
- **🌊 Streaming Generation**: Real-time token generation with adaptive adjustments
- **📈 Performance Monitoring**: Comprehensive metrics and insights tracking

### 🤖 Advanced Model Support
- **🧠 Reasoning Model Detection**: Automatic detection and proper handling of thinking-capable models
- **🎨 Thinking Process Visualization**: Styled display of reasoning vs answer content
- **🔄 Multi-Model Architecture**: Seamless switching between model families
- **⚙️ Dynamic Quantization**: Support for 4-bit, 8-bit, QAT, and DWQ variants

## 📊 Supported Models & Architectures

### 🔬 Reasoning Models (Thinking Capable)
| Model Family | Parameters | Quantization | Reasoning | Specialization |
|--------------|------------|--------------|-----------|----------------|
| **DeepSeek-R1 Distilled** | 8B-32B | 4-bit | ✅ | R1 Distillation |
| **DeepSeek-R1-0528** | 8B | 4-bit, 8-bit, DWQ | ✅ | Latest R1 |
| **Qwen3** | 32B | 4-bit | ✅ | Base Model |
| **Llama-4-Scout** | 17B | 4-bit | ✅ | Scout Specialized |

### 🛠️ Specialized Models
| Model Family | Parameters | Quantization | Specialization |
|--------------|------------|--------------|----------------|
| **Qwen 2.5 Uncensored** | 7B | 4-bit | Uncensored |
| **Qwen 2.5/2.5.1 Coder** | 7B, 32B | 4-bit | Code Generation |

### 🖼️ Vision-Language Models (VLMs)
| Model Family | Parameters | Quantization | Type | Description |
|--------------|------------|--------------|------|-------------|
| **Gemma 3** | 4B, 12B | 4-bit, QAT, DWQ | VLM | Google's vision-language model |
| **LLaVA** | 7B, 13B | 4-bit | VLM | Multi-modal chat |
| **Qwen2-VL** | 2B, 7B | 4-bit | VLM | Advanced vision-language |
| **Phi-3.5-vision** | 4.2B | 4-bit | VLM | Efficient multi-modal |

## 🚀 Quick Start

### Installation
```bash
# Initialize environment and download models
./scripts/init.sh
./scripts/download.models.sh

# For Vision-Language Model support (optional)
pip install mlx-vlm
```

### Launch Interface
```bash
# Interactive launcher
./run.sh

# Direct Streamlit launch
streamlit run py/streamlit_chat.py
```

## 🎛️ Hot Model Inference (HMI) Features

### Real-time Parameter Adaptation
```python
# Parameters can be hot-swapped during conversation
hmi.hot_update_params(
    temperature=0.8,        # Increase creativity
    top_p=0.9,             # Adjust nucleus sampling  
    repetition_penalty=1.2  # Reduce repetition
)
```

### Dynamic LoRA Adapters
```python
# Create adapters based on feedback patterns
adapter_name = hmi.create_feedback_adapter(recent_feedback)
hmi.lora_adapter.blend_adapters([adapter_name], [0.1])
```

### Session Management
- **📅 Automatic Session IDs**: `session_20250131_143052`
- **💾 Auto-save**: Every 10 feedback entries or 30 seconds
- **🔄 Session Restoration**: Load previous adaptations and parameters
- **📊 Performance Tracking**: Historical metrics and trends

### Multi-dimensional Feedback System
```python
feedback = FeedbackSignal(
    response_quality=0.8,   # Overall quality
    coherence=0.9,          # Logical flow
    relevance=0.7,          # Topic relevance
    creativity=0.6,         # Creative content
    factuality=0.8,         # Factual accuracy
    user_satisfaction=0.8   # User satisfaction
)
```

## 🎨 Advanced UI/UX Features

### Reasoning Model Display
- **🧠 Thinking Process Visualization**: Dracula-inspired styled thinking boxes
- **🎯 Automatic Model Detection**: Reasoning vs non-reasoning model handling based on models.json flags
- **⚡ Real-time Generation Metrics**: Live tokens/second display with fixed positioning
- **🛑 Generation Control**: Stop button during streaming
- **📄 JSON Beautification**: Automatic detection and formatting of JSON responses

### Vision-Language Model Interface
- **📷 Image Upload**: Streamlined image upload with auto-close functionality
- **🖼️ Image Preview**: Pending image indicator above chat input
- **🔄 Smart File Reset**: Dynamic key system for proper file uploader clearing
- **📍 Proper Layout**: Chat input positioned correctly between metrics and image upload

### Interactive Controls
- **🎛️ HMI Control Panel**: Session management, parameter adjustment, performance insights
- **📊 Real-time Feedback Panel**: 6-dimensional feedback scoring with auto-minimize
- **⚙️ Advanced Sampling Parameters**: Temperature, top-p, top-k, min-p controls
- **🔄 Model Quantization UI**: Built-in MLX quantization interface with VLM support
- **✏️ Message Editing**: Edit and delete functionality for chat messages
- **🗑️ Message Management**: Delete individual messages from conversation history

## 🔧 Technical Architecture

### Core Components

#### HotModelInference Class
```python
class HotModelInference:
    def __init__(self, model_path: str, session_id: Optional[str] = None):
        self.lora_adapter = DynamicLoRAAdapter()
        self.feedback_history: List[FeedbackSignal] = []
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```

#### Dynamic LoRA System
```python
# Low-Rank Adaptation Formula
output = input + (input @ LoRA_A @ LoRA_B) * (alpha/rank) * weight
```

#### Adaptive Generation
```python
def adaptive_generate(self, prompt: str, feedback_callback=None):
    \"\"\"Generate with real-time adaptation based on feedback\"\"\"
    for token in stream_generate(...):
        yield token  # Streaming output
        
        # Dynamic parameter adjustment
        if self.current_params.dynamic_temperature:
            new_temp = max(0.1, current_temp * temp_decay)
            self.hot_update_params(temperature=new_temp)
```

### Session Persistence
Sessions are automatically saved to `hf_models/hmi_sessions/` with complete state:
```python
session_data = {
    'session_id': 'session_20250131_143052',
    'current_params': {InferenceParams},
    'feedback_history': [FeedbackSignal],
    'lora_adapters': {adapter_matrices},
    'active_adapters': [adapter_names],
    'param_performance': {metrics}
}
```

## 📈 Performance & Benefits

### Memory Efficiency
- **Single Model Load**: Base model loaded once, parameters hot-swapped
- **LoRA Overhead**: Minimal memory for adapter matrices (~1MB per adapter)
- **Session Storage**: Compact pickle files for complete state persistence

### Speed Optimization  
- **No Model Reload**: Parameter changes take milliseconds vs. minutes
- **Streaming Generation**: Real-time token output with live adaptation
- **Efficient Caching**: Generation cache for performance analysis

### Adaptation Quality
- **Feedback-Driven**: Continuous improvement through user input
- **Multi-Metric Learning**: Composite scores considering quality, coherence, relevance
- **Historical Awareness**: Learning from accumulated feedback history

## 🛠️ Advanced Workflows

### HMI Session Workflow
1. **Enable HMI**: Toggle HMI checkbox in sidebar
2. **Load Model**: Select reasoning model (e.g., DeepSeek-R1)
3. **Start Conversation**: Automatic session creation
4. **Real-time Adaptation**: Parameters adjust based on quality
5. **Provide Feedback**: Use feedback panel for explicit learning
6. **Session Persistence**: Automatic saving and restoration

### Parameter Optimization Workflow  
1. **Baseline Testing**: Start with default parameters
2. **Performance Monitoring**: Track metrics and feedback scores
3. **Hot Parameter Updates**: Real-time adjustments
4. **LoRA Creation**: System creates adapters from feedback patterns
5. **Best Configuration**: System identifies optimal parameters

### Model Comparison Workflow
1. **Load Model A**: Start with reasoning model + HMI session
2. **Establish Baseline**: Gather performance data
3. **Switch Model B**: Load different model with new HMI session  
4. **Compare Sessions**: Use session management to switch
5. **Performance Analysis**: Compare metrics and adaptations

## 🎯 HMI Control Panel Features

### Session Management
- **Current Session Display**: Shows datetime ID
- **Auto-save Toggle**: Configurable automatic saving
- **Force Save Button**: Manual session saving
- **Session Listing**: Browse and load previous sessions
- **Session Metrics**: Feedback count, adapter count

### Parameter Controls
- **Real-time Sliders**: Temperature, top-p, repetition penalty
- **Dynamic Temperature**: Automatic temperature decay
- **Hot Update Button**: Immediate parameter changes
- **Performance Insights**: Real-time effectiveness tracking

### Performance Insights
- **Average Feedback Score**: Last 20 responses
- **Score Trend Analysis**: Improving/declining trends
- **Active Adapter Count**: Current LoRA adapters
- **Best Parameter Combinations**: Optimal configurations

## 🔄 MLX Quantization System

### Built-in Quantization Interface
```python
quantizer = MLXQuantizer(MODELS_PATH)
quantizer.quantize_model(
    hf_model_path="microsoft/Phi-3-mini-4k-instruct",
    output_name="phi-3-mini-4bit",
    q_bits=4,
    qat=False,
    dwq=False
)
```

### Vision-Language Model Quantization
```bash
# Using custom converter for VLMs
python py/custom_mlx_convert.py \
  --hf-path Qwen/Qwen2-VL-2B-Instruct \
  --mlx-path hf_models/Qwen2-VL-2B-Instruct-4bit \
  --quantize \
  --q-bits 4 \
  --model-type vlm

# Or use the UI with Model Type: "vlm"
```

### Quantization Options
- **Standard 4-bit**: General purpose quantization
- **8-bit**: Higher precision, larger memory usage
- **QAT (Quantization Aware Training)**: Training-optimized quantization
- **DWQ (Dynamic Weight Quantization)**: Advanced quantization method
- **Mixed Quantization**: Variable bits (2-6, 3-6, 4-6) for different layers

## 📁 Project Structure

```
kdk-mlx/
├── py/                              # Core Python modules
│   ├── streamlit_chat.py           # 🎯 Main Streamlit interface with HMI & VLM
│   ├── hot_model_inference.py      # 🚀 CORE HMI SYSTEM  
│   ├── mlx_quantizer.py           # ⚙️ MLX quantization wrapper
│   ├── custom_mlx_convert.py      # 🖼️ Custom converter with VLM support
│   ├── DeepSeekR1.*.py            # DeepSeek R1 interfaces
│   ├── Qwen*.py                   # Qwen model interfaces  
│   ├── llama*.py                  # Llama model interfaces
│   ├── proxy_forwarder.py         # HTTP proxy server
│   └── legacy/wip/                # Development archives
├── hf_models/                      # Model storage and configuration
│   ├── models.json                # 📊 Model metadata with reasoning & VLM flags
│   ├── hmi_sessions/              # 💾 HMI session storage (auto-created)
│   └── [model-directories]/       # Quantized model files
├── scripts/                       # Utility scripts
│   ├── download.models.sh         # Model downloading
│   ├── init.sh                   # Environment setup
│   ├── safetensors.lfs.sh        # LFS management
│   └── precodex.sh               # Development environment
├── proxyman/                      # Proxyman session files
├── CLAUDE.md                      # 📖 Comprehensive technical documentation
├── README.md                      # 📋 This file
├── requirements.txt               # Python dependencies
└── run.sh                        # 🚀 Interactive launcher
```

## 🔧 Requirements

### System Requirements
- **Apple Silicon Mac** (M1/M2/M3/M4/M4 Pro/Max/Ultra)
- **macOS Monterey 12.0+** (Recommended: macOS Sonoma 14.0+)
- **Python 3.10+**
- **Memory**: 16GB+ RAM (32GB+ recommended for 32B models)
- **Storage**: 50GB+ for model collection

### Key Dependencies
```
mlx==0.25.1                 # Apple MLX framework
mlx-lm==0.24.0             # MLX language models
mlx-vlm                    # MLX vision-language models (optional)
streamlit==1.45.0          # Web interface
transformers==4.51.3       # Hugging Face transformers
torch==2.5.1               # PyTorch (for compatibility)
numpy>=1.24.0              # Numerical computations
```

## 🚀 Usage Examples

### Basic Chat with HMI
```bash
# Launch Streamlit interface
streamlit run py/streamlit_chat.py

# Enable HMI in sidebar
# Load a reasoning model (DeepSeek-R1, Qwen3, Llama-4-Scout)
# Start conversation with automatic adaptation
```

### Vision-Language Model Usage
```bash
# Quantize a VLM first
python py/custom_mlx_convert.py \
  --hf-path llava-hf/llava-v1.6-mistral-7b-hf \
  --mlx-path hf_models/llava-v1.6-mistral-7b-hf-4bit \
  --quantize --q-bits 4 --model-type vlm

# Add to models.json with "model_type": "vlm"
# Launch interface and select the VLM (marked with 🖼️)

# Using the VLM:
# 1. Load a VLM model (e.g., Gemma 3, LLaVA)
# 2. Click "📷 Add Image" button below chat input
# 3. Upload your image
# 4. Type your message and send
# 5. Image upload area auto-closes after sending
```

### Command Line Interfaces
```bash
# DeepSeek R1 models
python py/DeepSeekR1.Llama.8b.py      # 8B Llama-based
python py/DeepSeekR1.Qwen.32b.py      # 32B Qwen-based

# Qwen models  
python py/Qwen3.32B.py                # Qwen3 32B
python py/llama.7b.py                 # Qwen 7B
python py/llama.jb.py                 # Qwen JB variant

# Llama models
python py/llama4.Scout.17B.py         # Llama-4-Scout 17B
```

### Proxy Server
```bash
# HTTP proxy for API integration
python py/proxy_forwarder.py
# Forwards requests through localhost:8080 on port 9000
```

## 🛠️ Utility Scripts

### Model Management
```bash
# Download all configured models  
./scripts/download.models.sh

# Initialize environment
./scripts/init.sh

# Split large safetensors for Git LFS
./scripts/safetensors.lfs.sh split [MODEL_NAME]

# Recombine split files
./scripts/safetensors.lfs.sh combine [MODEL_NAME]

# Development environment setup
./scripts/precodex.sh
```

## 🖼️ Vision-Language Model Configuration

### Adding VLMs to models.json
```json
{
  "llava-v1.6-mistral-7b-hf-4bit": {
    "parameters": "7B",
    "quantization": "4bit",
    "quantization_options": ["standard"],
    "base_family": "LLaVA",
    "version": "1.6",
    "type": "vlm",
    "model_type": "vlm",  // Required for VLM support
    "description": "Vision-Language Model"
  }
}
```

### VLM Features in UI
- **🖼️ Icon**: VLMs are marked with an image icon in model selector
- **Auto-detection**: System automatically loads with mlx-vlm when available
- **Image Support**: Can process both text and image inputs
- **No HMI**: Currently HMI is disabled for VLMs (text-only feature)
- **📷 Streamlined Upload**: Image upload button appears below chat input
- **🔄 Auto-close**: Upload interface closes automatically after sending
- **🎯 Smart State Management**: Pending images properly cleared after use
- **📍 Optimized Layout**: Natural flow with chat input → image upload → feedback

## 🎯 Advanced Features

### Multi-Adapter Blending
```python
# Blend multiple LoRA adapters simultaneously
hmi.lora_adapter.blend_adapters(
    adapter_names=['feedback_adapter_123', 'user_preference_456'],
    weights=[0.7, 0.3]
)
```

### Dynamic Temperature Decay
```python
# Automatic temperature adjustment during generation
if hmi.current_params.dynamic_temperature:
    new_temp = max(0.1, current_temp * temp_decay)
    hmi.hot_update_params(temperature=new_temp)
```

### Real-time Feedback Integration
```python
def real_time_feedback_callback(partial_response):
    if quality_detected_poor:
        return FeedbackSignal(low_scores)  # Triggers adaptation
    return None
```

## 📊 Performance Metrics

The HMI system provides comprehensive tracking:
- **Generation Speed**: Real-time tokens/second monitoring
- **Adaptation Effectiveness**: Feedback score trends over time  
- **Parameter Optimization**: Best-performing configuration identification
- **Session Continuity**: Long-term learning persistence across restarts
- **Memory Efficiency**: LoRA adapter overhead monitoring
- **Model Reasoning Capabilities**: Automatic detection and handling

## 🆕 Recent Updates

### UI/UX Improvements
- **✅ Fixed Reasoning Model Detection**: Automatic detection based on `reasoning` flag in models.json
- **✅ Enhanced Image Handling**: Proper PIL Image type handling for VLMs
- **✅ Layout Optimization**: Chat input properly positioned at bottom with natural flow
- **✅ State Management**: Fixed pending image persistence issues
- **✅ JavaScript Integration**: Auto-close functionality for upload expanders
- **✅ Error Handling**: Robust SVD error handling in performance insights

### Technical Enhancements
- **🔧 Dynamic Layer Shape Extraction**: MLX model introspection for LoRA adapters
- **🔧 Fixed Indentation Issues**: Proper code block structure in generation logic
- **🔧 Improved Type Checking**: Better handling of image types (PIL vs bytes)
- **🔧 Session State Cleanup**: Proper initialization and clearing of states

## 🔮 Future Enhancements

### Planned Features
- **Multi-Model Sessions**: Compare multiple models within single session
- **Advanced Adapter Types**: Specialized adapters for coding, reasoning, creativity
- **Distributed Training**: Multi-user feedback aggregation
- **Model-Specific Optimization**: Custom adaptation strategies per architecture
- **Advanced Analytics**: Detailed performance insights and optimization suggestions
- **VLM HMI Support**: Extend Hot Model Inference to vision-language models

## 🤝 Contributing

This repository represents advanced research in interactive machine learning and model adaptation. The Hot Model Inference system brings hot reload development practices to AI inference, creating a foundation for continuous, user-driven model improvement.

---

<div align="center">

## 📝 License
[License](LICENSE.md)

</div>

---

**KDK-MLX** - Developed by Kodiak LLC - Pioneering the future of interactive machine learning with Hot Model Inference on Apple Silicon 🚀
