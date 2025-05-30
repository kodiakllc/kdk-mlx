# KDK MLX

A collection of optimized MLX implementations for running large language models locally on Apple Silicon.

## Overview

This repository provides multiple interfaces for interacting with quantized language models using Apple's MLX framework. It includes command-line chat interfaces, Streamlit web UI, and proxy forwarding capabilities for various model families.

## Supported Models

### DeepSeek R1 Family
- `DeepSeek-R1-Distill-Llama-8B-4bit` - 8B parameter Llama-based model
- `DeepSeek-R1-Distill-Qwen-14B-4bit` - 14B parameter Qwen-based model  
- `DeepSeek-R1-Distill-Qwen-32B-4bit` - 32B parameter Qwen-based model
- `DeepSeek-R1-0528-Qwen3-8B-4bit` - Latest 8B Qwen3-based model
- `DeepSeek-R1-0528-Qwen3-8B-4bit-DWQ` - DWQ quantized variant
- `DeepSeek-R1-0528-Qwen3-8B-8bit` - 8-bit quantized variant

### Qwen Family
- `Qwen2.5-7B-Instruct-Uncensored-4bit` - Uncensored 7B instruction model
- `Qwen2.5.1-Coder-7B-Instruct-4bit` - Code-specialized 7B model
- `Qwen2.5-Coder-32B-Instruct-4bit` - Large code model
- `Qwen3-32B-4bit` - Latest 32B general model

### Gemma 3 Family
- `gemma-3-4b-it-4bit` - 4B instruction-tuned model
- `gemma-3-12b-it-4bit` - 12B instruction-tuned model
- `gemma-3-4b-it-qat-4bit` - QAT quantized 4B model
- `gemma-3-12b-it-qat-4bit` - QAT quantized 12B model
- `gemma-3-4b-it-4bit-DWQ` - DWQ quantized 4B model
- `gemma-3-12b-it-4bit-DWQ` - DWQ quantized 12B model

### Llama Family
- `meta-llama-Llama-4-Scout-17B-16E-4bit` - Llama 4 Scout 17B model

## Installation

1. **Initialize the environment:**
   ```bash
   ./scripts/init.sh
   ```

2. **Download models:**
   ```bash
   ./scripts/download.models.sh
   ```

## Usage

### Quick Start
Run the interactive menu to select and launch any Python interface:
```bash
./run.sh
```

### Available Interfaces

#### Command Line Chat Interfaces
- **DeepSeek R1 Llama 8B:** `python py/DeepSeekR1.Llama.8b.py`
- **DeepSeek R1 Qwen 32B:** `python py/DeepSeekR1.Qwen.32b.py`
- **Qwen3 32B:** `python py/Qwen3.32B.py`
- **Qwen 7B:** `python py/llama.7b.py`
- **Qwen JB:** `python py/llama.jb.py`
- **Llama 4 Scout:** `python py/llama4.Scout.17B.py`

#### Web Interface
**Streamlit Chat UI:**
```bash
streamlit run py/streamlit_chat.py
```
Features:
- Model selection dropdown
- Real-time token generation metrics
- Thinking process visualization
- Advanced sampling parameter controls
- Conversation management

#### Proxy Forwarder
**HTTP Proxy Server:**
```bash
python py/proxy_forwarder.py
```
Forwards requests through `localhost:8080` on port `9000`.

## Features

### Advanced Chat Capabilities
- **Thinking Process Display**: Models with `<think>...</think>` tags show reasoning process
- **Colored Terminal Output**: Enhanced readability with ANSI color codes
- **Streaming Generation**: Real-time token-by-token response generation
- **Conversation Memory**: Maintains chat history across interactions

### Model Management
- **Lazy Loading**: Models load only when needed
- **Memory Optimization**: Efficient model switching and garbage collection
- **Quantization Support**: 4-bit, 8-bit, QAT, and DWQ variants

### Sampling Controls
- **Temperature**: Randomness control (0.0-2.0)
- **Top-p**: Nucleus sampling
- **Top-k**: Token limit filtering
- **Min-p**: Minimum probability threshold
- **Repetition Penalty**: Reduces repetitive output

## Utility Scripts

### Model File Management
Split large safetensors files for Git LFS:
```bash
./scripts/safetensors.lfs.sh split [MODEL_NAME]
```

Recombine split files:
```bash
./scripts/safetensors.lfs.sh combine [MODEL_NAME]
```

### Development Tools
**Precodex Setup** (for development environment):
```bash
./scripts/precodex.sh
```

## Requirements

- **Python 3.10+**
- **Apple Silicon Mac** (M1/M2/M3/M4)
- **MLX Framework**
- **macOS Monterey 12.0+**

Key dependencies:
- `mlx==0.25.1`
- `mlx-lm==0.24.0`
- `streamlit==1.45.0`
- `transformers==4.51.3`
- `torch==2.5.1`

## Performance

All models are optimized for Apple Silicon with:
- **4-bit quantization** for memory efficiency (except DeepSeek-0528-Qwen3-8B-8bit)
- **Streaming generation** for responsive interactions
- **Token-per-second metrics** displayed in real-time
- **Lazy loading** to minimize startup time

## Project Structure

```
kdk-mlx/
├── py/                     # Python implementations
│   ├── DeepSeekR1.*.py    # DeepSeek model interfaces
│   ├── Qwen*.py           # Qwen model interfaces
│   ├── llama*.py          # Llama model interfaces
│   ├── streamlit_chat.py  # Web UI
│   └── proxy_forwarder.py # HTTP proxy
├── scripts/               # Utility scripts
├── hf_models/            # Downloaded model files
├── requirements.txt      # Python dependencies
└── run.sh               # Interactive launcher
```
