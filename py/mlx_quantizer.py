import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from mlx_lm import convert
import streamlit as st

class MLXQuantizer:
    """Wrapper for MLX model quantization with Streamlit integration"""
    
    def __init__(self, models_path: str):
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
    
    def quantize_model(
        self,
        hf_model_path: str,
        output_name: str,
        q_bits: int = 4,
        q_group_size: int = 64,
        qat: bool = False,
        dwq: bool = False,
        quantize_embeddings: bool = False,
        dtype: str = "float16",
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Quantize a model using MLX
        
        Args:
            hf_model_path: HuggingFace model path or local path
            output_name: Name for the output directory
            q_bits: Quantization bits (4 or 8)
            q_group_size: Group size for quantization
            qat: Enable Quantization Aware Training
            dwq: Enable Distilled Weight Quantization
            quantize_embeddings: Also quantize embedding layers
            dtype: Output data type
            progress_callback: Function to call for progress updates
            
        Returns:
            bool: Success status
        """
        try:
            output_path = self.models_path / output_name
            
            if progress_callback:
                progress_callback("Starting quantization...")
            
            # Build quantization method string
            quantization_method = "q4" if q_bits == 4 else "q8"
            if qat:
                quantization_method += "_qat"
            elif dwq:
                quantization_method += "_dwq"
            
            # Use MLX convert function
            convert(
                hf_path=hf_model_path,
                mlx_path=str(output_path),
                quantize=True,
                q_bits=q_bits,
                q_group_size=q_group_size,
                quantize_embeddings=quantize_embeddings,
                dtype=dtype
            )
            
            if progress_callback:
                progress_callback("Quantization completed successfully!")
                
            return True
            
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error during quantization: {str(e)}")
            return False
    
    def quantize_model_cli(
        self,
        hf_model_path: str,
        output_name: str,
        q_bits: int = 4,
        q_group_size: int = 64,
        qat: bool = False,
        dwq: bool = False,
        quantize_embeddings: bool = False,
        dtype: str = "float16",
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Quantize a model using CLI (alternative method)
        """
        try:
            output_path = self.models_path / output_name
            
            cmd = [
                "python", "-m", "mlx_lm.convert",
                "--hf-path", hf_model_path,
                "--mlx-path", str(output_path),
                "--quantize",
                "--q-bits", str(q_bits),
                "--q-group-size", str(q_group_size),
                "--dtype", dtype
            ]
            
            # Add quantization method flags
            if qat:
                cmd.append("--qat")
            elif dwq:
                cmd.append("--dwq")  # Check if this is the correct flag
                
            if quantize_embeddings:
                cmd.append("--quantize-embeddings")
            
            if progress_callback:
                progress_callback("Starting CLI quantization...")
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if progress_callback:
                progress_callback("CLI quantization completed successfully!")
                
            return True
            
        except subprocess.CalledProcessError as e:
            if progress_callback:
                progress_callback(f"CLI quantization failed: {e.stderr}")
            return False
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error during CLI quantization: {str(e)}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a quantized model"""
        model_path = self.models_path / model_name
        if not model_path.exists():
            return {"exists": False}
        
        info = {"exists": True, "path": str(model_path)}
        
        # Check for config.json
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                info["config"] = config
            except:
                pass
        
        # Get file sizes
        total_size = 0
        safetensors_files = list(model_path.glob("*.safetensors*"))
        for file in safetensors_files:
            total_size += file.stat().st_size
        
        info["size_mb"] = total_size / (1024 * 1024)
        info["num_files"] = len(safetensors_files)
        
        return info
    
    def list_available_models(self) -> List[str]:
        """List all available quantized models"""
        if not self.models_path.exists():
            return []
        
        models = []
        for item in self.models_path.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                models.append(item.name)
        
        return sorted(models)
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a quantized model"""
        try:
            model_path = self.models_path / model_name
            if model_path.exists():
                shutil.rmtree(model_path)
                return True
            return False
        except Exception:
            return False

def create_streamlit_quantizer_ui(quantizer: MLXQuantizer):
    """Create Streamlit UI for model quantization"""
    
    st.subheader("🔧 Model Quantization")
    
    with st.expander("Quantize New Model", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            hf_path = st.text_input(
                "HuggingFace Model Path",
                placeholder="microsoft/DialoGPT-medium",
                help="Enter the HuggingFace model path or local path"
            )
            
            output_name = st.text_input(
                "Output Model Name",
                placeholder="DialoGPT-medium-4bit",
                help="Name for the quantized model directory"
            )
            
            q_bits = st.selectbox("Quantization Bits", [4, 8], index=0)
            q_group_size = st.selectbox("Group Size", [32, 64, 128], index=1)
        
        with col2:
            dtype = st.selectbox("Data Type", ["float16", "bfloat16"], index=0)
            
            # Quantization method - mutually exclusive
            quant_method = st.radio(
                "Quantization Method",
                ["Standard", "QAT", "DWQ"],
                help="QAT: Quantization Aware Training, DWQ: Distilled Weight Quantization"
            )
            
            quantize_embeddings = st.checkbox("Quantize Embeddings", value=False)
            use_cli = st.checkbox("Use CLI Method", value=False, 
                                help="Use command line interface instead of Python API")
        
        if st.button("Start Quantization", type="primary"):
            if not hf_path or not output_name:
                st.error("Please provide both model path and output name")
                return
            
            # Set quantization flags
            qat = quant_method == "QAT"
            dwq = quant_method == "DWQ"
            
            # Create progress components
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(message: str):
                status_text.text(message)
                if "completed" in message.lower():
                    progress_bar.progress(100)
                elif "starting" in message.lower():
                    progress_bar.progress(25)
                elif "error" in message.lower():
                    progress_bar.progress(0)
                else:
                    progress_bar.progress(50)
            
            # Perform quantization
            if use_cli:
                success = quantizer.quantize_model_cli(
                    hf_model_path=hf_path,
                    output_name=output_name,
                    q_bits=q_bits,
                    q_group_size=q_group_size,
                    qat=qat,
                    dwq=dwq,
                    quantize_embeddings=quantize_embeddings,
                    dtype=dtype,
                    progress_callback=progress_callback
                )
            else:
                success = quantizer.quantize_model(
                    hf_model_path=hf_path,
                    output_name=output_name,
                    q_bits=q_bits,
                    q_group_size=q_group_size,
                    qat=qat,
                    dwq=dwq,
                    quantize_embeddings=quantize_embeddings,
                    dtype=dtype,
                    progress_callback=progress_callback
                )
            
            if success:
                st.success(f"✅ Model '{output_name}' quantized successfully!")
                st.rerun()
            else:
                st.error("❌ Quantization failed. Check the logs above.")
    
    # Display existing models
    st.subheader("📦 Available Quantized Models")
    
    models = quantizer.list_available_models()
    if not models:
        st.info("No quantized models found. Quantize your first model above!")
        return
    
    for model in models:
        with st.expander(f"📁 {model}", expanded=False):
            info = quantizer.get_model_info(model)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**Path:** `{info['path']}`")
                st.write(f"**Size:** {info['size_mb']:.1f} MB")
                st.write(f"**Files:** {info['num_files']} safetensors files")
            
            with col2:
                if st.button(f"🗑️ Delete", key=f"delete_{model}"):
                    if quantizer.delete_model(model):
                        st.success(f"Deleted {model}")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete {model}")
            
            with col3:
                if st.button(f"📋 Copy Path", key=f"copy_{model}"):
                    st.code(info['path'])