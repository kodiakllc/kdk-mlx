# Custom MLX converter with modified get_model_path usage
# Compatible with both mlx_lm and mlx_vlm models

import argparse
import glob
import shutil
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Try to import from both mlx_lm and mlx_vlm
try:
    from mlx_lm.utils import (
        create_model_card,
        dequantize_model,
        fetch_from_hub,
        get_model_path,
        quantize_model,
        save_config,
        save_weights,
        upload_to_hub,
    )
    MLX_LM_AVAILABLE = True
except ImportError:
    MLX_LM_AVAILABLE = False

try:
    from mlx_vlm.utils import (
        create_model_card as vlm_create_model_card,
        dequantize_model as vlm_dequantize_model,
        fetch_from_hub as vlm_fetch_from_hub,
        get_model_path as vlm_get_model_path,
        quantize_model as vlm_quantize_model,
        save_config as vlm_save_config,
        save_weights as vlm_save_weights,
        upload_to_hub as vlm_upload_to_hub,
    )
    MLX_VLM_AVAILABLE = True
except ImportError:
    MLX_VLM_AVAILABLE = False

# Check that at least one is available
if not MLX_LM_AVAILABLE and not MLX_VLM_AVAILABLE:
    raise ImportError("Neither mlx_lm nor mlx_vlm is installed. Please install at least one.")


def custom_get_model_path(hf_path: str, revision: Optional[str] = None, model_type: str = "auto") -> Path:
    """
    Custom model path handler that can work with local paths directly
    without trying to download from Hugging Face.
    Supports both text and vision models.
    """
    # If it's already a local path, just return it
    if Path(hf_path).exists():
        return Path(hf_path)
    
    # Otherwise fall back to the appropriate get_model_path
    if model_type == "vlm" and MLX_VLM_AVAILABLE:
        return vlm_get_model_path(hf_path, revision=revision)
    elif model_type == "lm" and MLX_LM_AVAILABLE:
        return get_model_path(hf_path, revision=revision)
    else:
        # Auto-detect based on what's available
        if MLX_LM_AVAILABLE:
            return get_model_path(hf_path, revision=revision)
        elif MLX_VLM_AVAILABLE:
            return vlm_get_model_path(hf_path, revision=revision)
        else:
            raise RuntimeError("No suitable get_model_path function available")


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module
) -> Callable[[str, nn.Module, dict], Union[bool, dict]]:

    if recipe == "mixed_2_6":
        low_bits = 2
    elif recipe == "mixed_3_6":
        low_bits = 3
    elif recipe == "mixed_4_6":
        low_bits = 4
    else:
        raise ValueError("Invalid quant recipe {recipe}")
    high_bits = 6
    group_size = 64

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
        config: dict,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """

        if not hasattr(module, "to_quantized"):
            return False

        index = (
            int(path.split(".")[layer_location])
            if len(path.split(".")) > layer_location
            else 0
        )
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
        if "v_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}
        if "down_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}
        if "lm_head" in path:
            return {"group_size": group_size, "bits": high_bits}

        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate


QUANT_RECIPES = ["mixed_2_6", "mixed_3_6", "mixed_4_6"]

MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]


def custom_convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: Optional[str] = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[
        Union[Callable[[str, nn.Module, dict], Union[bool, dict]], str]
    ] = None,
    use_custom_path: bool = True,  # New parameter to control path handling
    model_type: str = "auto",  # New parameter: "lm", "vlm", or "auto"
):
    # Check the save path is empty
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists."
            " Please delete the file/directory or specify a new path to save to."
        )

    print("[INFO] Loading")
    
    # Determine which functions to use
    use_vlm = False
    if model_type == "vlm":
        use_vlm = True
        if not MLX_VLM_AVAILABLE:
            raise RuntimeError("mlx_vlm is not installed but model_type='vlm' was specified")
    elif model_type == "lm":
        use_vlm = False
        if not MLX_LM_AVAILABLE:
            raise RuntimeError("mlx_lm is not installed but model_type='lm' was specified")
    else:  # auto
        # Try to detect based on model config or default to what's available
        use_vlm = MLX_VLM_AVAILABLE and not MLX_LM_AVAILABLE
    
    # Select appropriate functions
    if use_vlm:
        _fetch_from_hub = vlm_fetch_from_hub
        _get_model_path = vlm_get_model_path
        _quantize_model = vlm_quantize_model
        _dequantize_model = vlm_dequantize_model
        _save_weights = vlm_save_weights
        _save_config = vlm_save_config
        _create_model_card = vlm_create_model_card
        _upload_to_hub = vlm_upload_to_hub
        print("[INFO] Using MLX-VLM functions")
    else:
        _fetch_from_hub = fetch_from_hub
        _get_model_path = get_model_path
        _quantize_model = quantize_model
        _dequantize_model = dequantize_model
        _save_weights = save_weights
        _save_config = save_config
        _create_model_card = create_model_card
        _upload_to_hub = upload_to_hub
        print("[INFO] Using MLX-LM functions")
    
    # Use custom or standard model path handling
    if use_custom_path:
        model_path = custom_get_model_path(hf_path, revision=revision, model_type=model_type)
        print(f"[INFO] Using custom model path: {model_path}")
    else:
        model_path = _get_model_path(hf_path, revision=revision)
    
    model, config, tokenizer = _fetch_from_hub(model_path, lazy=True)

    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, model)

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    weights = dict(tree_flatten(model.parameters()))
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)

        if hasattr(model, "cast_predicate"):
            cast_predicate = model.cast_predicate()
        else:
            cast_predicate = lambda _: True
        weights = {
            k: v.astype(dtype) if cast_predicate(k) else v for k, v in weights.items()
        }

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = _quantize_model(
            model, config, q_group_size, q_bits, quant_predicate=quant_predicate
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = _dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    del model
    _save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    _save_config(config, config_path=mlx_path / "config.json")

    _create_model_card(mlx_path, hf_path)

    if upload_repo is not None:
        _upload_to_hub(mlx_path, upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format with custom path handling"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the non-quantized parameters. Defaults to config.json's `torch_dtype` or the current model weights dtype.",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use-standard-path",
        help="Use standard get_model_path instead of custom path handling.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--model-type",
        help="Model type: 'lm' for language models, 'vlm' for vision-language models, 'auto' to auto-detect.",
        type=str,
        choices=["lm", "vlm", "auto"],
        default="auto",
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    
    # Convert use_standard_path to use_custom_path
    args_dict = vars(args)
    use_custom_path = not args_dict.pop('use_standard_path', False)
    
    custom_convert(**args_dict, use_custom_path=use_custom_path)


if __name__ == "__main__":
    main()