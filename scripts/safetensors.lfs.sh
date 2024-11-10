#!/bin/bash
# Usage: ./scripts/safetensors.lfs.sh split|combine [MODEL_NAME]

ACTION=$1
MODEL_NAME=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../hf_models"

# Function to display usage
usage() {
    echo "Usage: $0 split|combine [MODEL_NAME]"
    exit 1
}

# Check if ACTION is provided
if [[ -z "$ACTION" ]]; then
    usage
fi

# Validate ACTION
if [[ "$ACTION" != "split" && "$ACTION" != "combine" ]]; then
    echo "Error: Invalid action '$ACTION'. Must be 'split' or 'combine'."
    usage
fi

# Function to split a single model
split_model() {
    local model_dir="$1"
    if [[ -d "$model_dir" ]]; then
        find "$model_dir" -type f -name "*.safetensors" | while read -r FILE; do
            echo "Splitting $FILE..."
            split -b 1024m "$FILE" "$FILE.part"
            rm "$FILE"
        done
    else
        echo "Error: Model directory '$model_dir' does not exist."
    fi
}

# Function to combine a single model
combine_model() {
    local model_dir="$1"
    if [[ -d "$model_dir" ]]; then
        find "$model_dir" -type f -name "*.safetensors.part*" | while read -r PART_FILE; do
            BASE_FILE="${PART_FILE%.part*}"
            if [[ ! -f "$BASE_FILE" ]]; then
                echo "Combining parts for $BASE_FILE..."
                cat "$BASE_FILE".part* > "$BASE_FILE"
                rm "$BASE_FILE".part*
            fi
        done
    else
        echo "Error: Model directory '$model_dir' does not exist."
    fi
}

# If MODEL_NAME is provided, perform action on that model only
if [[ -n "$MODEL_NAME" ]]; then
    MODEL_DIR="${MODELS_DIR}/$MODEL_NAME"
    if [[ "$ACTION" == "split" ]]; then
        split_model "$MODEL_DIR"
    elif [[ "$ACTION" == "combine" ]]; then
        combine_model "$MODEL_DIR"
    fi
else
    # No MODEL_NAME provided, perform action on all models
    if [[ "$ACTION" == "split" ]]; then
        find "$MODELS_DIR" -type d -mindepth 1 -maxdepth 1 | while read -r DIR; do
            split_model "$DIR"
        done
    elif [[ "$ACTION" == "combine" ]]; then
        find "$MODELS_DIR" -type d -mindepth 1 -maxdepth 1 | while read -r DIR; do
            combine_model "$DIR"
        done
    fi
fi
