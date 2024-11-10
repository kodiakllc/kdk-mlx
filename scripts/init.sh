#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../"

python3 -m venv ${PROJECT_ROOT}/mlx_env
source ${PROJECT_ROOT}/mlx_env/bin/activate
pip install -r ${PROJECT_ROOT}/requirements.txt
