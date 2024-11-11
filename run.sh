#!/bin/bash

echo "Activating virtual environment ..."
source mlx_env/bin/activate

echo "Running chatbot-ui.llama.py 🚀"
python3.10 py/chatbot-ui.llama.py
