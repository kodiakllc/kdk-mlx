#!/bin/bash

echo "Activating virtual environment ..."
source mlx_env/bin/activate

echo "Listing Python files in py/ directory:"
files=(py/*.py)
PS3="Please select a Python file to run: "

select file in "${files[@]}"; do
    if [ -n "$file" ]; then
        echo "You selected: $file :rocket:"
        # if file contains streamlit, run it with streamlit
        if grep -q "streamlit" "$file"; then
            echo "Running with streamlit ..."
            streamlit run "$file"
        else
            echo "Running with python ..."
            python3.11 "$file"
        fi
    else
        echo "Invalid selection. Please try again."
    fi
done
