#!/bin/bash

echo "Activating virtual environment ..."
source mlx_env/bin/activate

echo "Listing Python files in py/ directory:"
files=(py/*.py)
PS3="Please select a Python file to run: "

select file in "${files[@]}"; do
    if [ -n "$file" ]; then
        echo "You selected: $file 🚀"
        python3.10 "$file"
        break
    else
        echo "Invalid selection. Please try again."
    fi
done

