#!/bin/bash

# Function to clean up and kill the Python server on exit
cleanup() {
  echo “Stopping Python server...”
  kill $PYTHON_PID
  wait $PYTHON_PID 2>/dev/null
  cd “$ORIGINAL_DIR” || exit
}

# Ensure cleanup is called on script exit
trap cleanup EXIT

# Save the directory where the script was called from
ORIGINAL_DIR=$(pwd)

# Kill any existing Python processes
killall -9 Python 2>/dev/null || true

# Activate the virtual environment
source $HOME/local/kdk-mlx/mlx_env/bin/activate

# Start the Python server in the background
python3.10 $HOME/local/kdk-mlx/py/proxy_forwarder.py &
PYTHON_PID=$!

export HTTP_PROXY=http://localhost:8080
export HTTPS_PROXY=http://localhost:8080
export NO_PROXY=localhost,127.0.0.1

# Change to the project directory for build and copy steps
cd $HOME/local/codex/codex-cli || exit

# Build the project
nvm exec 22.11.0 pnpm build || exit 1

# Dynamically determine the global bin directory using the Node.js binary path
NODE_BIN=$(nvm exec 22.11.0 which node | tail -1)
if [ -z “$NODE_BIN” ]; then
  echo “Error: Unable to find Node.js binary.”
  exit 1
fi
PNPM_HOME=$(dirname “$NODE_BIN”)
export PNPM_HOME
export PATH=“$PNPM_HOME:$PATH”
export OPENAI_API_KEY=“sk-...“;

# Move the built files to the global bin directory
echo “Removing the old CLI from $PNPM_HOME/codex ...”
rm -f $PNPM_HOME/codex
echo “Moving the built CLI to $PNPM_HOME/codex ...”
cp -f bin/codex.js $PNPM_HOME/codex.js
mv $PNPM_HOME/codex.js $PNPM_HOME/codex

# Move the package.json to the global bin directory
echo “Removing the old package.json from $PNPM_HOME/codex ...”
rm -f $PNPM_HOME/../package.json
echo “Moving the package.json to $PNPM_HOME/../package.json ...”
cp -f package.json $PNPM_HOME/../package.json

# Move the built CLI to the global dist directory
echo “Removing the old CLI from $PNPM_HOME/../dist/cli.js ...”
rm -f $PNPM_HOME/../dist/cli.js
echo “Moving the built CLI to $PNPM_HOME/../dist/cli.js ...”
cp -f dist/cli-dev.js $PNPM_HOME/../dist/cli-dev.js
mv $PNPM_HOME/../dist/cli-dev.js $PNPM_HOME/../dist/cli.js

# Return to the original directory before running codex
cd “$ORIGINAL_DIR” || exit

echo “Codex installation completed.”

# run codex and ensure to kill the server if codex exits
nvm exec 22.11.0 codex --provider=oai
