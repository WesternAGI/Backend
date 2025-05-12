#!/bin/zsh
# Test script for aiagent.handler.cli using venv in the bot directory
# Usage: zsh test_aiagnet.sh (from anywhere)

set -e
# Get the absolute path to the bot root directory
SCRIPT_DIR="$(cd -- "${0:A:h}" && pwd)"
BOT_DIR="${SCRIPT_DIR}/.."
VENV_DIR="$BOT_DIR/venv"
PYTHON=python3

cd "$BOT_DIR"

# Create virtual environment if it does not exist
# if [ ! -d "$VENV_DIR" ]; then
#   $PYTHON -m venv "$VENV_DIR"
# fi

source "$VENV_DIR/bin/activate"

# # Install dependencies if requirements.txt exists
# if [ -f requirements.txt ]; then
#   pip install -r requirements.txt
# fi

echo "Installing dependencies..."


# Arguments to test
TEST_FILE="memory/loader.py"
TEST_QUERY="What does this file do?"

# 1. Test: Positional query only
$PYTHON -m aiagent.handler.cli "$TEST_QUERY"
echo "[PASS] Positional query only"

# 2. Test: --file argument
$PYTHON -m aiagent.handler.cli --file "$TEST_FILE" "$TEST_QUERY"
echo "[PASS] --file argument"

# 3. Test: --json argument
$PYTHON -m aiagent.handler.cli --json "$TEST_QUERY"
echo "[PASS] --json argument"

# 4. Test: --max-tokens argument
$PYTHON -m aiagent.handler.cli --max-tokens 32 "$TEST_QUERY"
echo "[PASS] --max-tokens argument"

# 5. Test: --temperature argument
$PYTHON -m aiagent.handler.cli --temperature 0.2 "$TEST_QUERY"
echo "[PASS] --temperature argument"

# 6. Test: --verbose argument
$PYTHON -m aiagent.handler.cli --verbose "$TEST_QUERY"
echo "[PASS] --verbose argument"

# 7. Test: --output argument
OUTFILE="/tmp/aiagnet_cli_test_output.txt"
$PYTHON -m aiagent.handler.cli --output "$OUTFILE" "$TEST_QUERY"
if [ -s "$OUTFILE" ]; then
  echo "[PASS] --output argument (output file created)"
else
  echo "[FAIL] --output argument (output file not created)"
  exit 1
fi
rm -f "$OUTFILE"

echo "All aiagent.handler.cli argument tests passed."
