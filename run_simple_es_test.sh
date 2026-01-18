#!/bin/bash
# Simple ES Training Test Runner
# 
# This script tests that ES (Evolution Strategies) training works correctly
# by training a minimal 1-layer model on a simple copy task.
#
# Requirements:
# - PyTorch must be installed
# - Model should reduce loss within 20 ES updates

set -e  # Exit on error

echo "=============================================="
echo "Simple ES Training Test"
echo "=============================================="
echo ""

# Detect if running on Google Colab
if [ -d "/content" ] && ([ -d "/content/sample_data" ] || python3 -c "import google.colab" 2>/dev/null); then
    echo "✓ Detected Google Colab environment"
    PYTHON_CMD="python"
else
    # Try to use uv if available, otherwise fall back to python3
    if command -v uv &> /dev/null; then
        PYTHON_CMD="uv run python"
    else
        PYTHON_CMD="python3"
    fi
fi

echo "Using Python command: $PYTHON_CMD"
echo ""

# Run the test
$PYTHON_CMD -m tests.test_es_simple

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    exit 0
else
    echo ""
    echo "❌ Test failed!"
    exit 1
fi

