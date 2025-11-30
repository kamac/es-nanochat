#!/bin/bash

# Setup script for EGGROLL (ES) pretraining
# Based on speedrun.sh but stops before the actual training phase
# Gets everything ready: environment, tokenizer, and dataset

# Usage:
#   bash setup_for_pretraining.sh
# 
# After this completes, you can run ES training with:
#   python verify_es.py  # Quick test
#   bash test_es_training.sh  # Full test
#   python -m scripts.base_train --depth=20  # Real training

set -e  # Exit on error

echo "======================================================================="
echo "NANOCHAT EGGROLL (ES) PRETRAINING SETUP"
echo "======================================================================="
echo ""
echo "This script will:"
echo "  1. Set up Python environment (uv)"
echo "  2. Install dependencies"
echo "  3. Install Rust/Cargo"
echo "  4. Build rustbpe tokenizer"
echo "  5. Download pretraining dataset"
echo "  6. Train tokenizer"
echo "  7. Evaluate tokenizer"
echo ""
echo "After completion, you can run ES pretraining."
echo ""
echo "======================================================================="
echo ""

# Default intermediate artifacts directory
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"
echo "✓ Base directory: $NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv

echo ""
echo "Step 1/7: Setting up Python environment with uv..."
echo "-------------------------------------------------------------------"

# Install uv (if not already installed)
if command -v uv &> /dev/null; then
    echo "✓ uv already installed"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the path for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Detect if running on Google Colab
# Check for /content directory and either google.colab module or sample_data
if [ -d "/content" ] && ([ -d "/content/sample_data" ] || python3 -c "import google.colab" 2>/dev/null); then
    echo "✓ Detected Google Colab environment"
    IS_COLAB=true
    
    # On Colab, skip venv - install directly into system Python (already has PyTorch + CUDA)
    echo "Installing dependencies into system Python..."
    uv pip install -e . --system
    
    echo "✓ Python environment ready (system Python)"
else
    IS_COLAB=false
    
    # Create .venv local virtual environment (if it doesn't exist)
    if [ -d ".venv" ]; then
        echo "Removing existing virtual environment..."
        rm -rf .venv
    fi

    echo "Creating virtual environment with system site-packages..."
    # Use system site-packages to access system PyTorch (avoids NVSHMEM issues)
    # Must use same Python version as system (3.12) to access system packages
    uv venv --python /usr/bin/python3.12 --system-site-packages

    # Install the repo dependencies (will use system torch, install other deps)
    echo "Installing dependencies..."
    uv pip install -e .

    # Activate venv
    echo "Activating virtual environment..."
    source .venv/bin/activate

    echo "✓ Python environment ready"
fi

# -----------------------------------------------------------------------------
# wandb setup (optional)

echo ""
echo "Step 2/7: Configuring wandb (optional)..."
echo "-------------------------------------------------------------------"

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
    echo "ℹ  Using dummy wandb (no logging)"
    echo "   To enable wandb: export WANDB_RUN=<run_name> before running"
else
    echo "✓ wandb run name: $WANDB_RUN"
fi

# Initialize report system
echo "Initializing report system..."
python -m nanochat.report reset
echo "✓ Report system ready"

# -----------------------------------------------------------------------------
# Rust/Cargo installation

echo ""
echo "Step 3/7: Installing Rust/Cargo..."
echo "-------------------------------------------------------------------"

if command -v cargo &> /dev/null; then
    echo "✓ Rust/Cargo already installed ($(rustc --version))"
else
    echo "Installing Rust/Cargo..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "✓ Rust/Cargo installed"
fi

# Ensure cargo is in PATH for this session
export PATH="$HOME/.cargo/bin:$PATH"

# -----------------------------------------------------------------------------
# Build rustbpe tokenizer

echo ""
echo "Step 4/7: Building rustbpe tokenizer..."
echo "-------------------------------------------------------------------"

echo "Building tokenizer (this may take a minute)..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "✓ rustbpe tokenizer built"

# -----------------------------------------------------------------------------
# Download dataset

echo ""
echo "Step 5/7: Downloading pretraining dataset..."
echo "-------------------------------------------------------------------"

# Download the first ~2B characters of pretraining dataset
# Each data shard is ~250M chars, ~100MB compressed
# We download 8 shards initially (~800MB) for tokenizer training
echo "Downloading initial dataset (8 shards, ~800MB)..."
python -m nanochat.dataset -n 8

# Start downloading the full dataset needed for pretraining in the background
# For d20 model (561M params), Chinchilla says we need 20X tokens = 11.2B tokens
# At ~4.8 chars/token, that's ~54B chars, or ~216 shards (round to 240 for safety)
# This is ~24GB of data
echo ""
echo "Starting background download of full dataset (240 shards, ~24GB)..."
echo "This will continue in the background while tokenizer trains..."
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!

echo "✓ Initial dataset downloaded"
echo "ℹ  Full dataset downloading in background (PID: $DATASET_DOWNLOAD_PID)"

# -----------------------------------------------------------------------------
# Train tokenizer

echo ""
echo "Step 6/7: Training tokenizer..."
echo "-------------------------------------------------------------------"

echo "Training tokenizer on 2B characters (this will take a few minutes)..."
echo "Vocab size: 65536 (2^16)"

python -m scripts.tok_train --max_chars=2000000000

echo "✓ Tokenizer trained"

# -----------------------------------------------------------------------------
# Evaluate tokenizer

echo ""
echo "Step 7/7: Evaluating tokenizer..."
echo "-------------------------------------------------------------------"

python -m scripts.tok_eval

echo "✓ Tokenizer evaluated"

# -----------------------------------------------------------------------------
# Wait for full dataset download

echo ""
echo "Waiting for full dataset download to complete..."
echo "-------------------------------------------------------------------"

wait $DATASET_DOWNLOAD_PID

echo "✓ Full dataset downloaded (240 shards, ~24GB)"

# -----------------------------------------------------------------------------
# Run ES verification tests (optional but recommended)

echo ""
echo "Running ES implementation tests..."
echo "-------------------------------------------------------------------"

if python -m pytest tests/test_eggroll.py -v; then
    echo "✓ All ES tests passed"
else
    echo "⚠️  Some ES tests failed - check output above"
    echo "   You can still proceed with training, but results may vary"
fi

# -----------------------------------------------------------------------------
# Summary and next steps

echo ""
echo "======================================================================="
echo "SETUP COMPLETE!"
echo "======================================================================="
echo ""
echo "Summary:"
echo "  ✓ Python environment ready (.venv)"
echo "  ✓ Dependencies installed"
echo "  ✓ Rust/Cargo installed"
echo "  ✓ rustbpe tokenizer built and trained"
echo "  ✓ Pretraining dataset downloaded (240 shards, ~24GB)"
echo "  ✓ ES implementation tests passed"
echo ""
echo "Base directory: $NANOCHAT_BASE_DIR"
echo "Virtual environment: .venv (activated)"
echo ""
echo "======================================================================="
echo "NEXT STEPS: ES PRETRAINING"
echo "======================================================================="
echo ""
echo "Option 1: Quick verification (2 minutes, 2-4GB VRAM)"
echo "  python verify_es.py"
echo ""
echo "Option 2: Small test training (10 minutes, 4-8GB VRAM)"
echo "  bash test_es_training.sh"
echo ""
echo "Option 3: Full ES pretraining - Single GPU"
echo "  python -m scripts.base_train \\"
echo "    --depth=20 \\"
echo "    --population_size=256 \\"
echo "    --sigma=0.01 \\"
echo "    --es_lr=0.02 \\"
echo "    --es_rank=1 \\"
echo "    --chunk_size=8 \\"
echo "    --run=es_pretrain"
echo ""
echo "Option 4: Full ES pretraining - Multi-GPU (8 GPUs)"
echo "  torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \\"
echo "    --depth=20 \\"
echo "    --population_size=256 \\"
echo "    --sigma=0.01 \\"
echo "    --es_lr=0.02 \\"
echo "    --es_rank=1 \\"
echo "    --chunk_size=8 \\"
echo "    --run=es_pretrain"
echo ""
echo "Note: Make sure virtual environment is activated:"
echo "  source .venv/bin/activate"
echo ""
echo "For more options, see:"
echo "  - VERIFICATION_GUIDE.md (testing and verification)"
echo "  - EGGROLL_IMPLEMENTATION_GUIDE.md (full ES documentation)"
echo ""
echo "To monitor training:"
echo "  - Watch GPU: nvidia-smi"
echo "  - Follow logs: tail -f training.log"
echo "  - Check reports: $NANOCHAT_BASE_DIR/report/report.md"
echo ""
echo "======================================================================="

