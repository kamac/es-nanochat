#!/bin/bash
# ES Training Script
# Automatically uses all available GPUs

set -e  # Exit on error

# Add project root to PYTHONPATH for torchrun compatibility
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Disable NVSHMEM and peer GPU memory access (not needed for ES training)
# ES training doesn't need GPU-to-GPU peer memory access since each rank
# evaluates its own population subset independently
export TORCH_NCCL_NVSHMEM_ENABLE=0
export NCCL_NVSHMEM_DISABLE=1
export NCCL_P2P_DISABLE=1  # Disable peer-to-peer memory access over NVLink
export CUDA_VISIBLE_DEVICES_PEER_ACCESS_ENABLED=0  # Disable CUDA peer access

# Detect if running on Google Colab
if [ -d "/content" ] && ([ -d "/content/sample_data" ] || python3 -c "import google.colab" 2>/dev/null); then
    echo "✓ Detected Google Colab environment"
    IS_COLAB=true
    PYTHON_CMD="python"  # Use system Python directly
    TORCHRUN_CMD="torchrun"  # Use system torchrun directly
else
    IS_COLAB=false
    PYTHON_CMD="uv run python"  # Use uv to run Python
    TORCHRUN_CMD="uv run torchrun"  # Use uv to run torchrun
fi

# Wandb configuration (set to "dummy" to disable wandb logging)
# To enable wandb, you need to authenticate first:
#   1. Run: wandb login
#   2. Or set: export WANDB_API_KEY=your_api_key_here
# Get your API key from: https://wandb.ai/authorize
WANDB_PROJECT=${WANDB_PROJECT:-"nanochat"}  # wandb project name
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-"es_test"}  # prefix for wandb run names

# Common training parameters (for 90gb vram)
COMMON_ARGS="
    --depth=4
    --max_seq_len=256
    --total_batch_size=32768
    --device_batch_size=16
    --population_size=4096
    --sigma=0.01
    --es_lr=0.01
    --chunk_size=16
    --eval_every=50
    --core_metric_every=-1
    --sample_every=100
    --save_every=100
"

echo "=========================================="
echo "ES Training Test"
echo "=========================================="
echo ""

# Check if we should run distributed test
NUM_GPUS=${NUM_GPUS:-0}  # Default to 0 (auto-detect)

if [ "$NUM_GPUS" -eq 0 ]; then
    # Auto-detect GPUs
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        echo "Auto-detected $NUM_GPUS GPUs"
    else
        NUM_GPUS=1
        echo "nvidia-smi not found, assuming 1 GPU"
    fi
fi

echo "Using $NUM_GPUS GPU(s)"
echo ""

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU - use python directly
    $PYTHON_CMD -m scripts.base_train \
        --run=${WANDB_RUN_PREFIX} \
        $COMMON_ARGS
else
    # Multiple GPUs - use torchrun
    $TORCHRUN_CMD --nproc_per_node=$NUM_GPUS scripts/base_train.py \
        --run=${WANDB_RUN_PREFIX} \
        $COMMON_ARGS
fi

echo ""
echo "Training completed"
