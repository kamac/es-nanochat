#!/bin/bash
# ES Training Test Script
# Tests both single-GPU and multi-GPU (distributed) training

set -e  # Exit on error

# Disable NVSHMEM (not needed for ES training)
export TORCH_NCCL_NVSHMEM_ENABLE=0
export NCCL_NVSHMEM_DISABLE=1

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

# Common training parameters (for 40gb vram)
COMMON_ARGS="
    --depth=32
    --max_seq_len=512
    --device_batch_size=16
    --population_size=512
    --sigma=0.1
    --es_lr=0.1
    --chunk_size=2
    --eval_every=10
    --core_metric_every=-1
    --sample_every=100
    --save_every=-1
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

echo ""
echo "=========================================="
echo "Test 1: Single-GPU Training"
echo "=========================================="
echo ""

$PYTHON_CMD -m scripts.base_train \
    --run=${WANDB_RUN_PREFIX}_single \
    $COMMON_ARGS

echo ""
echo "✅ Single-GPU test completed"
echo ""

# Run distributed test if multiple GPUs available
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "=========================================="
    echo "Test 2: Multi-GPU Training (2 GPUs)"
    echo "=========================================="
    echo ""
    
    $TORCHRUN_CMD --nproc_per_node=2 scripts/base_train.py \
        --run=${WANDB_RUN_PREFIX}_2gpu \
        $COMMON_ARGS
    
    echo ""
    echo "✅ 2-GPU test completed"
    echo ""
fi

if [ "$NUM_GPUS" -ge 4 ]; then
    echo "=========================================="
    echo "Test 3: Multi-GPU Training (4 GPUs)"
    echo "=========================================="
    echo ""
    
    $TORCHRUN_CMD --nproc_per_node=4 scripts/base_train.py \
        --run=${WANDB_RUN_PREFIX}_4gpu \
        $COMMON_ARGS
    
    echo ""
    echo "✅ 4-GPU test completed"
    echo ""
fi

if [ "$NUM_GPUS" -ge 8 ]; then
    echo "=========================================="
    echo "Test 4: Multi-GPU Training (8 GPUs)"
    echo "=========================================="
    echo ""
    
    $TORCHRUN_CMD --nproc_per_node=8 scripts.base_train.py \
        --run=${WANDB_RUN_PREFIX}_8gpu \
        $COMMON_ARGS
    
    echo ""
    echo "✅ 8-GPU test completed"
    echo ""
fi

echo "=========================================="
echo "All Tests Passed! ✅"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Single-GPU: ✅"
if [ "$NUM_GPUS" -ge 2 ]; then
    echo "  - 2-GPU: ✅"
fi
if [ "$NUM_GPUS" -ge 4 ]; then
    echo "  - 4-GPU: ✅"
fi
if [ "$NUM_GPUS" -ge 8 ]; then
    echo "  - 8-GPU: ✅"
fi
echo ""
echo "To run specific GPU count, use:"
echo "  NUM_GPUS=1 ./test_es_training.sh  # Single-GPU only"
echo "  NUM_GPUS=2 ./test_es_training.sh  # Up to 2 GPUs"
echo "  NUM_GPUS=4 ./test_es_training.sh  # Up to 4 GPUs"
echo "  NUM_GPUS=8 ./test_es_training.sh  # Up to 8 GPUs"
