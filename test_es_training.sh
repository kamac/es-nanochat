#!/bin/bash
# Disable NVSHMEM (not needed for single-GPU training)
export TORCH_NCCL_NVSHMEM_ENABLE=0
export NCCL_NVSHMEM_DISABLE=1

uv run python -m scripts.base_train \
    --run=es_test \
    --depth=20 \
    --max_seq_len=1024 \
    --device_batch_size=16 \
    --population_size=512 \
    --sigma=0.5 \
    --es_lr=0.02 \
    --es_rank=1 \
    --chunk_size=1 \
    --eval_every=10 \
    --core_metric_every=-1 \
    --sample_every=100 \
    --save_every=-1

