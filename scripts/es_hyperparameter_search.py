"""
ES Hyperparameter Search Script

Performs hyperparameter search for ES training on the actual target model architecture.
Similar to tests/test_es_simple.py but uses:
- Real model architecture (depth=32, max_seq_len=512, etc.)
- Real data loader (not synthetic)
- More realistic hyperparameter ranges

Run with:
    python -m scripts.es_hyperparameter_search
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import sys
import math
from contextlib import nullcontext

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.egroll import es_update_vectorized


def compute_loss(model, x, y):
    """Compute cross-entropy loss"""
    logits = model(x)
    # Flatten for loss computation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = y.view(-1)
    
    loss = torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        reduction='mean'
    )
    return loss


def train_with_hyperparams(model, train_loader, val_loader, token_bytes, population_size, sigma, es_lr, 
                           chunk_size, num_steps, base_seed, device, device_type, 
                           eval_every=10, verbose=False):
    """
    Train model with given hyperparameters and return results.
    
    Returns:
        dict with keys: initial_loss, final_loss, loss_reduction, success, losses, val_losses
    """
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    
    # Reset model to initial state (reinitialize weights)
    model.init_weights()
    
    # Initial loss on validation set
    model.eval()
    with torch.inference_mode(), autocast_ctx:
        # Evaluate on a few validation batches
        val_bpb = evaluate_bpb(model, val_loader, 5, token_bytes)
        initial_loss = val_bpb  # Use validation bpb as initial loss
    
    losses = []
    val_losses = []
    
    # Get first training batch
    x, y = next(train_loader)
    
    for step in range(num_steps):
        model.eval()
        
        # Evaluate population
        with torch.inference_mode(), autocast_ctx:
            fitnesses, seeds = model.evaluate_population(
                x, y,
                population_size=population_size,
                sigma=sigma,
                base_seed=base_seed,
                step=step,
                world_size=1,
                ddp_rank=0,
                chunk_size=chunk_size
            )
        
        # Check fitness variation (critical for ES)
        if step == 0:
            fitness_std = fitnesses.std().item()
            if fitness_std < 1e-6:
                if verbose:
                    print("ERROR: No fitness variation!")
                return {
                    'initial_loss': initial_loss,
                    'final_loss': initial_loss,
                    'loss_reduction': 0.0,
                    'success': False,
                    'losses': [initial_loss] * num_steps,
                    'val_losses': [initial_loss] * num_steps,
                    'error': 'no_fitness_variation'
                }
        
        # Apply ES update
        es_update_vectorized(
            model,
            fitnesses,
            seeds,
            es_lr,
            weight_decay=0.0,
            chunk_size=chunk_size,
            idx=x
        )
        
        # Get next batch
        x, y = next(train_loader)
        
        # Compute current loss on training batch (from average fitness)
        avg_fitness = fitnesses.mean().item()
        train_loss = -avg_fitness  # fitness = -loss
        losses.append(train_loss)
        
        # Evaluate on validation set periodically
        if step % eval_every == 0 or step == num_steps - 1:
            model.eval()
            with torch.inference_mode(), autocast_ctx:
                val_bpb = evaluate_bpb(model, val_loader, 5, token_bytes)
                val_losses.append((step, val_bpb))
        
        if verbose and (step % 20 == 0 or step == num_steps - 1):
            print(f"  Step {step:3d}: train_loss={train_loss:.6f}, val_bpb={val_bpb:.6f}")
    
    # Final validation loss
    model.eval()
    with torch.inference_mode(), autocast_ctx:
        final_val_bpb = evaluate_bpb(model, val_loader, 5, token_bytes)
    
    final_loss = final_val_bpb
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    success = final_loss < initial_loss * 0.97  # 3% reduction threshold
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'success': success,
        'losses': losses,
        'val_losses': val_losses
    }


def main():
    """Main hyperparameter search function"""
    
    print("="*60)
    print("ES Hyperparameter Search (Target Model Architecture)")
    print("="*60)
    print()
    
    # Setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    
    if ddp_world_size > 1:
        print("ERROR: This script only supports single-GPU training")
        print("Please run without torchrun")
        sys.exit(1)
    
    print(f"Device: {device_type}")
    print()
    
    # Model architecture (from es_training.sh)
    depth = 32
    max_seq_len = 512
    device_batch_size = 32
    
    # Derive model config (same as base_train.py)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    num_layers = depth
    model_dim = depth * 64  # aspect ratio 64
    num_heads = max(1, (model_dim + 127) // 128)  # head dim 128
    num_kv_heads = num_heads  # 1:1 GQA ratio
    
    print(f"Model config:")
    print(f"  depth: {depth}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  device_batch_size: {device_batch_size}")
    print(f"  vocab_size: {vocab_size:,}")
    print(f"  num_layers: {num_layers}")
    print(f"  model_dim: {model_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    
    # Create model
    model_config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim
    )
    
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  num_params: {num_params:,}")
    print()
    
    # Create data loaders
    train_loader = tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split="train", device=device
    )
    val_loader = tokenizing_distributed_data_loader(
        device_batch_size, max_seq_len, split="val", device=device
    )
    
    # Get token_bytes for evaluation
    token_bytes = get_token_bytes(device=device)
    
    print(f"Data loaders initialized")
    print()
    
    # Hyperparameter search space
    # More conservative ranges for the larger model
    es_lr_values = [0.05]
    sigma_values = [0.1]
    population_sizes = [64, 128, 512]
    chunk_sizes = [1]  # Conservative chunk sizes for large model
    num_steps_values = [10]  # Fewer steps for faster search
    # rank=1 is hardcoded for optimization (removed es_rank parameter)
    base_seed = 42
    
    # LR scaling for population size (same as base_train.py)
    reference_population = 256  # Baseline population for es_lr tuning
    
    print("Hyperparameter Search:")
    print(f"  sigma: {sigma_values}")
    print(f"  es_lr (base): {es_lr_values}")
    print(f"  population_size: {population_sizes}")
    print(f"  chunk_size: {chunk_sizes}")
    print(f"  num_steps: {num_steps_values}")
    print(f"  rank: 1 (hardcoded for optimization)")
    print(f"  reference_population: {reference_population} (for LR scaling)")
    print(f"  Note: es_lr will be scaled by sqrt(pop/{reference_population})")
    print()
    print("Searching...")
    print("-" * 60)
    
    results = []
    
    for sigma in sigma_values:
        for es_lr_base in es_lr_values:
            for population_size in population_sizes:
                for chunk_size in chunk_sizes:
                    for num_steps in num_steps_values:
                        # Ensure chunk_size doesn't exceed population_size
                        actual_chunk_size = min(chunk_size, population_size)
                        
                        # Apply sqrt scaling to learning rate based on population size
                        scaling_factor = math.sqrt(population_size / reference_population)
                        es_lr_scaled = es_lr_base * scaling_factor
                        
                        print(f"\nTesting: sigma={sigma:.3f}, es_lr_base={es_lr_base:.3f}, population_size={population_size}, chunk_size={actual_chunk_size}, num_steps={num_steps}")
                        print(f"  → es_lr_scaled={es_lr_scaled:.6f} (scaling_factor={scaling_factor:.4f})")
                        
                        try:
                            result = train_with_hyperparams(
                                model, train_loader, val_loader, token_bytes,
                                population_size=population_size,
                                sigma=sigma,
                                es_lr=es_lr_scaled,  # Use scaled LR
                                chunk_size=actual_chunk_size,
                                num_steps=num_steps,
                                base_seed=base_seed,
                                device=device,
                                device_type=device_type,
                                eval_every=10,
                                verbose=False
                            )
                            
                            result['sigma'] = sigma
                            result['es_lr_base'] = es_lr_base  # Store base LR for reporting
                            result['es_lr_scaled'] = es_lr_scaled  # Store scaled LR
                            result['scaling_factor'] = scaling_factor
                            result['population_size'] = population_size
                            result['chunk_size'] = actual_chunk_size
                            result['num_steps'] = num_steps
                            results.append(result)
                        
                            status = "✅ PASS" if result['success'] else "❌ FAIL"
                            print(f"  {status} | Initial: {result['initial_loss']:.4f} → Final: {result['final_loss']:.4f} | Reduction: {result['loss_reduction']:.2f}%")
                        except Exception as e:
                            print(f"  ❌ ERROR: {e}")
                            results.append({
                                'sigma': sigma,
                                'es_lr_base': es_lr_base,
                                'es_lr_scaled': es_lr_scaled,
                                'scaling_factor': scaling_factor,
                                'population_size': population_size,
                                'chunk_size': actual_chunk_size,
                                'num_steps': num_steps,
                                'initial_loss': float('inf'),
                                'final_loss': float('inf'),
                                'loss_reduction': -100.0,
                                'success': False,
                                'error': str(e)
                            })
    
    print("-" * 60)
    print()
    
    # Analyze results
    successful_configs = [r for r in results if r.get('success', False)]
    all_reductions = [r['loss_reduction'] for r in results if 'loss_reduction' in r]
    
    print("="*60)
    print("Search Results Summary")
    print("="*60)
    print()
    print(f"Total configurations tested: {len(results)}")
    print(f"Successful configurations: {len(successful_configs)}")
    if results:
        print(f"Success rate: {100 * len(successful_configs) / len(results):.1f}%")
    print()
    
    if successful_configs:
        # Find best configuration
        best = max(successful_configs, key=lambda r: r['loss_reduction'])
        
        print("✅ Best Configuration:")
        print(f"  sigma: {best['sigma']:.3f}")
        print(f"  es_lr_base: {best['es_lr_base']:.3f}")
        print(f"  es_lr_scaled: {best['es_lr_scaled']:.6f} (scaling_factor={best['scaling_factor']:.4f})")
        print(f"  population_size: {best['population_size']}")
        print(f"  chunk_size: {best['chunk_size']}")
        print(f"  num_steps: {best['num_steps']}")
        print(f"  Initial loss: {best['initial_loss']:.6f}")
        print(f"  Final loss: {best['final_loss']:.6f}")
        print(f"  Loss reduction: {best['loss_reduction']:.2f}%")
        print()
        
        # Show top 5
        top5 = sorted(successful_configs, key=lambda r: r['loss_reduction'], reverse=True)[:5]
        print("Top 5 Configurations:")
        for i, r in enumerate(top5, 1):
            print(f"  {i}. sigma={r['sigma']:.3f}, lr_base={r['es_lr_base']:.3f}, pop={r['population_size']}, chunk={r['chunk_size']}, steps={r['num_steps']}, reduction={r['loss_reduction']:.2f}%")
        print()
        
        # Overall test passes if at least one config succeeded
        overall_success = True
        print("✅ SEARCH COMPLETE: At least one hyperparameter configuration succeeded!")
    else:
        print("❌ SEARCH FAILED: No hyperparameter configuration succeeded")
        print()
        if results:
            print("Best attempt:")
            best_attempt = max(results, key=lambda r: r.get('loss_reduction', -100))
            print(f"  sigma: {best_attempt.get('sigma', 'N/A')}")
            print(f"  es_lr_base: {best_attempt['es_lr_base']:.3f}")
            print(f"  es_lr_scaled: {best_attempt['es_lr_scaled']:.6f} (scaling_factor={best_attempt['scaling_factor']:.4f})")
            print(f"  population_size: {best_attempt['population_size']}")
            print(f"  chunk_size: {best_attempt['chunk_size']}")
            print(f"  num_steps: {best_attempt['num_steps']}")
            print(f"  Loss reduction: {best_attempt.get('loss_reduction', -100):.2f}%")
            if 'error' in best_attempt:
                print(f"  Error: {best_attempt['error']}")
            print()
        overall_success = False
    
    # Show distribution of results
    if all_reductions:
        print("Loss Reduction Statistics:")
        print(f"  Mean: {sum(all_reductions) / len(all_reductions):.2f}%")
        print(f"  Min: {min(all_reductions):.2f}%")
        print(f"  Max: {max(all_reductions):.2f}%")
        print()
    
    print("="*60)
    
    # Cleanup
    compute_cleanup()
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

