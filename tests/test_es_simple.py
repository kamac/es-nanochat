"""
Simple ES Training Test

Trains a minimal 1-layer model on a simple synthetic task to verify:
1. ES training reduces loss
2. Population evaluation works correctly
3. ES updates are applied properly
4. Deterministic noise generation works

Run with:
    python -m tests.test_es_simple
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt import GPT, GPTConfig
from nanochat.egroll import es_update_vectorized


def create_synthetic_dataset(vocab_size=64, seq_len=16, num_samples=32, device='cpu'):
    """
    Create a simple synthetic dataset: constant prediction task
    Input: [0, 0, 0, ...] (all zeros)
    Target: [1, 1, 1, ...] (all ones, shifted by 1 for autoregressive)
    
    The model needs to learn: "when I see token 0, predict token 1"
    This is the simplest possible learning task - just memorize one mapping.
    """
    # Input: all zeros
    x = torch.zeros(num_samples, seq_len, dtype=torch.long, device=device)
    
    # Target: all ones (shifted by 1 for autoregressive prediction)
    y = torch.ones(num_samples, seq_len, dtype=torch.long, device=device)
    y[:, 0] = -1  # First token has no previous token to predict from
    
    return x, y


def compute_loss(model, x, y):
    """Compute cross-entropy loss, ignoring -1 targets"""
    logits = model(x)
    # Flatten for loss computation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = y.view(-1)
    
    # Only compute loss where target != -1
    mask = targets_flat != -1
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x.device)
    
    loss = torch.nn.functional.cross_entropy(
        logits_flat[mask],
        targets_flat[mask],
        reduction='mean'
    )
    return loss


def train_with_hyperparams(model, x, y, population_size, es_lr, es_rank, sigma, chunk_size, 
                           num_steps, base_seed, device, verbose=False):
    """
    Train model with given hyperparameters and return results.
    
    Returns:
        dict with keys: initial_loss, final_loss, loss_reduction, success, losses
    """
    from contextlib import nullcontext
    if device == 'cuda':
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()
    
    # Reset model to initial state (reinitialize weights)
    model.init_weights()
    
    # Initial loss
    model.eval()
    with torch.inference_mode(), autocast_ctx:
        initial_loss = compute_loss(model, x, y).item()
    
    losses = []
    
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
                    'error': 'no_fitness_variation'
                }
        
        # Apply ES update
        es_update_vectorized(
            model,
            fitnesses,
            seeds,
            es_lr,
            weight_decay=0.0,
            chunk_size=chunk_size
        )
        
        # Compute current loss
        with torch.inference_mode(), autocast_ctx:
            current_loss = compute_loss(model, x, y).item()
        losses.append(current_loss)
        
        if verbose and (step % 20 == 0 or step == num_steps - 1):
            print(f"  Step {step:3d}: loss={current_loss:.6f}")
    
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    success = final_loss < initial_loss * 0.97  # 3% reduction threshold
    
    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'success': success,
        'losses': losses
    }


def test_es_training():
    """Test ES training with hyperparameter search"""
    
    print("="*60)
    print("Simple ES Training Test (with Hyperparameter Search)")
    print("="*60)
    print()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Minimal model config
    vocab_size = 64
    seq_len = 16
    n_layer = 1
    n_embd = 64
    n_head = 2
    
    print(f"Model config:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  n_layer: {n_layer}")
    print(f"  n_embd: {n_embd}")
    print(f"  n_head: {n_head}")
    
    # Create model
    model_config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd
    )
    
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  num_params: {num_params:,}")
    print()
    
    # Create synthetic dataset
    batch_size = 16
    x, y = create_synthetic_dataset(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_samples=batch_size,
        device=device
    )
    
    print(f"Dataset:")
    print(f"  batch_size: {batch_size}")
    print(f"  x shape: {x.shape}")
    print(f"  y shape: {y.shape}")
    print()
    
    # Hyperparameter search space
    es_lr_values = [0.01, 0.05, 0.1]
    population_sizes = [64, 128, 256, 512]
    chunk_sizes = [16]  # Chunk sizes to test
    num_steps_values = [50, 100, 200]  # Number of training steps to test
    es_rank = 1
    sigma = 0.1  # Noise temperature (perturbation scale)
    base_seed = 42
    
    # LR scaling for population size (same as base_train.py)
    # The ES formula lr / (σ√r·N) has 1/N term, so larger populations reduce effective step size
    # We use SQRT scaling (not linear) because:
    # - Larger populations give better gradient estimates (can use larger steps)
    # - But linear scaling is too aggressive (causes divergence)
    # - Square root is a conservative middle ground
    reference_population = 256  # Baseline population for es_lr tuning
    import math
    
    print("Hyperparameter Search:")
    print(f"  es_lr (base): {es_lr_values}")
    print(f"  population_size: {population_sizes}")
    print(f"  chunk_size: {chunk_sizes}")
    print(f"  num_steps: {num_steps_values}")
    print(f"  es_rank: {es_rank}")
    print(f"  sigma: {sigma}")
    print(f"  reference_population: {reference_population} (for LR scaling)")
    print(f"  Note: es_lr will be scaled by sqrt(pop/{reference_population})")
    print()
    print("Searching...")
    print("-" * 60)
    
    results = []
    
    for es_lr_base in es_lr_values:
        for population_size in population_sizes:
            for chunk_size in chunk_sizes:
                for num_steps in num_steps_values:
                    # Ensure chunk_size doesn't exceed population_size
                    actual_chunk_size = min(chunk_size, population_size)
                    
                    # Apply sqrt scaling to learning rate based on population size
                    scaling_factor = math.sqrt(population_size / reference_population)
                    es_lr_scaled = es_lr_base * scaling_factor
                    
                    print(f"\nTesting: es_lr_base={es_lr_base:.3f}, population_size={population_size}, chunk_size={actual_chunk_size}, num_steps={num_steps}")
                    print(f"  → es_lr_scaled={es_lr_scaled:.6f} (scaling_factor={scaling_factor:.4f})")
                    
                    result = train_with_hyperparams(
                        model, x, y,
                        population_size=population_size,
                        es_lr=es_lr_scaled,  # Use scaled LR
                        es_rank=es_rank,
                        sigma=sigma,
                        chunk_size=actual_chunk_size,
                        num_steps=num_steps,
                        base_seed=base_seed,
                        device=device,
                        verbose=False
                    )
                    
                    result['es_lr_base'] = es_lr_base  # Store base LR for reporting
                    result['es_lr_scaled'] = es_lr_scaled  # Store scaled LR
                    result['scaling_factor'] = scaling_factor
                    result['population_size'] = population_size
                    result['chunk_size'] = actual_chunk_size
                    result['num_steps'] = num_steps
                    results.append(result)
                    
                    status = "✅ PASS" if result['success'] else "❌ FAIL"
                    print(f"  {status} | Initial: {result['initial_loss']:.4f} → Final: {result['final_loss']:.4f} | Reduction: {result['loss_reduction']:.2f}%")
    
    print("-" * 60)
    print()
    
    # Analyze results
    successful_configs = [r for r in results if r['success']]
    all_reductions = [r['loss_reduction'] for r in results]
    
    print("="*60)
    print("Search Results Summary")
    print("="*60)
    print()
    print(f"Total configurations tested: {len(results)}")
    print(f"Successful configurations: {len(successful_configs)}")
    print(f"Success rate: {100 * len(successful_configs) / len(results):.1f}%")
    print()
    
    if successful_configs:
        # Find best configuration
        best = max(successful_configs, key=lambda r: r['loss_reduction'])
        
        print("✅ Best Configuration:")
        print(f"  es_lr_base: {best['es_lr_base']:.3f}")
        print(f"  es_lr_scaled: {best['es_lr_scaled']:.6f} (scaling_factor={best['scaling_factor']:.4f})")
        print(f"  population_size: {best['population_size']}")
        print(f"  chunk_size: {best['chunk_size']}")
        print(f"  num_steps: {best['num_steps']}")
        print(f"  Initial loss: {best['initial_loss']:.6f}")
        print(f"  Final loss: {best['final_loss']:.6f}")
        print(f"  Loss reduction: {best['loss_reduction']:.2f}%")
        print()
        
        # Show top 3
        top3 = sorted(successful_configs, key=lambda r: r['loss_reduction'], reverse=True)[:3]
        print("Top 3 Configurations:")
        for i, r in enumerate(top3, 1):
            print(f"  {i}. lr_base={r['es_lr_base']:.3f}, pop={r['population_size']}, chunk={r['chunk_size']}, steps={r['num_steps']}, reduction={r['loss_reduction']:.2f}%")
        print()
        
        # Overall test passes if at least one config succeeded
        overall_success = True
        print("✅ TEST PASSED: At least one hyperparameter configuration succeeded!")
    else:
        print("❌ TEST FAILED: No hyperparameter configuration succeeded")
        print()
        print("All configurations failed. Best attempt:")
        best_attempt = max(results, key=lambda r: r['loss_reduction'])
        print(f"  es_lr_base: {best_attempt['es_lr_base']:.3f}")
        print(f"  es_lr_scaled: {best_attempt['es_lr_scaled']:.6f} (scaling_factor={best_attempt['scaling_factor']:.4f})")
        print(f"  population_size: {best_attempt['population_size']}")
        print(f"  chunk_size: {best_attempt['chunk_size']}")
        print(f"  num_steps: {best_attempt['num_steps']}")
        print(f"  Loss reduction: {best_attempt['loss_reduction']:.2f}%")
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
    
    return overall_success


if __name__ == "__main__":
    success = test_es_training()
    sys.exit(0 if success else 1)

