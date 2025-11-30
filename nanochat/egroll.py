"""
EGGROLL (Evolution Strategies) implementation for nanochat.

This module provides:
1. Low-rank noise generation with deterministic seeds
2. ES parameter update rules (vectorized for performance)
3. Seed coordination helpers for distributed training
4. Stable hashing for layer name offsets

Key implementation constraints:
- Uses chunk-level RNG for performance (100-1000x faster than per-member)
- Regenerates noise on-the-fly (no storage of perturbations)
- Deterministic via per-call torch.Generator (not global RNG state)
"""

import torch
import math
import hashlib


def generate_lowrank_noise_factors(param_shape, rank, seed, device='cpu'):
    """
    Generate low-rank noise factors (A, B) using deterministic seed.
    
    For parameter W with shape (m, n):
    - Returns A ∈ R^(m×r), B ∈ R^(n×r)
    - Perturbation is E = (sigma / sqrt(rank)) * A @ B.T
    - Uses per-call Generator for thread-safety and performance
    
    CRITICAL: Uses torch.Generator (NOT global torch.manual_seed):
    - No global RNG state pollution
    - Thread-safe and distributed-safe
    - Much faster (no RNG state save/restore overhead)
    
    NOTE: For production with large populations, use vectorized chunk-level
    generation directly in _linear_batched_lowrank (generates batches of A, B).
    This helper is provided for reference and distributed ES updates.
    
    Args:
        param_shape: (m, n) shape of parameter (e.g., [out_features, in_features])
        rank: Low-rank dimension r
        seed: Deterministic seed for this specific perturbation
        device: torch device
    
    Returns:
        A, B: Low-rank factors
    """
    m, n = param_shape
    
    # Create per-call generator (thread-safe, no global state)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    # Generate factors with i.i.d. N(0,1) entries
    A = torch.randn(m, rank, generator=gen, device=device)
    B = torch.randn(n, rank, generator=gen, device=device)
    
    return A, B


def generate_fullrank_noise(param_shape, seed, device='cpu'):
    """
    Generate full-rank noise for 1D parameters (bias, LayerNorm).
    Uses per-call Generator for thread-safety and performance.
    """
    # Create per-call generator (thread-safe, no global state)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    noise = torch.randn(param_shape, generator=gen, device=device)
    
    return noise


def normalize_layer_name(name):
    """
    Normalize parameter name to match forward pass layer naming convention.
    
    Forward pass uses names like:
    - "transformer.wte" (embedding)
    - "h.0.attn.c_q" (attention layers)
    
    named_parameters() returns:
    - "transformer.wte.weight" (embedding)
    - "transformer.h.0.attn.c_q.weight" (attention layers)
    
    This function strips suffixes and adjusts prefixes to ensure consistent naming.
    
    Args:
        name: Full parameter name (e.g., "transformer.h.0.attn.c_q.weight")
    
    Returns:
        Normalized name matching forward pass convention
    """
    # Remove ".weight" or ".bias" suffix if present
    if name.endswith(".weight"):
        name = name[:-len(".weight")]
    elif name.endswith(".bias"):
        name = name[:-len(".bias")]
    
    # Special case: embedding layer keeps "transformer." prefix
    # Other layers need "transformer.h" -> "h" conversion
    if name.startswith("transformer.h."):
        name = name[len("transformer."):]  # Remove "transformer." prefix
    
    return name


def stable_hash_name(name):
    """
    Compute stable hash of layer name for seed offsets.
    
    CRITICAL: Python's hash() is randomized per-process by default,
    causing non-reproducible results across runs. Use stable hash instead.
    
    Args:
        name: Layer name string (e.g., "h.0.attn.c_q")
    
    Returns:
        Stable 32-bit integer hash
    """
    # Use MD5 for stable hash across runs and processes
    hash_bytes = hashlib.md5(name.encode()).digest()
    # Take first 4 bytes as uint32
    return int.from_bytes(hash_bytes[:4], byteorder='little')


def compute_perturbation_seed(base_seed, step, world_size, population_size, ddp_rank, member_idx):
    """
    Compute deterministic seed for a specific perturbation.
    
    Ensures unique seed per (step, ddp_rank, member) combination.
    
    CRITICAL: Assumes population_size % world_size == 0 (population evenly divisible)
    
    Args:
        base_seed: Global base seed for reproducibility (e.g., 42)
        step: Current training step number
        world_size: Number of distributed ranks (1 for single-GPU)
        population_size: Total population size across all ranks
        ddp_rank: Current rank ID (0 to world_size-1, 0 for single-GPU)
        member_idx: Local member index within this rank's subset (0 to pop_per_rank-1)
    
    Returns:
        seed: Unique deterministic seed
    """
    # CRITICAL: Ensure population divides evenly across ranks
    assert population_size % world_size == 0, \
        f"population_size ({population_size}) must be divisible by world_size ({world_size})"
    
    population_per_rank = population_size // world_size
    global_member_idx = ddp_rank * population_per_rank + member_idx
    
    # Ensure uniqueness: base + step offset + global member offset
    # Step contributes: step * population_size (non-overlapping blocks per step)
    # Member contributes: global_member_idx (unique within step)
    return base_seed + step * population_size + global_member_idx


@torch.no_grad()
def es_update_vectorized(model, fitnesses, seeds, sigma, lr, rank, weight_decay=0.0, update_chunk_size=256):
    """
    ES update with vectorized chunked noise generation.
    
    Processes multiple members per chunk using batched ops for performance.
    
    Args:
        model: Model to update
        fitnesses: (N,) tensor of fitness scores
        seeds: (N,) tensor of seeds
        sigma: Noise temperature
        lr: Learning rate
        rank: Low-rank dimension
        weight_decay: Weight decay coefficient
        update_chunk_size: Process this many members at once (balance speed and memory)
    """
    # Normalize fitnesses
    norm_fit = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
    N = len(fitnesses)
    device = fitnesses.device
    
    for name, p in model.named_parameters():
        if p.ndim == 2:  # Matrix parameters
            # CRITICAL: Normalize name to match forward pass convention
            normalized_name = normalize_layer_name(name)
            layer_hash = stable_hash_name(normalized_name)
            m, n = p.shape
            update = torch.zeros_like(p.data)
            
            # Process population in chunks
            for chunk_start in range(0, N, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, N)
                chunk_size = chunk_end - chunk_start
                
                # Get chunk of seeds and fitnesses
                seeds_chunk = seeds[chunk_start:chunk_end]
                norm_fit_chunk = norm_fit[chunk_start:chunk_end]
                
                # VECTORIZED: Generate noise for entire chunk with ONE generator
                # CRITICAL: Derive from first member's seed (same as forward pass)
                chunk_seed = int(seeds_chunk[0].item()) + layer_hash
                gen = torch.Generator(device=device)
                gen.manual_seed(chunk_seed)
                
                # Two big randn calls instead of many small ones
                # [chunk_size, m, rank] and [chunk_size, n, rank]
                # CRITICAL: Use same dtype as parameters to match forward pass noise generation
                A_chunk = torch.randn(chunk_size, m, rank, generator=gen, device=device, dtype=p.dtype)
                B_chunk = torch.randn(chunk_size, n, rank, generator=gen, device=device, dtype=p.dtype)
                
                # Scale A by fitness: [chunk_size, m, rank]
                # CRITICAL: Cast norm_fit to same dtype as noise to avoid einsum dtype mismatch
                A_scaled = A_chunk * norm_fit_chunk.to(p.dtype).view(chunk_size, 1, 1)
                
                # Compute batched outer products and sum using einsum (most efficient)
                chunk_update = torch.einsum('cmi,cni->mn', A_scaled, B_chunk)
                update.add_(chunk_update)
            
            # Scale and apply update
            scaling = lr / (sigma * math.sqrt(rank) * N)
            p.data.add_(update, alpha=scaling)
            
            # Weight decay
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)
        
        elif p.ndim == 1:
            # FOR PRETRAINING: Update 1D params
            # FOR FINE-TUNING: Skip if frozen (uncomment continue below)
            # continue
            
            # Vectorized full-rank update
            # CRITICAL: Normalize name to match forward pass convention
            normalized_name = normalize_layer_name(name)
            layer_hash = stable_hash_name(normalized_name)
            update = torch.zeros_like(p.data)
            
            for chunk_start in range(0, N, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, N)
                seeds_chunk = seeds[chunk_start:chunk_end]
                norm_fit_chunk = norm_fit[chunk_start:chunk_end]
                chunk_size_actual = chunk_end - chunk_start
                
                # CRITICAL: Derive chunk seed from first member (same as forward pass)
                chunk_seed = int(seeds_chunk[0].item()) + layer_hash
                gen = torch.Generator(device=device)
                gen.manual_seed(chunk_seed)
                # CRITICAL: Use same dtype as parameters to match forward pass noise generation
                epsilon_chunk = torch.randn(chunk_size_actual, *p.shape, generator=gen, device=device, dtype=p.dtype)
                
                # CRITICAL: Cast norm_fit to same dtype to avoid type mismatch
                weighted = epsilon_chunk * norm_fit_chunk.to(p.dtype).view(chunk_size_actual, *([1] * len(p.shape)))
                update.add_(weighted.sum(dim=0))
            
            scaling = lr / (sigma * N)
            p.data.add_(update, alpha=scaling)
            
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)


@torch.no_grad()
def accumulate_micro_population_vectorized(model, fitnesses, seeds, sigma, rank, 
                                          accumulated_updates, update_chunk_size=256):
    """
    Accumulate fitness-weighted perturbations using chunk-level RNG and batched operations.
    
    Used for temporal accumulation to achieve large effective population sizes.
    """
    norm_fit = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
    M = len(fitnesses)
    device = fitnesses.device
    
    for name, p in model.named_parameters():
        if p.ndim == 2:
            # CRITICAL: Normalize name to match forward pass convention
            normalized_name = normalize_layer_name(name)
            layer_hash = stable_hash_name(normalized_name)
            m, n = p.shape
            
            if name not in accumulated_updates:
                accumulated_updates[name] = torch.zeros_like(p.data)
            
            # Process in chunks
            for chunk_start in range(0, M, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, M)
                chunk_size = chunk_end - chunk_start
                
                seeds_chunk = seeds[chunk_start:chunk_end]
                norm_fit_chunk = norm_fit[chunk_start:chunk_end]
                
                # VECTORIZED: Generate noise for chunk with ONE generator
                # CRITICAL: Derive from first member's seed (same as forward pass)
                chunk_seed = int(seeds_chunk[0].item()) + layer_hash
                gen = torch.Generator(device=device)
                gen.manual_seed(chunk_seed)
                
                # Two big randn calls
                # CRITICAL: Use same dtype as parameters to match forward pass noise generation
                A_chunk = torch.randn(chunk_size, m, rank, generator=gen, device=device, dtype=p.dtype)
                B_chunk = torch.randn(chunk_size, n, rank, generator=gen, device=device, dtype=p.dtype)
                # CRITICAL: Cast norm_fit to same dtype to avoid einsum dtype mismatch
                A_scaled = A_chunk * norm_fit_chunk.to(p.dtype).view(chunk_size, 1, 1)
                
                # Accumulate using einsum (most efficient)
                chunk_update = torch.einsum('cmi,cni->mn', A_scaled, B_chunk)
                accumulated_updates[name].add_(chunk_update)
        
        elif p.ndim == 1:
            # FOR PRETRAINING: Accumulate 1D params
            # FOR FINE-TUNING: Skip if frozen
            # continue
            
            # CRITICAL: Normalize name to match forward pass convention
            normalized_name = normalize_layer_name(name)
            layer_hash = stable_hash_name(normalized_name)
            
            if name not in accumulated_updates:
                accumulated_updates[name] = torch.zeros_like(p.data)
            
            for chunk_start in range(0, M, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, M)
                chunk_size = chunk_end - chunk_start
                
                seeds_chunk = seeds[chunk_start:chunk_end]
                norm_fit_chunk = norm_fit[chunk_start:chunk_end]
                
                chunk_seed = int(seeds_chunk[0].item()) + layer_hash
                gen = torch.Generator(device=device)
                gen.manual_seed(chunk_seed)
                # CRITICAL: Use same dtype as parameters to match forward pass noise generation  
                epsilon_chunk = torch.randn(chunk_size, *p.shape, generator=gen, device=device, dtype=p.dtype)
                
                # CRITICAL: Cast norm_fit to same dtype to avoid type mismatch
                weighted = epsilon_chunk * norm_fit_chunk.to(p.dtype).view(chunk_size, *([1] * len(p.shape)))
                accumulated_updates[name].add_(weighted.sum(dim=0))
    
    return accumulated_updates


@torch.no_grad()
def es_update_with_accumulation(model, accumulated_updates, N_total, sigma, lr, rank, weight_decay=0.0):
    """
    Apply ES update after accumulating fitness-weighted perturbations across micro-populations.
    
    This is called ONCE after accumulating over K micro-populations.
    
    Args:
        model: Model to update
        accumulated_updates: Dict of accumulated fitness-weighted noise per parameter
        N_total: Total effective population size (sum of all micro-population sizes)
        sigma: Noise temperature
        lr: Learning rate
        rank: Low-rank dimension
        weight_decay: Weight decay coefficient
    """
    for name, p in model.named_parameters():
        if p.ndim == 2 and name in accumulated_updates:
            # Scale accumulated update by ES formula
            scaling = lr / (sigma * math.sqrt(rank) * N_total)
            p.data.add_(accumulated_updates[name], alpha=scaling)
            
            # Apply weight decay
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)
        
        elif p.ndim == 1:
            # FOR PRETRAINING: Update 1D params
            # FOR FINE-TUNING: Skip if frozen
            # continue
            
            if name in accumulated_updates:
                scaling = lr / (sigma * N_total)
                p.data.add_(accumulated_updates[name], alpha=scaling)
                
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

