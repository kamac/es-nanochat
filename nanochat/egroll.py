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
import hashlib


def generate_lowrank_noise_factors(param_shape, seed, device='cpu'):
    """
    Generate low-rank noise factors (A, B) using deterministic seed.
    
    For parameter W with shape (m, n):
    - Returns A ∈ R^m, B ∈ R^n (rank=1 vectors)
    - Perturbation is E = sigma * A @ B.T (sigma=1.0 implicit, absorbed into lr)
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
        seed: Deterministic seed for this specific perturbation
        device: torch device
    
    Returns:
        A, B: Low-rank factors (vectors when rank=1)
    """
    m, n = param_shape
    
    # Create per-call generator (thread-safe, no global state)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    
    # Generate factors with i.i.d. N(0,1) entries (rank=1: vectors, not matrices)
    A = torch.randn(m, generator=gen, device=device)
    B = torch.randn(n, generator=gen, device=device)
    
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
def es_update_vectorized(model, fitnesses, seeds, lr, weight_decay=0.0, update_chunk_size=256):
    """
    ES update with vectorized chunked noise generation (rank=1 optimized).
    
    Works for both single-GPU and multi-GPU (DDP) training automatically.
    Detects DDP and uses all_reduce when needed.
    
    Processes multiple members per chunk using batched ops for performance.
    
    Args:
        model: Model to update
        fitnesses: (N,) tensor of fitness scores
                   - Single-GPU: all fitnesses
                   - Multi-GPU: fitnesses from this rank only (will be used after global normalization)
        seeds: (N,) tensor of seeds corresponding to fitnesses
               - Single-GPU: all seeds
               - Multi-GPU: seeds from this rank only
        lr: Learning rate (effective step size, sigma already fused into lr)
        weight_decay: Weight decay coefficient
        update_chunk_size: Process this many members at once (balance speed and memory)
                          CRITICAL: Must match chunk_size used in model.evaluate_population()
                          to ensure noise consistency between forward and update!
    
    Note: For multi-GPU, caller should pass all_fitnesses (after all_gather) for normalization,
          but only local seeds. This function will extract the relevant local_fitnesses slice.
    Note: rank=1 is hardcoded for optimization (1/sqrt(1)=1.0, vectors instead of matrices).
    Note: sigma is fused into lr, so update formula is: W += (lr / N) * Σ fitness_i * A_i @ B_i^T
    """
    # Detect if we're in DDP mode
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()
    
    if is_distributed:
        world_size = dist.get_world_size()
        rank_id = dist.get_rank()
    else:
        world_size = 1
        rank_id = 0
    
    # Normalize fitnesses using full distribution
    norm_fit = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
    
    # For multi-GPU: extract local fitnesses corresponding to local seeds
    # For single-GPU: N_total == N_local, so this is just identity
    N_total = len(fitnesses)
    N_local = len(seeds)
    
    if is_distributed:
        # Extract local slice from normalized fitnesses
        start_idx = rank_id * N_local
        local_norm_fit = norm_fit[start_idx:start_idx + N_local]
    else:
        local_norm_fit = norm_fit
    
    device = fitnesses.device
    
    for name, p in model.named_parameters():
        if p.ndim == 2:  # Matrix parameters
            # CRITICAL: Normalize name to match forward pass convention
            normalized_name = normalize_layer_name(name)
            layer_hash = stable_hash_name(normalized_name)
            m, n = p.shape
            update = torch.zeros_like(p.data)
            
            # Process local population in chunks
            for chunk_start in range(0, N_local, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, N_local)
                chunk_size = chunk_end - chunk_start
                
                # Get chunk of seeds and fitnesses
                seeds_chunk = seeds[chunk_start:chunk_end]
                norm_fit_chunk = local_norm_fit[chunk_start:chunk_end]
                
                # VECTORIZED: Generate noise for entire chunk with ONE generator
                # CRITICAL: Derive from first member's seed (same as forward pass)
                chunk_seed = int(seeds_chunk[0].item()) + layer_hash
                gen = torch.Generator(device=device)
                gen.manual_seed(chunk_seed)
                
                # Two big randn calls instead of many small ones
                # rank=1: [chunk_size, m] and [chunk_size, n] (vectors, not matrices)
                # CRITICAL: Use same dtype as parameters to match forward pass noise generation
                A_chunk = torch.randn(chunk_size, m, generator=gen, device=device, dtype=p.dtype)
                B_chunk = torch.randn(chunk_size, n, generator=gen, device=device, dtype=p.dtype)
                
                # Scale A by fitness: [chunk_size, m]
                # CRITICAL: Cast norm_fit to same dtype as noise to avoid einsum dtype mismatch
                A_scaled = A_chunk * norm_fit_chunk.to(p.dtype).view(chunk_size, 1)
                
                # Compute batched outer products and sum using einsum (rank=1 optimized)
                # 'cm,cn->mn' is outer product: sum over chunk dimension
                chunk_update = torch.einsum('cm,cn->mn', A_scaled, B_chunk)
                update.add_(chunk_update)
            
            # Multi-GPU: all-reduce to combine contributions from all ranks
            if is_distributed:
                dist.all_reduce(update, op=dist.ReduceOp.SUM)
            
            # Scale and apply update (use N_total for proper scaling)
            # Update formula: W += (lr / N) * Σ fitness_i * A_i @ B_i^T
            # Note: sigma is fused into lr, so no division by sigma here
            # rank=1: sqrt(1)=1.0, so no sqrt division needed
            scaling = lr / N_total
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
            
            for chunk_start in range(0, N_local, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, N_local)
                seeds_chunk = seeds[chunk_start:chunk_end]
                norm_fit_chunk = local_norm_fit[chunk_start:chunk_end]
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
            
            # Multi-GPU: all-reduce to combine contributions
            if is_distributed:
                dist.all_reduce(update, op=dist.ReduceOp.SUM)
            
            # Scale and apply update (use N_total for proper scaling)
            # Update formula: W += (lr / N) * Σ fitness_i * epsilon_i
            # Note: sigma is fused into lr, so no division by sigma here
            scaling = lr / N_total
            p.data.add_(update, alpha=scaling)
            
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)


@torch.no_grad()
def accumulate_micro_population_vectorized(model, fitnesses, seeds, sigma,
                                          accumulated_updates, update_chunk_size=256):
    """
    Accumulate fitness-weighted perturbations using chunk-level RNG and batched operations.
    
    Used for temporal accumulation to achieve large effective population sizes.
    Note: sigma is used to scale perturbations in forward pass, then normalized in es_update_with_accumulation.
    Note: rank=1 is hardcoded for optimization.
    
    Args:
        model: Model (not modified, used for parameter iteration)
        fitnesses: (M,) tensor of fitness scores
        seeds: (M,) tensor of seeds
        sigma: Noise temperature (perturbation scale used in forward pass)
        accumulated_updates: Dict to accumulate updates into
        update_chunk_size: Process this many members at once
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
                
                # Two big randn calls (rank=1: vectors, not matrices)
                # CRITICAL: Use same dtype as parameters to match forward pass noise generation
                A_chunk = torch.randn(chunk_size, m, generator=gen, device=device, dtype=p.dtype)
                B_chunk = torch.randn(chunk_size, n, generator=gen, device=device, dtype=p.dtype)
                # CRITICAL: Cast norm_fit to same dtype to avoid einsum dtype mismatch
                A_scaled = A_chunk * norm_fit_chunk.to(p.dtype).view(chunk_size, 1)
                
                # Accumulate using einsum (rank=1 optimized)
                chunk_update = torch.einsum('cm,cn->mn', A_scaled, B_chunk)
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
def es_update_with_accumulation(model, accumulated_updates, N_total, sigma, lr, weight_decay=0.0):
    """
    Apply ES update after accumulating fitness-weighted perturbations across micro-populations.
    
    This is called ONCE after accumulating over K micro-populations.
    
    Args:
        model: Model to update
        accumulated_updates: Dict of accumulated fitness-weighted noise per parameter
        N_total: Total effective population size (sum of all micro-population sizes)
        sigma: Noise temperature (perturbation scale used in forward pass; not used in update scaling)
        lr: Learning rate (effective step size, sigma already fused into lr)
        weight_decay: Weight decay coefficient
    Note: rank=1 is hardcoded for optimization.
    Note: sigma is fused into lr, so update formula is: W += (lr / N) * accumulated_update
    """
    for name, p in model.named_parameters():
        if p.ndim == 2 and name in accumulated_updates:
            # Scale accumulated update by ES formula
            # Update formula: W += (lr / N) * accumulated_update
            # Note: sigma is fused into lr, so no division by sigma here
            # rank=1: sqrt(1)=1.0, so no sqrt division needed
            scaling = lr / N_total
            p.data.add_(accumulated_updates[name], alpha=scaling)
            
            # Apply weight decay
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)
        
        elif p.ndim == 1:
            # FOR PRETRAINING: Update 1D params
            # FOR FINE-TUNING: Skip if frozen
            # continue
            
            if name in accumulated_updates:
                # Update formula: W += (lr / N) * accumulated_update
                # Note: sigma is fused into lr, so no division by sigma here
                scaling = lr / N_total
                p.data.add_(accumulated_updates[name], alpha=scaling)
                
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

