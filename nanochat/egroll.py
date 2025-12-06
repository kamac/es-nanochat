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


# Cache for parameter -> name mapping (computed once per model)
_param_name_cache = {}

def get_parameter_name(param, model):
    """
    Get the parameter name from a parameter object by looking it up in the model.
    
    This avoids brittle string construction and ensures consistency between
    forward pass and ES update. Uses caching to avoid repeated lookups.
    
    Args:
        param: Parameter tensor (e.g., block.attn.c_q.weight)
        model: Model containing the parameter
    
    Returns:
        Full parameter name (e.g., "transformer.h.0.attn.c_q.weight")
    """
    # Use parameter object id as cache key (faster than tensor comparison)
    param_id = id(param)
    
    # Check cache first
    if param_id in _param_name_cache:
        return _param_name_cache[param_id]
    
    # Look up parameter name
    for name, p in model.named_parameters():
        if p is param:
            _param_name_cache[param_id] = name
            return name
    
    raise ValueError(f"Parameter not found in model. This should not happen.")


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
def _update_2d_parameter(p, seeds, local_norm_fit, layer_hash, chunk_size, device, N_local):
    """
    Update a 2D matrix parameter using low-rank (rank=1) ES update.
    
    Args:
        p: Parameter tensor [m, n]
        seeds: (N_local,) tensor of seeds
        local_norm_fit: (N_local,) tensor of normalized fitnesses
        layer_hash: Hash for this layer (computed from parameter name)
        chunk_size: Chunk size for processing (seeds must be iterated in same order as evaluate_population)
        device: Device for tensors
        N_local: Local population size
    
    Returns:
        update: [m, n] tensor of accumulated updates
    """
    m, n = p.shape
    update = torch.zeros_like(p.data)
    
    # Process local population in chunks
    for chunk_start in range(0, N_local, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N_local)
        chunk_size_actual = chunk_end - chunk_start
        
        # Get chunk of seeds and fitnesses
        seeds_chunk = seeds[chunk_start:chunk_end]
        norm_fit_chunk = local_norm_fit[chunk_start:chunk_end]
        
        # Generate noise for each member in chunk independently
        # CRITICAL: Each member must have independent noise based on their own seed
        A_chunk = torch.zeros(chunk_size_actual, m, device=device, dtype=p.dtype)
        B_chunk = torch.zeros(chunk_size_actual, n, device=device, dtype=p.dtype)
        
        for i in range(chunk_size_actual):
            member_seed = int(seeds_chunk[i].item()) + layer_hash
            gen = torch.Generator(device=device)
            gen.manual_seed(member_seed)
            A_chunk[i] = torch.randn(m, generator=gen, device=device, dtype=p.dtype)
            B_chunk[i] = torch.randn(n, generator=gen, device=device, dtype=p.dtype)
        
        # Scale A by fitness: [chunk_size, m]
        # CRITICAL: Cast norm_fit to same dtype as noise to avoid einsum dtype mismatch
        A_scaled = A_chunk * norm_fit_chunk.to(p.dtype).view(chunk_size_actual, 1)
        
        # Compute batched outer products and sum using einsum (rank=1 optimized)
        # 'cm,cn->mn' is outer product: sum over chunk dimension
        chunk_update = torch.einsum('cm,cn->mn', A_scaled, B_chunk)
        update.add_(chunk_update)
    
    return update


@torch.no_grad()
def _update_embedding_parameter(p, seeds, local_norm_fit, layer_hash, chunk_size, device, N_local, idx):
    """
    Update embedding parameter using sparse updates based on token indices.
    
    Only updates rows corresponding to tokens used in the forward pass.
    This ensures the ES invariant: noise used in forward pass matches noise used in update.
    
    Args:
        p: Parameter tensor [vocab_size, n_embd]
        seeds: (N_local,) tensor of seeds
        local_norm_fit: (N_local,) tensor of normalized fitnesses
        layer_hash: Hash for this layer (computed from parameter name)
        chunk_size: Chunk size for processing (seeds must be iterated in same order as evaluate_population)
        device: Device for tensors
        N_local: Local population size
        idx: [N_local, batch, seq] tensor of token indices
    
    Returns:
        update: [vocab_size, n_embd] tensor of accumulated updates
    """
    m, n = p.shape
    update = torch.zeros_like(p.data)
    
    # Process local population in chunks
    for chunk_start in range(0, N_local, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N_local)
        chunk_size_actual = chunk_end - chunk_start
        
        # Get chunk of seeds and fitnesses
        seeds_chunk = seeds[chunk_start:chunk_end]
        norm_fit_chunk = local_norm_fit[chunk_start:chunk_end]
        idx_chunk = idx[chunk_start:chunk_end]  # [chunk, batch, seq]
        
        # Generate noise for each member in chunk independently
        # CRITICAL: Each member must have independent noise based on their own seed
        A_chunk = torch.zeros(chunk_size_actual, m, device=device, dtype=p.dtype)
        B_chunk = torch.zeros(chunk_size_actual, n, device=device, dtype=p.dtype)
        
        for i in range(chunk_size_actual):
            member_seed = int(seeds_chunk[i].item()) + layer_hash
            gen = torch.Generator(device=device)
            gen.manual_seed(member_seed)
            A_chunk[i] = torch.randn(m, generator=gen, device=device, dtype=p.dtype)
            B_chunk[i] = torch.randn(n, generator=gen, device=device, dtype=p.dtype)
        
        # For each member in chunk, accumulate updates only for tokens used
        for c in range(chunk_size_actual):
            # Get tokens used by this member
            member_tokens = idx_chunk[c].flatten()  # [batch*seq]
            # Get A values for these tokens (matching forward pass)
            A_indexed = A_chunk[c][member_tokens]  # [batch*seq]
            # Scale by fitness
            A_scaled = A_indexed * norm_fit_chunk[c].to(p.dtype)
            # Compute outer product: A_indexed @ B^T for used tokens only
            # A_scaled: [batch*seq], B_chunk[c]: [n_embd]
            # Compute contribution: [batch*seq, n_embd]
            contribution = A_scaled.unsqueeze(-1) * B_chunk[c]
            # Sum contributions per unique token (handle duplicates)
            # Use scatter_add to accumulate contributions for each token
            unique_tokens, inverse_indices = torch.unique(member_tokens, return_inverse=True)
            # Sum contributions for each unique token
            token_contributions = torch.zeros(len(unique_tokens), n, device=device, dtype=p.dtype)
            token_contributions.scatter_add_(0, inverse_indices.unsqueeze(-1).expand_as(contribution), contribution)
            # Add to update
            update[unique_tokens] += token_contributions
    
    return update


@torch.no_grad()
def es_update_vectorized(model, fitnesses, seeds, lr, weight_decay=0.0, chunk_size=256, idx=None):
    """
    ES update with vectorized chunked noise generation (rank=1 optimized).
    
    Works for both single-GPU and multi-GPU (DDP) training automatically.
    Detects DDP and uses all_reduce when needed.
    
    Processes multiple members per chunk using batched ops for performance.
    
    Args:
        model: Model to update
        fitnesses: (N_total,) tensor of fitness scores
                   - Single-GPU: all fitnesses (N_total = population_size)
                   - Multi-GPU: all fitnesses from all ranks (after all_gather)
                                Caller must gather fitnesses using all_gather_into_tensor
                                before calling this function
        seeds: (N_local,) tensor of seeds corresponding to local population members
               - Single-GPU: all seeds (N_local = population_size = N_total)
               - Multi-GPU: seeds from this rank only (N_local = population_size // world_size)
                            This function extracts the corresponding local_fitnesses slice
        lr: Learning rate (effective step size, sigma already fused into lr)
        weight_decay: Weight decay coefficient
        chunk_size: Process this many members at once (balance speed and memory)
                   Note: chunk_size does not need to match evaluate_population's chunk_size.
                   As long as seeds are iterated in the same order, noise will be consistent.
        idx: Optional input token indices [batch, seq] for embedding updates
             Will be automatically broadcasted to [N_local, batch, seq]
             Only needed for embedding layer updates (transformer.wte.weight)
    
    Note: For multi-GPU, caller must gather all fitnesses using all_gather_into_tensor
          before calling this function. This ensures proper global normalization.
          The function then extracts the local slice corresponding to local seeds.
    Note: rank=1 is hardcoded for optimization (1/sqrt(1)=1.0, vectors instead of matrices).
    Note: sigma is fused into lr, so update formula is: W += (lr / N_total) * Σ fitness_i * A_i @ B_i^T
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
    
    # Broadcast idx to add population dimension if needed
    # _update_embedding_parameter expects idx: [N_local, batch, seq]
    # Callers pass [batch, seq] from dataloader, which we broadcast
    if idx is not None:
        if idx.ndim == 2:
            # Broadcast from [batch, seq] to [N_local, batch, seq]
            idx = idx.unsqueeze(0).expand(N_local, -1, -1)
        else:
            raise ValueError(f"idx must be 2D [batch, seq], got {idx.ndim}D with shape {idx.shape}")
    
    for name, p in model.named_parameters():
        if p.ndim == 2:  # Matrix parameters
            layer_hash = stable_hash_name(name)
            
            # Check if this is embedding layer and we have token indices
            is_embedding = name == 'transformer.wte.weight'
            if is_embedding and idx is not None:
                update = _update_embedding_parameter(
                    p, seeds, local_norm_fit, layer_hash, chunk_size, device, N_local, idx
                )
            else:
                update = _update_2d_parameter(
                    p, seeds, local_norm_fit, layer_hash, chunk_size, device, N_local
                )
            
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
