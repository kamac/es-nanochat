# EGGROLL Implementation Guide for nanochat

This document outlines the changes needed to implement EGGROLL (Evolution Strategies) training from the paper "Evolution Strategies at the Hyperscale" (https://eshyperscale.github.io/).

## Overview

EGGROLL **replaces** backpropagation with Evolution Strategies using low-rank perturbations. This is a complete replacement of the training mechanism - we remove gradient computation entirely and replace it with:
1. Low-rank perturbation sampling for each parameter matrix
2. Population-based evaluation of perturbed models
3. Fitness-weighted parameter updates

## Critical Implementation Constraints

### Memory: Do NOT Store Perturbations
**Problem**: For 262k population, storing all perturbations requires ~TB of memory (infeasible)
**Solution**: Use counter-based RNG with deterministic seeds
- Generate noise on-the-fly during forward pass
- Regenerate same noise during ES update using saved seeds
- Memory: O(model_size) not O(population × model_size)

⚠️ **CRITICAL CAVEAT**: The naive embedding perturbation approach (Section 3.1) violates this constraint by materializing `[pop, vocab, n_embd]` tensors. For production:
- **Option 1** (recommended): Freeze embeddings (don't perturb wte/wpe)
- **Option 2**: Use low-rank embedding perturbation (Section 3.1.1)
- **Do NOT** use naive full-rank embedding approach for realistic models

### Speed: Do NOT Loop Over Population
**Problem**: Sequential loop over 262k members is unbearably slow
**Solution**: Batch population as extra dimension with chunking
- Shape: `[chunk_size, batch_size, seq_len, hidden]`
- All operations batched (matmuls, activations, etc.)
- **CRITICAL**: Chunk at top level (in `evaluate_population`), not just in linear layers
- Process 64-256 members at a time to avoid OOM on logits

### RNG: Use Per-Call Generators, NOT Global State
**Problem**: `torch.manual_seed()` in hot path crushes performance and breaks distributed
**Solution**: Use per-call `torch.Generator` objects
- `gen = torch.Generator(device=device); gen.manual_seed(seed)`
- No global RNG state pollution
- Thread-safe and distributed-safe
- Much faster (no state save/restore overhead)

**Performance Strategy**: Use chunk-level generators to amortize overhead:
- Create one generator per layer per chunk (not per member)
- Pop=10k, chunk=64 → ~3k generator creations vs 200k (100× fewer)
- Generate noise in vectorized batches instead of loops
- Essential for production use with populations > 1k

### Hashing: Use Stable Hash, NOT Python `hash()`
**Problem**: Python's `hash()` is randomized per-process (non-reproducible)
**Solution**: Use stable hash function
- `hashlib.md5(name.encode()).digest()` → uint32
- Reproducible across runs and processes
- Critical for layer-specific seed offsets

### Distributed: Coordinated Seed Generation
**Problem**: Each rank needs different perturbations, but must be reproducible
**Solution**: Centralized deterministic seed formula via `compute_perturbation_seed()`
- `seed = base_seed + step * population_size + rank * pop_per_rank + member_idx`
- Single source of truth used by both single-GPU and distributed code
- Each rank evaluates different subset of population
- All-gather fitnesses, all-reduce fitness-weighted updates

## Key Changes Required

### 1. Add ES Training Methods (`nanochat/gpt.py`)

**CRITICAL**: **Keep `GPT.forward()` exactly as-is** - it's used for inference/evaluation and remains unchanged.

**Current**: `GPT.forward(idx, targets=None)` computes logits + optional CE loss for backprop.

**ES Approach**: Add new ES-specific methods that internally reuse `forward()` but apply low-rank perturbations.

#### Changes needed:

1. **Keep `GPT.forward()` unchanged**:
   - **DO NOT MODIFY**: Standard `forward(idx, targets=None)` method
   - It continues to work for inference and standard training
   - Returns logits (if `targets=None`) or loss (if `targets` provided)

2. **Add ES-specific evaluation methods**:
   - **Add**: `evaluate_population()` - evaluates population of perturbed models with chunking
   - **Add**: `_forward_batched_population()` - batched forward pass with perturbations
   - **Add**: `_forward_block_batched()` - transformer block with batched perturbations
   - **Add**: `_linear_batched_lowrank()` - linear layer with low-rank perturbations
   - Noise generation delegated to `egroll.generate_lowrank_noise_factors()`

3. **ES training uses new methods**:
   - **Training**: `fitnesses, seeds = model.evaluate_population(x, y, population_size, ...)` 
   - **Inference**: `logits = model.forward(x)` (unchanged, no perturbations)
   - Each population member gets different perturbations via deterministic seeds
   - Use batched operations with chunking to evaluate efficiently

### 2. Low-Rank Perturbation Generation (`nanochat/egroll.py` - NEW FILE)

**CRITICAL**: Do NOT store perturbations in memory. Use counter-based RNG with deterministic seeds to regenerate noise on-the-fly.

**Key implementation strategy**:
1. **Counter-based RNG**: Use deterministic seeds to regenerate noise instead of storing it
2. **Seed format**: Use centralized `compute_perturbation_seed(...)` helper (see below)
3. **On-the-fly generation**: Generate noise factors during forward pass, discard after use
4. **Memory overhead**: O(model_size) instead of O(population_size × model_size)

**Reference implementation**:
```python
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

def stable_hash_name(name):
    """
    Compute stable hash of layer name for seed offsets.
    
    CRITICAL: Python's hash() is randomized per-process by default,
    causing non-reproducible results across runs. Use stable hash instead.
    
    Args:
        name: Layer name string (e.g., "transformer.h.0.attn.c_attn")
    
    Returns:
        Stable 32-bit integer hash
    """
    # Use MD5 for stable hash across runs and processes
    hash_bytes = hashlib.md5(name.encode()).digest()
    # Take first 4 bytes as uint32
    return int.from_bytes(hash_bytes[:4], byteorder='little')
```

**Seed coordination**:
```python
def compute_perturbation_seed(base_seed, step, world_size, population_size, ddp_rank, member_idx):
    """
    Compute deterministic seed for a specific perturbation.
    
    Args:
        base_seed: Global base seed for reproducibility
        step: Current training step
        world_size: Number of distributed ranks
        population_size: Total population size
        ddp_rank: Current DDP rank ID (0 to world_size-1)
        member_idx: Index within this rank's population subset
    
    Returns:
        seed: Deterministic seed that is unique per (step, ddp_rank, member)
    """
    # Ensure unique seed per step, ddp_rank, and population member
    global_member_idx = ddp_rank * (population_size // world_size) + member_idx
    return base_seed + step * population_size + global_member_idx
```

**Memory savings**:
- **Old approach**: Store 262k × num_params × (A+B) = **~TB of memory**
- **New approach**: Generate on-the-fly = **~GB of model parameters only**

### 3. Which Parameters to Perturb and Update

**CRITICAL DECISION**: You must perturb exactly the parameters you plan to update with ES.

**The Problem**:
- If you perturb a parameter in forward pass, its noise affects fitness → ES can learn
- If you DON'T perturb a parameter but still update it, updates are pure noise (random walk)
- If you perturb but don't update, you waste compute on noise that doesn't help

**For Pretraining: Perturb ALL Parameters (RECOMMENDED)**

For training from scratch, perturb and update ALL learnable parameters:

**2D Matrix Parameters** (use low-rank perturbations):
- ✅ Attention matrices: Q, K, V projections, output projection → `E = (σ/√r) * A @ B^T`
- ✅ MLP matrices: both layers → `E = (σ/√r) * A @ B^T`
- ✅ LM head: output projection → `E = (σ/√r) * A @ B^T`

**1D Vector Parameters** (use full-rank perturbations):
- ✅ Embeddings: `wte` (token embeddings), `wpe` (position embeddings) → `ε ~ N(0, I)`
- ✅ LayerNorm: `weight`, `bias` parameters → `ε ~ N(0, I)`
- ✅ All linear biases → `ε ~ N(0, I)`

**Why perturb 1D parameters**:
- Essential for pretraining from random initialization
- These parameters ARE learnable and affect the model
- 1D params are tiny relative to 2D matrices (memory/compute negligible)
- Full-rank noise is fine for vectors (no low-rank needed)
- Skipping them means they never learn (frozen at random init)

**Implementation**: See detailed code in subsections below.

**For Fine-Tuning: Optionally Freeze 1D Parameters**

If fine-tuning from a pretrained checkpoint where embeddings/LayerNorm are already good:

Do NOT perturb or update (freeze):
- ❌ Embeddings: `wte`, `wpe` (already learned)
- ❌ LayerNorm: `weight`, `bias` (already learned)
- ❌ Biases: all bias vectors (optional, can keep frozen)

Only perturb and update:
- ✅ Attention matrices, MLP matrices, LM head (low-rank perturbations)

**Why freeze for fine-tuning**:
- Simpler implementation (fewer cases to handle)
- Embeddings are already pretrained
- Reduces computation slightly
- Most adaptation happens in 2D matrices

**Implementation**:
```python
# In es_update: skip 1D parameters entirely
for name, p in model.named_parameters():
    if p.ndim == 2:  # Only update 2D matrices
        # Generate low-rank noise, compute ES update
        ...
    elif p.ndim == 1:
        # Skip - frozen at pretrained values
        continue
```

**Recommendation**: 
- **Pretraining**: Perturb ALL parameters (including 1D with full-rank noise)
- **Fine-tuning**: Optionally freeze 1D parameters (simpler, often sufficient)

#### 3.1. Perturbing 1D Parameters (Embeddings, LayerNorm, Biases)

**⚠️ CRITICAL MEMORY WARNING**: The examples in this section use simplified approaches for clarity. For embedding tables (`wte`, `wpe`), the naive full-rank perturbation shown below **will OOM for realistic models** (see detailed discussion and scalable solutions at the end of this section).

**SIMPLE APPROACH (for small models / fine-tuning / proof-of-concept only)**:

For 1D parameters, use **full-rank vector noise** `ε ~ N(0, I)`:

**Token Embeddings** (`wte`) - **SIMPLIFIED, NOT PRODUCTION-READY**:
```python
# ⚠️ WARNING: This approach materializes [pop, vocab_size, n_embd] tensor
# For vocab_size=50k, n_embd=4096, pop=8: ~6.4 GB in fp16 just for this tensor
# Use ONLY for small models or when freezing embeddings
# See "Scalable Embedding Perturbation" section below for production approach

# In _forward_batched_population:
# wte shape: [vocab_size, n_embd]
# idx shape: [pop, batch, seq]

# Generate noise for wte (once per population member)
wte = self.transformer.wte.weight  # [vocab_size, n_embd]
wte_perturbed_list = []

for i in range(pop_size):
    # Generate full-rank noise for entire embedding table
    member_seed = int(seeds[i].item()) + stable_hash_name('wte')
    epsilon_wte = generate_fullrank_noise(wte.shape, member_seed, device)
    wte_perturbed = wte + sigma * epsilon_wte  # [vocab_size, n_embd]
    wte_perturbed_list.append(wte_perturbed)

# Stack: [pop, vocab_size, n_embd] ← THIS IS THE OOM PROBLEM
wte_perturbed_batch = torch.stack(wte_perturbed_list)

# Apply embedding lookup for each population member
x_list = []
for i in range(pop_size):
    x_i = F.embedding(idx[i], wte_perturbed_batch[i])  # [batch, seq, n_embd]
    x_list.append(x_i)
x = torch.stack(x_list)  # [pop, batch, seq, n_embd]
```

**Position Embeddings** (`wpe`) - **SIMPLIFIED, NOT PRODUCTION-READY**:
```python
# ⚠️ WARNING: Similar OOM risk, though smaller than wte
# wpe is typically max_seq_len × n_embd (e.g., 2048 × 4096 = 8M params)
# Still becomes [pop, 2048, 4096] which is significant for large pop

wpe = self.transformer.wpe.weight  # [max_seq_len, n_embd]
wpe_perturbed_list = []

for i in range(pop_size):
    member_seed = int(seeds[i].item()) + stable_hash_name('wpe')
    epsilon_wpe = generate_fullrank_noise(wpe.shape, member_seed, device)
    wpe_perturbed = wpe + sigma * epsilon_wpe
    wpe_perturbed_list.append(wpe_perturbed)

wpe_perturbed_batch = torch.stack(wpe_perturbed_list)  # [pop, max_seq_len, n_embd]

# Add position embeddings
pos = torch.arange(seq_len, device=device)
for i in range(pop_size):
    x[i] = x[i] + wpe_perturbed_batch[i][pos]
```

**RECOMMENDATION FOR EMBEDDINGS**:
- **Fine-tuning / LoRA-style**: Freeze embeddings entirely (`continue` in ES update for `wte`/`wpe`)
- **Small models**: Use simplified approach above (< 1B params, pop < 16)
- **Pretraining / large models**: Use scalable low-rank approach (see below)

---

#### 3.1.1. Scalable Embedding Perturbation (Production Approach)

**The Problem**: The naive approach above materializes `[pop, vocab_size, n_embd]` tensors:

```
vocab_size ≈ 50k, n_embd ≈ 4096
Embedding table: 50k × 4096 ≈ 205M params ≈ 0.8 GB (fp16) / 1.6 GB (fp32)

With population batching:
pop=8  → 8 × 0.8 GB ≈ 6.4 GB (fp16) just for wte_perturbed_batch
pop=32 → 25 GB
```

This violates the "O(model_size) not O(pop × model_size)" constraint.

**The Solution**: Treat embeddings as 2D matrices and apply **low-rank perturbations**, but only materialize the perturbation for the **specific indices being looked up** in the current batch.

**Conceptual Approach**:
- Embedding table `E ∈ R^{vocab × d}` is just a 2D matrix
- Low-rank perturbation: `ΔE = (σ/√r) A B^T` where `A ∈ R^{vocab × r}`, `B ∈ R^{d × r}`
- For population member `i`, perturbed embedding: `E_i = E + ΔE_i`
- **Key insight**: We only need `E_i[idx]` for the indices actually used in the batch
- Never materialize full `[pop, vocab, d]` tensor

**Implementation Strategy**:

```python
def _embedding_batched_lowrank(
    self, 
    idx: torch.Tensor,      # [pop, batch, seq] - integer indices
    embedding_table: nn.Embedding,  # weight: [vocab_size, n_embd]
    seeds: torch.Tensor,    # [pop]
    sigma: float,
    rank: int,
    layer_name: str,
    device: torch.device
) -> torch.Tensor:
    """
    Apply low-rank perturbation to embedding table, but only for indices in idx.
    
    Memory: O(pop × batch × seq × d + pop × unique_indices × r)
    NOT: O(pop × vocab × d)
    """
    pop_size = idx.shape[0]
    batch_size = idx.shape[1]
    seq_len = idx.shape[2]
    vocab_size, n_embd = embedding_table.weight.shape
    
    # Base embedding lookup (shared across population)
    # Flatten idx to do one big lookup: [pop, batch, seq] → [pop*batch*seq]
    idx_flat = idx.reshape(-1)  # [pop*batch*seq]
    base_emb = embedding_table(idx_flat)  # [pop*batch*seq, n_embd]
    base_emb = base_emb.view(pop_size, batch_size, seq_len, n_embd)  # [pop, batch, seq, n_embd]
    
    # Low-rank perturbation - ONLY for unique indices in this batch
    # Step 1: Find unique indices across the entire batch
    unique_idx = torch.unique(idx)  # [num_unique] - typically << vocab_size
    num_unique = unique_idx.shape[0]
    
    # Step 2: For each population member, generate low-rank factors for ONLY these indices
    # We'll use chunk-level generation for efficiency
    layer_hash = stable_hash_name(layer_name)
    
    # For simplicity, process entire population (can chunk if pop is huge)
    perturbation_list = []
    
    for i in range(pop_size):
        # Generate low-rank factors for this member
        # A_subset: [num_unique, rank], B: [n_embd, rank]
        member_seed = int(seeds[i].item()) + layer_hash
        gen = torch.Generator(device=device)
        gen.manual_seed(member_seed)
        
        # We only need A rows for unique_idx, not all vocab
        A_subset = torch.randn(num_unique, rank, generator=gen, device=device)
        B = torch.randn(n_embd, rank, generator=gen, device=device)
        
        # Compute perturbation for these indices: [num_unique, n_embd]
        delta_subset = (sigma / math.sqrt(rank)) * (A_subset @ B.T)  # [num_unique, n_embd]
        
        # Create index mapping: idx → position in unique_idx
        # This allows us to look up perturbations
        idx_to_unique = torch.zeros(vocab_size, dtype=torch.long, device=device)
        idx_to_unique[unique_idx] = torch.arange(num_unique, device=device)
        
        # Map batch indices to perturbations
        idx_flat_i = idx[i].reshape(-1)  # [batch*seq]
        unique_positions = idx_to_unique[idx_flat_i]  # [batch*seq]
        perturbation_i = delta_subset[unique_positions]  # [batch*seq, n_embd]
        perturbation_i = perturbation_i.view(batch_size, seq_len, n_embd)
        
        perturbation_list.append(perturbation_i)
    
    # Stack and add to base embeddings
    perturbations = torch.stack(perturbation_list)  # [pop, batch, seq, n_embd]
    x = base_emb + perturbations
    
    return x
```

**Memory Analysis**:
- Base embeddings: `O(pop × batch × seq × d)` - same as forward pass
- Unique indices: typically `O(batch × seq)` for natural language (high token diversity)
- Per-member factors: `O(num_unique × r + d × r)` per member
- For `pop=32, batch=8, seq=512, d=4096, r=128, num_unique≈2000`:
  - `A_subset`: 2k × 128 = 256k values per member → 32 × 256k × 4 bytes = 32 MB (fp32)
  - `B`: 4096 × 128 = 524k values per member → 32 × 524k × 4 bytes = 64 MB (fp32)
  - **Total**: ~100 MB vs ~25 GB for naive approach (250× reduction)

**Optimization - Vectorized Index Lookup**:

The above has a per-member loop. For even better performance, vectorize across population:

```python
# Alternative: Generate all low-rank factors at once using chunk-level RNG
layer_hash = stable_hash_name(layer_name)

# Use chunk-level generation (same pattern as _linear_batched_lowrank)
chunk_seed = int(seeds[0].item()) + layer_hash  # Chunk-level seed
gen = torch.Generator(device=device)
gen.manual_seed(chunk_seed)

# Generate for all members: [pop, num_unique, rank] and [pop, n_embd, rank]
A_subset_all = torch.randn(pop_size, num_unique, rank, generator=gen, device=device)
B_all = torch.randn(pop_size, n_embd, rank, generator=gen, device=device)

# Compute perturbations: [pop, num_unique, n_embd]
delta_all = (sigma / math.sqrt(rank)) * torch.einsum('pur,pdr->pud', A_subset_all, B_all)

# Now use advanced indexing to apply perturbations
# This requires careful index mapping (exercise for production implementation)
```

**Practical Recommendations**:

1. **For most users**: Freeze embeddings during ES optimization
   - Embeddings don't need much tuning in transfer learning
   - Avoids this complexity entirely
   - Add `continue` in ES update loop for `'wte'` and `'wpe'` parameter names

2. **For pretraining from scratch**: 
   - Start with frozen embeddings to validate rest of pipeline
   - Add low-rank embedding perturbation as optimization once core is stable
   - Use rank `r=32` or `r=64` for embeddings (lower than linear layers)

3. **Hybrid approach**:
   - Use low-rank for `wpe` (smaller: max_seq × d)
   - Freeze `wte` (much larger: vocab × d)
   - Position embeddings are more "learning-critical" than token embeddings

**ES Update for Low-Rank Embeddings**:

The update follows the same pattern as 2D low-rank parameters:

```python
# In es_update_vectorized, for embedding parameters:
if name == 'transformer.wte.weight' or name == 'transformer.wpe.weight':
    # Use same low-rank update as linear layers
    # See Section 3.2 for full implementation
    m, n = p.shape  # [vocab_size or max_seq_len, n_embd]
    
    # Initialize accumulators
    A_update = torch.zeros(m, rank, device=device)
    B_update = torch.zeros(n, rank, device=device)
    
    # Process in chunks
    for chunk_start in range(0, N, update_chunk_size):
        chunk_end = min(chunk_start + update_chunk_size, N)
        seeds_chunk = seeds[chunk_start:chunk_end]
        norm_fit_chunk = norm_fit[chunk_start:chunk_end]
        chunk_size = chunk_end - chunk_start
        
        # Generate same noise as forward pass
        chunk_seed = int(seeds_chunk[0].item()) + layer_hash
        gen = torch.Generator(device=device)
        gen.manual_seed(chunk_seed)
        
        A_chunk = torch.randn(chunk_size, m, rank, generator=gen, device=device)
        B_chunk = torch.randn(chunk_size, n, rank, generator=gen, device=device)
        
        # Weight by fitness
        A_scaled = A_chunk * norm_fit_chunk.view(chunk_size, 1, 1)
        
        # Accumulate using einsum
        A_update.add_(torch.einsum('cmi,cni->mi', A_scaled, B_chunk))
    
    # Apply ES gradient
    scaling = lr / (sigma * math.sqrt(rank) * N)
    p.data.add_(A_update @ B_update.T, alpha=scaling)
```

---

**LayerNorm Parameters**:

⚠️ **Note**: LayerNorm parameters use per-member RNG (not chunk-level like 2D params). This is acceptable because:
- LayerNorm parameters are small (2 × n_embd per layer)
- Total LayerNorm params << 1% of model
- Loop overhead is negligible vs matmul operations

```python
# In _forward_block_batched:
# LayerNorm has weight (gamma) and bias (beta), both shape [n_embd]

# Option A: Apply per-member LayerNorm (cleaner but requires manual implementation)
ln_weight = block.ln_1.weight  # [n_embd]
ln_bias = block.ln_1.bias  # [n_embd]

x_norm_list = []
for i in range(pop_size):
    # Generate noise for this member (per-member RNG acceptable here)
    seed_w = int(seeds[i].item()) + stable_hash_name(f'block{block_idx}.ln_1.weight')
    seed_b = int(seeds[i].item()) + stable_hash_name(f'block{block_idx}.ln_1.bias')
    eps_w = generate_fullrank_noise(ln_weight.shape, seed_w, device)
    eps_b = generate_fullrank_noise(ln_bias.shape, seed_b, device)
    
    # Perturbed LayerNorm parameters
    weight_pert = ln_weight + sigma * eps_w
    bias_pert = ln_bias + sigma * eps_b
    
    # Apply LayerNorm with perturbed params
    x_norm_i = F.layer_norm(
        x[i],  # [batch, seq, n_embd]
        normalized_shape=(n_embd,),
        weight=weight_pert,
        bias=bias_pert,
        eps=block.ln_1.eps
    )
    x_norm_list.append(x_norm_i)

x_norm = torch.stack(x_norm_list)  # [pop, batch, seq, n_embd]
```

**Recommendation**: For fine-tuning, consider freezing LayerNorm (add `continue` in ES update for LayerNorm params). Pretrained LayerNorm rarely needs much adjustment.

**Linear Biases**:

⚠️ **Note**: Like LayerNorm, biases use per-member RNG. This is fine because:
- Biases are vectors (out_features), not matrices
- Generate_fullrank_noise overhead is minimal for vectors
- Can easily switch to chunk-level if needed (see 1D parameter update section)

```python
# In _linear_batched_lowrank, after computing y = x @ (W + E)^T:
if bias is not None:
    # Generate per-member bias noise
    bias_perturbed_list = []
    for i in range(pop_size):
        member_seed = int(seeds[i].item()) + stable_hash_name(layer_name + '.bias')
        epsilon_bias = generate_fullrank_noise(bias.shape, member_seed, device)
        bias_pert = bias + sigma * epsilon_bias
        bias_perturbed_list.append(bias_pert)
    
    # Stack and add: [pop, out_features]
    bias_perturbed = torch.stack(bias_perturbed_list)
    # Broadcast over batch and seq dimensions
    y = y + bias_perturbed.view(pop_size, 1, 1, out_features)
```

**Alternative**: For large populations, you can use chunk-level RNG for biases too (same pattern as 1D parameter update in Section 3.2).

**Performance Note**: 
- Perturbing 1D parameters (LayerNorm, biases) adds per-member loop overhead
- This is negligible because:
  - Total 1D parameter count << 2D parameter count (typically < 1% of model)
  - RNG cost for small vectors is ~1-10 μs per member
  - Matmul operations dominate compute time (ms scale)
- **For populations > 10k**: Consider chunk-level RNG for 1D params (see Section 3.2)
- **For most use cases**: Per-member RNG is fine and simpler

#### 3.2. ES Update for 1D Parameters (Vectorized Implementation)

Regenerate the same full-rank noise and apply the fitness-weighted update using vectorized operations:

```python
# In es_update_vectorized:
elif p.ndim == 1:
    # Full-rank update for vectors (vectorized for performance)
    layer_hash = stable_hash_name(name)
    update = torch.zeros_like(p.data)
    
    # Process in chunks (same strategy as 2D parameters)
    for chunk_start in range(0, N, update_chunk_size):
        chunk_end = min(chunk_start + update_chunk_size, N)
        seeds_chunk = seeds[chunk_start:chunk_end]
        norm_fit_chunk = norm_fit[chunk_start:chunk_end]
        chunk_size = chunk_end - chunk_start
        
        # CRITICAL: Derive chunk seed from first member (same as forward pass)
        chunk_seed = int(seeds_chunk[0].item()) + layer_hash
        gen = torch.Generator(device=device)
        gen.manual_seed(chunk_seed)
        
        # Generate noise for entire chunk: [chunk_size, *param_shape]
        epsilon_chunk = torch.randn(chunk_size, *p.shape, generator=gen, device=device)
        
        # Weight by fitness and sum: [chunk_size, ...] * [chunk_size, 1, ...]
        weighted = epsilon_chunk * norm_fit_chunk.view(chunk_size, *([1] * len(p.shape)))
        update.add_(weighted.sum(dim=0))
    
    # Scale by ES formula (no sqrt(rank) for full-rank perturbations)
    scaling = lr / (sigma * N)
    p.data.add_(update, alpha=scaling)
    
    # Apply weight decay
    if weight_decay > 0:
        p.data.mul_(1 - lr * weight_decay)
```

**Key differences from 2D parameters**:
- No `sqrt(rank)` in scaling formula (full-rank noise)
- Noise shape: `[chunk_size, *param_shape]` instead of separate A, B factors
- Same chunk-level RNG strategy for consistency and performance

### 4. Add Batched Population Evaluation Method (`nanochat/gpt.py`)

**CRITICAL**: 
1. Keep the existing `forward()` method unchanged - it's used for inference/evaluation without perturbations
2. **Batch population members as an extra dimension** - do NOT loop sequentially over population
3. Use on-the-fly noise generation - do NOT store all perturbations in memory
4. **For pretraining**: Perturb ALL parameters (2D with low-rank, 1D with full-rank noise)
   - **For fine-tuning**: Optionally perturb only 2D parameters (freeze pretrained embeddings/LayerNorm)

**Batching strategy**:
- Input shape: `[batch_size, seq_len]` → `[population_size, batch_size, seq_len]`
- Activations: `[pop, batch, seq, hidden]` - population as extra batch dimension
- All matmuls become batched operations over population dimension
- Generate noise on-the-fly for each layer during forward pass

**Add** new ES-specific methods:

```python
@torch.inference_mode()  # CRITICAL: No gradients computed, saves VRAM
def evaluate_population(self, idx, targets, population_size, sigma, rank=1, base_seed=0, step=0, 
                       world_size=1, ddp_rank=0, chunk_size=8):
    """
    Evaluate a population of perturbed models (TRAINING ONLY) with CHUNKED batching.
    This method is ONLY used during ES training, not during inference/evaluation.
    
    CRITICAL: Processes population in CHUNKS to avoid OOM. Never materializes
    full [population_size, batch_size, seq_len, vocab_size] tensor at once.
    
    IMPORTANT: Does NOT store perturbations in memory. Uses deterministic seeds
    to regenerate noise on-the-fly during forward pass and again during ES update.
    
    Args:
        idx: Input token indices [batch_size, seq_len]
        targets: Target tokens [batch_size, seq_len]
        population_size: Number of population members to evaluate (must be divisible by world_size)
        sigma: Noise temperature (perturbation scale)
        rank: Low-rank perturbation rank (r in paper, typically 1)
        base_seed: Base random seed for reproducibility
        step: Current training step (for seed coordination)
        world_size: Number of distributed ranks (1 for single-GPU)
        ddp_rank: Current rank ID (0 for single-GPU)
        chunk_size: Process this many population members at once (default 8)
                    CRITICAL: With large vocab (~50k), logits are [chunk, batch, seq, vocab]
                    Memory usage: chunk * batch * seq * vocab * 4 bytes
                    Example: 8 * 8 * 1024 * 50000 * 4 = ~13 GB just for logits!
                    Start conservative (8-16), increase if memory allows
    
    Returns:
        fitnesses: (population_size,) tensor of fitness scores
        seeds: (population_size,) tensor of seeds used (for ES update)
    """
    device = idx.device
    batch_size, seq_len = idx.shape
    
    # CRITICAL: Verify population divides evenly (assertion in helper will catch this)
    # Generate seeds for each population member using centralized helper
    seeds = torch.zeros(population_size, dtype=torch.int64, device=device)
    for i in range(population_size):
        seeds[i] = compute_perturbation_seed(
            base_seed, step, world_size, population_size, ddp_rank, i
        )
    
    # Allocate output fitnesses
    fitnesses = torch.empty(population_size, device=device)
    
    # CRITICAL: Process population in chunks to avoid OOM
    # Never materialize [pop, B, T, vocab] at once!
    for chunk_start in range(0, population_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, population_size)
        chunk_pop_size = chunk_end - chunk_start
        
        # Get seeds for this chunk
        seeds_chunk = seeds[chunk_start:chunk_end]
        
        # Expand inputs to [chunk_size, batch_size, seq_len]
        idx_chunk = idx.unsqueeze(0).expand(chunk_pop_size, -1, -1)
        targets_chunk = targets.unsqueeze(0).expand(chunk_pop_size, -1, -1)
        
        # Forward pass with batched perturbations for this chunk only
        # Shape: [chunk_size, batch_size, seq_len, vocab_size]
        logits_chunk = self._forward_batched_population(idx_chunk, seeds_chunk, sigma, rank)
        
        # Compute loss for each population member in chunk
        # Reshape: [chunk * batch * seq, vocab] and [chunk * batch * seq]
        logits_flat = logits_chunk.reshape(-1, logits_chunk.size(-1))
        targets_flat = targets_chunk.reshape(-1)
        
        # Compute CE loss
        loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        # Reshape back: [chunk, batch, seq]
        loss_per_token = loss_flat.reshape(chunk_pop_size, batch_size, seq_len)
        # Average over batch and sequence: [chunk]
        loss_per_member = loss_per_token.mean(dim=[1, 2])
        
        # Fitness = negative loss
        fitnesses[chunk_start:chunk_end] = -loss_per_member
        
        # Chunk tensors will be freed here, keeping memory bounded
    
    return fitnesses, seeds

@torch.inference_mode()
def _forward_batched_population(self, idx, seeds, sigma, rank):
    """
    Forward pass with batched low-rank perturbations (TRAINING ONLY).
    
    All population members are processed in parallel using batched operations.
    Perturbations are generated on-the-fly using deterministic seeds.
    
    Args:
        idx: Input tokens [population_size, batch_size, seq_len]
        seeds: Seeds for each population member [population_size]
        sigma: Noise temperature
        rank: Low-rank perturbation rank
    
    Returns:
        logits: [population_size, batch_size, seq_len, vocab_size]
    """
    pop_size, batch_size, seq_len = idx.shape
    device = idx.device
    
    # Embedding: [pop, batch, seq, n_embd]
    # NOTE: For pretraining, perturb embeddings (wte, wpe) with full-rank noise
    # See Section 3.1 for complete implementation
    # Code below shows simplified structure using base embeddings
    x = self.transformer.wte(idx)  # [pop, batch, seq, n_embd]
    pos = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, seq]
    x = x + self.transformer.wpe(pos.expand(pop_size, batch_size, -1))
    
    # Transformer blocks with batched perturbations
    for block_idx, block in enumerate(self.transformer.h):
        x = self._forward_block_batched(x, block, block_idx, seeds, sigma, rank)
    
    # Final layer norm
    x = self.transformer.ln_f(x)
    
    # Output projection (LM head)
    logits = self._linear_batched_lowrank(
        x, self.lm_head.weight, None,
        layer_name='lm_head', seeds=seeds, sigma=sigma, rank=rank
    )
    
    return logits

def _forward_block_batched(self, x, block, block_idx, seeds, sigma, rank):
    """
    Forward through one transformer block with batched perturbations.
    
    CRITICAL: This is the most complex part of the implementation. Key challenges:
    1. Shape propagation: [pop, batch, seq, hidden] throughout
    2. Attention mask broadcasting: Must include population dimension
    3. Linear projections: Use _linear_batched_lowrank with correct layer names
    4. Seed coordination: Don't accidentally misalign seeds with population members
    
    Args:
        x: [pop, batch, seq, n_embd]
        block: TransformerBlock module
        block_idx: Index of this block (for seed coordination)
        seeds: [pop] seeds for noise generation
        sigma: Noise temperature
        rank: Low-rank dimension
    
    Returns:
        x: [pop, batch, seq, n_embd]
    """
    pop_size, batch_size, seq_len, n_embd = x.shape
    
    # === Self-Attention with Perturbations ===
    
    # Layer norm (applies independently per population member)
    x_norm = block.ln_1(x)  # [pop, batch, seq, n_embd]
    
    # QKV projection with perturbations
    # CRITICAL: Each population member gets different noise for QKV
    # Layer names must be unique per projection to get different noise
    qkv = self._linear_batched_lowrank(
        x_norm, 
        block.attn.c_attn.weight,  # [3*n_embd, n_embd]
        block.attn.c_attn.bias,
        layer_name=f"block{block_idx}.attn.c_attn",
        seeds=seeds,
        sigma=sigma,
        rank=rank
    )  # [pop, batch, seq, 3*n_embd]
    
    # Split into Q, K, V
    q, k, v = qkv.split(n_embd, dim=-1)  # Each: [pop, batch, seq, n_embd]
    
    # Reshape for multi-head attention
    # CRITICAL: Population dimension must be preserved throughout
    n_head = block.attn.n_head
    head_dim = n_embd // n_head
    
    # Reshape: [pop, batch, seq, n_head, head_dim]
    q = q.view(pop_size, batch_size, seq_len, n_head, head_dim)
    k = k.view(pop_size, batch_size, seq_len, n_head, head_dim)
    v = v.view(pop_size, batch_size, seq_len, n_head, head_dim)
    
    # Transpose for attention: [pop, batch, n_head, seq, head_dim]
    q = q.transpose(2, 3)
    k = k.transpose(2, 3)
    v = v.transpose(2, 3)
    
    # Compute attention scores: Q @ K^T
    # CRITICAL: Use torch.matmul which broadcasts over [pop, batch, n_head]
    att = torch.matmul(q, k.transpose(-2, -1))  # [pop, batch, n_head, seq, seq]
    att = att / math.sqrt(head_dim)
    
    # Apply causal mask
    # CRITICAL: Mask must broadcast over population dimension
    # Causal mask: [1, 1, 1, seq, seq] broadcasts to [pop, batch, n_head, seq, seq]
    if not hasattr(self, '_causal_mask_cache') or self._causal_mask_cache.size(-1) < seq_len:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.view(1, 1, 1, seq_len, seq_len)  # Broadcast shape
        self._causal_mask_cache = mask
    
    causal_mask = self._causal_mask_cache[:, :, :, :seq_len, :seq_len]
    att = att.masked_fill(causal_mask == 0, float('-inf'))
    
    # Softmax (applies per population member, independently)
    att = F.softmax(att, dim=-1)  # [pop, batch, n_head, seq, seq]
    
    # CRITICAL: Do NOT apply dropout during ES training
    # Dropout adds extra randomness not controlled by seeds, breaking reproducibility
    # ES perturbations provide exploration - dropout is redundant and harmful
    # If dropout exists, it should be disabled (model in eval mode or p=0)
    
    # Attention output: attention weights @ V
    y = torch.matmul(att, v)  # [pop, batch, n_head, seq, head_dim]
    
    # Transpose back: [pop, batch, seq, n_head, head_dim]
    y = y.transpose(2, 3)
    
    # Reshape: [pop, batch, seq, n_embd]
    y = y.contiguous().view(pop_size, batch_size, seq_len, n_embd)
    
    # Output projection with perturbations
    y = self._linear_batched_lowrank(
        y,
        block.attn.c_proj.weight,
        block.attn.c_proj.bias,
        layer_name=f"block{block_idx}.attn.c_proj",
        seeds=seeds,
        sigma=sigma,
        rank=rank
    )  # [pop, batch, seq, n_embd]
    
    # Residual connection (base parameters, same for all population members)
    x = x + y
    
    # === MLP with Perturbations ===
    
    # Layer norm
    x_norm = block.ln_2(x)  # [pop, batch, seq, n_embd]
    
    # First MLP layer (expand) with perturbations
    h = self._linear_batched_lowrank(
        x_norm,
        block.mlp.c_fc.weight,
        block.mlp.c_fc.bias,
        layer_name=f"block{block_idx}.mlp.c_fc",
        seeds=seeds,
        sigma=sigma,
        rank=rank
    )  # [pop, batch, seq, 4*n_embd] typically
    
    # Activation (GELU or similar)
    h = block.mlp.act(h)  # Applies independently per population member
    
    # Second MLP layer (project back) with perturbations
    h = self._linear_batched_lowrank(
        h,
        block.mlp.c_proj.weight,
        block.mlp.c_proj.bias,
        layer_name=f"block{block_idx}.mlp.c_proj",
        seeds=seeds,
        sigma=sigma,
        rank=rank
    )  # [pop, batch, seq, n_embd]
    
    # Residual connection
    x = x + h
    
    return x  # [pop, batch, seq, n_embd]

def _linear_batched_lowrank(self, x, weight, bias, layer_name, seeds, sigma, rank):
    """
    Batched linear transformation with low-rank perturbations.
    
    Applies: y = x @ (W + E)^T + b
    where E = (sigma / sqrt(rank)) * A @ B^T
    
    Efficient implementation:
    y = x @ W^T + (sigma / sqrt(rank)) * (x @ B) @ A^T + b
    
    CRITICAL: Uses stable hash for layer offsets (not Python hash())
    and per-call Generator for noise (not global RNG state).
    
    Args:
        x: Input [pop, ..., in_features]
        weight: Base weight [out_features, in_features]
        bias: Optional bias [out_features]
        layer_name: Name for seed coordination
        seeds: [pop] base seeds for each population member
        sigma: Noise temperature
        rank: Low-rank dimension
    
    Returns:
        y: Output [pop, ..., out_features]
    """
    import math
    from nanochat.egroll import generate_lowrank_noise_factors, stable_hash_name
    
    pop_size = x.shape[0]
    out_features, in_features = weight.shape
    device = x.device
    
    # Base transformation: x @ W^T
    # x: [pop, batch, seq, in_features]
    # W: [out_features, in_features]
    # Need to broadcast W across population dimension
    y_base = torch.matmul(x, weight.t())  # [pop, batch, seq, out_features]
    
    # CRITICAL: Use STABLE hash (not Python's hash which is randomized)
    layer_hash = stable_hash_name(layer_name)
    
    scaling = sigma / math.sqrt(rank)
    
    # Generate noise in chunks to balance memory and speed
    # Note: noise_chunk_size is for noise generation, NOT the same as evaluate_population chunk_size
    # - evaluate_population chunk_size: Controls logits memory (typically 8-16)
    # - noise_chunk_size here: Controls A, B generation memory (can be larger, e.g., 64)
    # Since A, B are much smaller than logits (rank << vocab), we can use larger chunks
    noise_chunk_size = min(pop_size, 64)  # Process 64 members at a time for noise
    y_pert = torch.zeros_like(y_base)
    
    for chunk_start in range(0, pop_size, noise_chunk_size):
        chunk_end = min(chunk_start + noise_chunk_size, pop_size)
        chunk_size_actual = chunk_end - chunk_start
        
        # VECTORIZED NOISE GENERATION (RECOMMENDED for production):
        # Create ONE generator per chunk, not per member
        # This is orders of magnitude faster for large populations
        
        # CRITICAL: Derive chunk seed from first member's seed (not just layer_hash)
        # This ensures noise depends on (base_seed, step, member_idx) via compute_perturbation_seed
        seeds_chunk = seeds[chunk_start:chunk_end]
        chunk_seed = int(seeds_chunk[0].item()) + layer_hash
        
        # Create ONE generator for entire chunk
        gen = torch.Generator(device=device)
        gen.manual_seed(chunk_seed)
        
        # Generate noise for ALL members in chunk at once (two big randn calls)
        # [chunk_size, out_features, rank] and [chunk_size, in_features, rank]
        A_chunk = torch.randn(chunk_size_actual, out_features, rank, generator=gen, device=device)
        B_chunk = torch.randn(chunk_size_actual, in_features, rank, generator=gen, device=device)
        
        # Apply perturbation: (x @ B) @ A^T
        # x_chunk: [chunk, batch, seq, in_features]
        # B_chunk: [chunk, in_features, rank]
        x_chunk = x[chunk_start:chunk_end]
        
        # Reshape for batched matmul: [chunk, batch*seq, in_features]
        orig_shape = x_chunk.shape
        x_chunk_flat = x_chunk.reshape(chunk_end - chunk_start, -1, in_features)
        
        # x @ B: [chunk, batch*seq, rank]
        xB = torch.bmm(x_chunk_flat, B_chunk)
        
        # (x @ B) @ A^T: [chunk, batch*seq, out_features]
        xBA = torch.bmm(xB, A_chunk.transpose(1, 2))
        
        # Reshape back and scale
        y_pert[chunk_start:chunk_end] = scaling * xBA.reshape(orig_shape[:-1] + (out_features,))
    
    y = y_base + y_pert
    
    if bias is not None:
        y = y + bias
    
    return y

**Chunk-Level RNG Strategy**:

The implementation uses **chunk-level RNG** for optimal performance:

```python
# Create ONE generator per chunk (amortize Generator overhead)
# CRITICAL: Derive seed from first member in chunk (preserves step/base_seed dependence)
seeds_chunk = seeds[chunk_start:chunk_end]
chunk_seed = int(seeds_chunk[0].item()) + layer_hash
gen = torch.Generator(device=device)  # ONE GENERATOR FOR ENTIRE CHUNK
gen.manual_seed(chunk_seed)
# Two big randn calls instead of many small ones
A_chunk = torch.randn(chunk_size_actual, out_features, rank, generator=gen, device=device)
B_chunk = torch.randn(chunk_size_actual, in_features, rank, generator=gen, device=device)
```

**Key properties**:
- Fully deterministic and step-dependent via `seeds_chunk[0]`
- Chunk seed depends on: `compute_perturbation_seed(base_seed, step, ..., chunk_start)`
- Changing `step` or `base_seed` changes noise (via seeds tensor)
- ~1-10 microseconds per chunk (not per member)
- Pop=262k, chunk=64: ~4096 chunks → ~4-40ms total
- Essential for populations > 1k and temporal accumulation (N_eff=262k)
```

---

#### **⚠️ CRITICAL: RNG Design Tradeoff - Chunk-Level vs Per-Member Seeds**

The current implementation has an important **conceptual inconsistency** between stated goals and actual behavior:

**Stated Goal**: "Counter-based RNG with deterministic per-member seeds"
- Noise should be a pure function of `(member_seed, layer_name)`
- You can regenerate noise for any member from its seed alone
- Reordering, subselecting, or rechunking seeds shouldn't change noise

**Actual Implementation**: Chunk-level RNG where noise depends on position within chunk
- `chunk_seed = int(seeds_chunk[0].item()) + layer_hash`
- All members in a chunk share one generator, noise sequence determines per-member values
- Member `i`'s noise depends on: its individual seed + its position within the current chunk + chunk size

**What This Means**:

✅ **Works correctly for**:
- Fixed evaluation order and chunk size throughout a run
- Forward pass and ES update using identical chunking scheme
- Deterministic given same (`base_seed`, `step`, chunking parameters)

❌ **Breaks if you**:
- Change chunk size between forward and update
- Reorder seeds (e.g., for antithetic sampling, elitism reordering)
- Evaluate a subset of seeds (e.g., for selective re-evaluation)
- Run different chunking on different GPUs in distributed setting
- Want to regenerate noise for member `i` without evaluating all preceding members

**Why Chunk-Level RNG Exists**: **Performance**

Per-member RNG is ~100-1000× slower:
```python
# Per-member (SLOW): Create N generators, N manual_seed calls
for i in range(pop_size):  # pop_size = 262k for temporal accumulation
    gen = torch.Generator(device=device)  # Overhead: ~100-1000 μs per call
    gen.manual_seed(int(seeds[i].item()) + layer_hash)
    A_i = torch.randn(m, rank, generator=gen, device=device)
    B_i = torch.randn(n, rank, generator=gen, device=device)
    # Total: 262k × 1000 μs = 262 seconds PER LAYER (infeasible)

# Chunk-level (FAST): Create N/chunk_size generators
for chunk_start in range(0, pop_size, chunk_size):  # chunk_size = 64
    gen = torch.Generator(device=device)  # Overhead: ~1000 μs per chunk
    gen.manual_seed(...)
    A_chunk = torch.randn(chunk_size, m, rank, generator=gen, device=device)
    B_chunk = torch.randn(chunk_size, n, rank, generator=gen, device=device)
    # Total: (262k/64) × 1 ms = 4 seconds per layer (acceptable)
```

**Three Options Going Forward**:

**Option 1: Accept Chunk-Level Guarantee (RECOMMENDED for now)**

Keep current implementation and **document the constraints**:

- ✅ Deterministic given fixed chunking scheme and seed ordering
- ✅ Forward and update use identical RNG (correct ES gradient)
- ✅ Fast enough for temporal accumulation (N_eff=262k)
- ⚠️ **Constraint**: Never change chunk size within a run
- ⚠️ **Constraint**: Never reorder seeds after compute_perturbation_seed
- ⚠️ **Constraint**: All distributed workers must use identical chunking

**Validation**: Before deploying advanced features (antithetic sampling, seed reordering), ensure they maintain chunk structure.

**Option 2: True Per-Member Seeds (High Cost)**

Make noise a pure function of `(member_seed, layer_name)`:

```python
# In _linear_batched_lowrank and es_update_vectorized
for i in range(chunk_size):
    member_seed = int(seeds_chunk[i].item()) + layer_hash  # Use each member's seed
    gen = torch.Generator(device=device)
    gen.manual_seed(member_seed)
    
    A_chunk[i] = torch.randn(m, rank, generator=gen, device=device)
    B_chunk[i] = torch.randn(n, rank, generator=gen, device=device)
```

**Tradeoffs**:
- ✅ Noise is pure function of member seed (fully flexible)
- ✅ Can reorder, subselect, rechunk freely
- ❌ Still requires per-member loop within chunks (slower)
- ❌ May be too slow for N_eff=262k (needs profiling)

**When to use**: If you need noise reuse, antithetic sampling, or advanced ES algorithms that reorder seeds.

**Option 3: Hybrid - Fast Base + Slow Refinement**

Use chunk-level for initial evaluation, per-member for selective re-evaluation:

```python
def evaluate_population_fast(seeds):  # Chunk-level
    # Fast path for initial fitness evaluation
    return evaluate_population(seeds, chunk_size=64)

def evaluate_population_exact(seeds):  # Per-member
    # Slow but deterministic for elites, antithetic pairs
    return evaluate_population(seeds, chunk_size=1, per_member_rng=True)
```

Use fast path for 99% of population, exact path for top-k elites when needed.

**Current Recommendation**: **Option 1**

The guide currently uses chunk-level RNG because:
1. Performance is essential for large populations
2. Most ES variants don't require seed reordering (evaluate, update, repeat)
3. Can revisit if you implement antithetic sampling or advanced evolution strategies

**If you implement seed reordering later**, you'll need to either:
- Ensure reordering happens BEFORE chunking and is consistent
- Switch to per-member RNG (Option 2) for those cases
- Use hybrid approach (Option 3)

---

**Key implementation points**:
- **Batching**: Population is extra dimension, all ops are batched
- **On-the-fly noise**: Generate using deterministic seeds, don't store
- **Memory**: O(model_size) not O(population × model_size)
- **Chunking**: Process population in chunks for memory management
- **Parallelism**: All population members computed simultaneously
- **Chunk-level RNG**: One generator per layer per chunk (not per member)
  - Essential for production: 100-1000× speedup
  - Amortizes Generator overhead across chunk members

**CRITICAL - Implementation Complexity Warning**:

The `_forward_block_batched()` method is **the most complex and bug-prone part** of this implementation. Key challenges:

1. **Shape Propagation**:
   - Must maintain `[pop, batch, seq, hidden]` throughout entire forward pass
   - Every tensor operation must handle 4D shapes correctly
   - Reshaping for multi-head attention must preserve population dimension
   - Easy to accidentally flatten dimensions and lose population structure

2. **Attention Mask Broadcasting**:
   - Causal mask must broadcast over population dimension
   - Shape: `[1, 1, 1, seq, seq]` → broadcasts to `[pop, batch, n_head, seq, seq]`
   - Wrong broadcasting will cause subtle attention bugs
   - Test carefully that all population members use same causal structure

3. **Layer Name Coordination**:
   - Each linear layer needs unique name for seed coordination
   - E.g., `"block0.attn.c_attn"` vs `"block0.attn.c_proj"` vs `"block0.mlp.c_fc"`
   - Same layer name in different blocks → same noise (bug!)
   - Must include block index in layer names

4. **Seed Alignment**:
   - Seeds tensor `[pop]` must stay aligned with population dimension
   - Don't accidentally shuffle or reorder population members
   - Attention ops (transpose, matmul) must preserve population as dimension 0
   - If population and batch get mixed, seeds won't match noise

5. **BatchNorm/LayerNorm Behavior**:
   - LayerNorm applies independently per population member (correct)
   - Statistics computed per (pop, batch, seq) sample over hidden dim
   - Not pooled across population (each member normalized separately)

**Common Bugs to Watch For**:

```python
# ❌ BAD: Reshaping that loses population dimension
q = q.view(batch_size, seq_len, n_head, head_dim)  # Lost pop!

# ✅ GOOD: Preserve population dimension
q = q.view(pop_size, batch_size, seq_len, n_head, head_dim)

# ❌ BAD: Mask doesn't broadcast over population
mask = torch.tril(torch.ones(seq_len, seq_len))  # Shape [seq, seq]

# ✅ GOOD: Mask broadcasts over population
mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, 1, seq_len, seq_len)

# ❌ BAD: Same layer name for different layers
qkv = self._linear_batched_lowrank(..., layer_name="attn")  # Ambiguous!

# ✅ GOOD: Unique layer names
q = self._linear_batched_lowrank(..., layer_name=f"block{block_idx}.attn.q_proj")

# ❌ BAD: Population and batch dimensions swapped
att = att.transpose(0, 1)  # Now [batch, pop, ...] - seeds misaligned!

# ✅ GOOD: Keep population as dim 0
att = att.transpose(2, 3)  # Transpose within [pop, batch, ...] structure
```

**Testing Strategy for `_forward_block_batched`**:

1. **Single member test**: Run with `pop_size=1`, compare to standard `forward()` without perturbations
2. **Shape test**: Print shapes at every step, verify `[pop, batch, seq, ...]` maintained
3. **Noise test**: Run with `sigma=0`, should give identical results for all population members
4. **Seed test**: Verify different population members get different outputs (when `sigma > 0`)
5. **Mask test**: Verify attention is causal for all population members

### 5. ES Update Rule (`nanochat/egroll.py`)

**Remove** all optimizer logic (AdamW, Muon). **Replace** with vectorized ES update using chunk-level RNG and batched operations.

**Implementation** (`es_update_vectorized`):

```python
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
    import math
    from nanochat.egroll import stable_hash_name
    
    # Normalize fitnesses
    norm_fit = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
    N = len(fitnesses)
    device = fitnesses.device
    
    for name, p in model.named_parameters():
        if p.ndim == 2:  # Matrix parameters
            layer_hash = stable_hash_name(name)
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
                A_chunk = torch.randn(chunk_size, m, rank, generator=gen, device=device)
                B_chunk = torch.randn(chunk_size, n, rank, generator=gen, device=device)
                
                # Scale A by fitness: [chunk_size, m, rank]
                A_scaled = A_chunk * norm_fit_chunk.view(chunk_size, 1, 1)
                
                # Compute batched outer products and sum
                # We want: Σ(fitness_i * A_i @ B_i^T) for this chunk
                # Efficient: reshape and use batched matmul
                
                # Option 1: Loop over chunk (still faster than looping over full population)
                for i in range(chunk_size):
                    update.addmm_(A_scaled[i], B_chunk[i].t(), beta=1.0, alpha=1.0)
                
                # Option 2 (more vectorized): Use einsum
                # update += torch.einsum('cmi,cni->mn', A_scaled, B_chunk)
            
            # Scale and apply update
            scaling = lr / (sigma * math.sqrt(rank) * N)
            p.data.add_(update, alpha=scaling)
            
            # Weight decay
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)
        
        elif p.ndim == 1:
            # FOR PRETRAINING: Update 1D params (full code in Section 3.2)
            # FOR FINE-TUNING: Skip if frozen (uncomment continue below)
            # continue
            
            # Vectorized full-rank update
            layer_hash = stable_hash_name(name)
            update = torch.zeros_like(p.data)
            
            for chunk_start in range(0, N, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, N)
                seeds_chunk = seeds[chunk_start:chunk_end]
                norm_fit_chunk = norm_fit[chunk_start:chunk_end]
                chunk_size_actual = chunk_end - chunk_start
                
                chunk_seed = int(seeds_chunk[0].item()) + layer_hash
                gen = torch.Generator(device=device)
                gen.manual_seed(chunk_seed)
                epsilon_chunk = torch.randn(chunk_size_actual, *p.shape, generator=gen, device=device)
                
                weighted = epsilon_chunk * norm_fit_chunk.view(chunk_size_actual, 1)
                update.add_(weighted.sum(dim=0))
            
            scaling = lr / (sigma * N)
            p.data.add_(update, alpha=scaling)
            
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)
```

**Alternative optimization**: For maximum performance, replace the per-chunk loop with einsum:

```python
# Instead of looping over chunk members with addmm_:
for i in range(chunk_size):
    update.addmm_(A_scaled[i], B_chunk[i].t(), beta=1.0, alpha=1.0)

# Use einsum for fully batched operation:
chunk_update = torch.einsum('cmi,cni->mn', A_scaled, B_chunk)
update.add_(chunk_update)
```

The einsum version provides maximum performance with O(params) Python iterations instead of O((N / chunk_size) × params).

#### 5.2. Accumulation for Temporal Approach

```python
def accumulate_micro_population_vectorized(model, fitnesses, seeds, sigma, rank, 
                                          accumulated_updates, update_chunk_size=256):
    """
    Accumulate fitness-weighted perturbations using chunk-level RNG and batched operations.
    """
    norm_fit = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
    M = len(fitnesses)
    device = fitnesses.device
    
    for name, p in model.named_parameters():
        if p.ndim == 2:
            layer_hash = stable_hash_name(name)
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
                A_chunk = torch.randn(chunk_size, m, rank, generator=gen, device=device)
                B_chunk = torch.randn(chunk_size, n, rank, generator=gen, device=device)
                A_scaled = A_chunk * norm_fit_chunk.view(chunk_size, 1, 1)
                
                # Accumulate using einsum (most efficient)
                chunk_update = torch.einsum('cmi,cni->mn', A_scaled, B_chunk)
                accumulated_updates[name].add_(chunk_update)
    
    return accumulated_updates
```

**Key optimizations**:
1. **Chunking**: Process `update_chunk_size` members at once (e.g., 256)
   - Balances memory and speed
   - Reduces Python iterations significantly
2. **Batched tensors**: Stack A and B into 3D tensors
   - Enables vectorized operations
3. **Einsum**: Compute batched outer products efficiently
   - `einsum('cmi,cni->mn', A, B)` computes Σ A_i @ B_i^T
   - Single optimized kernel

**Memory usage**:
- A_chunk: `[chunk_size, m, rank]` floats
- B_chunk: `[chunk_size, n, rank]` floats  
- For chunk_size=256, rank=1, m=n=4096: ~8 MB per chunk (negligible)

**Performance**: Essential for large populations and temporal accumulation (N_eff=262k).

### 6. Replace Training Loop (`scripts/base_train.py`)

**Remove** the entire backprop-based training step (lines 306-328):
```python
# REMOVE THIS ENTIRE SECTION:
for micro_step in range(grad_accum_steps):
    with autocast_ctx:
        loss = model(x, y)
    loss = loss / grad_accum_steps
    loss.backward()  # ❌ REMOVE
    x, y, dataloader_state_dict = next(train_loader)
# gradient clipping  # ❌ REMOVE
# optimizer.step()    # ❌ REMOVE
model.zero_grad()     # ❌ REMOVE
```

**Replace with** ES training step:

**IMPORTANT DECISION**: What does `grad_accum_steps` mean for ES?

**Option 1** (RECOMMENDED): Treat as "ES updates per logging step"
- Each micro_step is a separate ES update with its own population evaluation
- `grad_accum_steps` controls how many ES updates happen between logging/eval
- Simpler, conceptually cleaner
- Different semantics from gradient accumulation (not accumulating anything)

**Option 2**: Combine populations across micro_steps
- Evaluate population at each micro_step, accumulate fitness/noise across steps
- Single ES update per outer step using combined populations
- Closer to gradient accumulation semantics but more complex

**We use Option 1** (separate ES updates per micro_step):

```python
# ES hyperparameters
population_size = 256  # Start small for testing, scale up (paper uses 262144)
                       # CRITICAL: Must be divisible by ddp_world_size
sigma = 0.01  # Noise temperature: controls perturbation magnitude in forward pass
es_lr = 0.02  # Effective learning rate: tuned step size (independent of sigma)
rank = 1  # Low-rank perturbation rank (r in paper)
weight_decay = 0.0  # L2 regularization (keep from original)
base_seed = 42  # Base random seed for reproducibility
chunk_size = 8  # Process this many population members at once
                # CRITICAL: Logits are [chunk, batch, seq, vocab] - memory intensive!
                # Example: 8 * 8 * 1024 * 50k * 4 bytes = ~13 GB
                # Start conservative (8-16), increase if memory allows

# Rename for clarity: not "gradient accumulation", but "ES updates per outer step"
es_updates_per_step = grad_accum_steps  # Or use a new config variable

# Get distributed info (world_size=1, ddp_rank=0 for single-GPU)
from nanochat.common import get_dist_info
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

# CRITICAL: Verify population divides evenly across ranks
assert population_size % ddp_world_size == 0, \
    f"population_size ({population_size}) must be divisible by world_size ({ddp_world_size})"

# CRITICAL: Put model in eval mode to disable dropout (ES provides exploration)
model.eval()

for micro_step in range(es_updates_per_step):
    # CRITICAL: Use inference_mode to prevent gradient computation (saves VRAM)
    with torch.inference_mode(), autocast_ctx:
        # Evaluate population (no gradients computed, no backward needed!)
        # Returns fitness scores and seeds (NOT full perturbations)
        # CRITICAL: Uses centralized seed computation and chunked evaluation
        fitnesses, seeds = model.evaluate_population(
            x, y, 
            population_size=population_size,
            sigma=sigma,
            rank=rank,
            base_seed=base_seed,
            step=step * es_updates_per_step + micro_step,  # Unique seed per ES update
            world_size=ddp_world_size,
            ddp_rank=ddp_rank,
            chunk_size=chunk_size
        )
    
    # Get current learning rate (with warmup/warmdown)
    lrm = get_lr_multiplier(step)  # Keep LR scheduler
    current_lr = es_lr * lrm
    
    # Compute ES update (no backward pass needed, no gradients)
    # NOTE: This is OUTSIDE inference_mode to allow in-place param.data writes
    # CRITICAL: Uses vectorized update for performance
    es_update_vectorized(model, fitnesses, seeds, sigma, current_lr, rank, weight_decay)
    
    # Prefetch next batch
    x, y, dataloader_state_dict = next(train_loader)
```

**Alternative**: If you want true single-update-per-step semantics like gradient accumulation:

```python
# Accumulate fitnesses and seeds across micro_steps
all_fitnesses = []
all_seeds = []

for micro_step in range(es_updates_per_step):
    with torch.inference_mode(), autocast_ctx:
        fitnesses, seeds = model.evaluate_population(
            x, y, 
            population_size=population_size // es_updates_per_step,  # Split population
            sigma=sigma,
            rank=rank,
            base_seed=base_seed,
            step=step * es_updates_per_step + micro_step
        )
    all_fitnesses.append(fitnesses)
    all_seeds.append(seeds)
    x, y, dataloader_state_dict = next(train_loader)

# Combine populations
combined_fitnesses = torch.cat(all_fitnesses)
combined_seeds = torch.cat(all_seeds)

# Single ES update using combined populations
lrm = get_lr_multiplier(step)
current_lr = es_lr * lrm
es_update_vectorized(model, combined_fitnesses, combined_seeds, sigma, current_lr, rank, weight_decay)
```

**IMPORTANT NOTES**: 
- `torch.inference_mode()` during population evaluation prevents gradient computation (saves VRAM)
- `es_update_vectorized()` decorated with `@torch.no_grad()` does in-place `param.data` writes (safe, no autograd)
- No `loss.backward()` - ES doesn't use gradients at all
- No `model.zero_grad()` - no gradients to zero
- Model parameters updated directly via `param.data.add_()` or `param.data.mul_()`
- Standard `model.forward()` remains unchanged for inference (no perturbations)

**Key replacements**:
- **Remove**: `loss.backward()` - ES doesn't compute gradients
- **Remove**: Gradient clipping - not needed without gradients
- **Remove**: Optimizer state (AdamW/Muon) - ES is stateless
- **Remove**: All gradient computation - use `torch.inference_mode()` to save VRAM
- **Replace**: Training forward pass → `evaluate_population()` (with perturbations, batched)
- **Replace**: Optimizer step → Direct parameter update via ES rule (no gradients)
- **Keep**: Standard `forward()` method unchanged - used for inference
- **Keep**: Learning rate scheduler (`get_lr_multiplier()`) - apply to ES learning rate
- **Keep**: Weight decay - apply as AdamW-style decoupled weight decay during ES update
- **Clarify**: `grad_accum_steps` → `es_updates_per_step` (different semantics)

### 7. Scaling to Large Populations: Temporal Accumulation

**Challenge**: Hardware may not support 262K population members in one step.

**Solution**: Use **temporal population accumulation** - spread evaluation across multiple steps.

#### 7.1. Concept: Effective Population Size (N_eff)

Instead of evaluating all 262K members simultaneously, accumulate fitness-weighted perturbations across multiple "micro-populations":

**Traditional approach** (memory intensive):
- Evaluate N = 262,144 members in one step
- Compute ES update: `W += (lr / (sigma * sqrt(r) * N)) * Σ fitness_i * A_i @ B_i^T`

**Temporal accumulation** (memory efficient):
- Evaluate K micro-populations of size M each (e.g., K=128, M=2048)
- Accumulate: `sum += Σ fitness_i * A_i @ B_i^T` for each micro-population
- After K micro-populations: `N_eff = K * M = 262,144`
- Apply single ES update: `W += (lr / (sigma * sqrt(r) * N_eff)) * sum`

**Key insight**: ES update is just a sum - we can compute it incrementally!

#### 7.2. Implementation: Micro-Population Accumulation

```python
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
    import math
    from nanochat.egroll import stable_hash_name
    
    for name, p in model.named_parameters():
        if p.ndim == 2 and name in accumulated_updates:
            # Scale accumulated update by ES formula
            scaling = lr / (sigma * math.sqrt(rank) * N_total)
            p.data.add_(accumulated_updates[name], alpha=scaling)
            
            # Apply weight decay
            if weight_decay > 0:
                p.data.mul_(1 - lr * weight_decay)
        
        elif p.ndim == 1:
            # FOR PRETRAINING: Update 1D params (see Section 3.2)
            # FOR FINE-TUNING: Skip if frozen (uncomment continue below)
            # continue
            
            if name in accumulated_updates:
                scaling = lr / (sigma * N_total)
                p.data.add_(accumulated_updates[name], alpha=scaling)
                
                if weight_decay > 0:
                    p.data.mul_(1 - lr * weight_decay)

#### 7.3. Training Loop with Temporal Accumulation

```python
# ES hyperparameters
target_population = 262144      # Target effective population size
micro_population_size = 2048    # Per-step population (hardware-limited)
K = target_population // micro_population_size  # Number of micro-populations (128)

sigma = 0.01
es_lr = 0.02
rank = 1
weight_decay = 0.0
base_seed = 42
chunk_size = 8

# Accumulation buffers
accumulated_updates = {}
N_total = 0

# Get distributed info
ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
model.eval()

for step in range(max_steps):
    # Reset accumulation for this step
    accumulated_updates = {}
    N_total = 0
    
    # Accumulate over K micro-populations
    for micro_idx in range(K):
        # Evaluate one micro-population
        with torch.inference_mode(), autocast_ctx:
            fitnesses, seeds = model.evaluate_population(
                x, y,
                population_size=micro_population_size,
                sigma=sigma,
                rank=rank,
                base_seed=base_seed,
                step=step * K + micro_idx,  # Unique seed per micro-population
                world_size=ddp_world_size,
                ddp_rank=ddp_rank,
                chunk_size=chunk_size
            )
        
        # Accumulate fitness-weighted perturbations (vectorized for performance)
        accumulate_micro_population_vectorized(
            model, fitnesses, seeds, sigma, rank, accumulated_updates
        )
        N_total += len(fitnesses)
        
        # Prefetch next batch for next micro-population
        x, y, dataloader_state_dict = next(train_loader)
    
    # Apply single ES update using accumulated perturbations
    lrm = get_lr_multiplier(step)
    current_lr = es_lr * lrm
    es_update_with_accumulation(
        model, accumulated_updates, N_total, sigma, current_lr, rank, weight_decay
    )
    
    # Now N_total = 262144, one ES update applied
```

#### 7.4. Benefits and Trade-offs

**Benefits**:
- ✅ Get statistical benefits of large population (N_eff = 262k)
- ✅ Memory requirements of small population (M = 2k)
- ✅ No hardware changes needed
- ✅ Same ES estimator mathematically
- ✅ Can tune K and M independently

**Trade-offs**:
- ⚠️ More forward passes per update (K times)
- ⚠️ More data consumed per update (K batches)
- ⚠️ Slightly more complex implementation

**When to use**:
- Hardware can't handle large population in one step
- Want 262k population but have limited memory
- Training time is acceptable (K forward passes per update)

**When not to use**:
- Hardware can handle full population comfortably
- Want maximum speed (fewer forward passes)
- Data is limited (each update sees K batches instead of 1)

#### 7.5. Distributed + Temporal Accumulation

Combine both strategies for maximum scale:

```python
# Each of 8 GPUs evaluates micro_population_size // 8 members
# Accumulate across K micro-populations
# Total effective population: K * micro_population_size = 262k

# Benefits:
# - 8 GPUs: 8x faster per micro-population
# - Temporal: Can reach 262k effective population
# - Combined: Fast AND large population
```

### 8. Batching and Chunking Strategy

**Challenge**: Evaluating even micro-populations requires careful memory management.

**Solution**: Batch population as extra dimension, with optional chunking:

#### 8.1. Full Batching (If Memory Allows)
```python
# Input: [batch_size, seq_len]
# Expand to: [population_size, batch_size, seq_len]
# All activations: [pop, batch, seq, hidden]
# All operations batched over population dimension
```

**Pros**: Maximum speed (all members in parallel)
**Cons**: May OOM for large populations

#### 8.2. Chunked Batching (Memory-Efficient)
```python
# CRITICAL: Start with small chunk_size due to vocab dimension
# Logits memory: chunk_size * batch_size * seq_len * vocab_size * 4 bytes
# Example: 8 * 8 * 1024 * 50000 * 4 = ~13 GB just for logits!
chunk_size = 8  # Conservative default, increase if memory allows

for chunk_start in range(0, population_size, chunk_size):
    chunk_end = min(chunk_start + chunk_size, population_size)
    chunk_seeds = seeds[chunk_start:chunk_end]
    
    # Process this chunk in batched mode
    chunk_fitnesses = model._evaluate_chunk(idx, targets, chunk_seeds, sigma, rank)
    fitnesses[chunk_start:chunk_end] = chunk_fitnesses
```

**Pros**: Bounded memory usage
**Cons**: Slower than full batching (but still much faster than sequential)

**Memory calculation**:
- Logits: `chunk_size × batch_size × seq_len × vocab_size × 4` bytes
- Activations: `chunk_size × batch_size × seq_len × hidden_dim × 4` bytes (multiple layers)
- Chunk size 8 is conservative for vocab ~50k
- Chunk size 16-32 may work for smaller vocab or more memory
- Monitor GPU memory usage and adjust accordingly

#### 8.3. Low-Rank Linear Forward Pattern

**Efficient implementation** avoids materializing full perturbation:
```python
# For each linear layer:
# Perturbed output = x @ (W + E)^T where E = (sigma/sqrt(r)) * A @ B^T
# Efficient form: x @ W^T + (sigma/sqrt(r)) * (x @ B) @ A^T

# Batched over population:
# x: [pop, batch, seq, in_features]
# W: [out_features, in_features] (base parameter, same for all pop members)
# B: [pop, in_features, rank] (generated per-member)
# A: [pop, out_features, rank] (generated per-member)

# Step 1: Base transform (broadcast W across population)
y_base = x @ W.T  # [pop, batch, seq, out_features]

# Step 2: Perturbation
# Reshape x for batched matmul: [pop, batch*seq, in_features]
x_flat = x.reshape(pop, -1, in_features)
# x @ B: [pop, batch*seq, rank]
xB = torch.bmm(x_flat, B)
# (x @ B) @ A^T: [pop, batch*seq, out_features]
pert_flat = torch.bmm(xB, A.transpose(1, 2))
# Reshape and scale
pert = (sigma / sqrt(rank)) * pert_flat.reshape(pop, batch, seq, out_features)

# Final output
y = y_base + pert  # [pop, batch, seq, out_features]
```

**Key insight**: Base parameters `W` are shared across population (broadcast), only noise factors `A`, `B` differ per member

#### 8.4. Memory Optimization: Generate Noise in Chunks

Even with batched forward, generating noise for all population members at once may OOM:
```python
# Instead of:
# A = [population_size, out_features, rank]  # May be huge!

# Use chunked generation:
for chunk in chunks:
    # Generate A, B for chunk_size members only
    # Apply perturbations for this chunk
    # Discard noise after use
```

See `_linear_batched_lowrank()` implementation in Section 3 for full example.

### 9. Dropout and Other Stochasticity During ES Training

**CRITICAL**: Disable dropout and other random layers during ES training.

**The Problem**:
- ES uses deterministic noise controlled by seeds
- Forward pass should be reproducible: same seed → same fitness
- Dropout adds **extra randomness** not tied to seeds
- This breaks seed-based reproducibility and adds unwanted variance

**Why this matters**:
1. **Reproducibility**: Can't regenerate exact forward pass from seeds alone
2. **Variance**: Dropout variance adds to ES estimation variance (worse signal-to-noise)
3. **ES semantics**: ES perturbations already provide exploration/regularization

**Solution**: Put model in eval mode or disable dropout

**Option 1 (Recommended)**: Use eval mode during ES training
```python
# During ES training (in base_train.py)
model.eval()  # Disables dropout, batchnorm updates, etc.

with torch.inference_mode(), autocast_ctx:
    fitnesses, seeds = model.evaluate_population(...)
```

**Option 2**: Keep dropout p=0 in config
```python
# In model config
dropout = 0.0  # No dropout for ES training
```

**Note**: Standard `forward()` (for inference) can still use model in eval mode as usual.

### 10. Inference and Evaluation (No Perturbations)

**CRITICAL**: During inference/evaluation, perturbations are NOT applied.

- **Use**: Standard `model.forward()` method (unchanged)
- **Do NOT use**: `evaluate_population()` during inference
- **Behavior**: Model uses base parameters only, no perturbations
- **Example**: 
  ```python
  # During inference/evaluation:
  model.eval()
  with torch.inference_mode():
      logits = model(idx)  # Uses base parameters, no perturbations
      loss = model(idx, targets)  # Standard loss computation
  ```

### 11. Fitness Function and Variance Reduction Techniques

**Current**: Uses cross-entropy loss (lower is better).

**EGGROLL**: Uses fitness (higher is better).

**Implementation**:
```python
# Compute loss for each population member
loss = F.cross_entropy(logits, targets, reduction='mean')  # per member
fitness = -loss  # Negative loss = fitness

# Normalize fitnesses (baseline subtraction and variance scaling)
normalized_fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-8)
```

**Advanced techniques** (optional optimizations):

#### 11.1. Antithetic Sampling
**What**: Evaluate both `θ + ε` and `θ - ε` for each noise sample `ε`
**Why**: Massive variance reduction (often 2-4x improvement)
**Implementation**:
```python
# Instead of N independent samples, use N/2 antithetic pairs
# For each noise ε_i, evaluate:
#   - Member 2i: θ + σ * ε_i
#   - Member 2i+1: θ - σ * ε_i
# Both use the same noise (just flipped sign)
# This cancels out many sources of noise variance
```
**Status**: Optional optimization, can add if needed

#### 11.2. Rank-Based Fitness Shaping
**What**: Use fitness ranks instead of raw fitness values
**Why**: More robust to outliers and fitness scale
**Implementation**:
```python
# Instead of:
normalized_fitness = (fitness - fitness.mean()) / fitness.std()

# Use ranks:
ranks = torch.argsort(torch.argsort(fitness))  # ranks in [0, N-1]
# Map to centered distribution
normalized_fitness = (ranks.float() - (N-1)/2) / N
# Or use utility transformation: U(rank) = max(0, log(N/2 + 1) - log(rank + 1))
```
**Status**: Optional alternative, often more stable than z-scoring

#### 11.3. Noise Reuse ES
**What**: Reuse the same noise perturbations across multiple batches/sequences
**Why**: Amortize population evaluation cost, better signal-to-noise on objective
**How**: EGGROLL paper uses this for long sequences (evaluate same perturbed model on multiple batches)
**Implementation**:
```python
# Evaluate each perturbed model on K different batches
# Fitness = average loss across K batches
# Update uses fitness from all K batches
# Effective population: N models × K batches
```
**Status**: Not in initial implementation, but critical for scaling to large contexts

**Recommendation**: 
- Start with z-scored fitness (`(f - mean) / std`)
- Add antithetic sampling, rank-based shaping, and noise reuse as optimizations
- These are valuable but not required for correctness

### 12. Distributed Training Support

**Current**: Uses DDP with gradient synchronization.

**EGGROLL**: Needs population evaluation across ranks with coordinated random seeds.

**Changes**:

#### 12.1. Seed Coordination Across Ranks and Steps

**CRITICAL**: Proper seed coordination ensures:
1. Each rank generates different perturbations (no duplicate work)
2. Each step uses different perturbations (no noise reuse across steps unless intended)
3. Reproducibility when resuming from checkpoints

**Seed formula**:
```python
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
```

**Example**: 
- `base_seed=42`, `step=0`, `world_size=8`, `population_size=262144`, `ddp_rank=0`, `member_idx=0`
- Result: `seed = 42 + 0*262144 + 0*32768 + 0 = 42`
- DDP rank 1, member 0: `seed = 42 + 0*262144 + 1*32768 + 0 = 32810`
- Step 1, DDP rank 0, member 0: `seed = 42 + 1*262144 + 0 = 262186`

**Note on population_size**: Must be divisible by world_size. The helper asserts this.
- Good: `population_size=256, world_size=8` → 32 per rank
- Good: `population_size=262144, world_size=8` → 32768 per rank
- Bad: `population_size=100, world_size=8` → assertion error

#### 12.2. Each Rank Evaluates a Subset of the Population

**Implementation**:
- Split the population across ranks: `population_per_rank = population_size // world_size`
- Each rank evaluates its assigned subset with coordinated seeds
- Use batched evaluation (NOT sequential loop)
- Example:
  ```python
  import torch.distributed as dist
  from nanochat.common import get_dist_info
  
  ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
  
  @torch.inference_mode()
  def evaluate_population_distributed(model, idx, targets, population_size, sigma, rank, 
                                     base_seed, step, world_size, ddp_rank, chunk_size=8):
      """
      Evaluate population subset on this rank (BATCHED with CHUNKING).
      
      Each rank evaluates population_size // world_size members.
      Uses centralized compute_perturbation_seed to ensure coordination.
      
      CRITICAL: Same signature as single-GPU evaluate_population, just with
      different world_size and ddp_rank parameters.
      """
      # Calculate this rank's subset
      population_per_rank = population_size // world_size
      
      # CRITICAL: Use centralized seed computation (same as single-GPU)
      local_seeds = torch.zeros(population_per_rank, dtype=torch.int64, device=idx.device)
      for i in range(population_per_rank):
          local_seeds[i] = compute_perturbation_seed(
              base_seed, step, world_size, population_size, ddp_rank, i
          )
      
      # Allocate output
      local_fitnesses = torch.empty(population_per_rank, device=idx.device)
      
      # CRITICAL: Chunk population to avoid OOM (same chunking as single-GPU)
      for chunk_start in range(0, population_per_rank, chunk_size):
          chunk_end = min(chunk_start + chunk_size, population_per_rank)
          chunk_pop_size = chunk_end - chunk_start
          
          seeds_chunk = local_seeds[chunk_start:chunk_end]
          idx_chunk = idx.unsqueeze(0).expand(chunk_pop_size, -1, -1)
          targets_chunk = targets.unsqueeze(0).expand(chunk_pop_size, -1, -1)
          
          logits_chunk = model._forward_batched_population(idx_chunk, seeds_chunk, sigma, rank)
          
          # Compute loss for chunk
          logits_flat = logits_chunk.reshape(-1, logits_chunk.size(-1))
          targets_flat = targets_chunk.reshape(-1)
          loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none')
          loss_per_token = loss_flat.reshape(chunk_pop_size, idx.size(0), idx.size(1))
          loss_per_member = loss_per_token.mean(dim=[1, 2])
          
          local_fitnesses[chunk_start:chunk_end] = -loss_per_member
      
      return local_fitnesses, local_seeds
  ```

**Benefits**:
- Linear scaling: 8 GPUs = 8x faster population evaluation
- Memory efficient: Each rank only handles `population_size / world_size` members
- Parallel evaluation: All ranks work simultaneously
- No stored perturbations: Seeds are lightweight (one int per member)

#### 12.3. Synchronize Fitness Scores Across Ranks

**Implementation**:
- Use `torch.distributed.all_gather()` to collect fitness scores from all ranks
- Fitness scores are lightweight (one scalar per population member)
- All ranks need full fitness distribution for proper normalization
- Example:
  ```python
  def synchronize_fitnesses(local_fitnesses, world_size):
      """
      Gather fitness scores from all ranks.
      
      All ranks need the full fitness distribution to compute proper
      fitness normalization (mean and std across entire population).
      
      Args:
          local_fitnesses: (population_per_rank,) tensor on each rank
          world_size: Number of ranks
      
      Returns:
          all_fitnesses: (population_size,) tensor with fitnesses from all ranks
      """
      # Create list to hold gathered tensors
      gathered_fitnesses = [torch.zeros_like(local_fitnesses) for _ in range(world_size)]
      
      # Gather fitness scores from all ranks
      dist.all_gather(gathered_fitnesses, local_fitnesses)
      
      # Concatenate into single tensor
      all_fitnesses = torch.cat(gathered_fitnesses, dim=0)
      
      return all_fitnesses
  ```

**Alternative (more efficient)**: Use `torch.distributed.all_gather_into_tensor()`:
```python
# Pre-allocate output tensor
all_fitnesses = torch.zeros(population_size, device=local_fitnesses.device, dtype=local_fitnesses.dtype)
# Gather into pre-allocated tensor (single kernel, more efficient)
dist.all_gather_into_tensor(all_fitnesses, local_fitnesses)
```

**Why gather fitnesses**: 
- Fitness normalization requires global mean/std across all population members
- Communication cost is small: `population_size * sizeof(float)` bytes
- For 262k population: ~1 MB of data (negligible)

#### 12.4. Aggregate Updates Across Ranks (Using Seeds, Not Stored Perturbations)

**Implementation strategy**:
1. Each rank computes its local contribution to the parameter update
2. Use `all_reduce` with `SUM` to combine contributions from all ranks
3. Regenerate noise on-the-fly using seeds (don't store perturbations)

**Efficient approach**:
```python
@torch.no_grad()
def es_update_distributed(model, all_fitnesses, local_seeds, sigma, lr, rank, weight_decay, ddp_rank, ddp_world_size):
    """
    Compute ES update in distributed setting.
    
    Strategy: Each rank computes its local fitness-weighted noise contribution,
    then we sum contributions across ranks (equivalent to averaging over full population).
    
    Uses on-the-fly noise regeneration from seeds (no stored perturbations).
    
    CRITICAL: Uses stable hash and per-call Generator (not global RNG state).
    
    Args:
        model: Model to update
        all_fitnesses: (population_size,) global fitnesses from all_gather
        local_seeds: (population_per_rank,) seeds used by this rank
        sigma: Noise temperature
        lr: Learning rate
        rank: Low-rank perturbation rank
        weight_decay: Weight decay coefficient
        ddp_rank: This rank's ID
        ddp_world_size: Total number of ranks
    """
    import math
    import torch.distributed as dist
    from nanochat.egroll import generate_lowrank_noise_factors, generate_fullrank_noise, stable_hash_name
    
    # Normalize fitnesses using global statistics
    normalized_fitnesses = (all_fitnesses - all_fitnesses.mean()) / (all_fitnesses.std() + 1e-8)
    
    # Extract local fitnesses for this rank's population members
    population_per_rank = len(local_seeds)
    start_idx = ddp_rank * population_per_rank
    local_fitnesses = normalized_fitnesses[start_idx:start_idx + population_per_rank]
    
    device = local_fitnesses.device
    N = len(all_fitnesses)  # Total population size
    
    # For each parameter, compute local contribution and all_reduce
    for param_name, param in model.named_parameters():
        # CRITICAL: Use STABLE hash (not Python's hash which is randomized)
        layer_hash = stable_hash_name(param_name)
        
        if param.ndim == 2:  # Matrix parameters: low-rank perturbations
            m, n = param.shape
            
            # VECTORIZED: Process local population in chunks (same as single-GPU es_update_vectorized)
            # This is CRITICAL for performance with large populations
            local_update = torch.zeros_like(param.data)
            
            # Use same chunk size strategy as single-GPU update
            update_chunk_size = min(population_per_rank, 64)  # Balance memory and speed
            
            # Accumulate A and B factors (for einsum-based update)
            A_update = torch.zeros(m, rank, device=device)
            B_update = torch.zeros(n, rank, device=device)
            
            for chunk_start in range(0, population_per_rank, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, population_per_rank)
                chunk_size = chunk_end - chunk_start
                
                # Get chunk of seeds and fitnesses
                seeds_chunk = local_seeds[chunk_start:chunk_end]
                local_fit_chunk = local_fitnesses[chunk_start:chunk_end]
                
                # VECTORIZED: Generate noise for entire chunk with ONE generator
                # CRITICAL: Derive chunk seed from first member's seed (same as forward pass)
                chunk_seed = int(seeds_chunk[0].item()) + layer_hash
                gen = torch.Generator(device=device)
                gen.manual_seed(chunk_seed)
                
                # Two big randn calls instead of many small ones
                # [chunk_size, m, rank] and [chunk_size, n, rank]
                A_chunk = torch.randn(chunk_size, m, rank, generator=gen, device=device)
                B_chunk = torch.randn(chunk_size, n, rank, generator=gen, device=device)
                
                # Scale A by fitness: [chunk_size, m, rank]
                A_scaled = A_chunk * local_fit_chunk.view(chunk_size, 1, 1)
                
                # Accumulate using einsum (most efficient)
                # Computes: Σ(chunk) fitness_i * A_i @ B_i^T
                chunk_update = torch.einsum('cmi,cni->mn', A_scaled, B_chunk)
                local_update.add_(chunk_update)
            
            # All-reduce: sum local contributions from all ranks
            # After all_reduce, update contains: Σ(all ranks) Σ(local) f_i A_i B_i^T
            dist.all_reduce(local_update, op=dist.ReduceOp.SUM)
            
            # Scale by ES formula: (lr / (sigma * sqrt(rank) * N))
            scaling = lr / (sigma * math.sqrt(rank) * N)
            param.data.add_(local_update, alpha=scaling)
            
            # Apply weight decay
            if weight_decay > 0:
                param.data.mul_(1 - lr * weight_decay)
        
        elif param.ndim == 1:  # Vector parameters (bias, LayerNorm, embeddings)
            # FOR PRETRAINING: Update 1D parameters with full-rank noise
            # FOR FINE-TUNING: Skip if frozen (uncomment continue below)
            # continue
            
            # VECTORIZED: Process local population in chunks (same as single-GPU es_update_vectorized)
            local_update = torch.zeros_like(param.data)
            
            # Use same chunk size strategy as 2D params
            update_chunk_size = min(population_per_rank, 64)
            
            for chunk_start in range(0, population_per_rank, update_chunk_size):
                chunk_end = min(chunk_start + update_chunk_size, population_per_rank)
                chunk_size = chunk_end - chunk_start
                
                # Get chunk of seeds and fitnesses
                seeds_chunk = local_seeds[chunk_start:chunk_end]
                local_fit_chunk = local_fitnesses[chunk_start:chunk_end]
                
                # VECTORIZED: Generate noise for entire chunk with ONE generator
                # CRITICAL: Derive chunk seed from first member's seed (same as forward pass)
                chunk_seed = int(seeds_chunk[0].item()) + layer_hash
                gen = torch.Generator(device=device)
                gen.manual_seed(chunk_seed)
                
                # Generate noise for entire chunk: [chunk_size, *param_shape]
                epsilon_chunk = torch.randn(chunk_size, *param.shape, generator=gen, device=device)
                
                # Weight by fitness and sum: [chunk_size, ...] * [chunk_size, 1, ...]
                weighted = epsilon_chunk * local_fit_chunk.view(chunk_size, *([1] * len(param.shape)))
                local_update.add_(weighted.sum(dim=0))
            
            # All-reduce: sum contributions
            dist.all_reduce(local_update, op=dist.ReduceOp.SUM)
            
            # Scale by ES formula (no sqrt(rank) for full-rank)
            scaling = lr / (sigma * N)
            param.data.add_(local_update, alpha=scaling)
            
            # Apply weight decay
            if weight_decay > 0:
                param.data.mul_(1 - lr * weight_decay)
```

**Performance Notes**:

⚠️ **CRITICAL**: This implementation uses **chunked vectorized updates**, not per-member loops!

For large populations (e.g., 262k split over 8 GPUs = 32k per rank):
- **Per-member loop** (SLOW): 32k iterations × (generator creation + 2 randn + matmul) per parameter
  - Generator overhead: ~100-1000 μs per call
  - Total per parameter: 32k × 1ms = **32 seconds** (infeasible!)
- **Chunked vectorized** (FAST): (32k / 64) = 500 chunks
  - Total per parameter: 500 × 1ms = **0.5 seconds** (64× speedup)

The vectorized approach:
- Generates `A_chunk`, `B_chunk` as `[chunk_size, m, rank]`, `[chunk_size, n, rank]`
- Uses single `einsum('cmi,cni->mn', A_scaled, B_chunk)` instead of many `addmm_` calls
- Same pattern as single-GPU `es_update_vectorized` (Section 5)
- Essential for temporal accumulation with large effective populations

**Why SUM instead of AVG**:
- We sum local contributions (each rank has `population_size / world_size` members)
- Total after all_reduce: `Σ(all) f_i * noise_i` = sum over entire population
- Then we scale by `1/N` where `N = population_size` (total, not per-rank)
- Equivalent to computing full update and then dividing by total population size

**Communication cost**:
- All-reduce one tensor per parameter (same as DDP gradient sync)
- Same communication pattern as gradient-based training
- No need to communicate perturbations (regenerate from seeds)

**Key differences from gradient-based DDP**:
- **Gradients**: DDP averages gradients → ES sums fitness-weighted noise contributions
- **Synchronization**: `all_reduce(gradients, AVG)` → `all_reduce(noise_updates, SUM)` + `all_gather(fitnesses)`
- **Memory**: DDP stores gradients → ES stores only seeds, regenerates noise on-the-fly

### 13. Replace Checkpointing Logic

**Remove**: Optimizer state saving/loading (AdamW momentum, Muon buffers).

**Replace with**: ES-compatible checkpointing:
- **Remove**: `optimizer_data` from checkpoint save/load (ES is stateless)
- **Keep**: Model parameters only
- **Add**: Save `base_seed` and current `step` for reproducibility
  - Allows deterministic replay of exact noise sequence when resuming
  - Critical for distributed training (ensures all ranks use same seeds)

**Example**:
```python
# Save checkpoint
checkpoint = {
    'model': model.state_dict(),
    'step': step,
    'base_seed': base_seed,
    'config': config,
    # NO optimizer state needed
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model'])
step = checkpoint['step']
base_seed = checkpoint['base_seed']
```

### 14. Replace Hyperparameters

**Remove** these hyperparameters entirely:
- `embedding_lr`, `unembedding_lr`, `matrix_lr` → **REMOVE** (replaced by single `es_lr`)
- `grad_clip` → **REMOVE** (no gradients to clip)
- Optimizer-specific params (`momentum`, `betas`, `nesterov`, etc.) → **REMOVE**

**Keep** these hyperparameters (adapt for ES):
- `weight_decay` → **KEEP** - Apply as AdamW-style **decoupled weight decay** during ES update
- `warmup_ratio`, `warmdown_ratio` → **KEEP** - Adapt LR scheduler for ES
- `final_lr_frac` → **KEEP** - For LR warmdown

**Add** ES hyperparameters:
- `population_size`: Number of population members (paper uses 2^18 = 262144, may need to reduce for memory)
- `sigma`: Noise temperature - controls perturbation magnitude during forward pass (paper uses ~0.01)
- `es_lr`: Effective learning rate - tuned step size for parameter updates (independent of sigma)
- `rank`: Low-rank perturbation rank (paper uses 1)
- `base_seed`: Base random seed for deterministic noise generation (e.g., 42)

**Important clarifications**:

#### 14.1. Weight Decay Semantics
```python
# We implement AdamW-style DECOUPLED weight decay:
# W *= (1 - lr * weight_decay)
# This is NOT L2 regularization in the objective

# Equivalent to:
# W -= lr * weight_decay * W

# This is what AdamW does, and it's what we keep for ES
```
**Why decoupled weight decay**: It's what the original training used (AdamW/Muon both do this). We maintain the same regularization behavior.

#### 14.2. `sigma` vs `es_lr` - Independent Hyperparameters

**CRITICAL**: `sigma` and `es_lr` are **independent** and serve different roles:

**`sigma` (noise temperature)**:
- Controls perturbation magnitude during forward pass: `W_perturbed = W + (sigma/sqrt(r)) * A @ B^T`
- Affects how much models differ in the population
- Too large: models are too different, signal is noisy
- Too small: models are too similar, signal is weak
- Typical values: 0.001 - 0.1

**`es_lr` (effective learning rate)**:
- Controls step size in parameter space: `W += es_lr * (...)` 
- Standard learning rate semantics (like SGD/Adam lr)
- Typical values: **NOT comparable to gradient-based LRs** - expect to retune
- May be orders of magnitude different from original LR

**Update formula**:
```python
W += (es_lr / (sigma * sqrt(rank) * N)) * Σ fitness_i * A_i @ B_i^T
```

**Both appear in the formula** and do NOT cancel out:
- The `es_lr / (sigma * sqrt(rank) * N)` scaling is deliberate
- `es_lr` scales the overall step size
- `sigma` normalizes by the perturbation scale used in forward pass
- They must be tuned separately

**Tuning strategy**:
1. Start with `sigma = 0.01` (from paper)
2. Start with `es_lr = ???` (no good heuristic from gradient-based LR)
3. Try `es_lr` in range [0.001, 0.1] and sweep
4. Monitor loss curves and adjust both independently

## Implementation Priority

### Phase 1: Core Implementation
1. **Start small**: Use `population_size = 256` for initial testing
2. **Remove**: `loss.backward()`, optimizer setup, gradient clipping from `base_train.py`
3. **Add**: Batched population evaluation with chunk-level RNG (Section 4)
   - `evaluate_population()` with chunked batched forward
   - On-the-fly noise generation with chunk-level generators
   - `_linear_batched_lowrank()` with vectorized noise generation
4. **Add**: Vectorized ES update rule (Section 5)
   - `es_update_vectorized()` with chunk-level RNG
   - Regenerate noise from seeds during update
5. **Keep**: `forward()` method unchanged - used for inference
6. **Test**: Verify loss decreases with small population on small model
7. **Debug**: Check shapes, memory usage, seeds reproducibility

**Goal**: Production-ready implementation that scales efficiently

### Phase 2: Scale Population Size
1. **Scale up**: Gradually increase per-step population size (256 → 1K → 4K)
2. **Monitor**: Memory usage, speed, loss curves at each scale
3. **Optimize**: Adjust `chunk_size` and `update_chunk_size` for your hardware
4. **Tune**: `sigma` and `es_lr` hyperparameters

**Goal**: Find largest feasible per-step population size for your hardware

### Phase 3: Temporal Accumulation (Optional)
1. **Implement**: Temporal accumulation (Section 7)
   - `accumulate_micro_population_vectorized()` function
   - `es_update_with_accumulation()` function
2. **Configure**: Choose micro-population size M and K
   - M = per-step population (hardware-limited)
   - K = number of micro-populations
   - N_eff = K * M (target 262k or as needed)
3. **Monitor**: Training speed (K× more forward passes per update)

**Goal**: Achieve paper-scale effective population (262k) on limited hardware

### Phase 4: Distributed Training
1. **Add**: Distributed population evaluation (Section 12)
   - Split population across ranks using `compute_perturbation_seed()`
   - All-gather fitnesses
2. **Add**: Distributed ES update with `es_update_distributed()`
   - Compute local fitness-weighted updates
   - All-reduce to combine
3. **Test**: Verify same results as single-GPU (with same seeds)
4. **Scale**: Combine with temporal accumulation for full 262K effective population

**Goal**: Match or exceed paper's scale with distributed training

### Phase 5: Advanced Optimizations (Optional)
1. **Add**: Antithetic sampling (Section 11.1) - 2x variance reduction
2. **Add**: Rank-based fitness shaping (Section 11.2) - robustness
3. **Add**: Noise reuse ES (Section 11.3) - for long sequences

**Goal**: Match paper's full methodology

## Code Organization and Responsibilities

**Clear separation of concerns**:

**Implementation uses chunk-level RNG throughout**:
- Forward: `_linear_batched_lowrank` with chunk-level RNG (Section 4)
- Update: `es_update_vectorized` with chunk-level RNG (Section 5)
- Temporal: `accumulate_micro_population_vectorized` (Section 5.2)
- **CRITICAL**: All components use consistent chunk-level RNG scheme

### `nanochat/egroll.py` (NEW FILE)
**Owns**: Noise generation and ES update logic
- `generate_lowrank_noise_factors(param_shape, rank, seed, device)` - Generate (A, B) from seed (reference only)
- `generate_fullrank_noise(param_shape, seed, device)` - Generate full-rank noise for 1D params
- `compute_perturbation_seed(base_seed, step, world_size, pop_size, ddp_rank, member_idx)` - Seed coordination (CRITICAL)
- `stable_hash_name(name)` - Stable layer name hashing (CRITICAL)
- `es_update_vectorized(model, fitnesses, seeds, ...)` - Vectorized ES update (production implementation)
- `es_update_distributed(model, all_fitnesses, local_seeds, ...)` - Distributed ES update

**Does NOT**:
- Know about model architecture (GPT, transformer blocks)
- Implement forward passes

### `nanochat/gpt.py` (MODIFIED)
**Owns**: Model architecture and population evaluation
- `forward(idx, targets)` - **UNCHANGED** standard forward (inference, no perturbations)
- `evaluate_population(idx, targets, pop_size, sigma, rank, base_seed, step)` - **NEW** batched population eval
- `_forward_batched_population(idx, seeds, sigma, rank)` - **NEW** forward with batched perturbations
- `_forward_block_batched(x, block, block_idx, seeds, sigma, rank)` - **NEW** block-level batched forward
- `_linear_batched_lowrank(x, W, bias, layer_name, seeds, sigma, rank)` - **NEW** linear layer with noise

**Does NOT**:
- Implement noise generation (calls `egroll.generate_lowrank_noise_factors()`)
- Implement ES update logic (trainer calls `egroll.es_update_vectorized()`)

### `scripts/base_train.py` (MODIFIED)
**Owns**: Training loop orchestration
- Calls `model.evaluate_population()` to get fitnesses and seeds
- Calls `egroll.es_update()` or `egroll.es_update_distributed()` to update parameters
- Manages training loop, logging, checkpointing

**Does NOT**:
- Know how noise is generated (delegates to egroll)
- Know model architecture details (delegates to gpt)

### Design principle: **Single source of truth**
- **Noise generation**: `egroll.py` owns it, everyone calls into it
- **Forward pass**: `gpt.py` owns it, including batched population variant
- **ES update**: `egroll.py` owns it
- **Training loop**: `base_train.py` orchestrates by calling the above

## Troubleshooting and Common Issues

### Q: Training is extremely slow
**A**: Check these issues:
1. Are you looping over population sequentially? → Must use batched evaluation with chunking
2. Are you storing all perturbations? → Must use on-the-fly generation with seeds
3. Are you using global `torch.manual_seed()`? → Must use per-call `torch.Generator`
4. Verify chunk-level RNG is implemented correctly (Section 4)
   - One generator per layer per chunk, not per member
   - Derive chunk seed from `seeds_chunk[0]` plus `layer_hash`
5. Verify vectorized ES update is used (Section 5)
   - Must use `es_update_vectorized` with chunk-level RNG
   - Essential for temporal accumulation with N_eff=262k
6. Is population size too large? → Start with 256-1K for initial testing
7. Is chunk_size suboptimal?
   - Too small (e.g., 1-2): Inefficient batching
   - Too large (e.g., 64+): May OOM
   - Sweet spot: 8-16 for most cases

### Q: Out of memory (OOM)
**A**: Reduce memory usage:
1. **CRITICAL**: Are you chunking at `evaluate_population` level? → See Section 4
2. **CRITICAL**: Is `chunk_size` too large? → Logits are `[chunk, batch, seq, vocab]`
   - With vocab ~50k: even chunk=8 uses ~13GB for logits alone!
   - Start with chunk_size=8, increase only if memory allows
   - Calculate: `chunk * batch * seq * vocab * 4 bytes`
3. **CRITICAL**: Are you materializing `[pop, vocab, n_embd]` for embedding perturbations?
   - For vocab=50k, n_embd=4096, pop=8: ~6.4 GB just for wte_perturbed_batch
   - **Solution 1** (recommended): Freeze embeddings (`continue` for wte/wpe in ES update)
   - **Solution 2**: Use low-rank embedding perturbation (Section 3.1.1)
   - **Do NOT** use the naive full-rank embedding approach for production
4. Decrease `population_size` (start with 256, gradually increase)
5. Use distributed training to split population across GPUs (Section 9)
6. Reduce batch_size or seq_len
7. Ensure `torch.inference_mode()` is used during evaluation (no autograd overhead)
8. Check vocab size - large vocab dominates memory in logits tensor

### Q: Loss is not decreasing
**A**: Check these hyperparameters:
1. `es_lr` too small or too large → Try range [0.001, 0.1], sweep broadly
2. `sigma` too small (weak signal) or too large (noisy signal) → Try [0.001, 0.01, 0.1]
3. `population_size` too small → Try at least 256-1K for meaningful signal
4. Verify fitness normalization is working (check fitness.std() > 0)
5. Check that seeds are different per step (not reusing same noise)

### Q: Results not reproducible when resuming from checkpoint
**A**: Ensure proper seed management:
1. Save `base_seed` and `step` in checkpoint (Section 10)
2. Use `compute_perturbation_seed()` with correct step number
3. Use `stable_hash_name()` not Python `hash()` for layer offsets
4. In distributed: ensure all ranks use consistent seed formula

### Q: Results not reproducible across runs (even without checkpoint)
**A**: Check RNG and hash implementations:
1. **CRITICAL**: Are you using `torch.Generator(device).manual_seed(seed)` instead of global `torch.manual_seed()`?
2. **CRITICAL**: Are you using `stable_hash_name()` instead of Python's `hash()`?
3. **CRITICAL**: Is dropout disabled (`model.eval()` or `dropout=0`)?
4. Python's `hash()` is randomized per-process by default (non-reproducible)
5. Global `torch.manual_seed()` can be affected by other random operations
6. Dropout adds extra randomness not controlled by seeds
7. Use per-call generators for isolation and reproducibility

### Q: Distributed training gives different results than single-GPU
**A**: Check seed coordination:
1. Verify using centralized `compute_perturbation_seed()` in both paths
2. Check that all ranks use same `base_seed` and `step`
3. Verify all-gather and all-reduce ops are correct
4. Test with same population_size and seeds on both single and multi-GPU
5. Check that `stable_hash_name()` gives same results on all ranks

### Q: How do I tune `sigma` and `es_lr`?
**A**: They are independent hyperparameters:
1. Start with `sigma = 0.01` (from paper), try [0.001, 0.01, 0.1]
2. Start with `es_lr = 0.01`, sweep [0.001, 0.003, 0.01, 0.03, 0.1]
3. Grid search or successive halving
4. Monitor loss curves and fitness variance (fitness.std() should be > 0 but not huge)
5. `sigma` affects signal-to-noise, `es_lr` affects step size - tune both

### Q: Should I use antithetic sampling?
**A**: Optional optimization with significant benefits:
- Gives 2-4x variance reduction (massive win)
- Requires evaluating pairs (θ+ε, θ-ε) with same noise
- Doubles effective population at same compute cost
- Can add after core implementation is working

### Q: Can I use mixed precision (autocast)?
**A**: Yes, but be careful:
- `autocast` during population evaluation is fine (Section 5)
- Ensure ES update happens in full precision (float32)
- Fitness differences may be small, float16 could lose precision
- Test with and without to verify stability

### Q: How do I verify my implementation is correct?
**A**: Progressive verification:
1. **Test noise generation**: 
   - Regenerate with same chunk seed, check equality
   - Verify using chunk-level `torch.Generator` (not global `torch.manual_seed`)
   - Check that same seed produces same noise across calls
2. **Test stable hash**: 
   - Verify `stable_hash_name("layer.0")` gives same result across runs
   - Check NOT using Python `hash()` (which is randomized)
3. **Test seed propagation**: Verify chunk seed includes step-dependence
   - `chunk_seed = int(seeds_chunk[0].item()) + layer_hash`
   - Different steps should produce different chunk seeds
4. **Test ES update**: Verify forward and update use same RNG scheme
   - Both must derive chunk seed from `seeds` tensor
5. **Test distributed**: Compare single-GPU vs multi-GPU with same seeds
6. **Test loss decrease**: Should see loss decrease over time

### Q: `_forward_block_batched()` is giving wrong results / shapes don't match
**A**: This is the most complex part. Debug systematically:
1. **Print shapes at every step**: Add `print(x.shape)` after each operation
   - Should always be `[pop, batch, seq, hidden]` or variants with `n_head`
2. **Test with pop_size=1**: Should match standard forward (no perturbations)
3. **Test with sigma=0**: All population members should be identical
4. **Check attention mask**: Verify mask shape includes population dimension
   - Should be `[1, 1, 1, seq, seq]` or similar to broadcast correctly
5. **Check layer names**: Each layer should have unique name
   - E.g., `f"block{block_idx}.attn.c_attn"` not just `"attn"`
6. **Verify dimension order**: Population should always be dim 0
   - After transpose: still `[pop, ...]` not `[batch, pop, ...]`
7. **Test attention outputs**: 
   - With causal mask, position i should only attend to positions ≤ i
   - Same for all population members

### Q: Attention is not causal / seeing future tokens
**A**: Check mask broadcasting:
1. Mask must have shape that broadcasts over population: `[1, 1, 1, seq, seq]`
2. Mask applied to attention scores: `[pop, batch, n_head, seq, seq]`
3. Verify `masked_fill` with `-inf` before softmax
4. Test: token at position i should not affect gradients of positions < i

### Q: Different population members giving identical outputs (when sigma > 0)
**A**: Seeds/noise not being applied correctly:
1. Verify `sigma > 0` (not accidentally 0)
2. Check layer names are unique per layer (not reusing same name)
3. Verify `stable_hash_name()` gives different values for different layers
4. Check that `_linear_batched_lowrank()` is actually generating and applying noise
5. Print some noise samples: `print(A[0, :5, :])` to verify it's non-zero and different per member

### Q: Some parameters seem to be random-walking / not learning
**A**: Check parameter perturbation/update consistency:
1. **CRITICAL**: Only update parameters that were perturbed in forward pass
2. For **pretraining**: Perturb ALL params (2D with low-rank, 1D with full-rank noise)
3. For **fine-tuning**: Can optionally freeze 1D params (embeddings, LayerNorm, biases)
4. In `es_update()`: If you freeze 1D params, add `continue` for `p.ndim == 1`
5. Mismatch between perturbed and updated params causes random walk

### Q: Should I train embeddings and LayerNorm parameters?
**A**: Depends on your task:

**For pretraining from scratch**:
- ✅ YES - Perturb and update ALL parameters
- 2D matrices: Low-rank noise `(σ/√r) * A @ B^T`
- 1D vectors: Full-rank noise `σ * ε`, where `ε ~ N(0, I)`
- See Section 3 for detailed implementation
- 1D parameters are tiny (< 1% of model), negligible overhead
- Essential for learning from random initialization

**For fine-tuning pretrained models**:
- ❌ OPTIONAL - Can freeze 1D parameters
- Embeddings/LayerNorm already learned from pretraining
- Only update 2D matrices for adaptation
- Simpler implementation, slightly faster
- Add `continue` for `p.ndim == 1` in ES update

### Q: How do I tune chunk_size and population_size?
**A**: Balance memory and speed:

**chunk_size**:
- Default: 8 (conservative for large vocab ~50k)
- Memory: `chunk * batch * seq * vocab * 4` bytes for logits alone
- Example calculation: `8 * 8 * 1024 * 50000 * 4 = ~13 GB`
- Too small (1-4): Slow, poor GPU utilization
- Too large (32+): OOM risk, especially with large vocab
- Sweet spot: 8-16 for most cases

**population_size** (per step):
- Start small: 256 for initial testing
- Scale gradually: 256 → 1K → 4K → 16K
- Constraints:
  - Must be divisible by world_size (assertion will check)
  - Larger per-step = more compute per update
  - Hardware-limited: single GPU typically 1K-4K max
- **For 262k effective population**: Use temporal accumulation (see Section 6)

### Q: How can I achieve 262k population like in the paper?
**A**: Use **temporal accumulation** (Section 7):

**Problem**: 262k simultaneous members requires massive memory

**Solution**: Spread evaluation across time
```python
# Instead of:
population_size = 262144  # One giant step (OOM!)

# Do this:
micro_population_size = 2048   # Hardware-feasible
K = 128                        # Number of micro-populations
# Accumulate fitness-weighted perturbations across K steps
# Effective population: N_eff = K * 2048 = 262,144
```

**Benefits**:
- Same statistical properties as 262k simultaneous
- Memory of 2k population, benefits of 262k
- No hardware changes needed

**Trade-off**:
- K times more forward passes per ES update
- K times more data per update

**Recommendation**: Start with per-step population your hardware can handle (1K-4K), then use temporal accumulation to reach target effective population.

**Example configurations**:
- Conservative: M=1024, K=256 → N_eff=262k, uses 1k population per step
- Balanced: M=2048, K=128 → N_eff=262k, uses 2k population per step
- Aggressive: M=4096, K=64 → N_eff=262k, uses 4k population per step

### Q: population_size not divisible by world_size error
**A**: The population must divide evenly across ranks:
- Good: `population_size=256, world_size=8` → 32 per rank
- Good: `population_size=262144, world_size=8` → 32768 per rank  
- Bad: `population_size=100, world_size=8` → error!
- Fix: Choose population_size that's a multiple of world_size
- Common multiples: 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144

### Q: Training hangs or deadlocks in distributed setting
**A**: Check synchronization:
1. Are all ranks calling all_gather/all_reduce at the same time?
2. Is one rank OOMing silently?
3. Check that all ranks have same population_per_rank
4. Verify barrier synchronization if needed

### Q: Training doesn't improve or uses same noise every step
**A**: Check that chunk-level RNG is deriving from seeds tensor:

**CRITICAL BUG**: If your chunk-level RNG does this:
```python
# ❌ WRONG: Noise doesn't depend on step/base_seed
chunk_seed = layer_hash + chunk_start
```

Every training step uses identical noise! This breaks ES completely.

**FIX**: Derive chunk seed from first member's seed:
```python
# ✅ CORRECT: Noise depends on (base_seed, step, member_idx)
seeds_chunk = seeds[chunk_start:chunk_end]
chunk_seed = int(seeds_chunk[0].item()) + layer_hash
```

**Symptoms**:
- Loss plateaus immediately or doesn't improve
- Different steps show identical fitness distributions
- `seeds` array is computed but changing `base_seed` or `step` has no effect on noise

**Root cause**: The `seeds` tensor (computed via `compute_perturbation_seed`) encodes step-dependence, but if RNG ignores it and only uses `layer_hash + chunk_start`, noise is constant across steps.

**Verify the fix**:
1. Change `base_seed` between runs → should get different results
2. Print `chunk_seed` in forward and update → should be identical for same step/chunk
3. Print `chunk_seed` across steps → should be different

### Q: Forward pass and ES update use different noise (correctness bug)
**A**: Ensure both use the same RNG scheme:

**Problem**: Forward and update must regenerate identical noise from seeds.

**Solution**: Both must use chunk-level RNG with same seed derivation:
- `_linear_batched_lowrank`: `chunk_seed = int(seeds_chunk[0].item()) + layer_hash`
- `es_update_vectorized`: same formula
- `accumulate_micro_population_vectorized`: same formula

**Symptoms**:
- Training diverges or shows random behavior
- Loss doesn't correlate with fitness scores
- Updates seem random even with deterministic seeds

### Q: RNG overhead is killing performance
**A**: Verify you're using chunk-level RNG correctly:

**Correct approach (chunk-level generators)**:
```python
# Chunk-level Generator: one per layer per chunk
for chunk in chunks:  # num_chunks iterations (e.g., 16 instead of 1024)
    # CRITICAL: Derive from first member's seed (preserves step-dependence)
    seeds_chunk = seeds[chunk_start:chunk_end]
    chunk_seed = int(seeds_chunk[0].item()) + layer_hash
    gen = torch.Generator(device=device)  # ONE generator for entire chunk
    gen.manual_seed(chunk_seed)
    # Two big randn calls instead of many small ones
    A_chunk = torch.randn(chunk_size, m, r, generator=gen, device=device)
    B_chunk = torch.randn(chunk_size, n, r, generator=gen, device=device)
    # Amortizes Generator overhead across chunk_size members
    # Pop=10k, chunk=64, 20 layers = 3.2k calls = 3-32ms overhead
```

**Common mistakes**:
- Creating generator per member (200k+ calls for large populations)
- Using global `torch.manual_seed()` and saving/restoring RNG state
- Not deriving chunk seed from `seeds` tensor

**Performance**: Chunk-level approach achieves ~3-32ms RNG overhead for pop=10k.

## Critical Implementation Checklist

Before deploying EGGROLL, verify these critical points:

### ✅ RNG Correctness
- [ ] Chunk-level RNG derives from `seeds` tensor: `chunk_seed = int(seeds_chunk[0].item()) + layer_hash`
- [ ] NOT using `chunk_seed = layer_hash + chunk_start` (missing step-dependence)
- [ ] `stable_hash_name()` used for layer offsets (not Python `hash()`)
- [ ] Per-call `torch.Generator` used (not global `torch.manual_seed()`)

### ✅ Forward/Update Consistency
- [ ] Forward pass and ES update both use chunk-level RNG
- [ ] Both derive chunk seed: `int(seeds_chunk[0].item()) + layer_hash`
- [ ] Using `es_update_vectorized` with matching RNG scheme
- [ ] Temporal accumulation uses same chunk-level RNG

### ✅ Seed Management
- [ ] `compute_perturbation_seed()` used for all seed generation
- [ ] `base_seed`, `step`, `member_idx` all contribute to seeds
- [ ] Seeds change across training steps (verify with print)
- [ ] Seeds saved in checkpoints for reproducibility

### ✅ Parameter Consistency
- [ ] Only perturb parameters you plan to update
- [ ] **For pretraining**: ALL parameters perturbed (2D low-rank, 1D full-rank)
- [ ] **For fine-tuning**: Optionally freeze 1D params (embeddings, LayerNorm, biases)
- [ ] Perturbation in forward matches update calculation
- [ ] If 1D params frozen: add `continue` for `p.ndim == 1` in ES update

### ✅ Performance
- [ ] Using chunk-level RNG throughout (forward and update)
- [ ] Using `es_update_vectorized` with batched operations
- [ ] Chunking enabled in `evaluate_population` to avoid OOM
- [ ] `torch.inference_mode()` used during population evaluation

### ✅ Distributed (if applicable)
- [ ] `population_size % world_size == 0` (assertion checks this)
- [ ] All ranks use same `base_seed` and `step`
- [ ] Seed coordination uses `compute_perturbation_seed()` on all ranks
- [ ] Fitnesses gathered via `all_gather`, updates via `all_reduce`

## References

- Paper: "Evolution Strategies at the Hyperscale" - https://eshyperscale.github.io/
- Key insight: Low-rank perturbations `A @ B.T` reduce memory from `O(mn)` to `O(r(m+n))` while still allowing high-rank updates through population averaging
- ES background: "Natural Evolution Strategies" (Wierstra et al., 2014)
- Distributed ES: "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" (Salimans et al., 2017)
