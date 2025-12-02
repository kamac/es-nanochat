"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import print0

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    # Handle both 4D [batch, n_head, seq, head_dim] and 5D [pop, batch, seq, n_head, head_dim]
    assert x.ndim in [4, 5], f"Expected 4D or 5D tensor, got {x.ndim}D"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], dim=-1) # re-assemble on last dimension
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

    # ===== EGGROLL (Evolution Strategies) Methods =====
    # These methods are used ONLY during ES training, not during standard inference

    @torch.inference_mode()
    def evaluate_population(self, idx, targets, population_size, sigma, base_seed=0, step=0,
                           world_size=1, ddp_rank=0, chunk_size=8):
        """
        Evaluate a population of perturbed models (TRAINING ONLY) with CHUNKED batching.
        This method is ONLY used during ES training, not during inference/evaluation.
        
        CRITICAL: Processes population in CHUNKS to avoid OOM. Never materializes
        full [population_size, batch_size, seq_len, vocab_size] tensor at once.
        
        IMPORTANT: Does NOT store perturbations in memory. Uses deterministic seeds
        to regenerate noise on-the-fly during forward pass and again during ES update.
        
        DISTRIBUTED: Each rank evaluates only its subset of the population.
        For example, with 8 GPUs and population_size=512:
        - Rank 0: evaluates members 0-63 (64 members)
        - Rank 1: evaluates members 64-127 (64 members)
        - ... and so on
        
        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target tokens [batch_size, seq_len]
            population_size: TOTAL population size across all ranks (must be divisible by world_size)
            sigma: Noise temperature (perturbation scale)
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
            fitnesses: (population_per_rank,) tensor of LOCAL fitness scores
            seeds: (population_per_rank,) tensor of LOCAL seeds used (for ES update)
        
        Note: rank=1 is hardcoded for optimization.
        Perturbations are scaled by sigma: E = sigma * A @ B^T
        """
        from nanochat.egroll import compute_perturbation_seed
        
        device = idx.device
        batch_size, seq_len = idx.shape
        
        # CRITICAL: Verify population divides evenly across ranks
        assert population_size % world_size == 0, \
            f"population_size ({population_size}) must be divisible by world_size ({world_size})"
        
        # Calculate local population size for this rank
        population_per_rank = population_size // world_size
        
        # Generate seeds for LOCAL population members only
        # Each rank generates different seeds based on its ddp_rank
        seeds = torch.zeros(population_per_rank, dtype=torch.int64, device=device)
        for i in range(population_per_rank):
            seeds[i] = compute_perturbation_seed(
                base_seed, step, world_size, population_size, ddp_rank, i
            )
        
        # Allocate output fitnesses for LOCAL population only
        fitnesses = torch.empty(population_per_rank, device=device)
        
        # CRITICAL: Process LOCAL population in chunks to avoid OOM
        # Never materialize [pop, B, T, vocab] at once!
        for chunk_start in range(0, population_per_rank, chunk_size):
            chunk_end = min(chunk_start + chunk_size, population_per_rank)
            chunk_pop_size = chunk_end - chunk_start
            
            # Get seeds for this chunk
            seeds_chunk = seeds[chunk_start:chunk_end]
            
            # Expand inputs to [chunk_size, batch_size, seq_len]
            idx_chunk = idx.unsqueeze(0).expand(chunk_pop_size, -1, -1)
            targets_chunk = targets.unsqueeze(0).expand(chunk_pop_size, -1, -1)
            
            # Forward pass with batched perturbations for this chunk only
            # Shape: [chunk_size, batch_size, seq_len, vocab_size]
            logits_chunk = self._forward_batched_population(idx_chunk, seeds_chunk, sigma)
            
            # Compute loss for each population member in chunk
            # Reshape: [chunk * batch * seq, vocab] and [chunk * batch * seq]
            logits_flat = logits_chunk.reshape(-1, logits_chunk.size(-1))
            targets_flat = targets_chunk.reshape(-1)
            
            # Compute CE loss
            loss_flat = F.cross_entropy(logits_flat, targets_flat, reduction='none', ignore_index=-1)
            # Reshape back: [chunk, batch, seq]
            loss_per_token = loss_flat.reshape(chunk_pop_size, batch_size, seq_len)
            # Average over batch and sequence: [chunk]
            loss_per_member = loss_per_token.mean(dim=[1, 2])
            
            # Fitness = negative loss
            fitnesses[chunk_start:chunk_end] = -loss_per_member
            
            # Chunk tensors will be freed here, keeping memory bounded
        
        return fitnesses, seeds

    @torch.inference_mode()
    def _forward_batched_population(self, idx, seeds, sigma):
        """
        Forward pass with batched low-rank perturbations (TRAINING ONLY, rank=1 optimized).
        
        All population members are processed in parallel using batched operations.
        Perturbations are generated on-the-fly using deterministic seeds.
        
        Args:
            idx: Input tokens [population_size, batch_size, seq_len]
            seeds: Seeds for each population member [population_size]
            sigma: Noise temperature (perturbation scale)
        
        Returns:
            logits: [population_size, batch_size, seq_len, vocab_size]
        
        Note: rank=1 is hardcoded for optimization.
        Perturbations are scaled by sigma: E = sigma * A @ B^T
        """
        pop_size, batch_size, seq_len = idx.shape
        device = idx.device
        
        # Token Embedding with low-rank perturbations: [pop, batch, seq, n_embd]
        # Treat embeddings as 2D matrix [vocab, n_embd] and use low-rank perturbations
        # This is memory-efficient and matches the ES update for 2D parameters
        
        # Apply perturbed embedding: E_perturbed = E + (sigma/sqrt(r)) * A @ B^T
        # Then index: x = E_perturbed[idx]
        # Efficient computation: x = E[idx] + (sigma/sqrt(r)) * (A[idx] @ B^T)
        
        base_emb = self.transformer.wte(idx)  # [pop, batch, seq, n_embd] - base embeddings
        
        # Apply low-rank perturbations using the same helper as linear layers
        # We'll use _linear_batched_lowrank but need to handle indexing
        # Alternative: directly compute perturbation for indexed embeddings
        
        from nanochat.egroll import stable_hash_name
        
        # Use the standard embedding (no perturbation to embedding weights)
        # But we'll perturb the outputs by treating each sequence position independently
        # This gives us the effect of perturbing embeddings without storing full [pop, vocab, n_embd]
        
        layer_hash = stable_hash_name('transformer.wte')
        vocab_size, n_embd = self.transformer.wte.weight.shape
        # rank=1: sqrt(1)=1.0, so no sqrt scaling needed
        # Perturbations are scaled by sigma: E = sigma * A @ B^T
        
        x = base_emb.clone()
        
        # Generate noise for entire population at once
        # Derive chunk seed from first member's seed
        chunk_seed = int(seeds[0].item()) + layer_hash
        gen = torch.Generator(device=device)
        gen.manual_seed(chunk_seed)
        
        # Generate low-rank factors for the embedding matrix (rank=1: vectors)
        # A: [pop, vocab], B: [pop, n_embd]
        # Generate in the same dtype as base_emb for consistency
        A_chunk = torch.randn(pop_size, vocab_size, generator=gen, device=device, dtype=base_emb.dtype)
        B_chunk = torch.randn(pop_size, n_embd, generator=gen, device=device, dtype=base_emb.dtype)
        
        # For each member, gather A values for its tokens and compute perturbation
        for c in range(pop_size):
            # A_indexed: [batch, seq] (rank=1: vector, not matrix)
            A_indexed = A_chunk[c][idx[c]]
            # B: [n_embd] (rank=1: vector)
            # Perturbation: sigma * (A_indexed @ B^T) = sigma * [batch, seq] @ [n_embd] = [batch, seq, n_embd]
            # Using outer product: A_indexed.unsqueeze(-1) * B_chunk[c] broadcasts correctly
            emb_pert = sigma * (A_indexed.unsqueeze(-1) * B_chunk[c])
            x[c] += emb_pert
        
        # Initial norm
        x = norm(x)  # Applies independently per population member
        
        # Rotary embeddings for this sequence length
        T = seq_len
        assert T <= self.cos.size(1), f"Sequence length grew beyond rotary cache: {T} > {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]  # Broadcasts over population dimension
        
        # Transformer blocks with batched perturbations
        for block_idx, block in enumerate(self.transformer.h):
            x = self._forward_block_batched(x, block, block_idx, seeds, sigma, cos_sin)
        
        # Final norm
        x = norm(x)
        
        # Output projection (LM head) with perturbations
        logits = self._linear_batched_lowrank(
            x, self.lm_head.weight, None,
            layer_name='lm_head', seeds=seeds, sigma=sigma
        )
        
        # Apply softcap (same as standard forward)
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        
        return logits

    def _forward_block_batched(self, x, block, block_idx, seeds, sigma, cos_sin):
        """
        Forward through one transformer block with batched perturbations (rank=1 optimized).
        
        Args:
            x: [pop, batch, seq, n_embd]
            block: Block module
            block_idx: Index of this block (for seed coordination)
            seeds: [pop] seeds for noise generation
            sigma: Noise temperature (perturbation scale)
            cos_sin: Tuple of (cos, sin) rotary embeddings
        
        Returns:
            x: [pop, batch, seq, n_embd]
        
        Note: rank=1 is hardcoded for optimization.
        Perturbations are scaled by sigma: E = sigma * A @ B^T
        """
        pop_size, batch_size, seq_len, n_embd = x.shape
        device = x.device
        
        # === Self-Attention with Perturbations ===
        
        # Norm (applies independently per population member)
        x_norm = norm(x)  # [pop, batch, seq, n_embd]
        
        # Q, K, V projections with perturbations
        n_head = block.attn.n_head
        n_kv_head = block.attn.n_kv_head
        head_dim = block.attn.head_dim
        
        # Query projection with perturbations
        q = self._linear_batched_lowrank(
            x_norm,
            block.attn.c_q.weight,
            None,
            layer_name=f"h.{block_idx}.attn.c_q",
            seeds=seeds,
            sigma=sigma
        )  # [pop, batch, seq, n_head * head_dim]
        
        # Key projection with perturbations
        k = self._linear_batched_lowrank(
            x_norm,
            block.attn.c_k.weight,
            None,
            layer_name=f"h.{block_idx}.attn.c_k",
            seeds=seeds,
            sigma=sigma
        )  # [pop, batch, seq, n_kv_head * head_dim]
        
        # Value projection with perturbations
        v = self._linear_batched_lowrank(
            x_norm,
            block.attn.c_v.weight,
            None,
            layer_name=f"h.{block_idx}.attn.c_v",
            seeds=seeds,
            sigma=sigma
        )  # [pop, batch, seq, n_kv_head * head_dim]
        
        # Reshape for multi-head attention
        q = q.view(pop_size, batch_size, seq_len, n_head, head_dim)
        k = k.view(pop_size, batch_size, seq_len, n_kv_head, head_dim)
        v = v.view(pop_size, batch_size, seq_len, n_kv_head, head_dim)
        
        # Apply rotary embeddings (broadcasts over population dimension)
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # QK norm (applies independently per population member)
        q = norm(q)
        k = norm(k)
        
        # Transpose for attention: [pop, batch, n_head, seq, head_dim]
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)
        
        # Compute attention scores: Q @ K^T
        # For GQA, we need to handle the case where n_head != n_kv_head
        # We'll use manual attention computation to handle batched population
        
        # Expand k, v to match query heads if needed (GQA)
        if n_head != n_kv_head:
            repeats = n_head // n_kv_head
            k = k.repeat_interleave(repeats, dim=2)
            v = v.repeat_interleave(repeats, dim=2)
        
        # Attention: [pop, batch, n_head, seq_q, seq_k]
        att = torch.matmul(q, k.transpose(-2, -1))
        att = att / math.sqrt(head_dim)
        
        # Apply causal mask (broadcasts over population dimension)
        # Mask shape: [1, 1, 1, seq, seq] -> broadcasts to [pop, batch, n_head, seq, seq]
        if not hasattr(self, '_causal_mask_cache') or self._causal_mask_cache.size(-1) < seq_len:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            mask = mask.view(1, 1, 1, seq_len, seq_len)
            self._causal_mask_cache = mask
        
        causal_mask = self._causal_mask_cache[:, :, :, :seq_len, :seq_len]
        att = att.masked_fill(~causal_mask, float('-inf'))
        
        # Softmax (applies per population member independently)
        att = F.softmax(att, dim=-1)
        
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
            None,
            layer_name=f"h.{block_idx}.attn.c_proj",
            seeds=seeds,
            sigma=sigma
        )  # [pop, batch, seq, n_embd]
        
        # Residual connection
        x = x + y
        
        # === MLP with Perturbations ===
        
        # Norm
        x_norm = norm(x)  # [pop, batch, seq, n_embd]
        
        # First MLP layer (expand) with perturbations
        h = self._linear_batched_lowrank(
            x_norm,
            block.mlp.c_fc.weight,
            None,
            layer_name=f"h.{block_idx}.mlp.c_fc",
            seeds=seeds,
            sigma=sigma
        )  # [pop, batch, seq, 4*n_embd]
        
        # Activation (ReLU^2)
        h = F.relu(h).square()
        
        # Second MLP layer (project back) with perturbations
        h = self._linear_batched_lowrank(
            h,
            block.mlp.c_proj.weight,
            None,
            layer_name=f"h.{block_idx}.mlp.c_proj",
            seeds=seeds,
            sigma=sigma
        )  # [pop, batch, seq, n_embd]
        
        # Residual connection
        x = x + h
        
        return x  # [pop, batch, seq, n_embd]

    def _linear_batched_lowrank(self, x, weight, bias, layer_name, seeds, sigma):
        """
        Batched linear transformation with low-rank perturbations (rank=1 optimized).
        
        Applies: y = x @ (W + E)^T + b
        where E = sigma * A @ B^T (rank=1: no sqrt scaling needed)
        
        Efficient implementation for rank=1:
        y = x @ W^T + sigma * (x @ B) * A + b
        where A is [chunk, out_features] and B is [chunk, in_features] (vectors)
        
        CRITICAL: Uses stable hash for layer offsets (not Python hash())
        and per-call Generator for noise (not global RNG state).
        
        Args:
            x: Input [pop, ..., in_features]
            weight: Base weight [out_features, in_features]
            bias: Optional bias [out_features] (usually None in this model)
            layer_name: Name for seed coordination
            seeds: [pop] base seeds for each population member
            sigma: Noise temperature (perturbation scale)
        
        Returns:
            y: Output [pop, ..., out_features]
        
        Note: rank=1 is hardcoded for optimization.
        Perturbations are scaled by sigma: E = sigma * A @ B^T
        """
        from nanochat.egroll import stable_hash_name
        
        pop_size = x.shape[0]
        out_features, in_features = weight.shape
        device = x.device
        
        # Base transformation: x @ W^T
        # x: [pop, batch, seq, in_features]
        # W: [out_features, in_features]
        # Need to broadcast W across population dimension
        # Ensure weight dtype matches input dtype
        weight = weight.to(x.dtype)
        y_base = torch.matmul(x, weight.t())  # [pop, batch, seq, out_features]
        
        # CRITICAL: Use STABLE hash (not Python's hash which is randomized)
        layer_hash = stable_hash_name(layer_name)
        
        # rank=1: sqrt(1)=1.0, so no sqrt scaling needed
        # Perturbations are scaled by sigma: E = sigma * A @ B^T
        
        # Generate noise for entire population at once
        # CRITICAL: Derive chunk seed from first member's seed (not just layer_hash)
        # This ensures noise depends on (base_seed, step, member_idx) via compute_perturbation_seed
        chunk_seed = int(seeds[0].item()) + layer_hash
        
        # Create ONE generator for entire population
        gen = torch.Generator(device=device)
        gen.manual_seed(chunk_seed)
        
        # Generate noise for ALL members at once (rank=1: vectors, not matrices)
        # [pop_size, out_features] and [pop_size, in_features]
        # Generate in the same dtype as x for consistency
        A_chunk = torch.randn(pop_size, out_features, generator=gen, device=device, dtype=x.dtype)
        B_chunk = torch.randn(pop_size, in_features, generator=gen, device=device, dtype=x.dtype)
        
        # Apply perturbation for rank=1: sigma * (x @ B) * A^T
        # Paper: xi(µ + σEi) = xiµ + σ * (xiBi) * A^T_i
        # For rank=1: xiBi is batched vector-vector dot product (scalars)
        # Then batched scalar-vector multiplication with A^T_i
        
        # Reshape for batched matmul: [pop, batch*seq, in_features]
        orig_shape = x.shape
        x_flat = x.reshape(pop_size, -1, in_features)
        
        # xiBi: batched vector-vector dot product to get scalars (inexpensive per paper)
        # x_flat: [pop, batch*seq, in_features]
        # B_chunk: [pop, in_features]
        # Use bmm for efficient batched matrix-vector multiplication
        # Result: [pop, batch*seq] (scalars per position)
        xB = torch.bmm(x_flat, B_chunk.unsqueeze(-1)).squeeze(-1)
        
        # (xiBi) * A^T_i: batched scalar-vector multiplication
        # xB: [pop, batch*seq] (scalars)
        # A_chunk: [pop, out_features] (A^T_i when viewed as row vector)
        # Result: [pop, batch*seq, out_features]
        # Broadcast: [pop, batch*seq, 1] * [pop, 1, out_features]
        xBA = sigma * (xB.unsqueeze(-1) * A_chunk.unsqueeze(1))
        
        # Reshape back
        y_pert = xBA.reshape(orig_shape[:-1] + (out_features,))
        
        y = y_base + y_pert
        
        if bias is not None:
            bias = bias.to(x.dtype)
            y = y + bias
        
        return y
