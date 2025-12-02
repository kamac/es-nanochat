"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = 20 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 2048 # max context length
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization (EGGROLL - Evolution Strategies)
device_batch_size = 32 # sequences per forward pass (set to not OOM)
# ES hyperparameters
population_size = 256 # ES population size per update (start small, scale up; paper uses 262144)
sigma = 0.1 # Noise temperature (perturbation scale; use 0.1 for bfloat16, 0.01 for float32)
es_lr = 0.02 # ES learning rate (effective step size)
# rank=1 is hardcoded for optimization (removed es_rank parameter)
weight_decay = 0.0 # weight decay (applied as decoupled weight decay like AdamW)
base_seed = 42 # base random seed for deterministic noise generation
chunk_size = 8 # ES population chunk size (memory vs speed tradeoff; start conservative)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
save_every = -1 # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# ES training hyperparameters
# Each ES update: population_size models evaluate the same batch, then parameters are updated once
tokens_per_batch = device_batch_size * max_seq_len  # tokens per forward pass
tokens_per_es_update = tokens_per_batch  # all population members see the same batch
print0(f"Tokens per ES update: {device_batch_size} seqs × {max_seq_len} = {tokens_per_batch:,}")
print0(f"Population size: {population_size}")
print0(f"ES evaluations per update: {population_size} models × {tokens_per_batch:,} tokens = {population_size * tokens_per_batch:,} total forward pass tokens")

# Automatic learning rate scaling for population size
# The ES formula lr / (σ√r·N) has 1/N term, so larger populations reduce effective step size
# We use SQRT scaling (not linear) because:
# - Larger populations give better gradient estimates (can use larger steps)
# - But linear scaling is too aggressive (causes divergence)
# - Square root is a conservative middle ground
reference_population = 256  # Baseline population for es_lr tuning
es_lr_base = es_lr  # Store original for logging
import math
scaling_factor = math.sqrt(population_size / reference_population)
es_lr = es_lr * scaling_factor
print0(f"ES learning rate (base): {es_lr_base}")
print0(f"ES learning rate (scaling factor): {scaling_factor:.4f} (sqrt scaling)")
print0(f"ES learning rate (scaled for pop={population_size}): {es_lr:.6f}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    # ES training: no optimizer state to load (ES is stateless)
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=False, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy
    # Verify base_seed matches if resuming (critical for reproducibility)
    if "base_seed" in meta_data and meta_data["base_seed"] != base_seed:
        print0(f"WARNING: Resuming with different base_seed! Checkpoint: {meta_data['base_seed']}, Current: {base_seed}")

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations (ES updates): {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * tokens_per_batch))
    print0(f"Calculated number of ES updates from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // tokens_per_batch
    print0(f"Calculated number of ES updates from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = tokens_per_batch * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {tokens_per_batch * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")
print0(f"Note: Each ES update requires {population_size} forward passes (one per population member)")

# -----------------------------------------------------------------------------
# ES Training: No optimizer needed (ES is stateless, updates parameters directly)
# Verify population size is divisible by world size
assert population_size % ddp_world_size == 0, \
    f"population_size ({population_size}) must be divisible by ddp_world_size ({ddp_world_size})"
print0(f"ES population size: {population_size} ({population_size // ddp_world_size} per rank)")
print0(f"ES sigma (noise temperature): {sigma}")
print0(f"ES learning rate: {es_lr} (auto-scaled from base {es_lr_base} for population {population_size})")
print0(f"ES low-rank dimension: 1 (hardcoded for optimization)")
print0(f"ES chunk size: {chunk_size}")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler (applies to ES learning rate)
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    # FLOPs calculation for ES: each step does population_size forward passes
    flops_so_far = num_flops_per_token * tokens_per_batch * population_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [], # ES training: no optimizer states (ES is stateless)
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "base_seed": base_seed, # save base_seed for reproducibility
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # Single ES training step: evaluate population, compute update, apply
    # CRITICAL: Model in eval mode (disables dropout, ES provides exploration)
    model.eval()
    orig_model.eval()
    
    synchronize()
    t0 = time.time()
    
    # CRITICAL: Use inference_mode to prevent gradient computation (saves VRAM)
    with torch.inference_mode(), autocast_ctx:
        # Evaluate population on current batch
        # All population members evaluate the SAME batch (x, y)
        # Returns LOCAL fitness scores and seeds (NOT full perturbations)
        # In distributed: each rank evaluates population_size // world_size members
        fitnesses, seeds = model.evaluate_population(
            x, y,
            population_size=population_size,
            sigma=sigma,
            base_seed=base_seed,
            step=step,  # Unique seed per ES update
            world_size=ddp_world_size,
            ddp_rank=ddp_rank,
            chunk_size=chunk_size
        )
    
    # Synchronize fitnesses across all ranks (DDP only)
    # All ranks need full fitness distribution for proper normalization
    if ddp:
        import torch.distributed as dist
        # Gather fitnesses from all ranks into a single tensor
        # Each rank contributes population_size // world_size fitnesses
        all_fitnesses = torch.empty(population_size, device=device, dtype=fitnesses.dtype)
        dist.all_gather_into_tensor(all_fitnesses, fitnesses)
        if step == 0 and master_process:
            print0(f"Distributed ES: Each rank evaluated {len(fitnesses)} members, gathered {len(all_fitnesses)} total")
    else:
        all_fitnesses = fitnesses
    
    # Compute and apply ES update (outside inference_mode to allow in-place param.data writes)
    # Get current learning rate (with warmup/warmdown)
    lrm = get_lr_multiplier(step)
    current_lr = es_lr * lrm
    
    # Import ES update function
    from nanochat.egroll import es_update_vectorized
    
    # Apply ES update (works for both single-GPU and multi-GPU automatically)
    # NOTE: This modifies parameters directly via param.data
    # CRITICAL: update_chunk_size MUST EQUAL chunk_size from evaluate_population!
    # This ensures RNG seed alignment between forward pass and parameter update.
    # If these don't match, each fitness will be correlated with the WRONG noise,
    # completely breaking the ES gradient estimator and causing "loss not going down" issues.
    # For multi-GPU: pass all_fitnesses (global) and seeds (local), function handles the rest
    if step == 0 and master_process:
        print0(f"[Diagnostic] ES update: chunk_size={chunk_size}, len(seeds)={len(seeds)}, len(all_fitnesses)={len(all_fitnesses)}")
    es_update_vectorized(orig_model, all_fitnesses, seeds, current_lr, weight_decay, 
                        update_chunk_size=chunk_size)
    
    # For logging: estimate loss from average fitness
    # Use all_fitnesses for accurate global loss (includes all ranks)
    avg_fitness = all_fitnesses.mean().item()
    train_loss = -avg_fitness  # fitness = -loss
    
    # Diagnostic: Check if population has variation (critical for ES!)
    if step < 5:  # Only log first few steps
        fitness_std = all_fitnesses.std().item()
        fitness_range = (all_fitnesses.max() - all_fitnesses.min()).item()
        if master_process:
            print0(f"  [Diagnostic] Fitness std={fitness_std:.6f}, range={fitness_range:.6f}, unique_losses={len(torch.unique(-all_fitnesses))}")
            if fitness_std < 1e-6:
                print0(f"  [WARNING] No fitness variation! All population members identical!")
                print0(f"  [WARNING] ES cannot learn. Check: sigma too small? perturbations not applied?")
            elif fitness_std < 0.01:
                print0(f"  [WARNING] Very small fitness variation. Consider increasing sigma.")
    
    # Prefetch next batch for next ES update
    x, y, dataloader_state_dict = next(train_loader)
    
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(tokens_per_batch / dt)  # tokens per second (data throughput)
    # ES effective throughput: each forward pass is done population_size times
    es_evals_per_sec = population_size / dt  # population evaluations per second
    flops_per_sec = num_flops_per_token * tokens_per_batch * population_size / dt  # total FLOPs including all population members
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # ES training: no gradient norm to log
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | pop_eval/sec: {es_evals_per_sec:.1f} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "ES population size": population_size,
        "ES sigma (noise temperature)": sigma,
        "ES learning rate (base)": es_lr_base,
        "ES learning rate (scaled)": es_lr,
        "ES LR scaling factor": scaling_factor,
        "ES LR scaling method": "sqrt",
        "ES rank": 1,
        "Calculated number of ES updates": num_iterations,
        "Number of training tokens (data consumed)": total_tokens,
        "Tokens : Params ratio": tokens_per_batch * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
