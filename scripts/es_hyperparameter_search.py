"""
ES Hyperparameter Search Script

Performs hyperparameter search by running base_train.py with different configs.
Each trial runs a limited number of steps and we compare initial vs final val BPB.

Run with:
    python -m scripts.es_hyperparameter_search
"""

import os
import sys
import re
import subprocess
import math


def run_trial(population_size, sigma, es_lr, chunk_size, num_steps,
              depth=8, max_seq_len=256, device_batch_size=4, total_batch_size=1024,
              eval_every=10, verbose=False):
    """
    Run a single trial using base_train.py as subprocess.

    Returns:
        dict with keys: initial_bpb, final_bpb, bpb_reduction, success, error
    """
    # Build command line args
    cmd = [
        sys.executable, "-m", "scripts.base_train",
        f"--depth={depth}",
        f"--max_seq_len={max_seq_len}",
        f"--device_batch_size={device_batch_size}",
        f"--total_batch_size={total_batch_size}",
        f"--num_iterations={num_steps}",
        f"--population_size={population_size}",
        f"--sigma={sigma}",
        f"--es_lr={es_lr}",
        f"--chunk_size={chunk_size}",
        f"--eval_every={eval_every}",
        f"--eval_tokens={5*524288}",  # Smaller eval for faster trials
        "--core_metric_every=-1",  # Disable core metric
        "--test_mode=True",  # Disable sampling and checkpointing
        "--run=dummy",  # No wandb
    ]

    if verbose:
        print(f"  Running: {' '.join(cmd[-10:])}")  # Show last 10 args for brevity

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None,  # No timeout
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        output = result.stdout + result.stderr

        if verbose:
            # Print last few lines of output
            lines = output.strip().split('\n')
            for line in lines[-20:]:
                print(f"    {line}")

        # Parse validation BPB values from output
        # Format: "Step 00000 | Validation bpb: 12.3456"
        bpb_pattern = r"Step (\d+) \| Validation bpb: ([\d.]+)"
        matches = re.findall(bpb_pattern, output)

        if not matches:
            return {
                'initial_bpb': float('inf'),
                'final_bpb': float('inf'),
                'bpb_reduction': -100.0,
                'success': False,
                'error': f'No validation BPB found in output. Return code: {result.returncode}'
            }

        # First match is initial (step 0), last match is final
        initial_bpb = float(matches[0][1])
        final_bpb = float(matches[-1][1])
        bpb_reduction = (initial_bpb - final_bpb) / initial_bpb * 100

        # Success if we reduced BPB by at least 1%
        success = final_bpb < initial_bpb * 0.99

        return {
            'initial_bpb': initial_bpb,
            'final_bpb': final_bpb,
            'bpb_reduction': bpb_reduction,
            'success': success,
            'error': None
        }

    except Exception as e:
        return {
            'initial_bpb': float('inf'),
            'final_bpb': float('inf'),
            'bpb_reduction': -100.0,
            'success': False,
            'error': str(e)
        }


def main():
    """Main hyperparameter search function"""

    print("="*60)
    print("ES Hyperparameter Search (via base_train.py)")
    print("="*60)
    print()

    # Model settings (small for fast trials)
    depth = 4
    max_seq_len = 256
    device_batch_size = 16
    total_batch_size = 8192

    # Hyperparameter search space
    es_lr_values = [0.01, 0.15, 0.02]
    sigma_values = [0.01, 0.02, 0.05]
    population_sizes = [16]
    chunk_sizes = [16]
    num_steps_values = [10]
    eval_every = 5  # Eval at start, middle, and end

    print(f"Model config:")
    print(f"  depth: {depth}")
    print(f"  max_seq_len: {max_seq_len}")
    print(f"  device_batch_size: {device_batch_size}")
    print(f"  total_batch_size: {total_batch_size}")
    print()

    print("Hyperparameter Search:")
    print(f"  sigma: {sigma_values}")
    print(f"  es_lr: {es_lr_values}")
    print(f"  population_size: {population_sizes}")
    print(f"  chunk_size: {chunk_sizes}")
    print(f"  num_steps: {num_steps_values}")
    print(f"  eval_every: {eval_every}")
    print()
    print("Searching...")
    print("-" * 60)

    results = []

    for sigma in sigma_values:
        for es_lr in es_lr_values:
            for population_size in population_sizes:
                for chunk_size in chunk_sizes:
                    for num_steps in num_steps_values:
                        actual_chunk_size = min(chunk_size, population_size)

                        print(f"\nTesting: sigma={sigma:.3f}, es_lr={es_lr:.4f}, pop={population_size}, chunk={actual_chunk_size}, steps={num_steps}")

                        result = run_trial(
                            population_size=population_size,
                            sigma=sigma,
                            es_lr=es_lr,
                            chunk_size=actual_chunk_size,
                            num_steps=num_steps,
                            depth=depth,
                            max_seq_len=max_seq_len,
                            device_batch_size=device_batch_size,
                            total_batch_size=total_batch_size,
                            eval_every=eval_every,
                            verbose=True
                        )

                        result['sigma'] = sigma
                        result['es_lr'] = es_lr
                        result['population_size'] = population_size
                        result['chunk_size'] = actual_chunk_size
                        result['num_steps'] = num_steps
                        results.append(result)

                        if result['error']:
                            print(f"  ❌ ERROR: {result['error']}")
                        else:
                            status = "✅ PASS" if result['success'] else "❌ FAIL"
                            print(f"  {status} | Initial: {result['initial_bpb']:.4f} → Final: {result['final_bpb']:.4f} | Reduction: {result['bpb_reduction']:.2f}%")

    print("-" * 60)
    print()

    # Analyze results
    successful_configs = [r for r in results if r.get('success', False)]
    all_reductions = [r['bpb_reduction'] for r in results if r.get('error') is None]

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
        best = max(successful_configs, key=lambda r: r['bpb_reduction'])

        print("✅ Best Configuration:")
        print(f"  sigma: {best['sigma']:.3f}")
        print(f"  es_lr: {best['es_lr']:.4f}")
        print(f"  population_size: {best['population_size']}")
        print(f"  chunk_size: {best['chunk_size']}")
        print(f"  num_steps: {best['num_steps']}")
        print(f"  Initial BPB: {best['initial_bpb']:.6f}")
        print(f"  Final BPB: {best['final_bpb']:.6f}")
        print(f"  BPB reduction: {best['bpb_reduction']:.2f}%")
        print()

        # Show top 5
        top5 = sorted(successful_configs, key=lambda r: r['bpb_reduction'], reverse=True)[:5]
        print("Top 5 Configurations:")
        for i, r in enumerate(top5, 1):
            print(f"  {i}. sigma={r['sigma']:.3f}, lr={r['es_lr']:.4f}, pop={r['population_size']}, reduction={r['bpb_reduction']:.2f}%")
        print()

        overall_success = True
        print("✅ SEARCH COMPLETE: At least one hyperparameter configuration succeeded!")
    else:
        print("❌ SEARCH FAILED: No hyperparameter configuration succeeded")
        print()
        if results:
            # Find best attempt even if it failed
            valid_results = [r for r in results if r.get('error') is None]
            if valid_results:
                best_attempt = max(valid_results, key=lambda r: r.get('bpb_reduction', -100))
                print("Best attempt:")
                print(f"  sigma: {best_attempt.get('sigma', 'N/A')}")
                print(f"  es_lr: {best_attempt.get('es_lr', 'N/A')}")
                print(f"  population_size: {best_attempt['population_size']}")
                print(f"  BPB reduction: {best_attempt.get('bpb_reduction', -100):.2f}%")
                print()
        overall_success = False

    # Show distribution of results
    if all_reductions:
        print("BPB Reduction Statistics:")
        print(f"  Mean: {sum(all_reductions) / len(all_reductions):.2f}%")
        print(f"  Min: {min(all_reductions):.2f}%")
        print(f"  Max: {max(all_reductions):.2f}%")
        print()

    print("="*60)

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
