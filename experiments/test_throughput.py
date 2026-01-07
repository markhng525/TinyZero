#!/usr/bin/env python3
"""
Throughput Verification Test for gpu_memory_utilization

This script tests the hypothesis that increasing gpu_memory_utilization
(which increases KV cache size) leads to higher tokens/sec throughput.

SGLang uses `mem_fraction_static` as the parameter name internally.
veRL maps `gpu_memory_utilization` -> `mem_fraction_static`.

Usage:
    python test_throughput.py [--model MODEL] [--batch-size BATCH] [--max-tokens MAX]
"""

import argparse
import time
import gc
import torch
from typing import List, Dict, Any

# Test configurations
GPU_MEMORY_UTILIZATIONS = [0.5, 0.7, 0.8]
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_NEW_TOKENS = 128

# Sample prompts for generation
SAMPLE_PROMPTS = [
    "Using the numbers 1, 2, 3, 4 and the operations +, -, *, /, find a way to make 24.",
    "Using the numbers 5, 5, 5, 1 and the operations +, -, *, /, find a way to make 24.",
    "Using the numbers 8, 3, 8, 3 and the operations +, -, *, /, find a way to make 24.",
    "Using the numbers 6, 6, 6, 6 and the operations +, -, *, /, find a way to make 24.",
    "Using the numbers 2, 3, 4, 5 and the operations +, -, *, /, find a way to make 24.",
    "Using the numbers 1, 5, 5, 5 and the operations +, -, *, /, find a way to make 24.",
    "Using the numbers 3, 3, 8, 8 and the operations +, -, *, /, find a way to make 24.",
    "Using the numbers 4, 4, 7, 7 and the operations +, -, *, /, find a way to make 24.",
]


def create_prompt_batch(batch_size: int) -> List[str]:
    """Create a batch of prompts by cycling through samples."""
    prompts = []
    for i in range(batch_size):
        prompts.append(SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)])
    return prompts


def run_throughput_test(
    model_path: str,
    mem_fraction: float,
    batch_size: int,
    max_new_tokens: int,
    warmup_batches: int = 1,
    test_batches: int = 3,
) -> Dict[str, Any]:
    """
    Run throughput test at a specific mem_fraction_static level.

    Returns dict with:
        - mem_fraction_static (same as gpu_memory_utilization in veRL)
        - tokens_per_second
        - total_tokens
        - total_time
        - batch_size
        - max_new_tokens
    """
    from sglang import Engine

    print(f"\n{'='*60}")
    print(f"Testing mem_fraction_static = {mem_fraction} (gpu_memory_utilization)")
    print(f"{'='*60}")

    # Initialize engine with specified memory fraction
    print(f"Initializing SGLang engine with {mem_fraction*100:.0f}% GPU memory for KV cache...")

    engine = Engine(
        model_path=model_path,
        mem_fraction_static=mem_fraction,
        log_level="warning",
    )

    prompts = create_prompt_batch(batch_size)

    # Warmup
    print(f"Running {warmup_batches} warmup batch(es)...")
    for _ in range(warmup_batches):
        _ = engine.generate(prompts, sampling_params={"max_new_tokens": max_new_tokens})

    # Timed test
    print(f"Running {test_batches} timed batch(es) with {batch_size} samples each...")
    total_tokens = 0
    start_time = time.perf_counter()

    for batch_idx in range(test_batches):
        outputs = engine.generate(prompts, sampling_params={"max_new_tokens": max_new_tokens})
        # Count actual output tokens from the response
        batch_tokens = 0
        for out in outputs:
            # Use token count from output if available, else estimate from text
            if isinstance(out, dict) and "text" in out:
                # Rough approximation: ~4 chars per token
                batch_tokens += len(out["text"]) // 4
            elif hasattr(out, "text"):
                batch_tokens += len(out.text) // 4
        total_tokens += batch_tokens
        print(f"  Batch {batch_idx + 1}/{test_batches}: ~{batch_tokens} output tokens")

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate throughput
    tokens_per_second = total_tokens / total_time

    # Cleanup
    print("Shutting down engine...")
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    # Give some time for cleanup
    time.sleep(3)

    result = {
        "gpu_memory_utilization": mem_fraction,
        "tokens_per_second": tokens_per_second,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "test_batches": test_batches,
    }

    print(f"\nResults for mem_fraction_static={mem_fraction}:")
    print(f"  Total output tokens: {total_tokens}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {tokens_per_second:.1f} tokens/sec")

    return result


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary comparison of all results."""
    print("\n" + "="*70)
    print("THROUGHPUT COMPARISON SUMMARY")
    print("="*70)
    print(f"{'GPU Mem Util':<15} {'Tokens/sec':<15} {'Total Time':<15} {'Speedup':<15}")
    print("-"*70)

    baseline_throughput = results[0]["tokens_per_second"] if results else 1

    for r in results:
        speedup = r["tokens_per_second"] / baseline_throughput
        print(f"{r['gpu_memory_utilization']:<15.1%} {r['tokens_per_second']:<15.1f} {r['total_time']:<15.2f}s {speedup:<15.2f}x")

    print("-"*70)

    # Conclusions
    if len(results) >= 2:
        first = results[0]
        last = results[-1]
        improvement = (last["tokens_per_second"] / first["tokens_per_second"] - 1) * 100

        print(f"\nCONCLUSION:")
        if improvement > 5:
            print(f"  Increasing gpu_memory_utilization from {first['gpu_memory_utilization']:.0%} to {last['gpu_memory_utilization']:.0%}")
            print(f"  improved throughput by {improvement:.1f}%")
            print(f"  This confirms: larger KV cache -> more concurrent sequences -> higher tokens/sec")
        elif improvement < -5:
            print(f"  Throughput DECREASED by {-improvement:.1f}% with higher gpu_memory_utilization")
            print(f"  This may indicate memory pressure or other bottlenecks")
        else:
            print(f"  Throughput was relatively stable ({improvement:+.1f}%) across memory utilization levels")
            print(f"  The workload may be compute-bound rather than memory-bound")


def main():
    parser = argparse.ArgumentParser(description="Test throughput at different gpu_memory_utilization levels")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens per response")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup batches")
    parser.add_argument("--test-batches", type=int, default=3, help="Number of test batches")
    parser.add_argument("--utils", type=float, nargs="+", default=GPU_MEMORY_UTILIZATIONS,
                        help="GPU memory utilization levels to test (e.g., 0.5 0.7 0.8)")
    args = parser.parse_args()

    print("="*70)
    print("GPU MEMORY UTILIZATION THROUGHPUT TEST")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_tokens}")
    print(f"GPU memory utilization levels: {args.utils}")
    print(f"Test batches per level: {args.test_batches}")
    print("")
    print("This test verifies that increasing gpu_memory_utilization (mem_fraction_static)")
    print("improves generation throughput by allowing a larger KV cache.")

    results = []

    for util in sorted(args.utils):
        try:
            result = run_throughput_test(
                model_path=args.model,
                mem_fraction=util,
                batch_size=args.batch_size,
                max_new_tokens=args.max_tokens,
                warmup_batches=args.warmup,
                test_batches=args.test_batches,
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR at gpu_memory_utilization={util}: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing with next configuration...")
            continue

    if results:
        print_summary(results)
    else:
        print("\nNo successful test runs!")


if __name__ == "__main__":
    main()
