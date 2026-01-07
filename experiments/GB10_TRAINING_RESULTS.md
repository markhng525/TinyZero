# DGX Spark (GB10) GRPO Training Results

## Test Summary

**Date:** 2026-01-07
**Model:** Qwen/Qwen2.5-0.5B (494M parameters)
**Algorithm:** GRPO (Group Relative Policy Optimization)
**Backend:** SGLang with flashinfer (auto-selected for Blackwell)

## Hardware Profile

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| Compute Capability | SM 12.1, CUDA 13.0 |
| Memory | ~120GB unified (shared CPU/GPU) |
| Platform | ARM64 (aarch64) |
| PyTorch | 2.9.0+cu130 |
| SGLang | Built from source for SM 12.1 |

## Test Configuration

```yaml
Model: Qwen/Qwen2.5-0.5B
Batch Size: 32 (train), 32 (val)
Samples per prompt (n): 2
Max prompt length: 256
Max response length: 256
GPU memory utilization: 0.3
Algorithm: GRPO with KL loss (coef=0.001)
Test frequency: 1 (every step)
```

## Step 1 Results (First training step)

| Metric | Value |
|--------|-------|
| **Validation Score** | 4.2% (baseline) |
| **Policy Loss** | 0.056 |
| **Entropy Loss** | 1.421 |
| **KL Loss** | 0.001 |
| **Gradient Norm** | 1.120 |
| **Response Length (avg)** | 181.1 tokens |
| **Response Length (max)** | 256 tokens |
| **Prompt Length (avg)** | 143.6 tokens |

### Timing Breakdown (Step 1)

| Phase | Time | % of Total |
|-------|------|------------|
| Generation (rollout) | 96.6s | 12.9% |
| Reference log_prob | 2.3s | 0.3% |
| Advantage estimation | 0.03s | 0.0% |
| Actor update | 8.3s | 1.1% |
| **Validation/Testing** | **643.7s** | **85.7%** |
| **Total Step** | **750.9s** | 100% |

### Throughput Metrics

| Metric | Value |
|--------|-------|
| Tokens processed | 20,784 |
| Generation speed | 8.3 ms/token |
| Actor update speed | 0.4 ms/token |

## Key Findings

### 1. Training Loop Works End-to-End

The GRPO training loop successfully:
- Loads model with FSDP (auto-switches to NO_SHARD for single GPU)
- Initializes SGLang rollout engine with flashinfer backend
- Generates responses using temperature sampling
- Computes log probabilities for actor and reference
- Updates actor with policy gradient loss
- Runs validation and computes countdown task scores

### 2. Backend Selection

**Important:** Environment variables like `SGLANG_ATTENTION_BACKEND=triton` may NOT propagate to Ray workers correctly. SGLang auto-selected flashinfer for GB10, which appears to work well.

Log evidence:
```
WARNING:2026-01-07 01:11:45,635:Attention backend not explicitly specified.
Use flashinfer backend by default.
```

### 3. Validation is the Bottleneck

Validation takes **86% of step time** when `test_freq=1`. For faster iteration:
- Set `test_freq=5` or `test_freq=10` to validate less frequently
- Or reduce `val_batch_size` to speed up individual validation runs

### 4. Memory Usage

With conservative settings (gpu_memory_utilization=0.3):
- Model memory: ~2.76 GB allocated
- SGLang reserves ~3.4 GB initially
- Plenty of headroom in 120GB unified memory

## Recommended Optimizations

### For Faster Iteration (Development)

```bash
trainer.test_freq=10           # Validate every 10 steps instead of 1
data.val_batch_size=64         # Smaller validation batches
actor_rollout_ref.rollout.n=4  # 4 samples per prompt
data.train_batch_size=64       # Moderate batch size
```

### For Maximum Throughput (Production)

```bash
trainer.test_freq=20           # Validate every 20 steps
data.train_batch_size=256      # Large training batch
actor_rollout_ref.rollout.n=8  # 8 samples for robust GRPO
actor_rollout_ref.rollout.gpu_memory_utilization=0.5  # Use more memory
actor_rollout_ref.actor.ppo_micro_batch_size=16
```

### For Backend Comparison

To force a specific backend, set it programmatically in the SGLang engine initialization rather than via environment variables. The env var approach doesn't work reliably with Ray workers.

## Training Script Used

```bash
#!/bin/bash
# Minimal GRPO test for GB10
export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-0.5B
export DATA_DIR=/workspace/data/countdown
export SGLANG_KERNEL_DISABLE=1
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

/workspace/.venv.linux-aarch64/bin/python -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  algorithm.adv_estimator=grpo \
  trainer.test_freq=1 \
  trainer.total_epochs=1
```

## Known Issues

1. **PyTorch Warning:** GB10 (SM 12.1) exceeds PyTorch's officially supported range (8.0-12.0). This is a warning only and doesn't prevent operation.

2. **FA3 Import Errors:** Flash Attention 3 (FA3) isn't available for vision models. These are safely ignored as we're using text-only models.

3. **uv run Issue:** Using `uv run python` may reinstall CPU-only PyTorch. Always use the venv Python directly: `/workspace/.venv.linux-aarch64/bin/python`

## Conclusion

GRPO training on DGX Spark (GB10) works successfully with:
- SGLang inference engine (flashinfer backend auto-selected)
- FSDP training (NO_SHARD mode on single GPU)
- Qwen2.5-0.5B model
- Countdown arithmetic task

The main bottleneck is validation time. For longer training runs, increase `test_freq` to improve throughput significantly.
