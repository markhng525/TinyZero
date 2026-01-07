# Qwen2.5-1.5B GRPO Configuration Report for DGX Spark (GB10)

## Executive Summary

This report provides optimized configuration settings for training Qwen2.5-1.5B with GRPO on the NVIDIA DGX Spark (GB10) platform. Recommendations are based on empirical testing with Qwen2.5-0.5B and scaling analysis.

**Key insight:** Maximize **throughput (tokens/sec)** by using available VRAM for larger KV cache and batch sizes. The goal is tokens/hour, not steps/hour.

---

## Hardware Profile

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| Compute Capability | SM 12.1, CUDA 13.0 |
| Memory | ~120 GB unified (shared CPU/GPU) |
| Platform | ARM64 (aarch64) |
| PyTorch | 2.9.0+cu130 |
| SGLang Backend | flashinfer (auto-selected) |

---

## Empirical Baseline (Qwen2.5-0.5B)

### Test Configuration
```yaml
Model: Qwen2.5-0.5B (494M parameters)
Batch Size: 32
Samples per Prompt (n): 2
GPU Memory Utilization: 0.3
```

### Measured Results

| Metric | Value |
|--------|-------|
| Memory Allocated | 2.76 GB |
| Memory Reserved | 3.4 GB |
| Tokens per Step | 20,784 |
| Generation Time | 96.6s |
| Actor Update Time | 8.3s |
| Validation Time | 643.7s (86% of step!) |
| Total Step Time | 750.9s |
| Initial Validation Score | 4.2% |

### Timing Breakdown

| Phase | Time | % of Total |
|-------|------|------------|
| Generation (rollout) | 96.6s | 12.9% |
| Reference log_prob | 2.3s | 0.3% |
| Advantage computation | 0.03s | 0.0% |
| Actor update | 8.3s | 1.1% |
| **Validation** | **643.7s** | **85.7%** |

**Critical insight:** Validation dominated step time. This means "more steps" actually multiplies overhead, not reduces it. Set `test_freq=50+` to minimize wasted compute.

---

## Understanding gpu_memory_utilization

### What It Actually Controls

`gpu_memory_utilization` sets the fraction of GPU memory allocated to SGLang's **KV cache**:

```
KV Cache Size = gpu_memory_utilization × Total VRAM
             = 0.8 × 120 GB = 96 GB
```

### Theoretical vs Empirical Reality

**Theory** (what you'd expect on discrete GPUs):
```
gpu_memory_utilization ↑ → KV cache ↑ → concurrent sequences ↑ → tokens/sec ↑
```

**Empirical Reality on GB10** (see test results below):
```
gpu_memory_utilization: 0.5-0.7 → optimal throughput
gpu_memory_utilization: 0.8+   → throughput DECREASES
```

The GB10's unified memory architecture and the compute-bound nature of this workload mean that higher KV cache allocation doesn't improve throughput and actually hurts it due to memory pressure.

### Memory Budget (120 GB)

From stress test with Qwen2.5-1.5B:
- Model + Optimizer (FSDP): ~10 GB reserved
- At gpu_memory_utilization=0.7: ~96 GB total used
- **Available headroom: ~24 GB**

This means we can safely push to `gpu_memory_utilization=0.8`.

---

## Throughput-First Strategy

### The Correct Optimization Target

For RL training on a single powerful GPU:

| Metric | Why It Matters |
|--------|----------------|
| **Tokens/hour** | Total learning signal processed |
| Steps/hour | Irrelevant if steps are tiny |
| GPU utilization | Should be maximized |

### Why "More Steps" Is Wrong

The naive intuition "smaller batches = more updates = faster learning" fails because:

1. **Validation overhead**: Each step triggers validation overhead. With 86% validation time, more steps = more wasted compute.
2. **Fixed per-step costs**: Rollout initialization, weight sync, logging all happen per-step regardless of batch size.
3. **VRAM is free compute**: Unused VRAM is wasted capacity. Larger KV cache = higher tokens/sec at no extra cost.
4. **Gradient quality**: Larger batches give more stable gradients, not worse ones.

### The Correct Approach

| Action | Effect |
|--------|--------|
| Maximize `gpu_memory_utilization` | Higher tokens/sec via larger KV cache |
| Maximize batch size | More samples per step, less overhead ratio |
| Increase `n` | Better advantage estimation per prompt |
| Push `test_freq` way out | Minimize validation overhead |

---

## Recommended Configuration

### Production Training Script

```bash
#!/bin/bash
# Qwen2.5-1.5B GRPO Training for GB10
# Strategy: Maximize throughput via large KV cache and batches
set -x

export N_GPUS=1
export BASE_MODEL=Qwen/Qwen2.5-1.5B
export DATA_DIR=/workspace/data/countdown

/workspace/.venv.linux-aarch64/bin/python -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=64 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=16 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=50 \
  trainer.project_name=TinyZero-GB10 \
  trainer.experiment_name=grpo-qwen2.5-1.5b \
  trainer.total_epochs=15
```

### Configuration Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gpu_memory_utilization` | 0.7 | Empirically optimal for GB10 (see test results below) |
| `train_batch_size` | 256 | Large batch for throughput, stable gradients |
| `n` | 8 | Excellent advantage estimation (8 samples per prompt) |
| `ppo_mini_batch_size` | 128 | Half of train_batch for gradient accumulation |
| `ppo_micro_batch_size` | 16 | Efficient forward passes with available memory |
| `test_freq` | 50 | Minimize validation overhead (was 86% of time!) |
| `gradient_checkpointing` | True | Trade compute for memory |
| `total_epochs` | 15 | Full training run |

### Expected Performance

| Metric | Estimated Value |
|--------|-----------------|
| Samples per step | 2048 (256 × 8) |
| Tokens per step | ~500K+ |
| Generation throughput | Higher tokens/sec with 0.8 util |
| Step time (no val) | ~10-15 min |
| Steps per epoch | 128 (32768 / 256) |
| Validation frequency | Every 50 steps |

---

## Alternative Configurations

### Conservative (If OOM)

```yaml
train_batch_size: 128
n: 4
gpu_memory_utilization: 0.5
ppo_micro_batch_size: 8
```

### Maximum Throughput

```yaml
train_batch_size: 256
n: 8
gpu_memory_utilization: 0.5  # Tied with 0.7 for best throughput
ppo_micro_batch_size: 16
```

**Note:** Do NOT increase gpu_memory_utilization above 0.7 - empirical tests show throughput decreases.

---

## Verified Findings

1. **No special env vars needed**: `SGLANG_KERNEL_DISABLE` and `SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK` can be omitted
2. **flashinfer auto-selected**: SGLang chooses flashinfer for GB10 automatically
3. **FSDP → NO_SHARD**: Single GPU automatically uses no sharding
4. **PyTorch warning is safe**: SM 12.1 exceeds official support (8.0-12.0) but works
5. **gpu_memory_utilization=0.7 tested stable**: 96GB used with 24GB headroom

---

## Throughput Verification Test Results

### Production Batch Test (batch=256, max_tokens=256)

| GPU Mem Util | Tokens/sec | Speedup |
|--------------|------------|---------|
| **50%**      | **4972.9** | **1.00x** |
| **70%**      | **4987.5** | **1.00x** |
| 80%          | 4804.6     | 0.97x |
| 85%          | 4768.7     | 0.96x |
| 90%          | 4626.6     | 0.93x |

### Analysis

**Key Finding:** Throughput **decreased by 7%** as gpu_memory_utilization increased from 50% to 90%.

This confirms the workload is **compute-bound, not memory-bound**:
- 50% of 120GB = 60GB KV cache is more than sufficient for 256 sequences
- Higher utilization causes memory pressure overhead without benefit
- Peak throughput achieved at 50-70% utilization

### Why Higher Isn't Better

| Factor | Effect |
|--------|--------|
| Memory pressure | Higher utilization leaves less headroom for dynamic allocations |
| CUDA graph overhead | More memory captured reduces flexibility |
| Unified memory | GB10's unified architecture may have different tradeoffs than discrete GPUs |

### Recommendation: Use 0.5-0.7

For Qwen2.5-1.5B on GB10:
- **Optimal:** `gpu_memory_utilization=0.5` or `0.7` (tied for best throughput)
- **Avoid:** 0.8+ (measurable throughput loss, no benefit)
- **Headroom:** Lower utilization provides safety margin for variable-length generations

To run the test yourself:
```bash
/workspace/.venv.linux-aarch64/bin/python /workspace/experiments/test_throughput.py \
    --model Qwen/Qwen2.5-1.5B --batch-size 256 --max-tokens 256 \
    --utils 0.5 0.7 0.8 0.85 0.9
```

---

## Conclusion

For Qwen2.5-1.5B on GB10 with GRPO:
- **Use `gpu_memory_utilization=0.7`** - empirically optimal for this hardware
- Use **large batches** (256) with **`n=8`** for throughput and advantage quality
- Set **`test_freq=50`** to avoid the validation bottleneck (was 86% of time!)
- Focus on **tokens/hour**, not steps/hour

**Key empirical finding:** Higher gpu_memory_utilization (0.8+) actually *decreases* throughput on GB10 due to memory pressure. The workload is compute-bound, not memory-bound.
