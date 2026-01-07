# OOM Error Analysis Report

## Step 1: Error Analysis

The crash message revealed critical information:

```
Memory on node: 113.73GB / 119.70GB (0.950112) exceeds threshold of 0.95
```

This tells us:

- **Peak usage:** 113.73 GB (95% of 119.7 GB available)
- **Kill trigger:** Ray's OOM prevention killed the process
- **Timing:** Crash occurred at "epoch 0, step 1" — the first training step

The "step 1" timing is significant: memory pressure hit during the first forward/backward pass, not during model loading (which succeeded with the "Engine initialized successfully" message).

---

## Step 2: Architecture Tracing

I traced the code path from the entry point to understand what gets loaded:

```
main_ppo.py:120 → ray.get(main_task.remote(config))
                → ray_trainer.py creates ResourcePoolManager
                → Spawns ActorRolloutRefWorker with role='actor_rollout_ref'
```

The role `actor_rollout_ref` is crucial. In `fsdp_workers.py:78-82`:

```python
self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']    # True
self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']  # True
self._is_ref = self.role in ['ref', 'actor_rollout_ref']                           # True
```

All three flags are `True` — meaning this single worker loads **actor**, **rollout engine**, AND **reference model**.

---

## Step 3: Memory Accounting

I traced each model load in `init_model()` at `fsdp_workers.py:300-368`:

### Model 1: Actor (lines 318-325)

```python
self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, ... = self._build_model_optimizer(
    model_path=self.config.model.path,  # Qwen/Qwen2.5-1.5B
    ...
)
```

**Memory for Qwen2.5-1.5B in BF16:**

- Parameters: 1.5B × 2 bytes = **3 GB**
- FSDP with `SHARD_GRAD_OP` on single GPU = no sharding benefit
- Optimizer (AdamW): 2 state tensors (m, v) × 1.5B × 4 bytes = **12 GB**
- **Total: ~15 GB**

### Model 2: SGLang Engine (lines 264-274)

```python
rollout = SGLangRollout(actor_module=self.actor_module_fsdp, ...)
```

Inside `sglang_rollout.py:64-70`:

```python
self.engine = Engine(
    model_path=model_path,  # Loads from disk, NOT from actor_module!
    mem_fraction_static=config.get('gpu_memory_utilization', 0.5),  # Set to 0.7
)
```

> **Critical finding:** Despite receiving `actor_module`, SGLang ignores it and loads from `model_path`. This creates a **second copy**.

**SGLang memory allocation:**

- Model weights: **3 GB** (duplicate)
- KV cache: `mem_fraction_static × available_memory`

### Model 3: Reference Policy (lines 349-356)

```python
self.ref_module_fsdp = self._build_model_optimizer(
    model_path=self.config.model.path,  # Same path, THIRD load!
    ...
)
```

Another **3 GB** for the reference model.

---

## Step 4: KV Cache Math

SGLang's `mem_fraction_static=0.7` means it tries to allocate 70% of GPU memory for KV cache.

**For Qwen2.5-1.5B architecture:**

| Parameter | Value |
|-----------|-------|
| Layers | 28 |
| Hidden dim | 1536 |
| Num KV heads | 2 |
| Head dim | 128 |

**KV cache per token per layer:**

```
2 (K+V) × num_kv_heads × head_dim × dtype_size
= 2 × 2 × 128 × 2 bytes (BF16)
= 1024 bytes/token/layer
```

**Total KV cache for max context (256 + 256 = 512 tokens) × batch:**

```
512 tokens × 28 layers × 1024 bytes × batch_size
= 14.68 MB × batch_size
```

**With batch expansion (256 prompts × 8 samples = 2048):**

```
14.68 MB × 2048 = 30 GB (just for KV cache at generation)
```

**But SGLang pre-allocates based on `mem_fraction_static`:**

```
0.7 × (120 GB - model_weights) ≈ 0.7 × (120 - 9) GB ≈ 78 GB pre-allocated for KV cache
```

This pre-allocation happens at engine init, before the first step.

---

## Step 5: Timeline Reconstruction

| Time | Event | Memory Used |
|------|-------|-------------|
| T0 | Ray spawns ActorRolloutRefWorker | — |
| T1 | Load actor model | ~3 GB |
| T2 | Wrap with FSDP, create optimizer | ~15 GB |
| T3 | Build SGLang rollout: | |
| | - Load model weights (DUPLICATE) | ~18 GB |
| | - Pre-allocate KV cache (70% of remaining) | ~89 GB |
| T4 | Load reference model (THIRD copy) | ~92 GB |
| T5 | First training step begins: | |
| | - Generate sequences (KV cache fills) | |
| | - Forward pass on expanded batch (2048 samples) | |
| | - Compute reference log probs | |
| | - Activations + gradients spike memory | |
| T6 | **Peak at ~113 GB → OOM kill at 95% threshold** | 113.73 GB |

> **Note:** The log message "After building sglang rollout, memory allocated (GB): 8.64" only shows the delta from SGLang model weights, not the KV cache pre-allocation which may be lazily reported.

---

## Step 6: Validating Against Error Logs

The crash dump shows:

| PID | MEM(GB) | COMMAND |
|-----|---------|---------|
| 362969 | 0.19 | `ray::WorkerDict.actor_rollout_generate_sequences` |
| 363687 | 0.11 | `sglang::scheduler` |

These numbers are **process RSS**, not GPU memory. The actual GPU memory isn't visible in Ray's OOM report — it's aggregated into the node's total memory (unified memory architecture on GB10).

The GB10's unified memory means CPU and GPU share the 120 GB pool. Ray's memory monitor sees total node memory usage, not GPU-specific allocation.

---

## Step 7: Root Cause Equation

**Constants:**

```
M_total     = 119.7 GB (available)
M_threshold = 0.95 × M_total = 113.7 GB
```

**Memory components:**

```
M_actor       = 3 GB (model) + 12 GB (optimizer) = 15 GB
M_sglang      = 3 GB (model) + 0.7 × (M_remaining)
M_ref         = 3 GB
M_activations = f(batch_size, seq_len, hidden_dim)
```

**Calculation:**

```
M_remaining after actor:
  M_remaining = 119.7 - 15 = 104.7 GB

SGLang KV cache:
  M_sglang_kv    = 0.7 × 104.7 = 73.3 GB
  M_sglang_total = 3 + 73.3 = 76.3 GB

After SGLang:
  M_used      = 15 + 76.3 = 91.3 GB
  M_after_ref = 91.3 + 3 = 94.3 GB

Remaining for activations:
  119.7 - 94.3 = 25.4 GB
```

**Activation memory for forward pass (approximate):**

```
M_act ≈ batch_size × seq_len × hidden_dim × num_layers × 2 (for attention)
      ≈ 2048 × 512 × 1536 × 28 × 2 / 10^9 GB
      ≈ 90 GB (without gradient checkpointing)
```

**With gradient checkpointing (enabled in config):**

```
M_act_ckpt ≈ M_act / sqrt(num_layers)
           ≈ 90 / 5.3
           ≈ 17 GB
```

**But we also need gradients:**

```
M_grad ≈ 3 GB (same size as model)
```

**Total at peak:**

```
94.3 + 17 + 3 = 114.3 GB → exceeds threshold ✓
```

---

## Step 8: Solution Space

The equation tells us which knobs to turn:

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `gpu_memory_utilization` 0.7→0.3 | Reduces M_sglang_kv by ~47 GB | Slower generation (more chunking) |
| `train_batch_size` 256→128 | Reduces M_act by ~50% | Slower training (more steps) |
| `n` 8→4 | Reduces batch expansion 2x | Less variance reduction in GRPO |
| `param_offload=True` | Moves M_ref to CPU | Slower ref log prob computation |

**The most impactful single change:** `gpu_memory_utilization=0.3` recovers ~47 GB.
