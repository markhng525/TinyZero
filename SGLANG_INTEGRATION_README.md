# SGLang Integration for veRL HybridEngine

**Status**: Experimental - Separate Ray Actor mode is the recommended approach
**Date**: 2026-01-11
**Target Hardware**: DGX Spark (GB10) with ~120GB unified memory

---

## Executive Summary

This document captures the complete effort to integrate SGLang as a rollout engine for RLHF/GRPO training in the veRL framework. After extensive experimentation, we found that **Separate Ray Actor mode** is the recommended approach.

The key insight: **"Run SGLang as a separate Ray actor (not a subprocess spawned from another Ray actor)"**

### Key Findings

| Approach | Result | Issue |
|----------|--------|-------|
| Colocated Engine (subprocess) | Failed | SGLang's scheduler subprocess conflicts with Ray's CUDA context |
| `update_weights_from_tensor` | Failed* | Segfaults when called from Ray actor's subprocess |
| `update_weights_from_distributed` | Failed | "Duplicate GPU detected" - NCCL can't work parent/subprocess on same GPU |
| `update_weights_from_disk` | Partial | Works standalone, hangs in Ray subprocess |
| **Separate Ray Actor** | **Works** | SGLang as top-level Ray actor - clean CUDA context, in-memory weight sync |
| HTTP Server Mode | Works | Fallback option with higher overhead |

*`update_weights_from_tensor` works when SGLang runs as a **top-level** Ray actor (not subprocess)

---

## Code Changes Overview

### New Files Created

```
verl/third_party/sglang/
├── __init__.py          # Version detection, exports Engine/ServerEngine
├── config.py            # LoadFormat enum (dummy, auto, hf, dtensor)
├── engine.py            # Custom Engine wrapper with disk-based weight sync
├── server_engine.py     # HTTP server-based engine (recommended)
└── weight_loaders.py    # DTensor/HF weight loading utilities

verl/workers/sharding_manager/
├── fsdp_sglang.py       # FSDPSGLangShardingManager (in-process attempt)
└── separate_sglang.py   # SeparateSGLangShardingManager (HTTP server mode)

verl/workers/rollout/sglang_rollout/
└── sglang_ray_actor.py  # Ray actor wrapper for SGLang server
```

### Modified Files

| File | Changes |
|------|---------|
| `verl/workers/fsdp_workers.py` | Added `sglang_server` rollout mode support |
| `verl/workers/rollout/sglang_rollout/__init__.py` | Export new rollout classes |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | Added `SGLangServerRollout` class |
| `verl/workers/sharding_manager/__init__.py` | Export new sharding managers |
| `verl/trainer/ppo/ray_trainer.py` | Minor logging improvements |
| `verl/utils/debug/` | Added performance profiling utilities |
| `.devcontainer/Dockerfile` | FlashInfer backend configuration |

---

## Architecture Decision: Why Separate Ray Actor Mode

### The Root Cause Problem

SGLang uses a **subprocess architecture** where the scheduler (which holds model weights) runs in a separate process:

```
PROBLEMATIC: Colocated Engine (What Doesn't Work)
─────────────────────────────────────────────────
Ray Actor (CUDA Context A)
    └── SGLang Engine
        └── Scheduler Subprocess (CUDA Context B)  <-- CONFLICT!
            └── Model weights on GPU

When weight sync is called from Ray Actor → goes to subprocess → CUDA conflict!
```

When FSDP (in the Ray actor) tries to sync weights to SGLang:
1. **Pickle serialization** of CUDA tensors fails across process boundaries
2. **NCCL** can't establish communication - both processes claim the same GPU
3. **Disk-based** approach works but the subprocess's CUDA read can hang waiting for the parent

### The Separate Ray Actor Solution (Recommended)

```
SOLUTION: Separate Ray Actor (What Works)
─────────────────────────────────────────
Ray Actor 1 (FSDP Worker)          Ray Actor 2 (SGLang Inference)
├── FSDP Training                  ├── SGLang Engine (TOP-LEVEL!)
├── state_dict extraction          │   └── Scheduler Subprocess (OK - no conflict)
└── ray.put(state_dict) ──────────►├── sync_weights_from_dict()
                                   └── generate() ──────────────► results
    Ray Object Store                   └── Clean CUDA context!
    (zero-copy on same node)
```

The key insight: **SGLang as a TOP-LEVEL Ray actor** means its subprocess has a clean CUDA context.

Benefits:
- **Clean CUDA context** - SGLang subprocess doesn't conflict with Ray parent
- **In-memory weight sync** - `update_weights_from_tensor` works!
- **Zero-copy transfer** - Ray object store enables fast weight sync on same node
- **Still HybridEngine** - same GPU can be used for training and inference (time-multiplexed)

### Alternative: HTTP Server Mode (Fallback)

```
Process 1: Ray Actor          Process 2: SGLang Server (outside Ray)
├── FSDP Training            ├── HTTP Server (:30000)
├── Weight sync via HTTP ────┼──> /update_weights_from_disk
└── Generation via HTTP ─────┼──> /generate
```

This works but has higher overhead (HTTP + disk-based sync). Use only if Ray actor approach fails.

---

## File Details

### `verl/workers/rollout/sglang_rollout/sglang_ray_actor.py` (RECOMMENDED)

**SGLang as a Separate Top-Level Ray Actor** - the solution that works:

```python
@ray.remote(num_gpus=0)  # Don't request GPU from Ray - we share with FSDP worker
class SGLangInferenceActor:
    """SGLang runs as TOP-LEVEL Ray actor - clean CUDA context."""

    def __init__(self, model_path, ...):
        # SGLang Engine created here is TOP-LEVEL (not subprocess of another Ray actor)
        self.engine = sgl.Engine(model_path=model_path, ...)

    def sync_weights_from_dict(self, state_dict):
        # Uses update_weights_from_tensor - WORKS because clean CUDA context!
        named_tensors = [(name, tensor) for name, tensor in state_dict.items()]
        self.engine.update_weights_from_tensor(named_tensors)

    def offload_weights(self):
        # release_memory_occupation() frees ALL GPU memory (weights + KV cache)
        self.engine.release_memory_occupation()

    def generate(self, prompts, sampling_params):
        return self.engine.generate(prompts, sampling_params)


class SGLangActorRollout:
    """Rollout class that delegates to separate SGLang Ray actor."""

    def __init__(self, config, ...):
        # Create SGLang as SEPARATE Ray actor (not in same process!)
        self.inference_actor = SGLangInferenceActor.remote(model_path=...)
```

**Status**: Recommended approach. Uses in-memory weight sync via Ray object store.

### `verl/workers/sharding_manager/separate_sglang.py` (RECOMMENDED)

Sharding manager for Separate Ray Actor mode:

```python
class SeparateSGLangShardingManager:
    def __enter__(self):
        # 1. Extract full state_dict from FSDP (on rank 0)
        params = self.module.state_dict()

        # 2. Put in Ray object store (zero-copy on same node)
        params_ref = ray.put(params)

        # 3. Call SGLang actor's sync method via Ray RPC
        ray.get(self.sglang_actor.sync_weights_from_dict.remote(params_ref))

    def __exit__(self):
        # Offload weights from SGLang to free GPU for training
        ray.get(self.sglang_actor.offload_weights.remote())
```

**Status**: Working. Enables HybridEngine with clean CUDA isolation.

### `verl/third_party/sglang/engine.py` (Experimental)

Custom Engine wrapper for colocated (in-process) integration:

```python
class Engine:
    def sync_model_weights(self, actor_weights, load_format):
        # Disk-based weight sync (fallback for colocated mode)
        # 1. Save FSDP state dict as safetensors to temp dir
        # 2. Call engine.update_weights_from_disk(temp_dir)
```

**Status**: Works standalone but fails in Ray due to subprocess CUDA conflicts. Use Separate Ray Actor mode instead.

### `verl/third_party/sglang/server_engine.py` (Fallback)

HTTP server-based engine:

```python
class ServerEngine:
    def __init__(self, model_path, port=30000, ...):
        # Launch SGLang as completely separate HTTP server (outside Ray)

    def sync_model_weights(self, actor_weights, load_format):
        # POST to http://localhost:{port}/update_weights_from_disk
```

**Status**: Works but has higher overhead. Use only if Ray actor approach fails.

---

## Configuration

### Recommended Config (Separate Ray Actor Mode)

```yaml
actor_rollout_ref:
  rollout:
    name: sglang_actor          # Use separate Ray actor mode
    model_path: Qwen/Qwen2.5-1.5B-Instruct
    dtype: bfloat16
    gpu_memory_utilization: 0.5 # Can be higher with HybridEngine
    enforce_eager: false        # CUDA graphs enabled (SGLang default)
    tensor_model_parallel_size: 1
    response_length: 1024
    temperature: 1.0
```

### Alternative Config (HTTP Server Mode - Fallback)

```yaml
actor_rollout_ref:
  rollout:
    name: sglang_server         # Use HTTP server mode
    dtype: bfloat16
    gpu_memory_utilization: 0.3 # Conservative for GB10
    enforce_eager: true         # Disable CUDA graphs for stability
    server_port: 30000
```

### Known Incompatibilities

```bash
# DO NOT SET with SGLang (causes crash):
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SGLang's torch_memory_saver explicitly rejects expandable_segments
```

---

## Test Results

### HTTP Server Mode Training (Verified Working)

```
============================================================
TEST SUMMARY
============================================================
  standalone: PASSED
  training: PASSED (3 steps completed)
------------------------------------------------------------
Overall: PASSED

Training Metrics:
  actor/kl_loss: 0.001
  actor/entropy_loss: 1.404
  actor/pg_loss: -0.002
  actor/grad_norm: 0.114
  timing_s/gen: 9.051
  timing_s/update_actor: 2.206
  timing_s/step: 24.980
```

---

## Memory Considerations

### GB10 Memory Budget (~120GB unified)

With HTTP server mode, memory is split across phases:

**Training Phase** (SGLang server idle):
- FSDP Model: ~6GB
- Gradients: ~6GB
- Optimizer states: ~12GB (lazy load on first backward)
- Activations: ~20GB (with gradient checkpointing)
- **Total**: ~44GB

**Generation Phase** (training tensors idle):
- FSDP Model: ~6GB (shared)
- SGLang Model: ~6GB (separate load)
- KV Cache: ~40GB (at 30% utilization)
- **Total**: ~52GB

### Logits Memory Issue

Qwen2.5's vocabulary (151,936 tokens) causes large logits tensors:
- Logits: `batch × seq × vocab × 2 bytes`
- For batch=4, seq=1280: **1.56 GB per micro-batch**

**Solution**: Use fused linear cross-entropy kernel (see `experiments/FUSED_KERNEL_INTEGRATION.md`)

---

## Lessons Learned

### 1. Subprocess Architecture is Fundamental

SGLang's design requires the scheduler to run as a subprocess. This is not a bug - it's how SGLang achieves its performance. Any in-process integration attempt will face CUDA context conflicts.

### 2. HTTP Mode Has Acceptable Overhead

The ~10-50ms HTTP overhead per generation request is negligible compared to:
- Actual generation time: 1-10 seconds
- Training step time: 10-30 seconds

### 3. Weight Sync Options Have Trade-offs

| Method | Speed | Memory | Reliability |
|--------|-------|--------|-------------|
| NCCL | Fast | Low | Requires multi-GPU |
| Disk (safetensors) | Medium | Low | Works everywhere |
| HTTP + Disk | Medium | Low | Most reliable |

### 4. Load Format Matters

- `load_format='dummy'` does NOT support weight updates (DummyModelLoader limitation)
- `load_format='auto'` loads weights from disk first, then can update
- Use `auto` for HTTP server mode, despite the initial memory cost

---

## Future Work

### Option 1: Wait for SGLang In-Process Mode

If SGLang adds an in-process execution mode (no subprocess), the `engine.py` implementation should work.

### Option 2: Multi-GPU Setup

With multiple GPUs, NCCL-based sync should work:
- GPU 0: FSDP training
- GPU 1: SGLang inference

### Option 3: Fused Kernel Integration

Port the fused linear cross-entropy kernel from upstream veRL (PR #462) to reduce memory pressure, enabling larger batch sizes.

---

## References

### SGLang Documentation
- [update_weights_from_disk test](https://github.com/sgl-project/sglang/blob/main/test/srt/rl/test_update_weights_from_disk.py)
- [update_weights_from_distributed test](https://github.com/sgl-project/sglang/blob/main/test/srt/rl/test_update_weights_from_distributed.py)
- [Issue #3646: NCCL weight sync](https://github.com/sgl-project/sglang/issues/3646)

### veRL Documentation
- [vLLM HybridEngine reference](verl/workers/rollout/vllm_rollout/vllm_rollout.py)
- [FSDP-vLLM sharding manager](verl/workers/sharding_manager/fsdp_vllm.py)

### Related Documents
- `/workspace/SGLANG_HYBRIDENGINE_WALKTHROUGH.md` - Comprehensive educational walkthrough
- `experiments/FUSED_KERNEL_INTEGRATION.md` - Memory optimization via fused kernels

---

## Quick Start

### Using Separate Ray Actor Mode (Recommended)

```python
from verl.workers.rollout.sglang_rollout.sglang_ray_actor import SGLangActorRollout
from verl.workers.sharding_manager.separate_sglang import SeparateSGLangShardingManager

# Create rollout (SGLang runs as separate Ray actor)
rollout = SGLangActorRollout(
    actor_module=fsdp_module,  # For interface compatibility
    config=rollout_config,
    tokenizer=tokenizer,
    model_hf_config=model_config,
)

# Create sharding manager
sharding_manager = SeparateSGLangShardingManager(
    module=fsdp_module,
    sglang_actor=rollout.inference_actor,
)

# Training loop
for batch in dataloader:
    # Inference phase: sync weights to SGLang, generate
    with sharding_manager:
        outputs = rollout.generate_sequences(prompts)

    # Training phase: SGLang memory released, FSDP trains
    loss = compute_loss(outputs)
    loss.backward()
    optimizer.step()
```

### Config-based Usage

```python
# In your training config
config = {
    'actor_rollout_ref': {
        'rollout': {
            'name': 'sglang_actor',  # Use separate Ray actor mode
            'model_path': 'Qwen/Qwen2.5-1.5B-Instruct',
            'dtype': 'bfloat16',
            'gpu_memory_utilization': 0.5,
        }
    }
}
```

---

## Summary

The SGLang integration effort revealed that the key to success is **running SGLang as a top-level Ray actor** rather than a subprocess spawned from another Ray actor. This provides clean CUDA context isolation while still enabling efficient in-memory weight synchronization.

**Recommended approach**: Use `SGLangActorRollout` + `SeparateSGLangShardingManager`

**Key insight**: SGLang's subprocess architecture is fine - the problem was *nesting* it under another Ray actor. As a top-level Ray actor, SGLang works correctly with `update_weights_from_tensor` and full memory management APIs.

**Alternative**: If the Ray actor approach fails for any reason, HTTP Server mode (`sglang_server`) provides a reliable fallback with slightly higher overhead.
