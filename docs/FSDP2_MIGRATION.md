# FSDP2 Migration Guide for veRL

## Overview

This document provides a complete guide for migrating veRL from FSDP1 (`FullyShardedDataParallel` wrapper class) to FSDP2 (`fully_shard` function-based API) on DGX Spark (ARM64/GB10, CUDA 13.0, PyTorch 2.9.0+cu130).

**Implementation Decisions**:
- Direct replacement (no feature flag)
- Fresh start (no checkpoint conversion needed)
- Full migration (all files)

---

## Background

### Why FSDP2?

| Benefit | Description |
|---------|-------------|
| **Memory** | 7% lower peak GPU memory (DTensor vs FlatParameter) |
| **Throughput** | 1.5% faster average throughput |
| **Checkpointing** | Communication-free sharded saves |
| **Flexibility** | Better LoRA/partial freezing support |
| **Semantics** | Per-parameter sharding (easier debugging) |

### Key API Differences

| FSDP1 | FSDP2 |
|-------|-------|
| `FullyShardedDataParallel(model, ...)` | `fully_shard(model, ...)` |
| `MixedPrecision(...)` | `MixedPrecisionPolicy(...)` |
| `ShardingStrategy.FULL_SHARD` | `reshard_after_forward=True` |
| `ShardingStrategy.SHARD_GRAD_OP` | `reshard_after_forward=False` |
| `auto_wrap_policy` | Manual hierarchical `fully_shard()` |
| `FSDP.state_dict_type()` | `get_model_state_dict()` from DCP |
| `FSDP.summon_full_params()` | `model.unshard()` / `model.reshard()` |
| `use_orig_params=False` | Always True (DTensor) |
| `sync_module_states=True` | Not needed |
| `param._local_shard` | `param.data` (DTensor handles natively) |

---

## Files to Modify

| Priority | File | Complexity | Description |
|----------|------|------------|-------------|
| 1 | `verl/utils/fsdp_v2_utils.py` (new) | Medium | Create FSDP2 helpers |
| 2 | `verl/utils/fsdp_utils.py` | Medium | Remove `_local_shard`, update wrap policy |
| 3 | `verl/workers/fsdp_workers.py` | High | Main worker migration |
| 4 | `verl/trainer/fsdp_sft_trainer.py` | Medium | SFT trainer migration |
| 5 | `verl/workers/sharding_manager/fsdp_vllm.py` | High | vLLM integration |
| 6 | `verl/workers/actor/dp_actor.py` | Low | Type checks |
| 7 | `verl/workers/critic/dp_critic.py` | Low | Type checks |
| 8 | `verl/workers/rollout/hf_rollout.py` | Medium | `summon_full_params` removal |

---

## Step-by-Step Implementation

### Step 1: Create FSDP2 Utilities

**Create new file**: `verl/utils/fsdp_v2_utils.py`

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
FSDP2 utility functions for veRL.

FSDP2 (PyTorch 2.9+) uses a functional API with DTensor-based sharding
instead of the class-based FSDP1 wrapper.
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from transformers.trainer_pt_utils import get_module_class_from_name
from typing import Set, Optional


def get_transformer_layer_classes(
    module: nn.Module,
    config: Optional[dict] = None
) -> Set[type]:
    """
    Extract transformer layer classes for hierarchical FSDP2 wrapping.

    Args:
        module: The model to extract layer classes from
        config: Optional config with 'transformer_layer_cls_to_wrap' or 'min_num_params'

    Returns:
        Set of layer classes to wrap individually
    """
    if config is None:
        config = {}

    if config.get('disable', False):
        return set()

    # Try to get transformer layer classes from config or model
    default_transformer_cls_names = getattr(module, "_no_split_modules", None)
    transformer_layer_cls_names = config.get(
        "transformer_layer_cls_to_wrap",
        default_transformer_cls_names
    )

    if transformer_layer_cls_names is None:
        return set()

    transformer_cls_to_wrap = set()
    for layer_class_name in transformer_layer_cls_names:
        layer_cls = get_module_class_from_name(module, layer_class_name)
        if layer_cls is not None:
            transformer_cls_to_wrap.add(layer_cls)

    return transformer_cls_to_wrap


def apply_fsdp2_wrapping(
    model: nn.Module,
    mesh: DeviceMesh,
    transformer_layer_cls: Set[type],
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """
    Apply FSDP2 hierarchical wrapping to a model.

    FSDP2 modifies the model in-place (no new wrapper object).
    Wraps transformer layers first, then the root module.

    Args:
        model: The model to wrap
        mesh: DeviceMesh for sharding
        transformer_layer_cls: Set of layer classes to individually wrap
        mp_policy: Mixed precision policy (optional)
        reshard_after_forward: Whether to reshard after forward
            - True = FULL_SHARD equivalent (frees memory after forward)
            - False = SHARD_GRAD_OP equivalent (keeps params for backward)

    Returns:
        The same model instance (modified in-place, now also isinstance FSDPModule)
    """
    # First, wrap each transformer layer individually
    for name, module in model.named_modules():
        if type(module) in transformer_layer_cls:
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )

    # Then wrap the root model
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )

    return model


def create_mixed_precision_policy(
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> MixedPrecisionPolicy:
    """
    Create FSDP2 MixedPrecisionPolicy.

    Note: FSDP2 uses 'output_dtype' instead of FSDP1's 'buffer_dtype'.
    """
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
    )
```

---

### Step 2: Update Core Utilities

**File**: `verl/utils/fsdp_utils.py`

#### 2.1 Update imports (add at top)
```python
# Add these imports
from torch.distributed.fsdp import FSDPModule
```

#### 2.2 Update `offload_fsdp_param_and_grad` function (lines 93-100)

**Before**:
```python
def offload_fsdp_param_and_grad(module, offload_grad=False):
    for _, param in module.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to("cpu", non_blocking=True)
        param.data = param.data.to('cpu', non_blocking=True)
        if offload_grad and param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()
```

**After**:
```python
def offload_fsdp_param_and_grad(module, offload_grad=False):
    """
    Offload FSDP parameters and gradients to CPU.

    In FSDP2, parameters are DTensors. The .data attribute
    points directly to the local shard (no _local_shard attribute).
    """
    for _, param in module.named_parameters():
        # DTensor .data is the local shard in FSDP2
        param.data = param.data.to('cpu', non_blocking=True)
        if offload_grad and param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()
```

#### 2.3 Update `load_fsdp_param_and_grad` function (lines 103-110)

**Before**:
```python
def load_fsdp_param_and_grad(module, device_id, load_grad=False):
    for _, param in module.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to(device_id, non_blocking=True)
        param.data = param.data.to(device_id, non_blocking=True)
        if load_grad and param.grad is not None:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()
```

**After**:
```python
def load_fsdp_param_and_grad(module, device_id, load_grad=False):
    """
    Load FSDP parameters and gradients back to GPU.

    In FSDP2, parameters are DTensors. The .data attribute
    points directly to the local shard.
    """
    for _, param in module.named_parameters():
        # DTensor .data is the local shard in FSDP2
        param.data = param.data.to(device_id, non_blocking=True)
        if load_grad and param.grad is not None:
            param.grad = param.grad.to(device_id, non_blocking=True)
    torch.cuda.empty_cache()
```

#### 2.4 Remove `init_fn` function (lines 29-33)

FSDP2 doesn't need `param_init_fn` - it handles initialization differently.
Remove or deprecate this function.

---

### Step 3: Migrate Main Workers

**File**: `verl/workers/fsdp_workers.py`

#### 3.1 Update imports (around line 122)

**Before**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
```

**After**:
```python
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from verl.utils.fsdp_v2_utils import apply_fsdp2_wrapping, get_transformer_layer_classes, create_mixed_precision_policy
```

#### 3.2 Update `_build_model_optimizer` method (lines 182-222)

**Before** (mixed precision setup):
```python
mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

if self._is_ref:
    mixed_precision = None

auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))
```

**After**:
```python
if self._is_ref:
    mp_policy = None
else:
    mp_policy = create_mixed_precision_policy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=buffer_dtype,
    )

transformer_layer_cls = get_transformer_layer_classes(
    module=actor_module,
    config=fsdp_config.get('wrap_policy', None)
)
```

**Before** (FSDP wrapping, lines 212-222):
```python
actor_module_fsdp = FSDP(
    actor_module,
    param_init_fn=init_fn,
    use_orig_params=False,
    auto_wrap_policy=auto_wrap_policy,
    device_id=torch.cuda.current_device(),
    sharding_strategy=sharding_strategy,
    mixed_precision=mixed_precision,
    sync_module_states=True,
    device_mesh=self.device_mesh,
    forward_prefetch=False)
```

**After**:
```python
# Determine sharding strategy
reshard_after_forward = len(transformer_layer_cls) > 0  # FULL_SHARD if hierarchical wrapping

# Apply FSDP2 hierarchical wrapping (modifies in-place)
actor_module_fsdp = apply_fsdp2_wrapping(
    model=actor_module,
    mesh=self.device_mesh,
    transformer_layer_cls=transformer_layer_cls,
    mp_policy=mp_policy,
    reshard_after_forward=reshard_after_forward,
)
```

#### 3.3 Update checkpoint saving (find `save_checkpoint` method)

**Before**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
    state_dict = self.actor.actor_module.state_dict()
```

**After**:
```python
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions

state_dict_options = StateDictOptions(
    full_state_dict=True,
    cpu_offload=True,
)
state_dict = get_model_state_dict(
    self.actor.actor_module,
    options=state_dict_options,
)
```

---

### Step 4: Migrate Sharding Manager

**File**: `verl/workers/sharding_manager/fsdp_vllm.py`

#### 4.1 Update imports

**Before**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig
```

**After**:
```python
from torch.distributed.fsdp import FSDPModule
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
```

#### 4.2 Update `__init__` method

**Before**:
```python
def __init__(self, module: FSDP, ...):
    if full_params:
        FSDP.set_state_dict_type(self.module,
                                 state_dict_type=StateDictType.FULL_STATE_DICT,
                                 state_dict_config=FullStateDictConfig())
    else:
        FSDP.set_state_dict_type(self.module,
                                 state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                 state_dict_config=ShardedStateDictConfig())
```

**After**:
```python
def __init__(self, module: FSDPModule, ...):
    # FSDP2 always returns sharded DTensor state dicts
    # No need to set state dict type - handled in __enter__
    self.module = module
    self.full_params = full_params
```

#### 4.3 Update `__enter__` method

**Before**:
```python
def __enter__(self):
    params = self.module.state_dict()
    load_format = 'hf' if self.full_params else 'dtensor'
    self.inference_engine.sync_model_weights(params, load_format=load_format)
```

**After**:
```python
def __enter__(self):
    if self.full_params:
        # Get full state dict for HF format
        options = StateDictOptions(full_state_dict=True, cpu_offload=False)
        params = get_model_state_dict(self.module, options=options)
        load_format = 'hf'
    else:
        # Get sharded state dict (DTensors)
        params = self.module.state_dict()
        # Convert DTensors to regular tensors if vLLM doesn't support DTensor
        params = {
            k: v.full_tensor() if hasattr(v, 'full_tensor') else v
            for k, v in params.items()
        }
        load_format = 'dtensor'

    self.inference_engine.sync_model_weights(params, load_format=load_format)
```

---

### Step 5: Update Type Checks

**Files**: `verl/workers/actor/dp_actor.py`, `verl/workers/critic/dp_critic.py`

#### 5.1 Update imports

**Before**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
```

**After**:
```python
from torch.distributed.fsdp import FSDPModule
```

#### 5.2 Update isinstance checks

**Before**:
```python
if isinstance(self.actor_module, FSDP):
    grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
```

**After**:
```python
if isinstance(self.actor_module, FSDPModule):
    grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
```

---

### Step 6: Migrate SFT Trainer

**File**: `verl/trainer/fsdp_sft_trainer.py`

Apply the same patterns as the worker migration:

1. Update imports (FSDP → fully_shard, FSDPModule)
2. Update mixed precision (MixedPrecision → MixedPrecisionPolicy)
3. Update model wrapping (use apply_fsdp2_wrapping)
4. Update checkpoint saving (use get_model_state_dict)

---

### Step 7: Update HF Rollout

**File**: `verl/workers/rollout/hf_rollout.py`

#### 7.1 Update imports

**Before**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
```

**After**:
```python
from torch.distributed.fsdp import FSDPModule
```

#### 7.2 Replace `summon_full_params`

**Before**:
```python
if isinstance(self.module, FSDP):
    param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
```

**After**:
```python
import contextlib

@contextlib.contextmanager
def unshard_for_inference(module):
    """Context manager to temporarily unshard FSDP2 parameters for inference."""
    if isinstance(module, FSDPModule):
        module.unshard()
    try:
        yield
    finally:
        if isinstance(module, FSDPModule):
            module.reshard()

# Usage:
if isinstance(self.module, FSDPModule):
    param_ctx = unshard_for_inference(self.module)
else:
    param_ctx = contextlib.nullcontext()
```

---

## Testing Checklist

After migration, verify:

- [ ] Unit tests pass for FSDP2 wrapping
- [ ] State dict extraction works correctly
- [ ] Offloading functions work with DTensor
- [ ] SFT training loop completes without errors
- [ ] PPO training loop completes without errors
- [ ] vLLM sharding manager syncs weights correctly
- [ ] Checkpoints can be saved and loaded
- [ ] Memory usage is equal or lower than FSDP1
- [ ] Training throughput is equal or higher than FSDP1
- [ ] All tests pass on DGX Spark (ARM64/GB10)

---

## Rollback

If issues are encountered, use git to revert:

```bash
git revert <commit-hash>
```

---

## References

- [PyTorch FSDP2 Documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [TorchTitan FSDP Guide](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)
- [HuggingFace FSDP1 vs FSDP2](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
