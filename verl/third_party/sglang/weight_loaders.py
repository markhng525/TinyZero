# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Weight loaders for SGLang, supporting DTensor and HF formats.

This module handles weight synchronization from FSDP actor to SGLang engine.
Unlike vLLM which has per-model weight loaders, SGLang uses a simpler
update_weights_from_tensor() API that accepts List[Tuple[str, Tensor]].
"""

from typing import Dict, List, Tuple, Any
import logging

import torch

logger = logging.getLogger(__name__)


def load_weights(
    actor_weights: Dict[str, torch.Tensor],
    engine: Any,  # SGLang Engine
    model_config: Any,
    load_format: str,
) -> None:
    """Load weights into SGLang engine.

    This is the main entry point for weight synchronization, matching
    the vLLM pattern from verl.third_party.vllm.

    Args:
        actor_weights: State dict from FSDP actor
        engine: SGLang Engine instance (raw, not wrapper)
        model_config: HuggingFace model config
        load_format: 'hf' for full params, 'dtensor' for sharded

    Raises:
        RuntimeError: If engine doesn't have update_weights_from_tensor API
    """
    if load_format == 'dtensor':
        named_tensors = _prepare_dtensor_weights(actor_weights, model_config)
    elif load_format == 'hf':
        named_tensors = _prepare_hf_weights(actor_weights, model_config)
    else:
        raise ValueError(f"Unknown load_format: {load_format}. Expected 'hf' or 'dtensor'")

    if len(named_tensors) == 0:
        logger.warning("No tensors to sync to SGLang")
        return

    logger.info(f"Syncing {len(named_tensors)} parameters to SGLang")

    # Synchronize CUDA before weight update to ensure all tensors are ready
    torch.cuda.synchronize()

    # Call SGLang's batch weight update API
    if hasattr(engine, 'update_weights_from_tensor'):
        engine.update_weights_from_tensor(named_tensors, flush_cache=True)
    else:
        raise RuntimeError(
            f"SGLang Engine doesn't have update_weights_from_tensor API. "
            f"Available methods: {[m for m in dir(engine) if not m.startswith('_')]}"
        )


def _prepare_hf_weights(
    actor_weights: Dict[str, torch.Tensor],
    model_config: Any,
) -> List[Tuple[str, torch.Tensor]]:
    """Prepare full HF-format weights for SGLang.

    Args:
        actor_weights: Full state dict from FSDP (no sharding)
        model_config: HuggingFace model config

    Returns:
        List of (parameter_name, tensor) tuples for SGLang
    """
    named_tensors = []

    for name, tensor in actor_weights.items():
        # Skip rotary embeddings (computed on the fly)
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue

        # Handle tied embeddings - skip lm_head if tied to embed_tokens
        if hasattr(model_config, 'tie_word_embeddings') and model_config.tie_word_embeddings:
            if "lm_head.weight" in name:
                continue

        # Ensure tensor is contiguous, on CUDA, and in bfloat16
        tensor = _ensure_tensor_format(tensor)
        named_tensors.append((name, tensor))

    return named_tensors


def _prepare_dtensor_weights(
    actor_weights: Dict[str, torch.Tensor],
    model_config: Any,
) -> List[Tuple[str, torch.Tensor]]:
    """Prepare sharded DTensor weights for SGLang.

    DTensors from FSDP sharded state dict need to be gathered to full tensors
    before passing to SGLang.

    Args:
        actor_weights: Sharded state dict from FSDP (may contain DTensors)
        model_config: HuggingFace model config

    Returns:
        List of (parameter_name, tensor) tuples for SGLang
    """
    from torch.distributed._tensor import DTensor

    named_tensors = []

    for name, tensor in actor_weights.items():
        # Skip rotary embeddings
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue

        # Handle tied embeddings
        if hasattr(model_config, 'tie_word_embeddings') and model_config.tie_word_embeddings:
            if "lm_head.weight" in name:
                continue

        # Convert DTensor to full tensor first
        if isinstance(tensor, DTensor):
            # full_tensor() gathers shards to create a regular tensor
            tensor = tensor.full_tensor()

        # Ensure proper format (detach, clone, contiguous, cuda, bfloat16)
        tensor = _ensure_tensor_format(tensor)
        named_tensors.append((name, tensor))

    return named_tensors


def _ensure_tensor_format(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is in the right format for SGLang.

    IMPORTANT: SGLang's scheduler runs in a separate subprocess. CUDA tensors
    serialized across process boundaries can cause segfaults in Ray environments.

    Solution: Pass CPU tensors to SGLang. Its scheduler will move them to CUDA
    after deserialization, avoiding cross-process CUDA tensor issues.

    Args:
        tensor: Input tensor (may be on CPU or GPU, any dtype)

    Returns:
        Tensor that is on CPU, contiguous, and in bfloat16
    """
    # Move to CPU first to avoid cross-process CUDA tensor issues
    tensor = tensor.detach().cpu()

    # Convert to bfloat16 on CPU
    tensor = tensor.to(dtype=torch.bfloat16)

    # Ensure contiguous memory layout for efficient serialization
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    return tensor
