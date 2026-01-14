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
Custom SGLang Engine wrapper for veRL HybridEngine.

This module wraps SGLang's Engine to match the verl.third_party.vllm.LLM interface:
1. Accept actor_module in __init__ (not used directly, weights synced later)
2. Support load_format='dummy' for weight-free initialization
3. Provide sync_model_weights() for batch weight updates
4. Provide offload_model_weights() for memory management
5. Provide init_cache_engine() / free_cache_engine() for KV cache control
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import logging

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast

from .config import normalize_load_format
from .weight_loaders import load_weights

logger = logging.getLogger(__name__)


class Engine:
    """SGLang Engine wrapper for veRL HybridEngine workflow.

    Unlike raw SGLang Engine which loads from model_path, this wrapper:
    1. Initializes with dummy weights (load_format='dummy')
    2. Syncs weights from FSDP actor via sync_model_weights()
    3. Manages memory via offload_model_weights() / resume patterns

    This matches the verl.third_party.vllm.LLM interface.
    """

    def __init__(
        self,
        actor_module: Union[nn.Module, Dict],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        model_hf_config: PretrainedConfig,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.35,
        enforce_eager: bool = False,
        load_format: str = 'dummy',
        **kwargs,
    ) -> None:
        """Initialize SGLang Engine for veRL HybridEngine workflow.

        Args:
            actor_module: FSDP-wrapped actor module (stored but not used directly,
                         weights are synced via sync_model_weights)
            tokenizer: HuggingFace tokenizer
            model_hf_config: HuggingFace model configuration
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Model dtype ('auto', 'float16', 'bfloat16')
            gpu_memory_utilization: Fraction of GPU memory for KV cache
            enforce_eager: If True, disable CUDA graphs (maps to disable_cuda_graph)
            load_format: Weight loading format ('dummy' for HybridEngine)
            **kwargs: Additional SGLang engine arguments
        """
        from sglang import Engine as SGLangEngine

        # Store config for later reference
        self.model_hf_config = model_hf_config
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self._load_format = load_format

        # Get model path for SGLang initialization
        # Even with load_format='dummy', SGLang needs model architecture info
        model_path = getattr(model_hf_config, '_name_or_path', None)
        if model_path is None:
            raise ValueError(
                "model_hf_config must have '_name_or_path' attribute for SGLang initialization. "
                "This should be the HuggingFace model identifier or path."
            )

        # Normalize load format for SGLang (only supports 'dummy', not 'dummy_dtensor')
        sglang_load_format = normalize_load_format(load_format)

        # Map veRL config to SGLang config
        sglang_kwargs = {
            'model_path': model_path,
            'tp_size': tensor_parallel_size,
            'dtype': dtype,
            'mem_fraction_static': gpu_memory_utilization,
            'disable_cuda_graph': enforce_eager,
            'load_format': sglang_load_format,
            'enable_memory_saver': True,  # Required for release/resume APIs
            'log_level': kwargs.get('log_level', 'warning'),
        }

        # Pass through additional kwargs
        for key in ['trust_remote_code', 'skip_tokenizer_init']:
            if key in kwargs:
                sglang_kwargs[key] = kwargs[key]

        logger.info(
            f"Initializing SGLang Engine: model_path={model_path}, "
            f"tp_size={tensor_parallel_size}, dtype={dtype}, "
            f"mem_fraction_static={gpu_memory_utilization}, "
            f"disable_cuda_graph={enforce_eager}, load_format={sglang_load_format}"
        )

        # Initialize underlying SGLang Engine
        self._engine = SGLangEngine(**sglang_kwargs)

        # Track state for memory management
        self._weights_offloaded = False
        self._cache_engine_freed = False

        logger.info("SGLang Engine initialized successfully")

    # === Weight Management APIs (vLLM interface) ===

    def sync_model_weights(
        self,
        actor_weights: Dict[str, torch.Tensor],
        load_format: str,
    ) -> None:
        """Synchronize weights from FSDP actor to SGLang engine.

        This is the primary weight sync method, matching vLLM's interface.

        Uses disk-based weight sync to avoid pickle serialization issues
        in Ray environments. This is more reliable than update_weights_from_tensor
        which can segfault when SGLang runs as a subprocess.

        Args:
            actor_weights: State dict from FSDP actor
            load_format: 'hf' for full params, 'dtensor' for sharded
        """
        import os
        import tempfile
        import shutil
        from safetensors.torch import save_file
        from torch.distributed._tensor import DTensor

        logger.info(f"Syncing model weights via disk (format: {load_format})")

        # Create temporary directory for weights
        temp_dir = tempfile.mkdtemp(prefix="verl_sglang_weights_")

        try:
            # Prepare weights for saving
            weights_to_save = {}
            for name, tensor in actor_weights.items():
                # Skip rotary embeddings
                if "rotary_emb" in name:
                    continue

                # Handle tied embeddings
                if hasattr(self.model_hf_config, 'tie_word_embeddings'):
                    if self.model_hf_config.tie_word_embeddings and "lm_head.weight" in name:
                        continue

                # Handle DTensors
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()

                # Move to CPU and convert to bfloat16
                tensor = tensor.detach().cpu().to(dtype=torch.bfloat16)

                # Ensure contiguous
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()

                weights_to_save[name] = tensor

            # Save as safetensors
            weight_path = os.path.join(temp_dir, "model.safetensors")
            save_file(weights_to_save, weight_path)
            logger.info(f"Saved {len(weights_to_save)} parameters to {weight_path}")

            # Clear weights_to_save to free memory
            del weights_to_save
            torch.cuda.empty_cache()

            # Call SGLang's update_weights_from_disk
            success, message = self.update_weights_from_disk(temp_dir)

            if not success:
                raise RuntimeError(f"Failed to update weights from disk: {message}")

            self._weights_offloaded = False
            logger.info("Weight sync complete")

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")

    def offload_model_weights(self) -> None:
        """Offload model weights to release GPU memory.

        This enables the HybridEngine pattern where inference weights
        are released during training phase.
        """
        if self._weights_offloaded:
            logger.debug("Weights already offloaded, skipping")
            return

        if hasattr(self._engine, 'release_memory_occupation'):
            # Use SGLang's native memory saver API
            logger.info("Releasing SGLang weight memory")
            self._engine.release_memory_occupation(tags=['weights'])
            self._weights_offloaded = True
        else:
            logger.warning(
                "SGLang Engine doesn't have release_memory_occupation API. "
                "Weight offloading not available."
            )

    # === Cache Engine APIs (vLLM interface) ===

    def init_cache_engine(self) -> None:
        """Initialize/resume KV cache engine.

        Call this before generation if free_cache_engine was called.
        """
        if not self._cache_engine_freed:
            logger.debug("Cache engine already initialized, skipping")
            return

        if hasattr(self._engine, 'resume_memory_occupation'):
            logger.info("Resuming SGLang KV cache and CUDA graphs")
            self._engine.resume_memory_occupation(tags=['kv_cache', 'cuda_graph'])
            self._cache_engine_freed = False
        else:
            logger.warning(
                "SGLang Engine doesn't have resume_memory_occupation API. "
                "Cache initialization not available."
            )

    def free_cache_engine(self) -> None:
        """Free KV cache to reclaim memory.

        Call this after generation to release memory for training.
        """
        if self._cache_engine_freed:
            logger.debug("Cache engine already freed, skipping")
            return

        if hasattr(self._engine, 'release_memory_occupation'):
            logger.info("Releasing SGLang KV cache and CUDA graphs")
            self._engine.release_memory_occupation(tags=['kv_cache', 'cuda_graph'])
            self._cache_engine_freed = True
        else:
            logger.warning(
                "SGLang Engine doesn't have release_memory_occupation API. "
                "Cache freeing not available."
            )

    # === Full Memory Management (combined) ===

    def release_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Release GPU memory for specified components.

        Args:
            tags: List of components to release ('kv_cache', 'weights', 'cuda_graph')
                  If None, releases all components.
        """
        if tags is None:
            tags = ['kv_cache', 'weights', 'cuda_graph']

        if hasattr(self._engine, 'release_memory_occupation'):
            logger.info(f"Releasing SGLang memory: {tags}")
            self._engine.release_memory_occupation(tags=tags)
            if 'weights' in tags:
                self._weights_offloaded = True
            if 'kv_cache' in tags:
                self._cache_engine_freed = True
        else:
            logger.warning("release_memory_occupation not available")

    def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Resume GPU memory for specified components.

        Args:
            tags: List of components to resume ('kv_cache', 'weights', 'cuda_graph')
                  If None, resumes all components.
        """
        if tags is None:
            tags = ['kv_cache', 'weights', 'cuda_graph']

        if hasattr(self._engine, 'resume_memory_occupation'):
            logger.info(f"Resuming SGLang memory: {tags}")
            self._engine.resume_memory_occupation(tags=tags)
            if 'weights' in tags:
                self._weights_offloaded = False
            if 'kv_cache' in tags:
                self._cache_engine_freed = False
        else:
            logger.warning("resume_memory_occupation not available")

    # === Generation APIs ===

    def generate(
        self,
        input_ids: List[List[int]],
        sampling_params: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences and return (tokens, log_probs).

        Args:
            input_ids: List of prompt token ID lists
            sampling_params: Sampling parameters dict

        Returns:
            Tuple of (response_ids, log_probs) tensors
            - response_ids: [batch_size, response_length]
            - log_probs: [batch_size, response_length] (zeros for now)
        """
        # Call underlying SGLang engine
        results = self._engine.generate(input_ids=input_ids, sampling_params=sampling_params)

        # Process outputs to tensors
        return self._process_outputs(results, sampling_params)

    def _process_outputs(
        self,
        results: Union[dict, List[dict]],
        sampling_params: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process SGLang output to (response_ids, log_probs) tensors.

        Args:
            results: SGLang generation results (dict or list of dicts)
            sampling_params: Sampling params used for generation

        Returns:
            Tuple of (response_ids, log_probs) tensors
        """
        # Handle both single result (dict) and batch results (list of dicts)
        if isinstance(results, dict):
            results = [results]

        # Extract output token IDs
        all_output_ids = []
        for result in results:
            output_ids = result.get('output_ids', [])
            all_output_ids.append(output_ids)

        # Determine max length for padding
        max_new_tokens = sampling_params.get('max_new_tokens', 1024)
        max_len = max(len(ids) for ids in all_output_ids) if all_output_ids else 0
        max_len = max(max_len, max_new_tokens)

        # Pad outputs
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        padded_outputs = []
        for output_ids in all_output_ids:
            if len(output_ids) < max_len:
                output_ids = list(output_ids) + [pad_token_id] * (max_len - len(output_ids))
            else:
                output_ids = list(output_ids)[:max_len]
            padded_outputs.append(output_ids)

        response_ids = torch.tensor(padded_outputs, dtype=torch.long)

        # Log probs: return zeros (actor will recompute)
        # This matches vLLM pattern where logprobs=0 is valid
        log_probs = torch.zeros_like(response_ids, dtype=torch.float32)

        return response_ids, log_probs

    # === Utility Methods ===

    def flush_cache(self) -> None:
        """Flush KV cache."""
        if hasattr(self._engine, 'flush_cache'):
            self._engine.flush_cache()

    def shutdown(self) -> None:
        """Shutdown the engine and release all resources."""
        logger.info("Shutting down SGLang Engine")
        if hasattr(self._engine, 'shutdown'):
            self._engine.shutdown()

    @property
    def engine(self) -> Any:
        """Access underlying SGLang engine (for advanced use)."""
        return self._engine

    # === NCCL-based Distributed Weight Sync APIs ===
    # These bypass pickle serialization and use NCCL directly

    def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str = "verl_weight_sync",
        backend: str = "nccl",
    ) -> Tuple[bool, str]:
        """Initialize NCCL process group for distributed weight sync.

        This sets up the receiving end of the NCCL broadcast. The training
        process should create the corresponding group using init_custom_process_group.

        Args:
            master_address: Address of rank 0 (training process)
            master_port: Port for NCCL rendezvous
            rank_offset: Starting rank for SGLang (usually 1, as training is rank 0)
            world_size: Total ranks (training + inference)
            group_name: Name for the process group
            backend: Communication backend ('nccl' recommended)

        Returns:
            Tuple of (success, message)
        """
        logger.info(
            f"Initializing weight update group: master={master_address}:{master_port}, "
            f"rank_offset={rank_offset}, world_size={world_size}, group_name={group_name}"
        )
        return self._engine.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

    def update_weights_from_distributed(
        self,
        names: List[str],
        dtypes: List[torch.dtype],
        shapes: List[Tuple[int, ...]],
        group_name: str = "verl_weight_sync",
    ) -> Tuple[bool, str]:
        """Receive weights via NCCL broadcast from training process.

        This receives weights that were broadcast by the training process.
        Must be called AFTER training process calls torch.distributed.broadcast().

        Args:
            names: List of parameter names to update
            dtypes: List of dtypes for each parameter
            shapes: List of shapes for each parameter
            group_name: Name of the process group (must match init_weights_update_group)

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Receiving {len(names)} weights from distributed (group={group_name})")
        return self._engine.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
        )

    def destroy_weights_update_group(self, group_name: str = "verl_weight_sync") -> Tuple[bool, str]:
        """Destroy the NCCL process group for weight sync.

        Should be called during shutdown to clean up resources.

        Args:
            group_name: Name of the process group to destroy

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Destroying weight update group: {group_name}")
        return self._engine.destroy_weights_update_group(group_name=group_name)

    # === Disk-based Weight Sync APIs ===
    # These save weights to disk and reload, avoiding NCCL coordination issues

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Update model weights from disk.

        This is the most reliable weight sync method as it avoids:
        - Pickle serialization issues (update_weights_from_tensor segfault in Ray)
        - NCCL coordination issues (update_weights_from_distributed with pre-init torch.distributed)

        The sharding manager saves FSDP state_dict to a temporary directory,
        then calls this method to reload the weights.

        Args:
            model_path: Path to directory containing safetensors weights
            load_format: Optional load format override

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Updating weights from disk: {model_path}")
        result = self._engine.update_weights_from_disk(
            model_path=model_path,
            load_format=load_format,
        )
        # SGLang returns a tuple (success, message, elapsed_time) for disk updates
        # but we just need (success, message)
        if isinstance(result, tuple) and len(result) >= 2:
            success, message = result[0], result[1]
        else:
            success, message = result, "Unknown result format"
        self._weights_offloaded = False
        logger.info(f"Weight update from disk: success={success}, message={message}")
        return success, message
