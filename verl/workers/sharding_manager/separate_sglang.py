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
Sharding manager for FSDP + Separate SGLang Ray Actor.

This manager synchronizes FSDP actor weights to a separate SGLang Ray actor
for inference. Unlike FSDPSGLangShardingManager which uses a colocated engine,
this manager communicates via Ray RPC to avoid CUDA context conflicts.

Architecture:
    FSDP Actor (training) <--> Ray Object Store <--> SGLang Ray Actor (inference)

Benefits:
- Clean CUDA context isolation (SGLang runs as top-level Ray actor)
- Avoids subprocess nesting issues
- Enables larger batch sizes due to better memory management
- Zero-copy weight transfer on same node via Ray object store

See: experiments/SGLANG_ARCHITECTURE_ANALYSIS.md
"""

import os
import logging
from typing import Any, Dict, Optional

import torch
import ray
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.debug import log_gpu_memory_usage
from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class SeparateSGLangShardingManager(BaseShardingManager):
    """Sharding manager that synchronizes FSDP actor weights to a separate SGLang Ray actor.

    This manager handles the HybridEngine pattern where:
    - Training runs on FSDP actor (in the main worker process)
    - Inference runs on a separate SGLang Ray actor (isolated CUDA context)

    Weight sync is done via Ray object store for efficient zero-copy transfer
    on the same node, or serialized transfer across nodes.

    Example usage:
        # In fsdp_workers.py _build_rollout()
        rollout = SGLangActorRollout(...)  # Creates SGLang Ray actor
        sharding_manager = SeparateSGLangShardingManager(
            module=self.actor_module_fsdp,
            sglang_actor=rollout.inference_actor,  # Ray actor handle
        )

        # In training loop
        with sharding_manager:
            # Weights are synced to SGLang actor
            output = rollout.generate_sequences(prompts)
        # Weights are offloaded from SGLang actor
    """

    def __init__(
        self,
        module: FSDP,
        sglang_actor: ray.actor.ActorHandle,
        model_config: Optional[Any] = None,
        device_mesh: DeviceMesh = None,
        sync_timeout: float = 300.0,
    ):
        """Initialize the separate SGLang sharding manager.

        Args:
            module: FSDP-wrapped actor module for training
            sglang_actor: Ray actor handle for SGLang inference
            model_config: Optional HuggingFace model config
            device_mesh: Optional device mesh for TP+DP parallelism
            sync_timeout: Timeout in seconds for weight sync RPC
        """
        self.module = module
        self.sglang_actor = sglang_actor
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.sync_timeout = sync_timeout

        # Always use full state dict for SGLang weight sync
        FSDP.set_state_dict_type(
            self.module,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(),
        )

        # RNG state management for TP consistency (matches vLLM pattern)
        self.torch_random_states = torch.cuda.get_rng_state()
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        # Track sync state
        self._weights_synced = False

        logger.info("SeparateSGLangShardingManager initialized")

    def __enter__(self):
        """Sync weights from FSDP to the separate SGLang Ray actor.

        Steps:
        1. Extract full state_dict from FSDP (on rank 0)
        2. Put state_dict in Ray object store
        3. Call SGLang actor's sync_weights_from_dict via RPC
        4. Wait for sync to complete
        5. Cleanup state_dict reference
        """
        log_gpu_memory_usage('Before state_dict() in separate sglang sharding manager', logger=logger)

        # Only rank 0 does the weight sync to avoid duplicated work
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if rank == 0:
            # Extract full state dict from FSDP
            params = self.module.state_dict()
            log_gpu_memory_usage('After state_dict() in separate sglang sharding manager', logger=logger)

            # Put state dict in Ray object store (zero-copy on same node)
            params_ref = ray.put(params)
            log_gpu_memory_usage('After ray.put() in separate sglang sharding manager', logger=logger)

            # Sync weights to SGLang actor via RPC
            try:
                sync_result = ray.get(
                    self.sglang_actor.sync_weights_from_dict.remote(params_ref),
                    timeout=self.sync_timeout
                )
                if sync_result:
                    self._weights_synced = True
                    logger.info("Weight sync to SGLang actor successful")
                else:
                    logger.error("Weight sync to SGLang actor returned False")
            except Exception as e:
                logger.error(f"Weight sync to SGLang actor failed: {e}")
                raise

            # Cleanup
            del params
            del params_ref
            torch.cuda.empty_cache()
            log_gpu_memory_usage('After cleanup in separate sglang sharding manager', logger=logger)

        # Synchronize all ranks before proceeding
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Set random states for TP consistency
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Offload weights from SGLang actor to free GPU memory for training.

        Steps:
        1. Call SGLang actor's offload_weights via RPC
        2. Restore random states
        3. Set module to training mode
        """
        log_gpu_memory_usage('Before SGLang offload in separate sharding manager', logger=logger)

        # Only rank 0 sends the offload command
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if rank == 0:
            try:
                offload_result = ray.get(
                    self.sglang_actor.offload_weights.remote(),
                    timeout=60.0
                )
                if offload_result:
                    logger.info("Weight offload from SGLang actor successful")
                else:
                    logger.warning("Weight offload from SGLang actor returned False")
            except Exception as e:
                logger.warning(f"Weight offload from SGLang actor failed: {e}")
                # Don't raise - offload failure shouldn't block training

        # Synchronize all ranks
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Restore random states
        if self.device_mesh is not None:
            torch.cuda.set_rng_state(self.torch_random_states)

        # Set module back to training mode
        self.module.train()
        torch.cuda.empty_cache()

        log_gpu_memory_usage('After SGLang offload in separate sharding manager', logger=logger)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """Preprocess data before sending to SGLang actor.

        For separate actor, we may need to handle data distribution differently.
        Currently passes through unchanged - SGLang actor handles its own batching.
        """
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Postprocess data received from SGLang actor.

        Currently passes through unchanged.
        """
        return data
