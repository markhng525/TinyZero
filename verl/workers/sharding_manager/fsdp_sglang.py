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
Sharding manager for FSDP + SGLang HybridEngine.

This manager synchronizes FSDP actor weights to SGLang Engine for inference.
It follows the same pattern as FSDPVLLMShardingManager:
- __enter__: Extract FSDP state_dict and sync to inference engine
- __exit__: Offload inference engine weights

Weight sync uses the engine's sync_model_weights() method which handles:
- In-process Engine: Direct weight update (may use update_weights_from_tensor)
- Server Engine: Disk-based weight update via HTTP
"""

import os
import logging

import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig, ShardedStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.debug import log_gpu_memory_usage
from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class FSDPSGLangShardingManager(BaseShardingManager):
    """Sharding manager that synchronizes FSDP actor weights to SGLang Engine.

    This follows the same pattern as FSDPVLLMShardingManager:
    - Uses FSDP state_dict extraction
    - Calls inference_engine.sync_model_weights() for weight sync
    - Manages random states for TP consistency

    The actual weight sync mechanism depends on the engine type:
    - verl.third_party.sglang.Engine: May use update_weights_from_tensor or NCCL
    - verl.third_party.sglang.ServerEngine: Uses disk + HTTP
    """

    def __init__(
        self,
        module: FSDP,
        inference_engine,  # Engine or ServerEngine from verl.third_party.sglang
        model_config,
        full_params: bool = True,
        device_mesh: DeviceMesh = None,
    ):
        """Initialize the FSDP-SGLang sharding manager.

        Args:
            module: FSDP-wrapped actor module
            inference_engine: SGLang Engine wrapper (Engine or ServerEngine)
            model_config: HuggingFace model config
            full_params: If True, use FULL_STATE_DICT (required for SGLang)
            device_mesh: Optional device mesh for TP+DP parallelism
        """
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Always use full state dict for SGLang weight sync
        # (SGLang doesn't support sharded loading like vLLM)
        self.full_params = True
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

        # Set critical NCCL environment variables
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"

    def __enter__(self):
        """Prepare SGLang for inference by syncing weights from FSDP.

        Steps:
        1. Extract full state_dict from FSDP
        2. Sync weights to SGLang engine
        3. Cleanup state_dict to free memory
        4. Set random states for TP consistency
        """
        log_gpu_memory_usage('Before state_dict() in sharding manager', logger=logger)

        # Extract full state dict from FSDP
        params = self.module.state_dict()
        log_gpu_memory_usage('After state_dict() in sharding manager', logger=logger)

        # Sync weights to SGLang engine
        # The engine's sync_model_weights handles the actual mechanism
        load_format = 'hf' if self.full_params else 'dtensor'
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        log_gpu_memory_usage('After sync_model_weights in sharding manager', logger=logger)

        # Cleanup state dict to free memory
        del params
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After cleanup in sharding manager', logger=logger)

        # Set random states for TP consistency (important for sampling)
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup after generation.

        Offloads SGLang weights to free GPU memory for training.
        """
        log_gpu_memory_usage('Before sglang offload in sharding manager', logger=logger)

        # Offload SGLang weights
        self.inference_engine.offload_model_weights()
        log_gpu_memory_usage('After sglang offload in sharding manager', logger=logger)

        # Switch back to training mode
        self.module.train()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """Preprocess data before generation.

        For SGLang with TP, we need to gather data across TP ranks.
        """
        # For now, return data as-is (single GPU or already gathered)
        # TODO: Add TP data gathering when TP>1 is supported
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Postprocess data after generation.

        For SGLang with TP, we need to broadcast results to all ranks.
        """
        # For now, return data as-is
        # TODO: Add TP data broadcast when TP>1 is supported
        return data
