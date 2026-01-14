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
SGLang rollout for veRL HybridEngine.

This follows the same pattern as vLLMRollout:
- Uses custom Engine wrapper from verl.third_party.sglang
- Exposes inference_engine for FSDPSGLangShardingManager
- Supports NCCL-based weight sync via update_weights_from_distributed
- Manages memory via offload_model_weights()

Weight sync uses NCCL instead of pickle to avoid segfaults in Ray.
See: experiments/SGLANG_WEIGHT_SYNC_RESEARCH.md
"""
import os
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove left padding from prompt token ids."""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class SGLangRollout(BaseRollout):
    """SGLang-based rollout for inference during RL training.

    This follows the same pattern as vLLMRollout:
    - Uses verl.third_party.sglang.Engine wrapper (not raw sglang.Engine)
    - Exposes self.inference_engine for FSDPSGLangShardingManager
    - Calls offload_model_weights() after init to reduce peak memory
    - Supports NCCL-based weight sync (no pickle serialization)
    """

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """Initialize SGLang rollout.

        Args:
            actor_module: The FSDP-wrapped actor module
            config: Rollout configuration
            tokenizer: Tokenizer for the model
            model_hf_config: HuggingFace model config
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Import our custom Engine wrapper (not raw sglang.Engine)
        from verl.third_party.sglang import Engine

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)

        # Get load_format from config (default to 'dummy' for HybridEngine)
        load_format = self.config.get('load_format', 'dummy')

        # Get enforce_eager (maps to disable_cuda_graph)
        enforce_eager = self.config.get('enforce_eager', True)

        print(f"[SGLangRollout] Initializing with tp_size={tensor_parallel_size}, "
              f"enforce_eager={enforce_eager}, load_format={load_format}")

        # Initialize SGLang Engine wrapper (matches vLLM LLM interface)
        self.inference_engine = Engine(
            actor_module=actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.get('dtype', 'bfloat16'),
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.35),
            enforce_eager=enforce_eager,
            load_format=load_format,
            log_level='warning',
        )

        # Offload model weights to reduce peak memory (like vLLM)
        self.inference_engine.offload_model_weights()

        # Default sampling params
        self.default_sampling_params = {
            'max_new_tokens': config.response_length,
            'temperature': config.get('temperature', 1.0),
            'top_p': config.get('top_p', 1.0),
            'top_k': config.get('top_k', -1),
        }

        print(f"[SGLangRollout] Engine initialized successfully")

    def _generate_batch(self, prompt_token_ids_list: List[List[int]], sampling_params: dict) -> tuple:
        """Generate responses for a batch of prompts.

        Returns:
            tuple: (response_ids, log_probs) as tensors
        """
        # Use the wrapper's generate method
        response_tensor, log_probs_tensor = self.inference_engine.generate(
            input_ids=prompt_token_ids_list,
            sampling_params=sampling_params,
        )
        return response_tensor, log_probs_tensor

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences from prompts.

        Args:
            prompts: DataProto containing input_ids, attention_mask, position_ids

        Returns:
            DataProto with generated sequences
        """
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)
        device = idx.device

        # Convert to list of token ids (remove padding)
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        # Handle do_sample flag
        do_sample = prompts.meta_info.get('do_sample', True)
        sampling_params = dict(self.default_sampling_params)

        if not do_sample:
            sampling_params.update({
                'temperature': 0,
                'top_p': 1.0,
                'top_k': -1,
            })

        # Override with kwargs
        for k, v in kwargs.items():
            if k in sampling_params:
                sampling_params[k] = v

        # Handle n > 1 (multiple samples per prompt)
        n_samples = self.config.get('n', 1)
        if n_samples > 1 and do_sample:
            # Repeat prompts n times
            expanded_idx_list = []
            for prompt_ids in idx_list:
                for _ in range(n_samples):
                    expanded_idx_list.append(prompt_ids)
            idx_list = expanded_idx_list
            batch_size = batch_size * n_samples
            idx = idx.repeat_interleave(n_samples, dim=0)
            attention_mask = attention_mask.repeat_interleave(n_samples, dim=0)
            position_ids = position_ids.repeat_interleave(n_samples, dim=0)

        # Generate
        response, log_probs = self._generate_batch(idx_list, sampling_params)
        response = response.to(device)
        log_probs = log_probs.to(device)

        # Pad response if needed
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, 0.0)

        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)

        # Update position_ids and attention_mask
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size
        )

        return DataProto(batch=batch)

    def shutdown(self):
        """Shutdown the SGLang engine."""
        if hasattr(self, 'inference_engine') and self.inference_engine is not None:
            self.inference_engine.shutdown()


class SGLangServerRollout(BaseRollout):
    """SGLang HTTP Server-based rollout for inference during RL training.

    This is an alternative to SGLangRollout that runs SGLang as a completely
    separate HTTP server process. This fully isolates SGLang from Ray's CUDA
    context, avoiding all the segfault and NCCL issues that occur with in-process
    engines.

    Key differences from SGLangRollout:
    - Uses ServerEngine instead of Engine
    - SGLang runs as separate process (not subprocess of Ray actor)
    - All communication via HTTP API
    - Weight sync via disk + HTTP endpoint

    Use this when:
    - Single GPU training where NCCL cannot work
    - Ray environment causing CUDA context conflicts
    - Need maximum isolation between training and inference
    """

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """Initialize SGLang HTTP Server rollout.

        Args:
            actor_module: The FSDP-wrapped actor module
            config: Rollout configuration
            tokenizer: Tokenizer for the model
            model_hf_config: HuggingFace model config
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Import ServerEngine (HTTP server-based)
        from verl.third_party.sglang import ServerEngine

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        load_format = self.config.get('load_format', 'dummy')
        enforce_eager = self.config.get('enforce_eager', True)

        print(f"[SGLangServerRollout] Initializing HTTP server mode with tp_size={tensor_parallel_size}, "
              f"enforce_eager={enforce_eager}, load_format={load_format}")

        # Initialize SGLang Server Engine
        self.inference_engine = ServerEngine(
            actor_module=actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.get('dtype', 'bfloat16'),
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.35),
            enforce_eager=enforce_eager,
            load_format=load_format,
        )

        # Wait for server to be ready
        self.inference_engine.wait_for_ready()

        # Offload model weights to reduce peak memory
        self.inference_engine.offload_model_weights()

        # Default sampling params
        self.default_sampling_params = {
            'max_new_tokens': config.response_length,
            'temperature': config.get('temperature', 1.0),
            'top_p': config.get('top_p', 1.0),
            'top_k': config.get('top_k', -1),
        }

        print(f"[SGLangServerRollout] HTTP Server initialized at {self.inference_engine.base_url}")

    def _generate_batch(self, prompt_token_ids_list: List[List[int]], sampling_params: dict) -> tuple:
        """Generate responses for a batch of prompts.

        Returns:
            tuple: (response_ids, log_probs) as tensors
        """
        response_tensor, log_probs_tensor = self.inference_engine.generate(
            input_ids=prompt_token_ids_list,
            sampling_params=sampling_params,
        )
        return response_tensor, log_probs_tensor

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences from prompts.

        Args:
            prompts: DataProto containing input_ids, attention_mask, position_ids

        Returns:
            DataProto with generated sequences
        """
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)
        device = idx.device

        # Convert to list of token ids (remove padding)
        idx_list = []
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        # Handle do_sample flag
        do_sample = prompts.meta_info.get('do_sample', True)
        sampling_params = dict(self.default_sampling_params)

        if not do_sample:
            sampling_params.update({
                'temperature': 0,
                'top_p': 1.0,
                'top_k': -1,
            })

        # Override with kwargs
        for k, v in kwargs.items():
            if k in sampling_params:
                sampling_params[k] = v

        # Handle n > 1 (multiple samples per prompt)
        n_samples = self.config.get('n', 1)
        if n_samples > 1 and do_sample:
            expanded_idx_list = []
            for prompt_ids in idx_list:
                for _ in range(n_samples):
                    expanded_idx_list.append(prompt_ids)
            idx_list = expanded_idx_list
            batch_size = batch_size * n_samples
            idx = idx.repeat_interleave(n_samples, dim=0)
            attention_mask = attention_mask.repeat_interleave(n_samples, dim=0)
            position_ids = position_ids.repeat_interleave(n_samples, dim=0)

        # Generate
        response, log_probs = self._generate_batch(idx_list, sampling_params)
        response = response.to(device)
        log_probs = log_probs.to(device)

        # Pad response if needed
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, 0.0)

        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)

        # Update position_ids and attention_mask
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size
        )

        return DataProto(batch=batch)

    def shutdown(self):
        """Shutdown the SGLang HTTP server."""
        if hasattr(self, 'inference_engine') and self.inference_engine is not None:
            self.inference_engine.shutdown()
