# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Minimal SGLang rollout adapter for verl (GB10 compatible)
"""
SGLang rollout that can be used as a drop-in replacement for vLLM rollout.
Designed for GB10 compatibility with SGLANG_KERNEL_DISABLE=1 and disable_cuda_graph=True.
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

# Set environment for GB10 compatibility before importing sglang
os.environ.setdefault('SGLANG_KERNEL_DISABLE', '1')
os.environ.setdefault('SGLANG_ATTENTION_BACKEND', 'triton')


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    """Remove left padding from prompt token ids."""
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class SGLangRollout(BaseRollout):
    """SGLang-based rollout for inference during RL training."""

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """Initialize SGLang rollout.

        Args:
            actor_module: The actor module (not used directly, weights synced separately)
            config: Rollout configuration
            tokenizer: Tokenizer for the model
            model_hf_config: HuggingFace model config
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Import sglang here after env vars are set
        from sglang import Engine

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)

        # Get model path from config or use the HF config
        model_path = getattr(model_hf_config, '_name_or_path', None)
        if model_path is None:
            # Try to get from actor module's config
            model_path = self.config.get('model_path', 'Qwen/Qwen2.5-1.5B')

        print(f"[SGLangRollout] Initializing with model_path={model_path}, tp_size={tensor_parallel_size}")

        # Initialize SGLang Engine with GB10-compatible settings
        self.engine = Engine(
            model_path=model_path,
            tp_size=tensor_parallel_size,
            disable_cuda_graph=True,  # Required for GB10
            log_level='warning',
            mem_fraction_static=config.get('gpu_memory_utilization', 0.5),
        )

        # Default sampling params
        self.default_sampling_params = {
            'max_new_tokens': config.response_length,
            'temperature': config.get('temperature', 1.0),
            'top_p': config.get('top_p', 1.0),
            'top_k': config.get('top_k', -1),
        }

        self._is_initialized = True
        print(f"[SGLangRollout] Engine initialized successfully")

    def _generate_batch(self, prompt_token_ids_list: List[List[int]], sampling_params: dict) -> tuple:
        """Generate responses for a batch of prompts.

        Returns:
            tuple: (response_ids, log_probs) as tensors
        """
        # SGLang generate expects list of prompts or token ids
        results = []
        for prompt_ids in prompt_token_ids_list:
            result = self.engine.generate(
                input_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            results.append(result)

        # Extract output token ids from results
        all_output_ids = []
        for result in results:
            output_ids = result.get('output_ids', [])
            all_output_ids.append(output_ids)

        # Pad to same length
        max_len = max(len(ids) for ids in all_output_ids) if all_output_ids else 0
        if max_len < self.config.response_length:
            max_len = self.config.response_length

        padded_outputs = []
        for output_ids in all_output_ids:
            if len(output_ids) < max_len:
                output_ids = output_ids + [self.pad_token_id] * (max_len - len(output_ids))
            else:
                output_ids = output_ids[:max_len]
            padded_outputs.append(output_ids)

        response_tensor = torch.tensor(padded_outputs, dtype=torch.long)
        # SGLang doesn't return log probs in the same way, use zeros as placeholder
        # The actor will recompute log probs anyway
        log_probs_tensor = torch.zeros_like(response_tensor, dtype=torch.float32)

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
        if hasattr(self, 'engine') and self.engine is not None:
            self.engine.shutdown()
            self._is_initialized = False
