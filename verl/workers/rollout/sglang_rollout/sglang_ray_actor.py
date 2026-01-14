# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Adapted for SGLang as separate Ray actor to avoid CUDA context conflicts
"""
SGLang Inference Ray Actor

This module implements SGLang rollout as a separate Ray actor, following the advice:
"Run SGLang as a separate Ray actor/service process (not Ray process spawning a subprocess)"

Key benefits:
1. SGLang runs as top-level Ray actor with clean CUDA context
2. No nesting of SGLang subprocesses under another Ray actor
3. Efficient data transfer via Ray object store
4. HybridEngine memory sharing still possible via weight sync
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import ray

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0)  # Don't request GPU from Ray - we share with FSDP worker
class SGLangInferenceActor:
    """
    SGLang inference engine running as a separate Ray actor.

    This avoids CUDA context conflicts by making SGLang the top-level process,
    not a subprocess spawned from another Ray actor.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.5,
        enforce_eager: bool = False,  # Default to CUDA graphs enabled (SGLang default)
        max_num_seqs: int = 256,
        log_level: str = "warning",
    ):
        """Initialize SGLang engine as top-level process."""
        import os
        import sglang as sgl

        # Ensure we use GPU 0 (same as FSDP worker)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        logger.info(f"Initializing SGLang inference actor for model: {model_path}")

        # SGLang now runs as top-level Ray actor - no CUDA context conflicts
        self.engine = sgl.Engine(
            model_path=model_path,
            tp_size=tensor_parallel_size,
            dtype=dtype,
            mem_fraction_static=gpu_memory_utilization,
            disable_cuda_graph=enforce_eager,
            enable_memory_saver=True,  # Enable weight offload APIs
            log_level=log_level,
        )

        self.model_path = model_path
        self._weights_synced = False
        self._weights_offloaded = False

        logger.info("SGLang inference actor initialized successfully")

    def generate(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Generate completions for prompts.

        Args:
            prompts: List of prompt strings
            sampling_params: Dict with temperature, max_new_tokens, n, etc.

        Returns:
            Tuple of (output_ids, log_probs) for each prompt
            - output_ids: List[List[List[int]]] - [batch][n_samples][token_ids]
            - log_probs: List[List[List[float]]] - [batch][n_samples][log_probs] (empty if not available)
        """
        n_samples = sampling_params.get('n', 1)

        # SGLang expects dict for sampling params
        sgl_params = {
            'temperature': sampling_params.get('temperature', 1.0),
            'max_new_tokens': sampling_params.get('max_new_tokens', 1024),
            'n': n_samples,
            'top_p': sampling_params.get('top_p', 1.0),
            'top_k': sampling_params.get('top_k', -1),
        }

        # Run generation
        # SGLang returns a flat list of dicts: [{'text': ..., 'output_ids': [...], 'meta_info': {...}}, ...]
        # When n > 1, it returns n outputs per prompt in sequence
        outputs = self.engine.generate(prompts, sgl_params)

        # Extract output IDs and log probs
        # SGLang output format: list of dicts with 'output_ids', 'text', 'meta_info'
        all_output_ids = []
        all_log_probs = []

        # Group outputs by prompt (n samples per prompt)
        for i in range(len(prompts)):
            prompt_output_ids = []
            prompt_log_probs = []
            for j in range(n_samples):
                idx = i * n_samples + j
                if idx < len(outputs):
                    output = outputs[idx]
                    # output is a dict with 'output_ids', 'text', 'meta_info'
                    prompt_output_ids.append(output.get('output_ids', []))
                    # SGLang doesn't return log probs by default
                    prompt_log_probs.append([])
            all_output_ids.append(prompt_output_ids)
            all_log_probs.append(prompt_log_probs)

        return all_output_ids, all_log_probs

    def sync_weights_from_dict(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Synchronize model weights from a state dict using direct tensor transfer.

        This is called when transitioning from training to inference phase.
        The state dict is transferred via Ray object store (zero-copy on same node).

        Uses update_weights_from_tensor for fast in-memory sync (vs disk-based).

        Args:
            state_dict: Model state dict from FSDP actor

        Returns:
            True if successful
        """
        import time
        start_time = time.time()

        logger.info(f"Syncing weights from state dict ({len(state_dict)} parameters)")

        # Resume memory if previously offloaded
        if self._weights_offloaded:
            self._resume_memory()

        # Prepare weights as list of (name, tensor) tuples for update_weights_from_tensor
        named_tensors = []
        for name, tensor in state_dict.items():
            if "rotary_emb" in name:
                continue  # Skip rotary embeddings

            # Ensure bfloat16 and contiguous
            tensor = tensor.to(dtype=torch.bfloat16)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            named_tensors.append((name, tensor))

        # Use direct tensor sync (much faster than disk-based)
        try:
            result = self.engine.update_weights_from_tensor(named_tensors)
            elapsed = time.time() - start_time
            self._weights_synced = True
            self._weights_offloaded = False
            logger.info(f"Weight sync successful via direct tensor transfer in {elapsed:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Weight sync via tensor failed: {e}")
            # Fallback to disk-based sync if tensor sync fails
            logger.info("Falling back to disk-based weight sync...")
            return self._sync_weights_from_disk(state_dict)

    def _sync_weights_from_disk(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """Fallback disk-based weight sync."""
        import tempfile
        import shutil
        from safetensors.torch import save_file

        temp_dir = tempfile.mkdtemp(prefix="sglang_weights_")
        try:
            weights_to_save = {}
            for name, tensor in state_dict.items():
                if "rotary_emb" in name:
                    continue
                if tensor.device.type != 'cpu':
                    tensor = tensor.cpu()
                tensor = tensor.to(dtype=torch.bfloat16)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                weights_to_save[name] = tensor

            weight_path = os.path.join(temp_dir, "model.safetensors")
            save_file(weights_to_save, weight_path)

            try:
                self.engine.update_weights_from_disk(model_path=temp_dir, load_format="auto")
                self._weights_synced = True
                self._weights_offloaded = False
                logger.info("Weight sync successful via disk fallback")
                return True
            except Exception as e:
                logger.error(f"Disk-based weight sync failed: {e}")
                return False
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def offload_weights(self) -> bool:
        """
        Offload ALL GPU memory (weights AND KV cache) for training phase.

        This releases ALL SGLang GPU memory to allow FSDP training to proceed
        without memory conflicts. For HybridEngine, we need to release everything,
        not just weights.

        Returns:
            True if successful
        """
        if self._weights_offloaded:
            logger.debug("Weights already offloaded")
            return True

        try:
            if hasattr(self.engine, 'release_memory_occupation'):
                # Release ALL memory (weights + KV cache) by not specifying tags
                self.engine.release_memory_occupation()  # No tags = release everything
                self._weights_offloaded = True
                import torch
                torch.cuda.empty_cache()
                logger.info("All GPU memory offloaded successfully")
                return True
            else:
                logger.warning("SGLang engine doesn't support memory offloading")
                return False
        except Exception as e:
            logger.error(f"Failed to offload memory: {e}")
            return False

    def _resume_memory(self) -> bool:
        """Resume ALL GPU memory (weights + KV cache) after offloading."""
        try:
            if hasattr(self.engine, 'resume_memory_occupation'):
                # Resume ALL memory by not specifying tags
                self.engine.resume_memory_occupation()  # No tags = resume everything
                self._weights_offloaded = False
                logger.info("All GPU memory resumed successfully")
                return True
            else:
                logger.warning("SGLang engine doesn't support memory resume")
                return False
        except Exception as e:
            logger.error(f"Failed to resume memory: {e}")
            return False

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage info."""
        torch.cuda.synchronize()
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
        }

    def shutdown(self):
        """Shutdown the engine."""
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception as e:
                logger.warning(f"Error during engine shutdown: {e}")
            self.engine = None


class SGLangActorRollout:
    """
    Rollout that uses SGLangInferenceActor as a separate Ray actor.

    This class provides the same interface as SGLangRollout but delegates
    inference to a separate Ray actor to avoid CUDA context conflicts.

    Key benefits:
    - Clean CUDA context isolation (SGLang runs as top-level Ray actor)
    - Avoids subprocess nesting issues that cause segfaults
    - Enables larger batch sizes due to better memory management
    - Zero-copy weight transfer on same node via Ray object store

    Usage in fsdp_workers.py:
        rollout = SGLangActorRollout(
            actor_module=self.actor_module_fsdp,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            model_hf_config=self.actor_model_config,
        )
        sharding_manager = SeparateSGLangShardingManager(
            module=self.actor_module_fsdp,
            sglang_actor=rollout.inference_actor,
        )
    """

    def __init__(
        self,
        actor_module,  # FSDP module (for interface compatibility, not used directly)
        config,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
        """
        Initialize rollout with separate SGLang Ray actor.

        Args:
            actor_module: FSDP-wrapped actor module (for interface compatibility)
            config: Rollout configuration (OmegaConf or dict)
            tokenizer: Tokenizer instance
            model_hf_config: HuggingFace model config
        """
        self.config = config
        self.tokenizer = tokenizer
        self.model_hf_config = model_hf_config
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Get model path from config
        model_path = config.get('model_path', None)
        if model_path is None:
            # Try to get from model_hf_config
            model_path = getattr(model_hf_config, '_name_or_path', None)
        if model_path is None:
            raise ValueError("model_path must be specified in config or model_hf_config")

        # Create SGLang inference actor as a separate Ray actor
        logger.info(f"Creating SGLang inference actor for model: {model_path}")
        self.inference_actor = SGLangInferenceActor.remote(
            model_path=model_path,
            tensor_parallel_size=config.get('tensor_model_parallel_size', 1),
            dtype=config.get('dtype', 'bfloat16'),
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.5),
            enforce_eager=config.get('enforce_eager', False),  # CUDA graphs enabled by default
            max_num_seqs=config.get('max_num_seqs', 256),
            log_level=config.get('log_level', 'warning'),
        )

        # Default sampling params
        self.default_sampling_params = {
            'max_new_tokens': config.get('response_length', 1024),
            'temperature': config.get('temperature', 1.0),
            'top_p': config.get('top_p', 1.0),
            'top_k': config.get('top_k', -1),
        }

        logger.info("SGLang Actor Rollout initialized")

    def _pre_process_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
        """Remove left padding from prompt token ids."""
        non_pad_index = torch.nonzero(prompt_token_ids != self.pad_token_id, as_tuple=False)
        if len(non_pad_index) == 0:
            return prompt_token_ids.tolist()
        return prompt_token_ids[non_pad_index[0][0]:].tolist()

    @torch.no_grad()
    def generate_sequences(self, prompts, **kwargs):
        """
        Generate sequences using the separate SGLang Ray actor.

        Args:
            prompts: DataProto containing input_ids, attention_mask, position_ids

        Returns:
            DataProto with generated sequences
        """
        from verl import DataProto
        from tensordict import TensorDict
        from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)
        device = idx.device

        # Convert to list of token ids (remove padding)
        idx_list = []
        for i in range(batch_size):
            idx_list.append(self._pre_process_inputs(idx[i]))

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
            sampling_params['n'] = n_samples
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
        else:
            sampling_params['n'] = 1

        # Decode to strings for SGLang
        prompt_strings = []
        for token_ids in idx_list:
            prompt_str = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            prompt_strings.append(prompt_str)

        # Call remote actor for generation
        output_ids_nested, log_probs_nested = ray.get(
            self.inference_actor.generate.remote(prompt_strings, sampling_params),
            timeout=300  # 5 minute timeout
        )

        # Process outputs - flatten if n > 1
        all_responses = []
        for prompt_outputs in output_ids_nested:
            if len(prompt_outputs) > 0:
                all_responses.append(prompt_outputs[0])  # Take first sample per prompt
            else:
                all_responses.append([])

        # Pad responses to response_length
        response_length = self.config.get('response_length', 1024)
        max_response_len = max(len(r) for r in all_responses) if all_responses else 0
        max_response_len = min(max_response_len, response_length)
        max_response_len = max(max_response_len, 1)  # At least 1 token

        padded_responses = []
        for resp in all_responses:
            if len(resp) < response_length:
                padding = [self.pad_token_id] * (response_length - len(resp))
                padded_responses.append(resp + padding)
            else:
                padded_responses.append(resp[:response_length])

        response = torch.tensor(padded_responses, dtype=torch.long, device=device)

        # Create log probs placeholder (will be recomputed by actor)
        log_probs = torch.zeros_like(response, dtype=torch.float)

        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)

        # Update position_ids and attention_mask
        response_len = response.size(1)
        delta_position_id = torch.arange(1, response_len + 1, device=device)
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

    def sync_weights(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """Sync weights to the SGLang actor."""
        state_dict_ref = ray.put(state_dict)
        return ray.get(self.inference_actor.sync_weights_from_dict.remote(state_dict_ref))

    def offload_weights(self) -> bool:
        """Offload weights from the SGLang actor."""
        return ray.get(self.inference_actor.offload_weights.remote())

    def get_memory_info(self) -> Dict[str, float]:
        """Get memory info from the SGLang actor."""
        return ray.get(self.inference_actor.get_memory_info.remote())

    def shutdown(self):
        """Shutdown the SGLang actor."""
        try:
            ray.get(self.inference_actor.shutdown.remote(), timeout=30)
        except Exception as e:
            logger.warning(f"Error during SGLang actor shutdown: {e}")
        try:
            ray.kill(self.inference_actor)
        except Exception as e:
            logger.warning(f"Error killing SGLang actor: {e}")
