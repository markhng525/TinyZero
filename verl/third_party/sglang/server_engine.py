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
HTTP Server-based SGLang Engine for veRL HybridEngine.

This module runs SGLang as a completely separate HTTP server process,
avoiding the CUDA context conflicts that occur when SGLang runs as a
subprocess of Ray actors.

Key differences from in-process Engine:
1. SGLang runs in a completely separate process (not subprocess of Ray)
2. Communication via HTTP API
3. Weight sync via disk + HTTP endpoint
4. No NCCL or pickle serialization issues

Usage:
    engine = ServerEngine(
        model_path="Qwen/Qwen2.5-0.5B",
        port=30000,
        ...
    )
    engine.wait_for_ready()

    # Sync weights
    engine.update_weights_from_disk("/path/to/weights")

    # Generate
    outputs = engine.generate(prompts, sampling_params)

    # Shutdown
    engine.shutdown()
"""

import os
import time
import json
import socket
import signal
import tempfile
import subprocess
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import requests
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find an available port for the SGLang server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class ServerEngine:
    """SGLang Engine running as HTTP server for veRL HybridEngine workflow.

    This avoids CUDA context conflicts by running SGLang in a completely
    separate process, not as a subprocess of Ray actors.

    IMPORTANT: Unlike the in-process Engine, this DOES NOT support load_format='dummy'.
    SGLang must load real weights from disk first for update_weights_from_disk to work.
    This means higher initial memory usage but reliable weight sync via disk.

    Trade-offs vs in-process Engine:
    + Complete CUDA context isolation (no segfaults in Ray)
    + Reliable weight sync via HTTP + disk
    - Higher initial memory (loads weights from HF first)
    - Slightly higher latency (HTTP overhead)
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
        port: Optional[int] = None,
        host: str = "127.0.0.1",
        **kwargs,
    ) -> None:
        """Initialize SGLang HTTP server.

        Args:
            actor_module: FSDP-wrapped actor module (not used directly)
            tokenizer: HuggingFace tokenizer
            model_hf_config: HuggingFace model configuration
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Model dtype
            gpu_memory_utilization: Fraction of GPU memory for KV cache
            enforce_eager: If True, disable CUDA graphs
            load_format: Weight loading format ('dummy' for HybridEngine)
            port: HTTP server port (auto-assigned if None)
            host: HTTP server host
            **kwargs: Additional arguments
        """
        self.model_hf_config = model_hf_config
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self._load_format = load_format

        # Get model path for SGLang initialization
        self.model_path = getattr(model_hf_config, '_name_or_path', None)
        if self.model_path is None:
            raise ValueError("model_hf_config must have '_name_or_path' attribute")

        # Server configuration
        self.host = host
        self.port = port if port is not None else find_free_port()
        self.base_url = f"http://{self.host}:{self.port}"

        # Server process handle
        self._server_process: Optional[subprocess.Popen] = None
        self._temp_weight_dir: Optional[str] = None

        # State tracking
        self._weights_offloaded = False
        self._server_ready = False

        # Launch server
        self._launch_server(
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            load_format=load_format,
        )

        logger.info(f"SGLang server engine initialized at {self.base_url}")

    def _launch_server(
        self,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        load_format: str,
    ) -> None:
        """Launch SGLang HTTP server as a separate process."""
        import sys
        # Build command - use sys.executable to ensure we use the same Python interpreter
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--tp-size", str(self.tensor_parallel_size),
            "--dtype", self.dtype,
            "--mem-fraction-static", str(gpu_memory_utilization),
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "warning",
        ]

        if enforce_eager:
            cmd.append("--disable-cuda-graph")

        # NOTE: Do NOT use load_format='dummy' for HTTP server mode!
        # SGLang's DummyModelLoader does not support weight updates via update_weights_from_disk.
        # We must load real weights initially, then sync FSDP weights via disk.
        # This means HTTP server mode has higher initial memory cost but works reliably.
        if load_format == 'dummy':
            logger.warning(
                "load_format='dummy' is not supported for HTTP server mode. "
                "SGLang will load model weights from disk first, then we'll sync FSDP weights."
            )
            # Don't add --load-format dummy - let SGLang load real weights

        # Set environment variables for NCCL
        env = os.environ.copy()
        env["NCCL_CUMEM_ENABLE"] = "0"
        env["NCCL_NVLS_ENABLE"] = "0"

        logger.info(f"Launching SGLang server: {' '.join(cmd)}")

        # Launch as completely separate process
        self._server_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new session
        )

        # Wait for server to be ready
        self._wait_for_ready(timeout=120)

    def _wait_for_ready(self, timeout: int = 120) -> None:
        """Wait for the server to become ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    self._server_ready = True
                    logger.info(f"SGLang server ready at {self.base_url}")
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        # Check if process died
        if self._server_process.poll() is not None:
            stdout, stderr = self._server_process.communicate()
            raise RuntimeError(
                f"SGLang server process died. "
                f"stdout: {stdout.decode()}, stderr: {stderr.decode()}"
            )

        raise TimeoutError(f"SGLang server failed to start within {timeout} seconds")

    def wait_for_ready(self, timeout: int = 120) -> None:
        """Public method to wait for server readiness."""
        if not self._server_ready:
            self._wait_for_ready(timeout)

    # === Weight Management APIs ===

    def sync_model_weights(
        self,
        actor_weights: Dict[str, torch.Tensor],
        load_format: str,
    ) -> None:
        """Synchronize weights from FSDP actor to SGLang engine.

        Saves weights to disk and uses HTTP endpoint for update.

        Args:
            actor_weights: State dict from FSDP actor
            load_format: 'hf' for full params, 'dtensor' for sharded
        """
        logger.info(f"Syncing model weights via disk (format: {load_format})")

        # Create temporary directory for weights
        if self._temp_weight_dir is None:
            self._temp_weight_dir = tempfile.mkdtemp(prefix="verl_sglang_weights_")

        # Prepare weights for saving
        weights_to_save = self._prepare_weights(actor_weights, load_format)

        # Save as safetensors
        weight_path = os.path.join(self._temp_weight_dir, "model.safetensors")
        save_file(weights_to_save, weight_path)
        logger.info(f"Saved {len(weights_to_save)} parameters to {weight_path}")

        # Call HTTP endpoint to update weights
        success, message = self.update_weights_from_disk(self._temp_weight_dir)

        if not success:
            raise RuntimeError(f"Failed to update weights via HTTP: {message}")

        self._weights_offloaded = False
        logger.info("Weight sync complete")

    def _prepare_weights(
        self,
        actor_weights: Dict[str, torch.Tensor],
        load_format: str,
    ) -> Dict[str, torch.Tensor]:
        """Prepare weights for saving to disk.

        Args:
            actor_weights: State dict from FSDP
            load_format: 'hf' or 'dtensor'

        Returns:
            Dict of cleaned parameter names to CPU tensors
        """
        from torch.distributed._tensor import DTensor

        prepared = {}

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

            prepared[name] = tensor

        return prepared

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Update model weights from disk via HTTP endpoint.

        Args:
            model_path: Path to directory containing safetensors weights
            load_format: Optional load format override

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Updating weights from disk: {model_path}")

        payload = {"model_path": model_path}
        if load_format:
            payload["load_format"] = load_format

        try:
            response = requests.post(
                f"{self.base_url}/update_weights_from_disk",
                json=payload,
                timeout=300,  # 5 minutes for large models
            )

            if response.status_code == 200:
                result = response.json()
                success = result.get("success", False)
                message = result.get("message", "Unknown")
                return success, message
            else:
                return False, f"HTTP {response.status_code}: {response.text}"

        except requests.exceptions.RequestException as e:
            return False, str(e)

    def offload_model_weights(self) -> None:
        """Offload model weights to release GPU memory.

        Uses SGLang's memory saver API via HTTP.
        """
        if self._weights_offloaded:
            logger.debug("Weights already offloaded, skipping")
            return

        try:
            response = requests.post(
                f"{self.base_url}/release_memory_occupation",
                json={"tags": ["weights"]},
                timeout=60,
            )
            if response.status_code == 200:
                self._weights_offloaded = True
                logger.info("Weights offloaded successfully")
            else:
                logger.warning(f"Failed to offload weights: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to offload weights: {e}")

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
        """
        # Convert sampling params to SGLang format
        sglang_params = {
            "max_new_tokens": sampling_params.get("max_new_tokens", 256),
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "top_k": sampling_params.get("top_k", -1),
        }

        # Build request payload
        payload = {
            "input_ids": input_ids,
            "sampling_params": sglang_params,
        }

        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=300,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Generation failed: {response.text}")

            results = response.json()
            return self._process_outputs(results, sampling_params)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Generation request failed: {e}")

    def _process_outputs(
        self,
        results: Union[dict, List[dict]],
        sampling_params: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process SGLang output to (response_ids, log_probs) tensors."""
        if isinstance(results, dict):
            results = [results]

        # Extract output token IDs
        all_output_ids = []
        for result in results:
            output_ids = result.get('output_ids', result.get('token_ids', []))
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

        # Return zeros for log_probs (actor will recompute)
        log_probs = torch.zeros_like(response_ids, dtype=torch.float32)

        return response_ids, log_probs

    # === Utility Methods ===

    def shutdown(self) -> None:
        """Shutdown the SGLang server."""
        logger.info("Shutting down SGLang server")

        if self._server_process is not None:
            try:
                # Send SIGTERM to the process group
                os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                self._server_process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                # Force kill if needed
                try:
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

            self._server_process = None

        # Cleanup temp directory
        if self._temp_weight_dir is not None:
            import shutil
            try:
                shutil.rmtree(self._temp_weight_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")
            self._temp_weight_dir = None

        self._server_ready = False

    def __del__(self):
        """Cleanup on destruction."""
        self.shutdown()

    @property
    def engine(self) -> None:
        """No direct engine access in server mode."""
        return None
