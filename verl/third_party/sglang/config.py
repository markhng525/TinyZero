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
Configuration for veRL SGLang integration.

This module defines load formats and configuration options following
the vLLM pattern from verl.third_party.vllm.
"""

from enum import Enum


class LoadFormat(str, Enum):
    """Load format for SGLang weight synchronization.

    Matches the vLLM LoadFormat pattern for consistency.
    """
    AUTO = 'auto'
    HF = 'hf'               # Full state dict from HuggingFace model
    DTENSOR = 'dtensor'     # Sharded DTensor from FSDP
    DUMMY = 'dummy'         # Initialize with random/empty weights (HybridEngine)
    DUMMY_HF = 'dummy_hf'   # Dummy init, expect HF format sync
    DUMMY_DTENSOR = 'dummy_dtensor'  # Dummy init, expect DTensor format sync


def normalize_load_format(load_format: str) -> str:
    """Normalize load format string to SGLang-compatible format.

    SGLang only supports 'dummy' for HybridEngine workflow,
    not 'dummy_dtensor' or 'dummy_hf' variants.

    Args:
        load_format: The load format string from config

    Returns:
        Normalized format ('dummy' or 'auto')
    """
    if load_format.startswith('dummy'):
        return 'dummy'
    return load_format
