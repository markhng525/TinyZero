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
veRL SGLang integration module.

This module provides a custom SGLang Engine wrapper for the HybridEngine
pattern, matching the interface of verl.third_party.vllm.

Key features:
- Accept actor_module in __init__ (not model_path)
- Support load_format='dummy' for weight-free initialization
- Provide sync_model_weights() for batch weight updates
- Provide offload_model_weights() for memory management
- Provide init_cache_engine() / free_cache_engine() for KV cache control
"""

# Version check - require SGLang 0.5.4+ for memory saver APIs
try:
    import sglang
    sglang_version = getattr(sglang, '__version__', '0.0.0')
except ImportError:
    raise ImportError(
        "SGLang is not installed. Please install it with: "
        "pip install sglang"
    )

# SGLang 0.5.4+ has update_weights_from_tensor and memory saver APIs
MIN_SGLANG_VERSION = "0.5.4"
from packaging import version
if version.parse(sglang_version) < version.parse(MIN_SGLANG_VERSION):
    raise ImportError(
        f"SGLang >= {MIN_SGLANG_VERSION} required for veRL integration, "
        f"but found {sglang_version}. Please upgrade: pip install -U sglang"
    )

from .config import LoadFormat, normalize_load_format
from .engine import Engine
from .server_engine import ServerEngine

__all__ = [
    'Engine',
    'ServerEngine',
    'LoadFormat',
    'normalize_load_format',
    'sglang_version',
]
