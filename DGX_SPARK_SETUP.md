# DGX Spark (GB10) Environment Setup Guide

**Last Updated:** 2026-01-06
**Status:** Working

---

## Table of Contents

1. [Environment Overview](#environment-overview)
2. [The Problem](#the-problem)
3. [The Solution](#the-solution)
4. [Current Working State](#current-working-state)
5. [Verification](#verification)
6. [Running Experiments](#running-experiments)
7. [Troubleshooting](#troubleshooting)
8. [Prevention & Maintenance](#prevention--maintenance)

---

## Environment Overview

### Hardware

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (Blackwell architecture) |
| Compute Capability | SM 12.1 |
| Platform | ARM64 (aarch64) |

### Software Stack

| Component | Version |
|-----------|---------|
| CUDA | 13.0 |
| Driver | 580.95.05 |
| Python | 3.12.3 |
| Base Image | `lmsysorg/sglang:spark` |
| uv | 0.9.7 |

### Package Architecture

The environment uses a **hybrid package strategy**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Virtual Environment               │
│                /workspace/.venv.linux-aarch64               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Venv site-packages (priority 1)                     │   │
│  │ - ray 2.53.0                                        │   │
│  │ - verl (local editable install)                     │   │
│  │ - transformers, datasets, etc.                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ System site-packages (priority 2, fallback)         │   │
│  │ - torch 2.9.0+cu130 (CUDA-enabled)                  │   │
│  │ - sglang 0.5.4.post2                                │   │
│  │ - sgl-kernel 0.3.16.post5 (GB10-compatible)         │   │
│  │ - torchvision, torchaudio                           │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle:** Critical CUDA packages (torch, sglang, sgl-kernel) come from the base image and must NOT be shadowed by venv installations.

---

## The Problem

### Symptom

When running SGLang-based rollouts, the following errors occurred:

```
ImportError: libnvrtc.so.12: cannot open shared object file: No such file or directory
```

or

```
ImportError: libc10_cuda.so: cannot open shared object file: No such file or directory
```

### Root Cause

The `uv sync` command installed incompatible packages from PyPI into the venv that **shadowed** the working system packages:

| Package | System (WORKS) | Venv (BROKEN) | Problem |
|---------|----------------|---------------|---------|
| sgl-kernel | 0.3.16.post5 | 0.3.20 | PyPI version built for CUDA 12, not CUDA 13 |
| sglang | 0.5.4.post2 | 0.5.7 | Version mismatch with sgl-kernel |
| torch | 2.9.0+cu130 | 2.9.1+cpu | PyPI version is CPU-only, no CUDA |

### Why This Happened

1. The `pyproject.toml` had `sglang = ["sglang[all]>=0.4.6"]` as an optional dependency
2. `uv sync --extra sglang` installed sglang from PyPI
3. PyPI packages lack:
   - CUDA 13.0 library support (they expect CUDA 12)
   - SM 12.1 (GB10) kernel support
   - ARM64-optimized CUDA binaries
4. Python's import system prioritizes venv packages over system packages

---

## The Solution

### What Was Done

1. **Removed sglang extra** from `pyproject.toml`
2. **Updated devcontainer** to not install `--extra sglang`
3. **Uninstalled shadowing packages** from venv:
   ```bash
   /workspace/.venv.linux-aarch64/bin/python -m pip uninstall sgl-kernel sglang torch -y
   ```
4. **Regenerated `uv.lock`** without sglang packages

### Why It Works

With the venv packages removed, Python falls back to the system site-packages (enabled via `include-system-site-packages = true` in `pyvenv.cfg`), which contain the properly built CUDA 13.0 / GB10-compatible versions.

---

## Current Working State

### Package Versions

| Package | Version | Location | Notes |
|---------|---------|----------|-------|
| torch | 2.9.0+cu130 | System | CUDA-enabled |
| sglang | 0.5.4.post2 | System | GB10-compatible |
| sgl-kernel | 0.3.16.post5 | System | Has SM121 kernels |
| ray | 2.53.0 | Venv | For distributed training |
| verl | local | Venv | RL training framework |

### Expected Warnings (Harmless)

1. **PyTorch SM Warning:**
   ```
   Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
   Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
   ```
   This is informational only - PyTorch works correctly on GB10.

2. **FA3 (FlashAttention 3) Warnings:**
   ```
   Ignore import error when loading sglang.srt.models.clip: Can not import FA3 in sgl_kernel.
   ```
   These only affect vision models, not text generation.

---

## Verification

### Quick Check (< 10 seconds)

```bash
/workspace/.venv.linux-aarch64/bin/python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
import sgl_kernel
print('sgl_kernel:', sgl_kernel.__version__)
import sglang
print('SGLang:', sglang.__version__)
import ray
print('Ray:', ray.__version__)
import verl
print('verl: OK')
"
```

**Expected Output:**
```
PyTorch: 2.9.0+cu130
CUDA: True
sgl_kernel: 0.3.16.post5
SGLang: 0.5.4.post2
Ray: 2.53.0
verl: OK
```

### SGLang Engine Test (~ 30 seconds)

```bash
/workspace/.venv.linux-aarch64/bin/python << 'EOF'
from sglang import Engine

engine = Engine(
    model_path="Qwen/Qwen2.5-0.5B",
    tp_size=1,
    disable_cuda_graph=True,
    log_level="warning",
)

result = engine.generate(
    prompt="The capital of France is",
    sampling_params={"max_new_tokens": 10, "temperature": 0}
)
print(f"Generation: {result['text']}")
engine.shutdown()
print("SUCCESS!")
EOF
```

**Expected Output:**
```
Generation:  Paris. It is the largest city in Europe and
SUCCESS!
```

### Full Stack Test

```bash
cd /workspace
/workspace/.venv.linux-aarch64/bin/python tutorial/reinforce.py
```

---

## Running Experiments

### GRPO Countdown Experiment

```bash
cd /workspace
bash experiments/run_grpo_countdown.sh
```

### Prerequisites Checklist

- [ ] Data files exist: `/workspace/data/countdown/train.parquet` and `test.parquet`
- [ ] Quick verification passes (see above)
- [ ] No other Ray clusters running (`ray stop` to clean up)

---

## Troubleshooting

### Error: `libnvrtc.so.12` or `libc10_cuda.so` not found

**Cause:** Venv has shadowing packages installed.

**Fix:**
```bash
# Check package locations
/workspace/.venv.linux-aarch64/bin/python -c "import sgl_kernel; print(sgl_kernel.__file__)"
/workspace/.venv.linux-aarch64/bin/python -c "import torch; print(torch.__file__)"

# Should show /usr/local/lib/python3.12/dist-packages/...
# If shows .venv path, uninstall:
/workspace/.venv.linux-aarch64/bin/python -m pip uninstall sgl-kernel sglang torch -y
```

### Error: `CUDA available: False` or `torch 2.9.1+cpu`

**Cause:** CPU-only PyTorch installed in venv.

**Fix:**
```bash
/workspace/.venv.linux-aarch64/bin/python -m pip uninstall torch -y
```

### Error: Ray cluster issues

**Fix:**
```bash
ray stop --force
# Wait a few seconds, then retry
```

### Error: Out of memory

**Fix:** Reduce batch size in experiment config or ensure no other processes are using GPU:
```bash
nvidia-smi  # Check for running processes
```

---

## Prevention & Maintenance

### DO NOT Run These Commands

These will reinstall broken packages and break the setup:

```bash
# DO NOT RUN:
pip install sglang
pip install sgl-kernel
pip install torch
uv sync --extra sglang
uv add sglang
```

### Safe Commands

These are safe to run:

```bash
# Update other packages
uv sync

# Add new packages (except torch/sglang/sgl-kernel)
uv add <package-name>

# Rebuild venv (will need to re-fix)
uv venv --system-site-packages
uv sync
```

### After Rebuilding Venv

If you rebuild the venv or devcontainer, verify that torch/sglang/sgl-kernel are coming from system:

```bash
/workspace/.venv.linux-aarch64/bin/python -c "
import torch; print('torch:', torch.__file__)
import sglang; print('sglang:', sglang.__file__)
import sgl_kernel; print('sgl_kernel:', sgl_kernel.__file__)
"
```

All paths should show `/usr/local/lib/python3.12/dist-packages/...`

If any show `.venv`, uninstall them:
```bash
/workspace/.venv.linux-aarch64/bin/python -m pip uninstall torch sglang sgl-kernel -y
```

---

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        DGX Spark (GB10)                            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    Docker Container                          │ │
│  │                  lmsysorg/sglang:spark                       │ │
│  │                                                              │ │
│  │  ┌────────────────────┐    ┌────────────────────┐           │ │
│  │  │  System Packages   │    │   Virtual Env      │           │ │
│  │  │  (CUDA 13.0 built) │◄───│ (.venv.linux-aarch64)          │ │
│  │  │                    │    │                    │           │ │
│  │  │  • torch 2.9.0+cu130    │  • ray 2.53.0     │           │ │
│  │  │  • sglang 0.5.4.post2   │  • verl (local)   │           │ │
│  │  │  • sgl-kernel 0.3.16    │  • transformers   │           │ │
│  │  │                    │    │  • datasets       │           │ │
│  │  └────────────────────┘    └────────────────────┘           │ │
│  │           │                         │                        │ │
│  │           └─────────┬───────────────┘                        │ │
│  │                     ▼                                        │ │
│  │         ┌───────────────────────┐                            │ │
│  │         │  Python 3.12 Runtime  │                            │ │
│  │         │  (include-system-     │                            │ │
│  │         │   site-packages=true) │                            │ │
│  │         └───────────────────────┘                            │ │
│  │                     │                                        │ │
│  └─────────────────────┼────────────────────────────────────────┘ │
│                        ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    CUDA 13.0 / Driver 580                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                        │                                          │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                 NVIDIA GB10 GPU (SM 12.1)                    │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## References

- Base image: `lmsysorg/sglang:spark`
- GPU: NVIDIA GB10 (Blackwell architecture, SM 12.1)
- Detailed investigation: `/workspace/DEPENDENCY_INVESTIGATION.md`
- Fix report: `/workspace/SGLANG_FIX_REPORT.md`
