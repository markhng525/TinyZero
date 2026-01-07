# DGX Spark / GB10 Dependency Investigation

## Date: 2026-01-06

## Environment Summary

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM 12.1 / compute capability 12.1) |
| CUDA | 13.0 |
| Platform | ARM64 (aarch64) |
| Base Image | `lmsysorg/sglang:spark` |
| Python | 3.12.3 |
| uv | 0.9.7 |

## Root Cause Found

**There are TWO conflicting sgl-kernel installations:**

### 1. System sgl-kernel (WORKS)
- **Version:** 0.3.16.post5
- **Location:** `/usr/local/lib/python3.12/dist-packages/sgl_kernel/`
- **Status:** Works correctly with CUDA 13.0 and GB10 (SM121)
- **Kernels:** Has SM121a kernels embedded in `common_ops.abi3.so` and `spatial_ops.abi3.so`

### 2. Venv sgl-kernel (BROKEN)
- **Version:** 0.3.20
- **Location:** `/workspace/.venv.linux-aarch64/lib/python3.12/site-packages/sgl_kernel/`
- **Status:** FAILS with error: `libnvrtc.so.12: cannot open shared object file`
- **Problem:** Built for CUDA 12, but we have CUDA 13.0
- **Kernels:** Has only `sm90/` and `sm100/` folders - NO SM121 support

## Why This Happened

1. The devcontainer `postCreateCommand` runs:
   ```bash
   uv venv --system-site-packages && uv sync --extra sglang --all-groups
   ```

2. The `pyproject.toml` has `sglang = ["sglang[all]>=0.4.6"]` as an optional dependency

3. `uv sync --extra sglang` installs sglang from PyPI, which pulls in sgl-kernel 0.3.20

4. This **shadows** the working system sgl-kernel 0.3.16.post5 from the base image

5. The PyPI sgl-kernel 0.3.20 was built for CUDA 12 (x86_64 or different ARM variant) and lacks:
   - CUDA 13 library support (looks for libnvrtc.so.12)
   - SM121 kernels (GB10's compute capability)

## Package Version Comparison

| Package | System (base image) | Venv (uv installed) |
|---------|---------------------|---------------------|
| sglang | 0.5.4.post2 | 0.5.7 |
| sgl-kernel | 0.3.16.post5 (WORKS) | 0.3.20 (BROKEN) |
| torch | 2.9.0+cu130 | (uses system) |
| ray | Not installed | 2.53.0 |
| verl | Not installed | Installed from local |

## Verification Commands

```bash
# System Python (works but no ray/verl):
python3 -c "import sgl_kernel; print(sgl_kernel.__version__)"  # 0.3.16.post5

# Venv Python (has ray/verl but broken sgl_kernel):
/workspace/.venv.linux-aarch64/bin/python -c "import sgl_kernel"  # FAILS
/workspace/.venv.linux-aarch64/bin/python -c "import ray; print(ray.__version__)"  # 2.53.0
```

## Solutions

### Option A: Use System sgl-kernel (Recommended)
Remove the venv's sgl-kernel so it falls back to the working system version:
```bash
/workspace/.venv.linux-aarch64/bin/pip uninstall sgl-kernel -y
```

### Option B: Fix uv.lock / pyproject.toml
Exclude sgl-kernel from the sglang extra and rely on system-site-packages.

### Option C: Create symlink for CUDA 13 → 12
```bash
ln -s /usr/local/cuda/lib64/libnvrtc.so.13 /usr/local/cuda/lib64/libnvrtc.so.12
```
(This is a hack and may cause other issues)

## Resolution (2026-01-06)

**Fix Applied:**
```bash
# Uninstalled both broken packages from venv
/workspace/.venv.linux-aarch64/bin/python -m pip uninstall sgl-kernel -y
/workspace/.venv.linux-aarch64/bin/python -m pip uninstall sglang -y
```

**Result:** The venv now falls back to system-site-packages for SGLang/sgl-kernel:
- sgl_kernel: 0.3.16.post5 (system) - WORKS with CUDA 13.0 and GB10
- sglang: 0.5.4.post2 (system) - Compatible with sgl_kernel 0.3.16.post5
- ray: 2.53.0 (venv) - Works
- verl: local install (venv) - Works

**Verification:**
```bash
/workspace/.venv.linux-aarch64/bin/python /workspace/test_full_stack.py
# Output: === All tests passed! ===
```

## Prevention Applied

- [x] Removed `sglang` extra from `pyproject.toml` (rely on base image)
- [x] Updated `.devcontainer/devcontainer.json` to remove `--extra sglang`
- [x] Regenerated `uv.lock` to exclude sglang/sgl-kernel packages
