# SGLang + veRL Fix Report for DGX Spark (GB10)

**Date:** 2026-01-06
**Status:** Fixed and ready for testing

---

## Executive Summary

The SGLang rollout was failing on DGX Spark (GB10) due to a dependency conflict. The `uv sync --extra sglang` command was installing incompatible versions of `sglang` and `sgl-kernel` from PyPI that shadowed the working versions from the base Docker image.

**Root Cause:** PyPI's `sgl-kernel` 0.3.20 requires CUDA 12 libraries (`libnvrtc.so.12`), but DGX Spark has CUDA 13.0.

**Fix Applied:** Removed the `sglang` extra dependency so the venv uses the compatible system packages from `lmsysorg/sglang:spark`.

---

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GB10 (SM 12.1) |
| CUDA | 13.0 |
| Platform | ARM64 (aarch64) |
| Base Image | `lmsysorg/sglang:spark` |
| Python | 3.12.3 |

---

## Working Package Versions

| Package | Version | Source |
|---------|---------|--------|
| sglang | 0.5.4.post2 | System (base image) |
| sgl-kernel | 0.3.16.post5 | System (base image) |
| **torch** | **2.9.0+cu130** | **System (base image)** - MUST NOT be shadowed by venv |
| ray | 2.53.0 | Venv (uv installed) |
| verl | local | Venv (uv installed) |

---

## Files Modified

1. **`pyproject.toml`** - Removed `sglang` extra dependency
2. **`.devcontainer/devcontainer.json`** - Removed `--extra sglang` from postCreateCommand
3. **`uv.lock`** - Regenerated without sglang packages

---

## How to Test

### Quick Verification (< 1 minute)

```bash
# Verify all packages import correctly
/workspace/.venv.linux-aarch64/bin/python -c "
import ray, sglang, sgl_kernel, verl
print('Ray:', ray.__version__)
print('SGLang:', sglang.__version__)
print('sgl_kernel:', sgl_kernel.__version__)
print('verl: OK')
"
```

**Expected output:**
```
Ray: 2.53.0
SGLang: 0.5.4.post2
sgl_kernel: 0.3.16.post5
verl: OK
```

### SGLang Engine Test (~ 30 seconds)

Create and run a test file:

```bash
cat > /tmp/test_sglang_engine.py << 'EOF'
#!/usr/bin/env python3
"""Test SGLang Engine on GB10"""

if __name__ == '__main__':
    from sglang import Engine

    print("Initializing SGLang Engine...")
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
    print("SUCCESS: SGLang Engine works!")
EOF

/workspace/.venv.linux-aarch64/bin/python /tmp/test_sglang_engine.py
```

**Expected output:**
```
Initializing SGLang Engine...
Generation:  Paris. It is the largest city in Europe and
SUCCESS: SGLang Engine works!
```

### Full veRL + Ray + SGLang Test (optional, longer)

To test the full RL training stack, run the tutorial:

```bash
cd /workspace
/workspace/.venv.linux-aarch64/bin/python tutorial/reinforce.py
```

---

## Notes for Next Session

1. **Always use the venv Python:** `/workspace/.venv.linux-aarch64/bin/python`

2. **PyTorch SM warning is expected and harmless:**
   ```
   Found GPU0 NVIDIA GB10 which is of cuda capability 12.1.
   Minimum and Maximum cuda capability supported by this version of PyTorch is (8.0) - (12.0)
   ```

3. **FA3 (FlashAttention 3) warnings are informational only** - they affect some vision models but not text generation.

4. **If you rebuild the devcontainer**, it will now correctly use system sglang packages.

5. **Do NOT run:** `pip install sglang`, `pip install torch`, or `uv sync --extra sglang` - this will reinstall broken CPU packages that shadow the working system CUDA packages.

---

## Troubleshooting

### If SGLang fails with `libnvrtc.so.12` or `libc10_cuda.so` error:
```bash
# Check which packages are being used
/workspace/.venv.linux-aarch64/bin/python -c "import sgl_kernel; print('sgl_kernel:', sgl_kernel.__file__)"
/workspace/.venv.linux-aarch64/bin/python -c "import torch; print('torch:', torch.__file__, torch.__version__)"

# sgl_kernel should show: /usr/local/lib/python3.12/dist-packages/sgl_kernel/__init__.py
# torch should show: /usr/local/lib/python3.12/dist-packages/torch/__init__.py 2.9.0+cu130

# If either shows .venv path or torch shows +cpu, uninstall the venv versions:
/workspace/.venv.linux-aarch64/bin/python -m pip uninstall sgl-kernel sglang torch -y
```

### If CUDA is not available in PyTorch:
```bash
# Check torch version - MUST be 2.9.0+cu130, NOT 2.9.1+cpu
/workspace/.venv.linux-aarch64/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# If it shows 2.9.1+cpu or CUDA=False, uninstall venv torch:
/workspace/.venv.linux-aarch64/bin/python -m pip uninstall torch -y
```

---

## References

- Full investigation: `/workspace/DEPENDENCY_INVESTIGATION.md`
- Base image: `lmsysorg/sglang:spark`
- GPU: NVIDIA GB10 (Blackwell architecture, SM 12.1)
