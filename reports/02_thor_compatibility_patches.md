# KernelBench on Jetson AGX Thor — Compatibility Patches

**Date:** 2026-03-18
**Host:** nvidia-thor-01
**KernelBench commit:** `423217d` (main)

---

## Summary

KernelBench required **two categories of changes** to run on Jetson AGX Thor (Blackwell, sm_110, aarch64, CUDA 13.0). Both are minimal and non-destructive.

---

## Patch 1: Relax Python version pin

**File:** `pyproject.toml` line 10
**Reason:** KernelBench pins `requires-python = "==3.10.*"` but the Jetson AGX Thor ships with Python 3.12.3 (Ubuntu 24.04). No Python 3.10 is available in the system repos. The codebase uses no 3.10-specific syntax — the pin is conservative.

```diff
diff --git a/pyproject.toml b/pyproject.toml
index f3f98ae..7aaff44 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -7,7 +7,7 @@ build-backend = "setuptools.build_meta"
 [project]
 name = "kernelbench"
 version = "0.2.0.dev0"
-requires-python = "==3.10.*"
+requires-python = ">=3.10"
 dependencies = [
     # Frameworks
     "torch>=2.9.0",
```

**Risk:** None observed. All imports and smoke tests pass on Python 3.12.3.

---

## Patch 2 (runtime, not code): Use torch cu130 instead of cu126

**Not a code change** — this is a wheel selection decision at install time.

**Reason:** The default PyTorch cu126 wheel (`torch==2.9.1+cu126`) only includes compiled kernels for sm_80–sm_90. Thor's sm_110 (Blackwell) is not in that set, producing:

```
NVIDIA Thor with CUDA capability sm_110 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_80 sm_90.
```

**Fix:** Install from the cu130 index which includes sm_110 support:

```bash
pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130
```

The cu130 wheel installs CUDA 13.0 runtime libraries that match the system driver exactly.

**Available alternatives tested:**

| Index | torch version | sm_110 support | aarch64 wheel |
|-------|--------------|----------------|---------------|
| cu124 | 2.5.1 max | No | Yes |
| cu126 | 2.9.1, 2.10.0 | No (sm_80-sm_90) | Yes |
| cu128 | 2.9.1, 2.10.0 | Yes | Not verified |
| **cu130** | **2.9.1, 2.10.0** | **Yes** | **Yes** |
| nightly/cu130 | 2.12.0.dev | Yes | Yes |

---

## No code patches needed for:

| Item | Status | Notes |
|------|--------|-------|
| `gpu_arch` handling | Already supported | `"Blackwell"` is in `NVIDIA_ARCHS` list (`utils.py:42`) |
| ARM/aarch64 assumptions | None found | All Python code is platform-neutral |
| CUDA JIT compilation | Works | nvcc 13.0 at `/usr/local/cuda-13.0/bin/nvcc` compiles for sm_110 |
| Unified memory (ATS) | Transparent | No code assumes discrete VRAM; PyTorch handles ATS natively |
| Timing infrastructure | Works | `cuda_event` timing method produces correct results |

---

## Environment Setup (non-destructive)

All changes are confined to `~/thor_kernelbench_work/`. Zero system files modified.

```bash
# 1. Create isolated venv
python3 -m venv ~/thor_kernelbench_work/venv

# 2. Install CUDA-enabled PyTorch (cu130 for sm_110 support)
~/thor_kernelbench_work/venv/bin/pip install 'torch==2.9.1' \
  --index-url https://download.pytorch.org/whl/cu130

# 3. Install core dependencies
~/thor_kernelbench_work/venv/bin/pip install numpy tqdm packaging ninja tomli \
  einops python-dotenv pydra-config tabulate datasets transformers openai litellm

# 4. Install KernelBench (after applying pyproject.toml patch)
~/thor_kernelbench_work/venv/bin/pip install -e ~/thor_kernelbench_work/KernelBench --no-deps

# 5. Run with Blackwell arch
export PATH=$PATH:/usr/local/cuda-13.0/bin
~/thor_kernelbench_work/venv/bin/python3 scripts/get_baseline_time_single_problem.py
```

**Rollback:** `rm -rf ~/thor_kernelbench_work/venv` + `git checkout pyproject.toml`

---

## Skipped dependencies (not needed for baseline timing)

| Package | Why skipped | When needed |
|---------|------------|-------------|
| `modal` | Cloud GPU eval only | `eval_from_generations_modal.py` |
| `triton` | Installed automatically with torch | `backend=triton` |
| `cupy-cuda12x` | CUDA 13 incompatible; `[gpu]` extra only | Experimental profiling |
| `nvidia-cutlass-dsl` | `[gpu]` extra only | `backend=cute` |
| `tilelang` | `[gpu]` extra only | `backend=tilelang` |
