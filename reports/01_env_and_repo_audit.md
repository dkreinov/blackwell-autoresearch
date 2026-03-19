# KernelBench on Jetson AGX Thor — Environment & Repo Audit

**Date:** 2026-03-18
**Host:** nvidia-thor-01
**Goal:** Establish what works, what blocks, and what to fix before running any benchmarks.

---

## 1. Hardware Summary

| Item | Value |
|------|-------|
| GPU | NVIDIA Thor |
| Architecture | Blackwell |
| Compute Capability | 11.0 (sm_110) |
| Memory model | ATS unified (no dedicated VRAM — shared with CPU) |
| Driver | 580.00 |
| CUDA version (driver) | 13.0 |
| CPU | ARM Cortex (aarch64), 14 cores, 2.6 GHz max |
| OS | Ubuntu 24.04.3 LTS, Linux 6.8.12-tegra |
| Platform | Tegra/Jetson SoC |
| Disk free | 564 GB (NVMe) |

---

## 2. CUDA Toolkit

| Item | Value |
|------|-------|
| nvcc path | `/usr/local/cuda-13.0/bin/nvcc` |
| nvcc in PATH | **No** — must export PATH |
| CUDA version | 13.0.48 (built Jul 16 2025) |
| CUDA symlink | `/usr/local/cuda` → `/etc/alternatives/cuda` → cuda-13.0 |

**Fix:** `export PATH=$PATH:/usr/local/cuda-13.0/bin`

---

## 3. Python Environment

| Item | Value |
|------|-------|
| System Python | `/usr/bin/python3` → Python **3.12.3** |
| Python 3.10 | **NOT installed** |
| uv | **NOT installed** |
| pip | 24.0 |

### KernelBench requires Python 3.10

`pyproject.toml`: `requires-python = "==3.10.*"`

**This is a hard blocker for `uv sync`.** Options:
1. Install Python 3.10 from deadsnakes PPA and relax the constraint to `>=3.10`
2. Use `pip install -e .` with Python 3.12 (ignores pyproject version pin)
3. Create a venv with Python 3.12 and install deps manually

The code itself has no 3.10-specific syntax — the pin is conservative. Python 3.12 is expected to work.

---

## 4. PyTorch Status

| Location | Version | CUDA build | cuda_available |
|----------|---------|-----------|----------------|
| System pip (`/home/nvidia/.local/`) | 2.9.1+**cpu** | None | **False** |

**CRITICAL BLOCKER: No CUDA-enabled PyTorch.**

KernelBench requires `torch.cuda.is_available() == True` for:
- All timing and benchmarking
- JIT compilation of CUDA extensions (`torch.utils.cpp_extension`)
- Kernel correctness evaluation

A Jetson/Tegra-compatible CUDA PyTorch wheel for aarch64 + CUDA 13.0 must be installed.
Source: NVIDIA Jetson PyTorch release page or build from source.

### Other installed packages

| Package | Version | Notes |
|---------|---------|-------|
| tensorrt | 10.13.3.9 | System pip |
| torchvision | 0.24.1+cpu | CPU only |
| numpy | 1.26.4 | OK |

---

## 5. KernelBench Repo Audit

**Cloned to:** `~/thor_kernelbench_work/KernelBench`
**Branch:** main
**Head commit:** `423217d` — update all legacy python commands to UV + document integration

### 5a. gpu_arch Handling

`src/kernelbench/utils.py:42`:
```python
NVIDIA_ARCHS = ["Maxwell", "Pascal", "Volta", "Turing", "Ampere", "Hopper", "Ada", "Blackwell"]
```

`set_gpu_arch(arch_list)` sets `TORCH_CUDA_ARCH_LIST` env var.

**Thor/Blackwell IS in the list.** Pass `gpu_arch=Blackwell` to all scripts.

Default in `generate_and_eval_single_sample.py`:
```python
self.gpu_arch = ["Ada"]   # must override to ["Blackwell"] for Thor
```

### 5b. sm_110 / Thor explicit support

- `NVIDIA_ARCHS` includes `"Blackwell"` ✓
- No explicit `sm_110` strings in src/ (not needed — arch name is sufficient)
- `eval_from_generations.py:54` Modal mapping does NOT include Thor/Blackwell:
  ```python
  gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], ...}
  ```
  Minor — only affects Modal cloud path, not local evaluation.
- `timing.py:138` L2 cache thrash comment mentions `Blackwell~192MB` — codebase is Blackwell-aware.

### 5c. x86 / ARM assumptions

- **No explicit x86 assumptions found** in Python source code.
- CUDA JIT via `torch.utils.cpp_extension` is arch-neutral (compiles on-device).
- Generated CUDA kernels use standard CUDA C++ — should compile for sm_110 with nvcc 13.0.

### 5d. ARM / CUDA 13 / torch extension risks

| Risk | Severity | Notes |
|------|---------|-------|
| `cupy-cuda12x` in requirements | Medium | CUDA 13 != CUDA 12; cupy-cuda12x likely fails. Only needed for `[gpu]` extras. |
| `triton` for aarch64 | High | PyPI triton wheels are x86_64 only. May need source build or skip entirely. |
| `nvidia-cutlass-dsl` | Unknown | Likely x86 wheel only. Part of `[gpu]` extras. |
| `tilelang` | Unknown | Experimental DSL, likely lacks aarch64 support. |
| `torch.compile` / Inductor | Medium | Inductor uses Triton internally — may fail on aarch64. |
| Unified memory (ATS) | Low-Medium | Timing differs from discrete GPUs; no `cudaMemcpy` between host/device. |
| Python 3.10 pin | High | System only has 3.12. Need to relax pin or install 3.10. |

### 5e. Core dependency assessment

| Package | Required for | Status on Thor |
|---------|-------------|---------------|
| torch (CUDA) | Everything | **MISSING** — CPU only |
| nvcc | Kernel compilation | Present, not in PATH |
| ninja | Fast JIT builds | pip-installable, likely OK |
| pydra-config | Config system | pip-installable, OK |
| litellm | LLM API calls | pip-installable, OK |
| triton | `[gpu]` extras, Inductor | aarch64 wheel uncertain |
| cupy-cuda12x | `[gpu]` extras | CUDA 13 incompatible |

---

## 6. Recommended Patch Plan (ranked by criticality)

### MUST fix before any run

1. **Install CUDA-enabled PyTorch for aarch64 + CUDA 13**
   - Check NVIDIA Jetson PyTorch index for aarch64 CUDA 13 wheel
   - Fallback: build from source with CUDA 13 flags
   - Constraint: must be torch >= 2.9.0 (per pyproject.toml)

2. **Add nvcc to PATH**
   ```bash
   echo 'export PATH=$PATH:/usr/local/cuda-13.0/bin' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Install core Python deps (skip gpu extras initially)**
   ```bash
   pip3 install --user pydra-config ninja tqdm packaging tomli numpy einops \
     python-dotenv litellm openai datasets transformers modal
   cd ~/thor_kernelbench_work/KernelBench
   pip3 install --user -e . --no-deps
   ```

4. **Pass gpu_arch=Blackwell in all benchmark commands**

### Fix for full benchmark (later)

5. **Triton for aarch64** — investigate NVIDIA wheel or skip triton backend
6. **cupy** — skip or use `cupy-cuda13x` if it exists
7. **Python 3.10** — optionally install via deadsnakes if strict compat required

---

## 7. Smoke Test Plan (next step)

Once CUDA torch is installed:

```bash
# 1. Verify CUDA torch
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Verify nvcc
/usr/local/cuda-13.0/bin/nvcc --version

# 3. Run single problem smoke test (level 1, problem 1, local dataset)
cd ~/thor_kernelbench_work/KernelBench
python3 scripts/run_and_check.py level=1 problem_id=1 dataset_src=local eval_mode=local gpu_arch=Blackwell
```

Do NOT start bulk runs until smoke test passes.

---

## 8. Pre-existing Work in This Directory

- `~/thor_kernelbench_work/thor_env_report.md` — detailed prior env inventory (Docker, TensorRT, network)
- `~/thor_kernelbench_work/KernelBench/` — freshly cloned main branch (2026-03-18)

---

## 9. Publishability Checklist

| Component | Status |
|-----------|--------|
| Environment report | DONE (this doc + thor_env_report.md) |
| Thor GPU identification (sm_110, Blackwell, CUDA 13) | DONE |
| Compatibility patch list | DONE (documented above) |
| Thor-specific patches with git diff | BLOCKED — need CUDA torch first |
| Baseline timings (torch eager) | BLOCKED |
| Pilot benchmark results | BLOCKED |
| Analysis writeup | In progress |

**Single unblocking action:** Install CUDA-enabled PyTorch aarch64 wheel for CUDA 13.0.

---

## 10. Phase Summary

### What worked
- SSH, nvidia-smi, disk, CPU info all accessible
- CUDA 13.0 toolkit installed at `/usr/local/cuda-13.0`
- KernelBench repo clones cleanly; Blackwell is in NVIDIA_ARCHS
- No x86-specific assumptions in Python source

### What failed / is blocked
- `torch.cuda.is_available()` → False (CPU-only build)
- `nvcc` not in PATH
- `uv` not installed; Python 3.10 not installed (pyproject.toml requires it)
- `cupy-cuda12x` incompatible with CUDA 13
- `triton` aarch64 wheels uncertain

### Exact commands run
```bash
ssh nvidia@nvidia-thor-01 "nvidia-smi; nvcc --version; uname -a; lscpu; python3 --version; pip3 --version"
ssh nvidia@nvidia-thor-01 "python3 -c 'import torch; print(torch.cuda.is_available())'"
ssh nvidia@nvidia-thor-01 "/usr/local/cuda-13.0/bin/nvcc --version"
ssh nvidia@nvidia-thor-01 "cd ~/thor_kernelbench_work && git clone https://github.com/ScalingIntelligence/KernelBench.git"
# Read: README.md, EVAL.md, scripts/generate_and_eval_single_sample.py,
#       scripts/generate_baseline_time.py, scripts/eval_from_generations.py,
#       src/kernelbench/utils.py, pyproject.toml
```

### Files changed
- `~/thor_kernelbench_work/KernelBench/` — cloned (no patches yet)
- `~/thor_kernelbench_work/01_env_and_repo_audit.md` — this file

### Recommended next step
**Phase 2:** Install CUDA-enabled PyTorch for Jetson AGX Thor (aarch64, CUDA 13.0), verify with smoke test, then install KernelBench core dependencies.
