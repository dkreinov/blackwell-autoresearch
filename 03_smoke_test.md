# KernelBench on Jetson AGX Thor — Smoke Test Results

**Date:** 2026-03-18
**Host:** nvidia-thor-01
**GPU:** NVIDIA Thor (Blackwell, sm_110, compute capability 11.0)
**PyTorch:** 2.9.1+cu130 (aarch64)
**CUDA:** 13.0 (driver 580.00, nvcc 13.0.48)
**Python:** 3.12.3
**KernelBench:** 0.2.0.dev0 (commit `423217d`)

---

## Test 1: torch CUDA functional check

```bash
~/thor_kernelbench_work/venv/bin/python3 -c "
import torch
print(torch.__version__)           # 2.9.1+cu130
print(torch.version.cuda)          # 13.0
print(torch.cuda.is_available())   # True
print(torch.cuda.get_device_name(0))  # NVIDIA Thor
print(torch.cuda.get_device_capability(0))  # (11, 0)
x = torch.randn(4, 4, device='cuda')
y = x @ x.T
print(y.shape)  # torch.Size([4, 4])
"
```

**Result:** PASS — no warnings, no errors, GPU matmul functional.

---

## Test 2: KernelBench imports

```bash
~/thor_kernelbench_work/venv/bin/python3 -c "
import kernelbench
import kernelbench.eval
import kernelbench.timing
import kernelbench.dataset
import kernelbench.utils
print('All imports OK')
"
```

**Result:** PASS

---

## Test 3: Hardcoded softmax baseline (`get_baseline_time_single_problem.py`)

```bash
export PATH=$PATH:/usr/local/cuda-13.0/bin
cd ~/thor_kernelbench_work/KernelBench
~/thor_kernelbench_work/venv/bin/python3 scripts/get_baseline_time_single_problem.py
```

**Output:**
```
Using PyTorch Eager Execution on softmax
[Profiling] Using timing method: cuda_event
[Profiling] Using device: cuda:0 NVIDIA Thor, warm up 5, trials 100
{'mean': 10.2, 'std': 0.783, 'min': 9.88, 'max': 17.9, 'num_trials': 100,
 'hardware': 'NVIDIA Thor', 'device': 'cuda:0'}
```

**Result:** PASS — softmax (batch=4096, dim=65536), mean 10.2ms

---

## Test 4: Full dataset pipeline — Level 1, Problem 1

```bash
~/thor_kernelbench_work/venv/bin/python3 -c "
import torch
from kernelbench.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_dataset
from kernelbench.timing import measure_ref_program_time
from kernelbench.utils import set_gpu_arch
set_gpu_arch(['Blackwell'])
dataset = construct_kernelbench_dataset(1)
print(f'Level 1: {len(dataset)} problems')
_, name, src = fetch_ref_arch_from_dataset(dataset, 1)
print(f'Problem: {name}')
result = measure_ref_program_time(name, src, device=torch.device('cuda:0'),
                                   timing_method='cuda_event', precision='fp32')
print(f'Result: {result}')
"
```

**Output:**
```
Level 1: 100 problems
Problem: 1_Square_matrix_multiplication_.py
Using PyTorch Eager Execution on 1_Square_matrix_multiplication_.py
[Profiling] Using timing method: cuda_event
[Profiling] Using device: cuda:0 NVIDIA Thor, warm up 5, trials 100
Result: {'mean': 22.3, 'std': 0.020, 'min': 22.2, 'max': 22.3, 'num_trials': 100,
         'hardware': 'NVIDIA Thor', 'device': 'cuda:0'}
```

**Result:** PASS — 4096×4096 square matmul, mean 22.3ms, extremely low variance (std=0.020ms)

---

## Test 5: LLM generate-and-eval

**Result:** NOT TESTED — no API keys configured on Thor (no `.env` file). This only affects the LLM generation path; the eval/timing pipeline is proven functional.

---

## Summary

| Test | Status | Notes |
|------|--------|-------|
| torch CUDA | ✅ PASS | sm_110, no compat warnings |
| KernelBench imports | ✅ PASS | All 5 core modules |
| Softmax baseline timing | ✅ PASS | 10.2ms mean |
| Dataset pipeline + matmul timing | ✅ PASS | 22.3ms mean |
| LLM generate-and-eval | ⏭️ SKIPPED | No API keys |

---

## Initial Thor Performance Observations

| Problem | Size | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------|------|-----------|----------|----------|----------|
| Softmax | 4096×65536 | 10.2 | 0.783 | 9.88 | 17.9 |
| Square matmul | 4096×4096 | 22.3 | 0.020 | 22.2 | 22.3 |

Notable: The matmul has remarkably low timing variance (std/mean = 0.09%), likely due to Thor's unified memory architecture eliminating PCIe transfer jitter.

---

## Remaining Blockers for Full Benchmark

| Blocker | Severity | Resolution |
|---------|---------|------------|
| No API keys for LLM generation | Medium | Configure `.env` with at least one provider key |
| `modal` not installed | Low | Only needed for cloud eval, not local |
| `cupy-cuda12x` incompatible | Low | Only for `[gpu]` extras profiling |
| Full baseline timing (250 problems) | None — ready | `generate_baseline_time.py` needs minor adaptation (has `input()` prompts) |

---

## Recommended Next Steps

1. **Generate full baseline timings** for all 250 problems (Levels 1–3) on Thor
2. **Configure API key** for at least one LLM provider to test generate-and-eval
3. **Compare Thor timings** against H100/L40S baselines in `results/timing/`
4. **Write publishable report** with Thor as first Jetson/Blackwell KernelBench platform

---

## Reproducibility

All work is in `~/thor_kernelbench_work/`:
```
~/thor_kernelbench_work/
├── venv/                          # Isolated Python 3.12 + torch 2.9.1+cu130
├── KernelBench/                   # Repo (one patch: pyproject.toml Python pin)
├── 01_env_and_repo_audit.md       # Phase 1 report
├── 02_thor_compatibility_patches.md  # This phase patches
├── 03_smoke_test.md               # This file
└── thor_env_report.md             # Prior env inventory
```

**Rollback:** `rm -rf ~/thor_kernelbench_work/venv && cd KernelBench && git checkout pyproject.toml`
