# Thor CUDA Kernel Autoresearch Program

## 1. Mission

Optimize CUDA kernels from KernelBench Level 1 on NVIDIA Jetson AGX Thor (Blackwell sm_110).

- **Primary metric**: speedup vs PyTorch baseline (higher = better)
- **Evaluation**: compile with nvcc (sm_110), test correctness (3 trials), benchmark (20 trials, cuda_event)
- **Current category**: ACTIVATIONS (13 problems: 19-22, 25-32, 88)
- **Iteration**: per-kernel optimization via `eval_kernel.py` — one problem at a time, round-robin schedule
- **One kernel per experiment**: each KernelBench problem has one PyTorch Model → write one optimized CUDA replacement (ModelNew)

### Autonomy
Run experiments continuously until externally interrupted. Do not pause for confirmation.

---

## 2. Thor Hardware

### 2.1 GPU

| Spec | Value | Source |
|---|---|---|
| GPU | NVIDIA Thor | nvidia-smi |
| Architecture | Blackwell | nvidia-smi (Product Architecture) |
| Compute Capability | 11.0 (sm_110) | torch.cuda.get_device_capability() |
| SMs | 20 | torch.cuda.get_device_properties() |
| CUDA Cores per SM | 128 (2,560 total) | Web specs |
| Tensor Cores | 96 (5th gen, FP8 capable) | Web specs |
| Warp Size | 32 | torch.cuda.get_device_properties() |
| Max Threads per Block | 1,024 | cudaDeviceGetAttribute |
| Max Threads per SM | 1,536 (48 warps) | torch.cuda.get_device_properties() |
| Max Blocks per SM | 24 | cudaDeviceGetAttribute |
| Max Registers per Thread | 255 | CUDA Programming Guide |
| Registers per SM | 65,536 | torch.cuda.get_device_properties() |
| Registers per Block | 65,536 | cudaDeviceGetAttribute |
| Shared Memory per Block | 48 KB default, 227 KB opt-in | torch (shared_memory_per_block, _optin) |
| Shared Memory per SM | 228 KB | torch (shared_memory_per_multiprocessor) |
| L2 Cache | 32 MB | torch (L2_cache_size) |
| GPU Clock (MAXN) | 1,575 MHz | sysfs gpu-gpc-0/cur_freq |
| GPU Clock (120W mode) | 1,386 MHz | Measured (power sweep) |
| FP32 Performance | 8.064 TFLOPS | Web specs |
| Addressing Mode | ATS | nvidia-smi -q |
| Integrated GPU | Yes (SoC, no discrete VRAM) | torch (is_integrated=1) |
| Cooperative Launch | Supported | cudaDeviceGetAttribute |
| Thread Block Clusters | Supported | PTX ISA 9.0 (.blocksareclusters) |
| PTX ISA | 9.0 | CUDA 13.0 |

### 2.2 Memory

| Spec | Value | Source |
|---|---|---|
| Type | LPDDR5X (unified CPU+GPU, shared DRAM) | Web specs |
| Size | 128 GB (125,772 MB usable) | /proc/meminfo, torch |
| Bus Width | 256-bit | Web specs |
| EMC Clock | 4,266 MHz | sysfs bpmp/debug/clk/emc/rate (sudo) |
| Theoretical Bandwidth | 273 GB/s | Web specs |
| PyTorch Allocator | native (cudaMalloc) | torch.cuda.get_allocator_backend() |

### 2.3 GPU L2 Caching by Allocation Type (CUDA 13.0)

| Allocation | GPU L2 Cached |
|---|---|
| cudaMalloc | Yes |
| Pageable host (malloc) | Yes |
| cudaHostAlloc (pinned) | No |
| cudaMallocManaged | No |
| cudaHostRegister | Yes |

[Source: CUDA for Tegra App Note, Section 3.2]

### 2.4 Coherency

- Two-way full coherency (Sysmem Full Coherency): CPU reads GPU cache, GPU reads GPU cache.
- Hardware-managed via SoC interconnect.

[Source: CUDA for Tegra App Note, CUDA 13.0 Blog]

### 2.5 Power

| Spec | Value | Source |
|---|---|---|
| Power Modes | MAXN (0), 120W (1), 90W (2), 70W (3) | nvpmodel.conf |
| GPU Power (MAXN, Level 1 benchmark) | 22.1W avg, 38.4W peak | Our power sweep |
| GPU Power (120W, Level 1 benchmark) | 20.8W avg, 36.3W peak | Our power sweep |
| GPU Temp (MAXN, Level 1 benchmark) | 50.6°C avg, 66.6°C peak | Our power sweep |

### 2.6 Software

| Component | Version | Source |
|---|---|---|
| Driver | 580.00 | nvidia-smi |
| CUDA | 13.0 | nvidia-smi |
| nvcc | 13.0.48 | nvcc --version |
| torch | 2.9.1+cu130 | Python |
| Python | 3.12.3 | system |
| OS | Ubuntu 24.04.3 LTS (aarch64, tegra) | system |

---

## 3. Compilation

- Target: `TORCH_CUDA_ARCH_LIST=11.0` (NOT "Blackwell" — that maps to sm_100/120, not sm_110)
- Flags: `-O3 --use_fast_math`
- Build: `torch.utils.cpp_extension.load_inline()`
- Correctness tolerance: atol=1e-2, rtol=1e-2

---

## 4. Baseline Data

[Source: baseline_level1.json, MAXN mode, 100 trials, cuda_event timing]

99/100 problems pass. Problem 95 (CrossEntropyLoss) fails on sm_110.

### By Category (PyTorch eager, FP32)

| Category | n | Median (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| matmul | 18 | 23.05 | 0.73 | 72.90 |
| conv | 35 | 29.20 | 8.32 | 181.00 |
| activation | 13 | 56.70 | 19.80 | 197.00 |
| normalization | 8 | 112.50 | 12.50 | 193.00 |
| pooling | 6 | 72.60 | 33.90 | 86.90 |
| reduction | 11 | 65.00 | 42.90 | 122.00 |
| softmax | 2 | 105.00 | 100.00 | 110.00 |
| loss | 5 | 69.10 | 47.40 | 122.00 |
| attention | 1 | 143.00 | 143.00 | 143.00 |

---

## 5. Activation Loop Protocol

### Category: Activations (13 problems)

All 13 are elementwise `x -> f(x)` ops on a 4096×393216 float32 tensor (except pid 88: 8192×8192).
Each problem has its own independently-evolving kernel file.

Current best speedups are tracked in `results/Thor_AGX/kernel_results.json` (source of truth).

| PID | Name | Baseline (ms) | Kernel File |
|-----|------|---------------|-------------|
| 19 | ReLU | 57.5 | kernels/p19_relu.py |
| 20 | LeakyReLU | 56.6 | kernels/p20_leakyrelu.py |
| 21 | Sigmoid | 56.6 | kernels/p21_sigmoid.py |
| 22 | Tanh | 56.8 | kernels/p22_tanh_act.py |
| 25 | Swish | 142.0 | kernels/p25_swish.py |
| 26 | GELU | 56.8 | kernels/p26_gelu.py |
| 27 | SELU | 56.8 | kernels/p27_selu.py |
| 28 | HardSigmoid | 56.7 | kernels/p28_hardsigmoid.py |
| 29 | Softplus | 56.5 | kernels/p29_softplus.py |
| 30 | Softsign | 197.0 | kernels/p30_softsign.py |
| 31 | ELU | 56.5 | kernels/p31_elu.py |
| 32 | HardTanh | 56.7 | kernels/p32_hardtanh.py |
| 88 | MinGPTNewGelu | 19.8 | kernels/p88_mingptnewgelu.py |

[Source: baseline_level1.json, MAXN mode, 100 trials, cuda_event timing]

### Files

- `kernels/p{pid}_{name}.py` — current best kernel (evolves in place, git tracks history)
- `scripts/eval_kernel.py` — single-problem eval: `python scripts/eval_kernel.py --pid <N> --kernel kernels/p{N}_*.py`
- `results/Thor_AGX/kernel_results.json` — per-problem best speedup + version history
- `findings.md` — per-problem experiment results and failures
- `schedule.json` — round-robin order and timing config

### Loop (run autonomously, do not pause)

**Schedule**: `schedule.json` defines the order and timing. Spend 1 hour per activation in round-robin.
Max 2 minutes per experiment (eval times out at 120s).

**Per-experiment cycle** (repeat until 1hr on current activation, then advance schedule):

1. Read `findings.md` section for current problem — what has been tried, what failed
2. Read `results/Thor_AGX/kernel_results.json` — current best speedup for this problem
3. Read current kernel file `kernels/p{pid}_{name}.py` — the code to improve
4. Invent ONE targeted change based on:
   - Thor hardware specs (Section 2)
   - What failed before (findings.md for this problem)
   - Ideas that haven't been tried yet
5. Write the modified kernel to `kernels/p{pid}_{name}_candidate.py`
6. Run: `python scripts/eval_kernel.py --pid <N> --kernel kernels/p{pid}_{name}_candidate.py`
   - Must complete in <120s or it's discarded
7. **If result is correct AND faster than current best**:
   - Overwrite `kernels/p{pid}_{name}.py` with the candidate
   - Update `results/Thor_AGX/kernel_results.json` (best_speedup, best_ms, iterations, history)
   - Update `findings.md` section: add row to table, note what worked
   - Commit on Thor: `git add kernels/p{pid}_{name}.py results/Thor_AGX/kernel_results.json findings.md`
     `git commit -m "p{pid}: {name} v{N} — <change>, {old}x -> {new}x"`
   - Delete the candidate file
8. **If result is slower, incorrect, or times out**:
   - Discard candidate, do NOT overwrite current best
   - Note the failure in findings.md (one line: what was tried and why it failed)
   - Delete the candidate file
9. Go to step 1 for the same problem (until 1hr elapsed, then advance schedule)

### Schedule Enforcement

- Track start time when switching to a new activation
- After 3600s on current activation: advance to next in `schedule.json` order
- After all 13: round-robin back to first
- Only the user decides when to stop the loop entirely

### Commit Format

```
p{pid}: {name} v{N} — {one-line change description}, {old_speedup}x -> {new_speedup}x
```

Only commit improvements. Discards are noted in findings.md but not committed.

---

## 6. Rules

### Allowed
- Write custom CUDA kernels using any standard CUDA features
- Use shared memory, warp shuffles, cooperative groups, vectorized loads
- Use torch.utils.cpp_extension for compilation
- Math approximations if numerically acceptable (correctness oracle catches mismatches)
- FP32 precision (matching baseline)

### NOT Allowed
- Modify the reference Model class, get_inputs(), or get_init_inputs()
- Use cuBLAS/cuDNN library calls that just wrap the same PyTorch operation
- Change the test/evaluation infrastructure
- Use features that require sm_120+ (Thor is sm_110)
- Overwrite a kernel file unless the new version is strictly better (correct + faster)

### Anti-Paralysis Rules
1. First tool call within first response — no multi-paragraph analysis before acting
2. No cycle estimation in prose — the benchmark is the oracle
3. One response = one action — every response must contain a tool call
4. If resuming from context summary: read kernel_results.json + schedule.json, pick current problem, execute immediately

---

## 7. Output Format

After each experiment (one problem, one change):

```
=== p{pid} {name} — attempt {N} ===
Idea: <one-line description of the change>
Result: OK {ms}ms {speedup}x | FAIL {reason}
vs current best: {old_speedup}x -> {new_speedup}x (+{pct}%) | no improvement
Action: committed v{N} | discarded
findings.md: updated | noted failure
Next: <what to try next for this problem>
===
```
