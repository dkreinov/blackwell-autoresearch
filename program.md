# Thor CUDA Kernel Autoresearch Program

## 1. Mission

Optimize CUDA kernels from KernelBench Level 1 on NVIDIA Jetson AGX Thor (Blackwell sm_110).

- **Primary metric**: speedup vs PyTorch baseline (higher = better)
- **Secondary metric**: performance per watt (speedup / GPU power in watts)
- **Evaluation**: compile with nvcc (sm_110), test correctness (5 trials), benchmark (100 trials, cuda_event)
- **One kernel per experiment**: each KernelBench problem has one PyTorch Model → write one optimized CUDA replacement (ModelNew)
- **Target**: beat PyTorch eager execution on Thor's unified memory architecture

### Autonomy
You are running an autonomous optimization loop. Do NOT pause to ask for confirmation. Run experiments continuously until externally interrupted. If you encounter a problem, fix it yourself or skip and move on.

---

## 2. Thor Hardware Reference

### 2.1 GPU

| Spec | Value | Source |
|---|---|---|
| GPU | NVIDIA Thor | nvidia-smi |
| Architecture | Blackwell (Tegra variant) | nvidia-smi |
| Compute Capability | 11.0 (sm_110) | torch.cuda.get_device_capability() |
| SMs | 20 (128 CUDA cores each = 2,560 total) | torch + web specs |
| Tensor Cores | 96 (5th gen) | Web specs |
| Warp Size | 32 | torch.cuda.get_device_properties() |
| Max Threads per SM | 1,536 | torch.cuda.get_device_properties() |
| Registers per SM | 65,536 | torch.cuda.get_device_properties() |
| Shared Memory per Block | 48 KB default, 227 KB opt-in max | torch.cuda.get_device_properties() |
| Shared Memory per SM | 228 KB | torch.cuda.get_device_properties() |
| L2 Cache | 32 MB | torch.cuda.get_device_properties() |
| GPU Clock | 1,575 MHz (MAXN), 1,386 MHz (120W mode) | Measured |
| FP32 Performance | 8.064 TFLOPS | Web specs |
| Addressing Mode | ATS (Address Translation Services) | nvidia-smi |
| Integrated GPU | Yes (is_integrated=1) — SoC, not discrete | torch.cuda.get_device_properties() |

### 2.2 Memory

| Spec | Value | Source |
|---|---|---|
| Type | LPDDR5X (unified — shared between CPU and GPU) | Web specs |
| Size | 128 GB (125,772 MB usable) | /proc/meminfo |
| Bus Width | 256-bit | Web specs |
| Clock | 4,266 MHz | tegrastats EMC_FREQ |
| Theoretical Bandwidth | 273 GB/s | Web specs |
| Measured Bandwidth | 229 GB/s (84% efficiency) | Our elementwise kernel test |

**Critical difference from H100:** H100 has HBM3 at 2,039 GB/s (discrete, separate from CPU). Thor has LPDDR5X at 229 GB/s (unified, shared with CPU). Thor's memory bandwidth is 8.9x lower than H100.

### 2.3 Power

| Spec | Value | Source |
|---|---|---|
| Power Modes | MAXN (0), 120W (1), 90W (2), 70W (3) | nvpmodel.conf |
| GPU Power (MAXN, benchmark load) | 22.1W avg, 38.4W peak | Our power sweep |
| GPU Power (120W, benchmark load) | 20.8W avg, 36.3W peak | Our power sweep |
| GPU Temp (MAXN, benchmark load) | 50.6°C avg, 66.6°C peak | Our power sweep |
| Sensors | VDD_GPU, VDD_CPU_SOC_MSS, VIN_SYS_5V0 | tegrastats |
| tegrastats format | SENSOR inst_mW/avg_mW | Measured |

### 2.4 Comparison with H100 PCIe

| Spec | Thor | H100 PCIe | Ratio |
|---|---|---|---|
| SMs | 20 | 114 | 0.18x |
| CUDA Cores | 2,560 | 14,592 | 0.18x |
| Memory Bandwidth | 229 GB/s | 2,039 GB/s | 0.11x |
| L2 Cache | 32 MB | 50 MB | 0.64x |
| FP32 TFLOPS | 8.064 | 51.2 | 0.16x |
| Memory | LPDDR5X unified | HBM3 discrete | Different |
| Baseline Median (Level 1) | 51.5 ms | 6.95 ms | 7.6x slower |

---

## 3. Unified Memory on Thor (ATS)

This section describes how Thor's memory differs from discrete GPUs. The LLM's CUDA training data is mostly from H100/A100 with HBM. Thor's unified memory changes which optimizations are effective.

### 3.1 Architecture

- CPU and iGPU share all SoC DRAM (LPDDR5X). There is no dedicated VRAM.
- Device memory, host memory, and unified memory are all on the same physical DRAM.
- There is no PCIe bus between CPU and GPU — no transfer latency, no cudaMemcpy overhead for data movement.
- cudaMemcpy between device pointers on Thor does not perform a DMA copy across a bus. It is still useful because it populates the GPU L2 cache.

[Source: CUDA for Tegra App Note]

### 3.2 Coherency

- Thor has "Sysmem Full Coherency" (two-way): CPU reads GPU cache, GPU reads CPU cache.
- Hardware-managed via SoC interconnect — no manual cache flush/invalidate needed.
- Full coherency is Thor-specific. Xavier and Orin only have one-way I/O coherency.

[Source: CUDA for Tegra App Note, CUDA 13.0 Blog]

### 3.3 Memory Caching by Type (Thor, CUDA 13.0)

| Memory Type | CPU Cache | GPU L2 Cache | Notes |
|---|---|---|---|
| cudaMalloc (device) | Not accessible | Cached | Best for GPU-only data |
| Pageable host (malloc) | Cached | Cached | Directly accessible from GPU kernels |
| Pinned host (cudaHostAlloc) | Cached | NOT cached | GPU uncached on Thor |
| Registered host (cudaHostRegister) | Cached | Cached | Best for shared CPU/GPU data |
| cudaMallocManaged | Cached | NOT cached | UVM driver uses IO coherency only in 13.0 |

[Source: CUDA for Tegra App Note, Section 3.2]

### 3.4 Critical Limitation

**cudaMallocManaged() allocations are NOT cached in GPU L2 in CUDA 13.0 on Thor.** Code using cudaMallocManaged will have worse GPU performance than code using cudaMalloc. The UVM driver currently selects IO coherency only for managed allocations.

[Source: CUDA 13.0 Blog]

### 3.5 Allocation Strategy (Thor, CUDA 13.0)

| Use Case | Recommended Allocation |
|---|---|
| GPU-only data | cudaMalloc — GPU L2 cached |
| Shared CPU/GPU data | cudaHostRegister on malloc'd memory — cached on both |
| Large buffers, GPU-heavy access | cudaMalloc — best GPU performance |
| Small transfer buffers | cudaHostAlloc (pinned) — CPU cached, GPU uncached |
| Do NOT use for GPU performance | cudaMallocManaged — NOT GPU-cached in 13.0 |

[Source: CUDA for Tegra App Note, Section 4]

### 3.6 Implications for Kernel Optimization

Because CPU and GPU share DRAM:
- Shared memory tiling to reduce global memory accesses has a DIFFERENT cost/benefit profile than on H100
- On H100: global memory = HBM (high bandwidth but still bottleneck) → shared memory tiling helps
- On Thor: global memory = LPDDR5X (lower bandwidth, but no PCIe penalty) → tiling benefit depends on L2 cache utilization
- The L2 cache (32 MB) can hold significant working sets — kernels that fit in L2 may not benefit from shared memory tiling
- Reduction operations optimized for HBM access patterns consistently hurt on Thor (see Section 4)

---

## 4. Transfer Study: Measured Results

We evaluated 63 CUDA kernels from Sakana AI's archive (optimized for H100) on Thor. These are empirical facts, not predictions.

[Source: Our evaluation using eval_sakana_kernels.py on Thor sm_110]

### 4.1 Overall Transfer Results

| Metric | Value |
|---|---|
| Tasks evaluated | 63 |
| Compiled + correct on Thor | 32 (51%) |
| Compile failures (sm_110 incompatibility) | 26 (41%) |
| Incorrect output | 5 (8%) |
| Of 32 correct: faster than PyTorch on Thor | 23 (72%) |
| Of 32 correct: slower than PyTorch on Thor | 9 (28%) |
| Thor speedup median | 1.195x |
| Thor speedup range | 0.349x to 39.953x |

### 4.2 What Transfers Well

| Category | n | Success Rate | Median Thor Speedup | Transfer Ratio |
|---|---|---|---|---|
| normalization | 4 | 4/4 (100%) | 2.349x | 0.944 |
| activation | 10 | 9/10 (90%) | 1.228x | 1.047 |
| softmax | 2 | 2/2 (100%) | 1.127x | 1.054 |
| loss | 5 | 4/5 (80%) | 1.864x | 0.700 |
| conv | 1 | 1/1 (100%) | 1.440x | 1.162 |

Normalization kernels transfer best. Activation speedups are nearly identical on Thor and H100 (ratio ~1.05).

### 4.3 What Backfires

| Category | n | Success Rate | Median Thor Speedup | Transfer Ratio |
|---|---|---|---|---|
| reduction | 9 | 2/9 (22%) | 0.750x | 0.350 |

7 out of 9 reduction kernels are SLOWER on Thor than PyTorch baseline. The H100-optimized shared memory tiling and warp-level reduction patterns hurt on Thor's unified LPDDR5X memory.

Specific backfire cases:
- Sum reduction: H100 1.47x → Thor 0.38x
- Max reduction: H100 1.50x → Thor 0.40x
- Argmin: H100 1.73x → Thor 0.62x
- cumsum: H100 2.21x → Thor 0.75x

### 4.4 Standout Successes

| Task | Op | H100 | Thor | Notes |
|---|---|---|---|---|
| 12 | Diagonal Matmul | 54.4x | 39.95x | Specialized algorithm wins on both |
| 88 | MinGPTNewGelu | 5.72x | 6.00x | Fused activation — MORE effective on Thor |
| 97 | CosineSimilarityLoss | 7.64x | 5.35x | Custom reduction + normalize |
| 40 | LayerNorm | 8.60x | 2.57x | Fused norm — transfers but less effective |

---

## 5. Baseline Performance Data

[Source: Our baseline_level1.json (MAXN mode, 100 trials, cuda_event timing)]

### 5.1 Thor Baseline by Category (PyTorch eager, FP32)

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

99/100 problems pass. Problem 95 (CrossEntropyLoss) fails: nll_loss kernel not compiled for sm_110.

### 5.2 Bandwidth-Bound Indicator

Activations (elementwise) on Thor: median 56.7ms for 16M float32 elements = ~4.5 GB/s effective.
Peak measured bandwidth: 229 GB/s. The gap indicates these kernels are not bandwidth-saturating — room for optimization via memory coalescing, vectorized loads, and launch configuration.

---

## 6. KernelBench Problem Format

### 6.1 Problem Structure

Each KernelBench Level 1 problem is a Python file containing:

```python
class Model(nn.Module):
    def __init__(self):       # Store parameters (weights, bias, etc.)
        ...
    def forward(self, *args): # The operation to optimize
        return torch.some_op(...)

def get_inputs():             # Returns list of input tensors
    return [torch.randn(...)]

def get_init_inputs():        # Returns list of Model constructor args
    return []
```

### 6.2 Expected Output

A Python file containing `ModelNew(Model)` that:
1. Compiles custom CUDA code via `torch.utils.cpp_extension.load_inline()` or `load()`
2. Calls the compiled CUDA `forward()` function in `ModelNew.forward()`
3. Produces output identical to `Model.forward()` (within atol=1e-2, rtol=1e-2)

### 6.3 CUDA Code Pattern

The CUDA code exports a `forward()` function via PYBIND11:

```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(...) { ... }

torch::Tensor forward(torch::Tensor input, ...) {
    // Launch kernel, return result
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "description");
}
```

### 6.4 Compilation

- Compiler: nvcc 13.0.48
- Required: `TORCH_CUDA_ARCH_LIST=11.0` (NOT "Blackwell" — that maps to sm_100/103/120/121, not sm_110)
- Flags: `-O3 --use_fast_math` recommended
- Build: `torch.utils.cpp_extension.load(name=..., sources=[...], extra_cuda_cflags=["-O3", "--use_fast_math"])`

---

## 7. Loop Protocol

### Before Each Experiment
1. Read `findings.md` — current best, what's been tried, active ideas, dead ends
2. Read the KernelBench problem (Model class, get_inputs, forward signature)
3. Pick ONE optimization idea (from findings.md or invent one based on hardware knowledge)

### Execute
4. Write the CUDA kernel + ModelNew Python wrapper
5. Compile on Thor (TORCH_CUDA_ARCH_LIST=11.0)
6. Test correctness (5 trials with random inputs, atol=1e-2)
7. Benchmark: 5 warmup + 100 trials, cuda_event timing
8. Read power: tegrastats VDD_GPU during benchmark

### Decide
9. **Improved** (faster than best so far): keep, record in findings.md, save kernel
10. **Same or worse**: discard, one-line note in findings.md dead ends, start next
11. **Compile error**: read error, fix if simple (typo/syntax), discard if fundamental approach problem
12. **Incorrect output**: discard immediately — math was wrong
13. **Hang/timeout**: kill after 120s, discard, move on

### Discipline
- ONE change per experiment
- Read compile errors carefully — they contain sm_110-specific information
- If ideas run out: simplify the kernel, remove unnecessary shared memory, try vectorized loads

---

## 8. Power Monitoring

### Reading Power During Benchmark

```bash
# Start tegrastats daemon (2-second interval)
tegrastats --start --interval 2000 --logfile /tmp/power.log

# ... run benchmark ...

# Stop and read
tegrastats --stop
grep VDD_GPU /tmp/power.log
```

### tegrastats Line Format

```
03-19-2026 07:10:07 RAM 44721/125772MB ... VDD_GPU 3960mW/3960mW VDD_CPU_SOC_MSS 7520mW/7520mW VIN 28796mW/28796mW
```

Each power sensor shows: `<instantaneous>mW/<running_average>mW`

### Performance per Watt

```
perf_per_watt = speedup_vs_baseline / gpu_power_watts
```

A kernel that achieves 2x speedup at 25W GPU power has perf_per_watt = 0.08.
A kernel that achieves 1.5x speedup at 15W GPU power has perf_per_watt = 0.10 (better).

---

## 9. Rules

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

### Anti-Paralysis Rules
1. First tool call within first response — no multi-paragraph analysis before acting
2. No cycle estimation in prose — the benchmark is the oracle
3. One response = one action — every response must contain a tool call
4. If resuming from context summary: read current state, pick next experiment, execute immediately

---

## 10. Output Format

After each experiment:

```
=== EXPERIMENT N ===
Problem: <task_id> <name>
Idea: <one-line description>
Change: <what the kernel does differently, 1-2 lines>
Compile: success | error: <message>
Correct: yes | no (max_diff=X)
Baseline: <X.XX>ms
Custom: <X.XX>ms
Speedup: <X.XX>x
Power: <X.X>W GPU avg
Perf/Watt: <X.XXX>
Status: keep | discard | crash
findings.md updated: yes | no
Next: <what to try next>
===
```
