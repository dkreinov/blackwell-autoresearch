# KernelBench Power & Clock Characterization -- NVIDIA Thor AGX

**Date:** 2026-03-19
**Hardware:** Thor_AGX (NVIDIA Thor, Blackwell sm_110, compute cap 11.0, aarch64)
**Software:** PyTorch 2.9.1+cu130, CUDA 13.0, Python 3.12.3
**Scope:** Level 1 (100 problems), fp32, cuda_event timing (5 warmup + 100 trials)

---

## Overview

This report characterizes KernelBench Level 1 performance across two power modes on the Jetson AGX Thor, with live power monitoring via `tegrastats`. It extends the MAXN baseline from `04_baseline_pilot.md` by adding power consumption measurements and a direct comparison against the 120W capped mode.

Two of four power modes (90W and 70W) require a full system reboot to activate their GPU power-gating configuration. Given the constraints of remote execution and the user's instruction not to disrupt the system, these modes are documented based on hardware specifications rather than measured data. See [Mode 2 and Mode 3: Reboot Requirement](#mode-2-and-mode-3-reboot-requirement) for details.

---

## Test Matrix

| Mode | ID | GPU Clock | CPU Clock | GPU PG | Tested | Method |
|------|----|-----------|-----------|--------|--------|--------|
| MAXN | 0 | 1575 MHz | 2601 MHz | minimal (64) | ✅ Full run | Live tegrastats |
| 120W | 1 | 1386 MHz | 2601 MHz | minimal (64) | ✅ Full run | Live tegrastats |
| 90W | 2 | 1530 MHz | 2601 MHz | heavy (15873) | ⚠️ Reboot required | Spec only |
| 70W | 3 | 1530 MHz | 1998 MHz | heavy (15873) | ⚠️ Reboot required | Spec only |

**Note on clock locking:** `jetson_clocks` was run after each mode switch to lock clocks at the mode's maximum before starting the benchmark.

---

## Run Conditions

| Parameter | MAXN | 120W |
|-----------|------|------|
| GPU GPC clock (confirmed) | 1575 MHz | 1386 MHz |
| CPU 0 clock | 2601 MHz | 2601 MHz |
| Thermal settle before run | 60 s | 60 s |
| tegrastats interval | 2000 ms | 2000 ms |
| tegrastats samples | 461 | 475 |
| GPU temp at start (idle) | ~34°C | ~35°C |

---

## Performance Summary

| Metric | MAXN | 120W | Delta |
|--------|------|------|-------|
| Problems completed | 99/100 | 99/100 | -- |
| Wall time (s) | 920.9 | 948.6 | +3.0% |
| Timing: min (ms) | 0.73 | 0.74 | +1.4% |
| Timing: median (ms) | 51.50 | 54.90 | +6.6% |
| Timing: mean (ms) | 59.01 | 61.63 | +4.4% |
| Timing: max (ms) | 197.00 | 198.00 | +0.5% |

**Key finding:** Reducing GPU clock by 12% (1575→1386 MHz) results in only a 6.6% median slowdown. Memory-bandwidth-bound operations are largely unaffected; compute-bound operations slow in proportion to the clock reduction.

---

## Power Consumption

All power values are from tegrastats averages over the full benchmark run.

| Sensor | MAXN (mean) | 120W (mean) | Delta |
|--------|-------------|-------------|-------|
| VDD_GPU (inst avg, mW) | 22,052 | 20,788 | −5.7% |
| VDD_GPU (peak inst, mW) | 38,365 | 36,334 | −5.3% |
| VDD_CPU_SOC_MSS (avg, mW) | 14,517 | 14,259 | −1.8% |
| VIN_SYS_5V0 (avg, mW) | 12,241 | 12,108 | −1.1% |

**Note on VIN (total board power):** The tegrastats log on this Thor system shows total board power (`VIN`) only as a single field name, distinct from `VIN_SYS_5V0`. The GPU-specific `VDD_GPU` sensor is the most reliable indicator of compute workload power.

---

## Thermal Profile

| Sensor | MAXN mean | MAXN max | 120W mean | 120W max |
|--------|-----------|----------|-----------|----------|
| GPU (°C) | 50.6 | 66.6 | 48.6 | 63.2 |
| CPU (°C) | 48.8 | 57.8 | 47.1 | 55.7 |
| TJ (junction, °C) | -- | -- | 48.9 | 63.1 |
| SoC 0-2 (°C) | -- | -- | 45.6 | 54.0 |

Both modes remained well within thermal limits (throttle threshold ~90°C). The 120W mode ran ~2°C cooler on average.

---

## Performance per Watt

Using GPU power as the denominator (most relevant for compute workload characterization):

| Mode | Problems/s | GPU Power (W) | Problems/s/kW |
|------|-----------|---------------|---------------|
| MAXN | 0.1075 | 22.05 | 4.88 |
| 120W | 0.1044 | 20.79 | 5.02 |

**120W mode delivers 2.9% better performance-per-watt than MAXN.** This is a common result in GPU characterization: running at maximum frequency is often not the most efficient operating point due to the superlinear relationship between voltage and frequency (dynamic power ∝ V² × f).

---

## Per-Category Timing: MAXN vs 120W

| Category | n | MAXN median (ms) | 120W median (ms) | Slowdown |
|----------|---|------------------|------------------|----------|
| matmul | 18 | 23.05 | 25.95 | +12.6% |
| conv | 35 | 29.20 | 32.30 | +10.6% |
| activation | 13 | 56.70 | 56.70 | 0.0% |
| normalization | 8 | 112.50 | 115.00 | +2.2% |
| pooling | 6 | 72.60 | 77.35 | +6.5% |
| reduction | 11 | 65.00 | 64.70 | −0.5% |
| softmax | 2 | 105.00 | 104.60 | −0.4% |
| loss | 5 | 69.10 | 70.10 | +1.4% |
| attention | 1 | 143.00 | 162.00 | +13.3% |

### Key Category Observations

1. **Compute-bound (matmul, attention) show the largest slowdown** (+12-13%) -- these are directly limited by FLOP throughput, which scales with GPU clock.
2. **Memory-bandwidth-bound (activation, reduction, softmax) are clock-insensitive** -- running at essentially the same speed because the bottleneck is memory access latency, not compute. Thor's unified memory bus is shared between modes.
3. **Normalization is barely affected** (+2.2%) -- mostly bandwidth-bound with reductions over small tensors.
4. **Convolutions sit in the middle** (+10.6%) -- a mix of GEMM-based (clock-sensitive) and bandwidth-sensitive operations.

---

## Mode 2 and Mode 3: Reboot Requirement

Attempting `sudo nvpmodel -m 2` (90W) produced:
```
NVPM WARN: Reboot required for changing to this power mode: 2
NVPM WARN: DO YOU WANT TO REBOOT NOW? enter YES/yes to confirm:
NVPM ERROR: bad input!
NVPM ERROR: optMask is 1, no request for power mode
```

Mode 2 and Mode 3 both use GPU power-gating mask 15873 (vs 64 for MAXN/120W), which requires disabling GPU compute partitions -- a hardware-level change that can only take effect at boot. The same applies to the CPU core count reduction (14→12 cores).

### Estimated Mode 2 (90W) Characteristics

Based on specs and extrapolation from MAXN/120W measurements:
- GPU clock: 1530 MHz (≈ same as 120W, but MAXN power level due to heavy PG)
- Expected GPU power: ~15–18W (PG reduces effective wattage dramatically)
- Expected timing: Compute-bound ops ~3% faster than 120W (clock 1530 vs 1386), bandwidth-bound ops unchanged
- CPU: Same as 120W for this benchmark (the 12-core limit doesn't affect single-threaded dispatch)

### Estimated Mode 3 (70W) Characteristics

- GPU clock: 1530 MHz
- CPU clock: 1998 MHz (23% lower than MAXN)
- Expected GPU power: ~15W
- Expected timing: GPU-bound ops similar to 90W; CPU-bound dispatch may add overhead

---

## Failure Analysis

Problem 95 (CrossEntropyLoss) failed in both measured modes:
- **Root cause:** `nll_loss_forward_reduce_cuda_kernel_2d_index` not implemented for `Float` on sm_110 (Blackwell) in PyTorch 2.9.1+cu130
- **Mode-invariant:** This is a kernel dispatch issue, not a power/clock issue
- **Resolution:** Expected to be fixed in PyTorch 2.10.x or later cu130 builds

---

## Recommendations

1. **Use 120W mode for extended benchmarking** -- 2.9% better performance-per-watt, 2°C cooler, minimal throughput loss. For workloads that are not compute-bound (most of Level 1), the difference is negligible.

2. **MAXN for latency-critical compute-bound workloads** -- Matmul, attention, convolution where every ms counts.

3. **Test modes 2/3 with a dedicated reboot sequence** -- Boot into mode 2, run benchmark, reboot into mode 3, run benchmark, reboot into MAXN. This can be automated via `cron` or a post-boot script.

4. **GPU clock sweep within MAXN** -- Since modes 2/3 require reboots, a more practical sweep is: set GPU min/max via sysfs (`/sys/class/devfreq/gpu-gpc-0/{min,max}_freq`) to lock specific clock points (25%, 50%, 75%, 100%) while staying in MAXN. This gives clean clock-vs-performance data without reboots.

---

## Files Produced

```
~/thor_kernelbench_work/results/Thor_AGX/
├── power_MAXN_level1_1-100.json     # MAXN mode, all 100 problems + power
├── power_120W_level1_1-100.json     # 120W mode, all 100 problems + power
├── tegrastats_MAXN.log              # Raw tegrastats output (MAXN run)
├── tegrastats_120W.log              # Raw tegrastats output (120W run)
└── baseline_level1.json             # Phase 3 reference (no power data)
```

---

## Methodology Notes

- **Script:** `run_power_sweep.py` -- sets mode, locks clocks, waits 60s thermal settle, starts tegrastats daemon, runs timing, stops tegrastats, parses power log.
- **tegrastats sampling:** 2-second interval throughout the benchmark run. Each sensor (`VDD_GPU`, `VDD_CPU_SOC_MSS`, `VIN_SYS_5V0`) provides instantaneous and running-average readings.
- **Timing method:** Same as Phase 3 -- `cuda_event`, 5 warmup trials, 100 timed trials per problem.
- **Clock verification:** GPU GPC clock confirmed via `/sys/class/devfreq/gpu-gpc-0/cur_freq` before each run.
