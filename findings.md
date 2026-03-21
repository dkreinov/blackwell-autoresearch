# Thor CUDA Kernel Autoresearch — Findings

## Current State

- Target: KernelBench Level 1 on Thor AGX (sm_110, unified LPDDR5X 229 GB/s)
- Best results: none yet — autoresearch not started
- Baseline: 99/100 problems pass at fp32 (median 51.5ms)
- Prior transfer study: 72% of Sakana H100-optimized kernels also speed up Thor

## Priority Problems (highest baseline time = most room to improve)

| ID | Problem | Baseline (ms) | Category | Transfer Study |
|---|---|---|---|---|
| 30 | Softsign | 197.0 | activation | Sakana: 1.78x |
| 38 | L1Norm | 193.0 | normalization | Sakana: 2.42x |
| 76 | conv_standard_1D_dilated_strided | 181.0 | conv | Sakana: compile fail |
| 36 | RMSNorm | 172.0 | normalization | Sakana: 2.28x |
| 63 | conv_standard_2D_large | 145.0 | conv | Sakana: compile fail |
| 97 | ScaledDotProductAttention | 143.0 | attention | Not in Sakana |
| 25 | Swish | 142.0 | activation | Sakana: 1.65x |
| 64 | conv_transposed_1D | 138.0 | conv | Sakana: compile fail |
| 34 | InstanceNorm | 135.0 | normalization | Sakana: compile fail |
| 92 | cumsum_exclusive | 122.0 | reduction | Sakana: compile fail |

## Completed Experiments

(none yet)

## Active Ideas

(to be filled by the agent based on program.md hardware knowledge)

## Dead Ends

(to be filled as experiments are discarded)

## Observations

- From transfer study: reduction ops consistently backfire on Thor (7/9 slower than PyTorch)
- From transfer study: normalization kernels transfer best (4/4 faster, median 2.35x)
- From transfer study: activations have near-unity transfer ratio (~1.05)
- Problem 12 (diagonal matmul): Thor 5x FASTER than H100 baseline — unified memory advantage
