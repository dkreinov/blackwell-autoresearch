# Thor CUDA Kernel Autoresearch -- fp16 Findings

Per-problem optimization history for fp16 precision. Each section tracks experiments for one kernel.
Current best = the kernel file in kernels/fp16/p{pid}_{name}.py.
Results source of truth: results/Thor_AGX/kernel_results_fp16.json.
Baselines: results/Thor_AGX/baseline_level1_fp16.json.
Eval: `python scripts/eval_kernel.py --pid <N> --kernel kernels/fp16/p{N}_*.py --precision fp16`

26/28 fp32 kernels use data_ptr<float>() and cannot work with fp16 inputs.
2 pure-PyTorch kernels (p92, p97) transferred directly -- dtype-agnostic.

---

### p19 ReLU (fp16)

fp16 baseline: 27.7ms (fp32 baseline: 57.5ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p20 LeakyReLU (fp16)

fp16 baseline: 27.8ms (fp32 baseline: 56.6ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p21 Sigmoid (fp16)

fp16 baseline: 27.5ms (fp32 baseline: 56.6ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p22 Tanh (fp16)

fp16 baseline: 27.8ms (fp32 baseline: 56.8ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p23 Softmax (fp16)

fp16 baseline: 36.1ms (fp32 baseline: 100.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 31.00 | 1.165x | online softmax, float4 (8 halfs), __ldcg pass1 + __ldlu+__stwt pass2, 8x unroll, 1024t |

- FAIL v2 (43.60ms): 512t -- worse, 1024t saturates bandwidth better for 4096-row problem
- FLOOR: 31.0ms (1.165x). 1024t optimal. Compute-bound by __expf calls in online reduce.

### p24 LogSoftmax (fp16)

fp16 baseline: 32.6ms (fp32 baseline: 110.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 31.00 | 1.052x | online logsoftmax, float4 (8 halfs), __ldcg pass1 + __ldlu+__stwt pass2, 8x unroll, 1024t |

### p25 Swish (fp16)

fp16 baseline: 69.0ms (fp32 baseline: 142.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 46.0 | 1.500x | half2 vectorized, naive loads |

### p26 GELU (fp16)

fp16 baseline: 27.7ms (fp32 baseline: 56.8ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p27 SELU (fp16)

fp16 baseline: 27.7ms (fp32 baseline: 56.8ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p28 HardSigmoid (fp16)

fp16 baseline: 27.8ms (fp32 baseline: 56.7ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p29 Softplus (fp16)

fp16 baseline: 28.6ms (fp32 baseline: 56.5ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p30 Softsign (fp16)

fp16 baseline: 96.9ms (fp32 baseline: 197.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 26.70 | 3.629x | float4 (8 halfs), float32 __fdividef, __ldlu+__stwt, 1024t exact-grid |

### p31 ELU (fp16)

fp16 baseline: 27.7ms (fp32 baseline: 56.5ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p32 HardTanh (fp16)

fp16 baseline: 27.7ms (fp32 baseline: 56.7ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

### p33 BatchNorm2d (fp16)

fp16 baseline: 64.5ms (fp32 baseline: 91.5ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 30.40 | 2.122x | 1 block/channel, 2-pass, float4 8x unroll, __ldcg pass1 + __ldlu+__stwt pass2, fmaf normalize, float32 weight/bias (must call .float() in forward since .half() converts params) |

### p34 InstanceNorm2d (fp16)

fp16 baseline: 82.5ms (fp32 baseline: 135.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 35.70 | 2.311x | float4 (8 halfs) loads, 8x unroll, float32 accum, 2-pass reduce, __ldcg+__stwt, 1024t |
| 3 | 34.70 | 2.378x | __ldcg pass1 (L2-only, keeps for pass2 re-read) + __ldlu+__stwt pass2 (evict-after-use + bypass write) |

### p35 GroupNorm (fp16)

fp16 baseline: 86.5ms (fp32 baseline: N/A)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 52.90 | 1.635x | 896 blocks (B*G), 1024t, 2-pass, float4 8x unroll, __ldcg+__ldlu+__stwt, channel index via j>>15 (HW4=32768=2^15), per-channel affine with float32 wt/bi |

### p36 RMSNorm (fp16)

fp16 baseline: 105.0ms (fp32 baseline: 172.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 41.60 | 2.524x | 1 thread/pos, 64ch into float32 regs via __ldg, float32 accumulation, 512t |
| 2 | 41.30 | 2.542x | __ldcg (L2-only cache) for reads -- bypasses L1 since 64ch*512*512*2=32MB >> L1=256KB |
| 3 | 41.00 | 2.561x | __ldcg reads + __stcs streaming writes (evict-first, output never re-read) |
| 4 | 40.50 | 2.593x | __ldlu (evict-after-single-use) reads + __stcs writes -- perfect: each element read once |
| 5 | 36.70 | 2.861x | __ldlu reads + __stwt (write-through bypass ALL caches) -- frees L2 entirely for reads |
| 6 | 36.60 | 2.869x | half2[32] channel packing (halves reg count), __ldlu+__stwt, 512t |

- FAIL v2 (incorrect): 1024 threads -- v[64] float needs 64 regs/thread, 1024t = 65536 regs at SM limit, register spilling causes corruption
- FAIL v3 (44.60ms): 256 threads -- fewer warps per block, worse latency hiding
- DISCARD v4 (41.60ms tied): half2[32] packed channels -- identical timing to v1, bandwidth floor reached
- FAIL v5 (205ms): warp-per-position (32 threads/pos, 2 channels/thread) -- breaks coalescing: threads access channels c*HW+lane strided by HW apart within a warp
- FAIL v6 (48.4ms): 2 positions per thread, 256t -- doubles register pressure, register spilling to L2
- FAIL v7 (46.5ms): 128 threads -- 6 blocks/SM but worse performance, 512t remains best
- FLOOR: 41.6ms (2.524x). All block sizes tried (128/256/512); 512t optimal. 1024t broken (65536 register limit).
- FAIL v9 (43.9ms): __ldlu+__stwt with 256t -- worse despite extra L2 parallelism
- FAIL v11 (tied 36.6ms): 1024t with half2[32] -- identical to 512t
- FAIL v12 (46.4ms): #pragma unroll 4 -- partial unroll breaks latency hiding, full unroll required
- FAIL v13 (37.7ms): __launch_bounds__(512,2) -- compiler hint hurts
- FAIL v14 (36.7ms): separate load/accumulate phases -- compiler already does this
- KEY INSIGHT: __ldlu+__stwt pattern: evict-after-read + bypass-all-caches for write = best for streaming kernels

### p37 FrobeniusNorm (fp16)

fp16 baseline: 64.4ms (fp32 baseline: 98.5ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 53.60 | 1.201x | 2-kernel: atomicAdd partial_sums into float array (1024×1024t), CPU rsqrt, normalize pass (__ldlu+__stwt+__hmul2) |

### p38 L1Norm (fp16)

fp16 baseline: 117.0ms (fp32 baseline: 193.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 51.70 | 2.263x | fused 2-pass L2-reuse, scalar __half loads, float32 accumulation, 8x unroll, 1024t |
| 2 | 42.00 | 2.786x | half2 packing via __halves2half2, __habs2, 4x half2 unroll, 1024t |

- FAIL v3 (42.80ms): branch on row parity for direct half2 loads on even rows -- no improvement, memory bandwidth bound not instruction bound
- FAIL v4 (43.20ms): 8x half2 unroll -- extra register pressure outweighs ILP gain, kernel near bandwidth floor

### p39 L2Norm (fp16)

fp16 baseline: 80.0ms (fp32 baseline: 118.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 41.50 | 1.928x | fused 2-pass, float4 (8 halfs), float32 accum sumsq, __ldcg pass1 + __ldlu+__stwt pass2, 8x unroll, 1024t |

### p88 MinGPTNewGelu (fp16)

fp16 baseline: 10.5ms (fp32 baseline: 19.8ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 1.18 | 8.898x | float4 (8 halfs), float32 tanhf, 512t exact-grid |

### p91 CumsumReverse (fp16)

fp16 baseline: 99.0ms (fp32 baseline: 110.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|

- FAIL v1 (INCORRECT, max_diff=2952): tile-based suffix scan, float32 accumulation -- PyTorch fp16 cumsum accumulates in fp16, causing ~14% underestimation at n=32768 (ref≈13568 vs float32≈16312). Our more-accurate result fails the correctness check.
- APPROACH: PyTorch flip+cumsum+flip fallback (identical to reference, max_diff=0). No speedup possible without exactly replicating fp16 accumulation semantics.

### p92 CumsumExclusive (fp16)

fp16 baseline: 102.0ms (fp32 baseline: 122.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 85.90 | 1.187x | fp32 kernel transferred (pure PyTorch cumsum+shift, dtype-agnostic) |

### p93 MaskedCumsum (fp16)

fp16 baseline: 73.1ms (fp32 baseline: 90.5ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 72.50 | 1.008x | PyTorch cumsum(x*mask) fallback (matches reference fp16 semantics) |

- FAIL custom CUDA v1 (INCORRECT, max_diff=1276): float32 accumulation diverges from PyTorch's fp16 cumsum at large N (same issue as p91/p93 cumsum operations)

### p94 MSELoss (fp16)

fp16 baseline: 55.1ms (fp32 baseline: 103.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 18.90 | 2.915x | float4 (8 halfs), 2048×1024t, atomicAdd global reduce, float32 accum, __ldcg both inputs |

### p96 HuberLoss (fp16)

fp16 baseline: 36.9ms (fp32 baseline: 69.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 18.90 | 1.952x | smooth_l1, float4 (8 halfs), 2048×1024t, atomicAdd global reduce, float32 accum |

### p97 ScaledDotProductAttention (fp16)

fp16 baseline: 52.1ms (fp32 baseline: 143.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 32.00 | 1.628x | fp32 kernel transferred (baddbmm+bmm+softmax, dtype-agnostic) |

### p100 HingeLoss (fp16)

fp16 baseline: 73.5ms (fp32 baseline: 122.0ms)

| Version | Time (ms) | Speedup | Change |
|---------|-----------|---------|--------|
| 1 | 8.91 | 8.249x | float4 (8 halfs), target L1-cached, atomicAdd reduce, 2048x1024t |

- FAIL v4 (42.00ms): 512t -- fewer concurrent memory requests per block, worse DRAM saturation than 1024t
- FAIL v5 (34.80ms): half2 FMA in pass2 instead of float32 fmaf -- tied, bandwidth floor dominates instruction savings
- FAIL v6 (34.70ms): 1 syncthreads (all threads reduce ws1/ws2 locally) -- tied, barrier overhead negligible vs memory latency
- FAIL v7 (34.70ms): 768t (24 warps/block, 2 blocks/SM possible) -- tied with 1024t, no benefit
- FAIL v8 (INCORRECT): #pragma unroll 4 on outer 8x loop -- 4*8=32 simultaneous float4 = 128 regs, register spill → corruption
- FAIL v9 (35.70ms): __ldcg for pass2 reads (no eviction) -- worse than __ldlu; L2 pollution from unreleased data hurts next block's pass1 caching
- KEY INSIGHT: __ldlu in pass2 is strictly better than __ldcg: eviction frees L2 for next block's pass1 data
- FAIL v10 (34.80ms): __launch_bounds__(1024,1) -- no benefit, compiler already uses full register budget
- FAIL v11 (34.90ms): 4x unroll -- fewer outstanding requests, worse than 8x (less ILP and memory pipelining)
- FLOOR: 34.7ms (2.378x). 2-pass with __ldcg+__ldlu+__stwt is optimal. 1024t max memory bandwidth. 8x unroll is optimal. Remaining gap from ~26ms theoretical is inherent DRAM+L2 bandwidth limit.
- **Dirty state cleanup** (2026-03-26): discarded stale candidates: p34_instancenorm_candidate.py
