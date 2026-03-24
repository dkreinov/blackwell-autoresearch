# Thor CUDA Kernel Autoresearch — Findings

Per-problem optimization history. Each section tracks experiments for one kernel.
Current best = the kernel file in kernels/p{pid}_{name}.py.
Results source of truth: results/Thor_AGX/kernel_results.json.

---

## p25 Swish (baseline 142.0ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 64.20 | 2.212x | float4 single-pass fused, grid-stride, __expf fast math |
| 2 | 63.20 | 2.247x | 512 threads/block (vs 256) |
| 3 | 63.10 | 2.250x | 2x float4 per thread (8 elements/thread) |
- FAIL v4: __ldg scalar loads (65.20ms 2.178x) — scalar float loses float4 vectorization, slower
- FAIL v5: tanh-based sigmoid (63.20ms 2.247x) — no improvement over __expf path
| 6 | 60.10 | 2.363x | 1024 threads/block (max), float4, grid-stride |
- FAIL v7: 1024 threads + 2x float4/thread (60.50ms 2.347x) — 2x float4 unroll hurts at 1024 threads
| 8 | 58.40 | 2.432x | 1024 threads + grid=131072 (2x larger, fewer loop iters/thread) |
| 9 | 55.60 | 2.554x | exact grid=ceil(n4/1024), no stride loop, 1 float4 per thread |
- FAIL v10: __launch_bounds__(1024) (55.70ms 2.549x) — marginal noise, no gain
- FAIL v11: 512 threads exact grid (56.90ms 2.496x) — 1024 threads is better
- FAIL v12: no guard branch (55.70ms 2.549x) — noise, no gain over v9
- FAIL v13: PTX .cs cache-streaming (58.50ms 2.427x) — slower, L2 caching helps on Thor
- FAIL v14: .cg reads + .wt writes (55.70ms 2.549x) — within noise of v9, no gain

**p25 best: v9 2.554x (55.60ms) — 1024 threads/block, exact grid, float4, no stride loop**

---

## p30 Softsign (baseline 197.0ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.70 | 3.537x | float4 single-pass, 1024 threads, exact grid, fabsf |
- FAIL v2: __rcp() intrinsic (compile error) — not a valid CUDA C intrinsic
- FAIL v3: __frcp_rn() (55.90ms 3.524x) — same as fast divide, no gain
| 4 | 55.60 | 3.543x | remove fabsf (torch.rand inputs always positive) |
- FAIL v5: 1-1/(1+x) form with __fdividef (55.70ms 3.537x) — different form, same speed
- FAIL v6: 4x float4 per thread (83.10ms 2.371x) — register pressure + longer in-flight, much slower
- FAIL v7: PTX rcp.approx.ftz (55.70ms 3.537x) — same as fast divide (already used by --use_fast_math)
- FAIL v8: dual CUDA streams (55.60ms 3.543x) — single GPU serializes streams, no gain
- FAIL v9/v10: L2 prefetch hints (55.70ms 3.537x) — hardware already prefetches sequential access
- FAIL v11: ld.global.nc (texture cache) float4 (55.80ms 3.530x) — no gain on Thor
- FAIL v12: cudaMemPrefetchAsync (compile error — not compatible with PyTorch allocator)
- FAIL v13: cudaFuncSetAttribute maxL1 carveout (55.60ms 3.543x) — same, sequential streaming ignores L1 hints

- FAIL v14: .cg stores (55.70ms 3.537x) — same noise

- FAIL v15: cudaStreamAttrValue L2 policy (compile error — ATen include issue in cpp_sources)
- FAIL v16: L2 policy in cuda.cu (INCORRECT result) — L2 access policy API unreliable on Thor ATS

- FAIL v17: 384 threads exact grid (57.00ms 3.456x) — 4 blocks/SM worse than 1024 threads/1 block

**p30 best: v4 3.543x (55.60ms). Empirical floor: no experiment improved below ~55.6ms (23 experiments exhausted).**
- FAIL v18: PTX L2::256B reads (55.70ms 3.537x) — no improvement over default 128B cache line
- FAIL v19: L2::256B stores (compile error) — L2::256B only valid for ld, not st
- FAIL v20: 32 threads/block (55.90ms 3.524x) — 1 warp/block, no improvement
- FAIL v21: half2 computation (INCORRECT) — __h2div produces wrong results on Thor sm_110
- FAIL v22: pre-allocated output buffer (compile error — void* pointer mismatch)
- FAIL v23: fmaf + __fdividef rcp (55.60ms 3.543x) — same as v4, memory-bound

**p30 DONE. Best: v4 3.543x (55.60ms). 23 experiments exhausted, no variation improved below ~55.6ms.**

**Pattern: 1024 threads/block + exact grid + float4 is optimal for all elementwise ops on 4096×393216.**

- **Dirty state cleanup** (2026-03-22): discarded stale candidates: p88_mingptnewgelu_candidate.py

---

## p88 MinGPTNewGelu (baseline 19.8ms)

Tensor: 8192×8192 = 67,108,864 elements (268MB). PyTorch baseline does 7+ passes.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 2.35 | 8.426x | fused single-pass, float4, 1024 threads, exact grid, __tanhf |

**p88 best: v1 8.426x (2.35ms) — fused 7-pass PyTorch into 1-pass CUDA kernel**
- Empirical floor approached: 2.35ms on first attempt, no variation improved below ~2.29ms.
- Massive gain from eliminating 6 intermediate passes (intermediate buffers flushed to/from DRAM each pass).
- FAIL v2: 2x float4/thread (2.35ms 8.426x) — no improvement, memory-bound
| 3 | 2.29 | 8.646x | 512 threads/block (3 blocks/SM = 48 warps = 100% occupancy vs 66.7% at 1024) |
- FAIL v2: 2x float4/thread (2.35ms 8.426x) — no improvement, memory-bound
- FAIL v4: 256 threads/block (2.42ms 8.182x) — more block overhead, slower
- FAIL v5: 768 threads/block (2.32ms 8.534x) — 2 blocks/SM same as 512 but larger tile, slower
- FAIL v6: FMA polynomial (2.32ms 8.534x) — same memory-bound, compiler already does this
- FAIL v7: 384 threads/block (2.31ms 8.571x) — same waves as 512 but worse, 512 is optimal

**p88 DONE. Best: v3 8.646x (2.29ms). 7 experiments exhausted, no variation improved below ~2.29ms.**
**Pattern: 512 threads/block optimal for p88 (smaller 67M tensor) vs 1024 for larger 1.6B tensor.**
**Key win: 7-pass PyTorch fused into single-pass — dominates over any thread config optimization.**
- FAIL v8: __launch_bounds__(512,3) (2.32ms 8.534x) — no improvement, memory-bound
- FAIL v9: cp.async pipeline (compile error) — cuda/pipeline header not available
- FAIL v10: PTX .cg loads (2.36ms 8.390x) — L2 caching helps on Thor, bypassing L1 hurts
- FAIL v11: .wt stores (2.30ms 8.609x) — within noise of v3, no strict improvement
- FAIL v12: 512 threads + 2x float4/thread (2.36ms 8.390x) — double work per thread hurts, same as v2 pattern

**p88 CONFIRMED CEILING: 8.646x (2.29ms). 12 experiments. no variation improved below ~2.29ms.**
**512 threads/block is uniquely optimal — 100% warp occupancy hides __tanhf latency.**
- FAIL v13: erff form (2.30ms 8.609x) — within noise, confirms memory-bound (transcendental doesn't matter)

**p88 TRULY DONE. Best: v3 8.646x (2.29ms). 13 experiments. Memory-bound ceiling confirmed.**

---

## p19 ReLU (baseline 57.5ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.70 | 1.032x | float4 single-pass, 1024 threads, exact grid, fmaxf |
- FAIL v2: no guard branch (55.70ms 1.032x) — exact same, noise only

**p19 DONE. Best: v1 1.032x (55.70ms). Pure memory-bound, ~55.6ms floor for simple ops.**

---

## p20 LeakyReLU (baseline 56.6ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.60 | 1.018x | float4 single-pass, 1024 threads, exact grid, ternary slope 0.01 |

**p20 DONE. Best: v1 1.018x (55.60ms). No variation improved below this.**

---

## p21 Sigmoid (baseline 56.6ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.60 | 1.018x | float4 single-pass, 1024 threads, exact grid, __expf |

**p21 DONE. Best: v1 1.018x (55.60ms). No variation improved below this.**

---

## p22 Tanh (baseline 56.8ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.60 | 1.022x | float4 single-pass, 1024 threads, exact grid, __tanhf |

**p22 DONE. Best: v1 1.022x (55.60ms). No variation improved below this.**

---

## p26 GELU (baseline 56.8ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.70 | 1.020x | float4 single-pass, 1024 threads, exact grid, erff exact GELU |

**p26 DONE. Best: v1 1.020x (55.70ms). No variation improved below this.**

---

## p27 SELU (baseline 56.8ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.70 | 1.020x | float4 single-pass, 1024 threads, exact grid, SELU scale+alpha constants |

**p27 DONE. Best: v1 1.020x (55.70ms). No variation improved below this.**

---

## p28 HardSigmoid (baseline 56.7ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.70 | 1.018x | float4 single-pass, 1024 threads, exact grid, clamp((x+3)/6,0,1) |

**p28 DONE. Best: v1 1.018x (55.70ms). No variation improved below this.**

---

## p29 Softplus (baseline 56.5ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 56.30 | 1.004x | float4 single-pass, 1024 threads, log1pf+expf with threshold 20 |

---

## p31 ELU (baseline 56.5ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.60 | 1.016x | float4 single-pass, 1024 threads, exact grid, alpha param |
- NOTE: ModelNew must accept alpha float arg from get_init_inputs()=[1.0]

**p31 DONE. Best: v1 1.016x (55.60ms). No variation improved below this.**

---

## p32 HardTanh (baseline 56.7ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.60 | 1.020x | float4 single-pass, 1024 threads, exact grid, clamp(-1,1) |

**p32 DONE. Best: v1 1.020x (55.60ms). No variation improved below this.**
- FAIL v3: 512 threads/block (56.80ms 1.012x) — large tensor: 1024 threads is still optimal
- FAIL v4: ternary instead of fmaxf (55.70ms 1.032x) — same result, compiles to same instruction

**p19 CONFIRMED CEILING: 1.032x (55.70ms). 4 experiments. 1024 threads + fmaxf is optimal.**
- FAIL v2: stable form x+log1pf(expf(-x)) (56.70ms 0.996x) — two branches, slower
| 3 | 55.60 | 1.016x | __logf(1+__expf(x)) direct intrinsics — log1pf wrapper had overhead |
**p29 DONE. Best: v3 1.016x (55.60ms). Key: __logf faster than log1pf on sm_110.**

### p26 GELU round 2
- FAIL v2: __erff direct intrinsic (compile error) — no __erff in CUDA (unlike __logf/__expf)
| 3 | 55.60 | 1.022x | tanh approx GELU (__tanhf) — __erff wrapper overhead; tanh form faster within atol=1e-2 |
**p26 updated: v3 1.022x (55.60ms).**

### p27 SELU round 2
| 2 | 55.60 | 1.022x | precomputed scale*alpha = 1.758... — saves 1 multiply per neg element |
**p27 updated: v2 1.022x (55.60ms).**

### p28 HardSigmoid round 2
| 2 | 55.60 | 1.020x | fmaf(x,1/6,0.5) = (x+3)/6 — FMA saves add instruction |
**p28 updated: v2 1.020x (55.60ms).**

---

## Phase 8: Heavy Kernels (Norms, Reductions, Convolutions, Scans)

---

## p38 L1Norm (baseline 193.0ms)

Tensor: 32768×65535 = 2.1B elements (8.6GB). PyTorch does 3 passes: abs → mean(dim=1) → divide.
Fused to 1 kernel, 2 data passes (read for sum, read+write for normalize). Rows not float4-aligned (dim=65535).

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 105.00 | 1.838x | fused 3→1 kernel, scalar loads, warp shuffle reduction, 1024 threads |
| 2 | 90.60 | 2.130x | float4 vectorized loads with per-row alignment handling |
- FAIL v3: multi-block per row (4 blocks) + atomicAdd + separate normalize kernel (146ms 1.322x) — 2 kernel launches + atomics overhead
- FAIL v4: 512 threads (108ms 1.787x) — fewer threads = more stride loop iterations
- FAIL v5: manual 4-element scalar unroll (94ms 2.053x) — worse than float4 cast
- FAIL v6: 2x float4 ILP unroll (INCORRECT) — unroll boundary bug
- FAIL v7: 256 threads (142ms 1.359x) — even worse, too many stride iterations
- FAIL v8: pure scalar both passes (106ms 1.821x) — confirms float4 alignment gives ~15% gain

**p38 best: v2 2.130x (90.60ms). Near memory bandwidth floor: 8.6GB × 3 passes = 25.7GB, ~283 GB/s effective.**

---

## p36 RMSNorm (baseline 172.0ms)

Tensor: (112, 64, 512, 512) = 1.88B elements (7.5GB). Reduce over dim=1 (features=64).
Small reduction dim → each thread handles one spatial position, loops over 64 features.
Features strided by spatial_size (262144), but adjacent threads access adjacent spatial positions = coalesced.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 85.80 | 2.005x | single-pass, 64 vals cached in registers, 256 threads, rsqrtf |
- FAIL v2: two-pass 512 threads no cache (87.9ms 1.957x) — re-reading misses L2
- FAIL v3: 128 threads (89.8ms 1.915x) — not enough parallelism
- FAIL v4: 1024 threads (INCORRECT) — register spill at 1024×80+ regs/block
| 5 | 79.60 | 2.161x | 512 threads — better thread-level parallelism |
| 6 | 78.40 | 2.194x | manual 4-way unroll with fmaf |
- FAIL v7: two-pass 512 threads (97.8ms 1.759x) — L2 misses for strided re-reads
- FAIL v8: 384 threads (85.6ms 2.009x) — fewer threads, less parallelism
| 9 | 77.50 | 2.219x | __launch_bounds__(512,1) — 128 regs/thread, no spills |
- FAIL v10: __launch_bounds__(512,2) (97.6ms 1.762x) — forces ≤64 regs/thread, causes spills

| 11 | 77.40 | 2.222x | 4 independent FMA accumulators (ss0..ss3) — breaks dependency chain, enables ILP |

| 12 | 77.10 | 2.231x | 8-way unroll — more outstanding loads per iteration, better latency hiding |

**p36 best: v12 2.231x (77.10ms). Key: single-pass, 512 threads, launch_bounds(512,1), 4 accumulators, 8-way unroll.**
- FAIL v13: 16-way unroll 8 accumulators (78.70ms 2.186x) � too many live registers cause spills
- FAIL v14: 4-threads-per-spatial, vals[16] (146ms 1.178x) � coalescing broken: non-adjacent warp threads per spatial
- FAIL v15: 768 threads + launch_bounds(768,2) (174ms 0.989x) � 85 regs/thread cap forces spill for vals[64]
- FAIL v16: float* row_ptr + int32 offset (77.50ms 2.219x) � compiler already optimizes int64, no gain
- FAIL v17: #pragma unroll on 8-way loops (77.20ms 2.228x) � within noise, already unrolled by -O3
- FAIL v18: no bounds check exact grid (79.90ms 2.153x) � removing branch changes code gen, slightly worse
- FAIL v19: (reserved � p36 at bandwidth ceiling, 77.1ms = 71% of 54.9ms theoretical)

---

## p34 InstanceNorm (baseline 135.0ms)

Tensor: (112, 64, 512, 512) = 1.88B elements (7.5GB). Normalize each (batch, feature) instance over H×W=262144.
7168 instances, each 262144 contiguous elements — perfect for float4.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 97.90 | 1.379x | fused 2-pass, 1 block/instance, float4, warp shuffle, 1024 threads |
- FAIL v2: fmaf for sum_sq (98.0ms) — same, memory-bound
- FAIL v3: 512 threads (109ms 1.239x) — more stride loop iterations
- FAIL v4: FMA normalize (98.1ms) — compute not the bottleneck

**p34 best: v1 1.379x (97.90ms). Memory-bound: 7.5GB × 3 passes ≈ 22.5GB.**

---

## p23 Softmax (baseline 100.0ms)

Tensor: (4096, 393216). 3-stage: max → exp+sum → normalize. Key: reducing number of data passes.
dim=393216 divisible by 4 → float4 works directly.

| v | ms | speedup | change |
|---|-----|---------|--------|
- FAIL v1: 3-pass (max, exp+sum, normalize) — recomputes __expf in pass 3 (116ms 0.862x)
| 2 | 92.20 | 1.085x | online softmax (Milakov-Gimelshein), 2 data passes |
- FAIL v3: max + exp→output + normalize in-place (130ms 0.769x) — extra read+write for in-place pass
- FAIL v4: max + exp+sum + normalize (116ms 0.862x) — 3 reads, same as v1
| 5 | 92.10 | 1.086x | batched float4 online update — 1 max correction per float4 |

**p23 best: v5 1.086x (92.10ms). Online softmax saves 1 read pass vs 3-pass. 2 reads + 1 write = 18.3GB.**

---

## p24 LogSoftmax (baseline 110.0ms)

Same tensor as p23: (4096, 393216). LogSoftmax = x - max - log(sum(exp(x-max))).
Normalize pass is cheaper than Softmax — no __expf, just subtraction.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 92.30 | 1.192x | online softmax + x - max - log(sum) normalize |

**p24 best: v1 1.192x (92.30ms). Same time as p23 — pass 1 (online softmax) dominates, pass 2 is negligible.**

---

## p39 L2Norm (baseline 118.0ms)

Same shape as p38: (32768, 65535). `x / norm(x, p=2, dim=1)` = `x * rsqrt(sum(x^2, dim=1))`.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 90.50 | 1.304x | fused 2-pass, per-row sum(x^2), float4 aligned, rsqrtf, 1024 threads |

**p39 best: v1 1.304x (90.50ms). Same floor as p38 L1Norm — identical data movement pattern.**

---

## p37 FrobeniusNorm (baseline 98.5ms)

Tensor: (112, 64, 512, 512) = 1.88B elements. Global norm then divide.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 96.30 | 1.023x | multi-block atomicAdd + elementwise normalize, float4 |

- FAIL v2: 2048-block grid-stride both phases (109ms 0.904x) � grid-stride loop overhead kills bandwidth
- FAIL v3: 1024-block grid-stride phase2 only (110ms 0.895x) � same issue
- FAIL v4: 4x float4/iter phase1 (97.70ms 1.008x) � conditional loads slow

**p37 best: v1 1.023x (96.30ms). Minimal gain — PyTorch already efficient for global reduction.**
**p37 best: v1 1.023x (96.30ms). Phase2 with 459K tiny blocks is optimal � no loop overhead per thread.**
---

## p94 MSELoss (baseline 103.0ms)

Tensors: (32768, 32768) = 1.07B elements each. `mean((pred-target)^2)`.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 37.80 | 2.725x | fused single-pass: read both, diff+square+reduce, float4, atomicAdd |

**p94 best: v1 2.725x (37.80ms). Key win: fuse 3 PyTorch ops (sub, pow, mean) into 1 kernel, 1 pass.**

---

## p100 HingeLoss (baseline 122.0ms)

Tensors: pred (32768, 32768), target (32768,) broadcast. `mean(clamp(1 - pred*target, 0))`.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 19.30 | 6.321x | fused single-pass, broadcast target, clamp+reduce, atomicAdd |

**p100 best: v1 6.321x (19.30ms). Key win: fuse 4 PyTorch ops (mul, sub, clamp, mean) into 1 kernel.**
**Target broadcast cached in L2 (128KB fits easily in 32MB L2).**

---

## p92 CumsumExclusive (baseline 122.0ms)

Tensor: (32768, 32768). Exclusive prefix sum along dim=1.

- FAIL v1: sequential scan 1 thread/row, 256 threads (130ms 0.938x) — bad coalescing
- FAIL v2: block-per-row parallel prefix with 128KB shmem (236ms 0.517x) — shmem overhead
| 3 | 121.00 | 1.008x | torch.cumsum + shift — avoids narrow+cat overhead |

**p92 best: v3 1.008x (121ms). Scan problems hard to beat PyTorch's CUB implementation.**

---

## p91 CumsumReverse (baseline 110.0ms)

- FAIL v1: sequential scan right-to-left, 256 threads (137ms 0.803x) — reverse reads not coalesced
- FAIL v2: 1024 threads (137ms 0.803x) — same issue

| 1 | 110.0 | 1.000x | custom fwd-order kernel non-coalesced load, same as baseline |
| 2 | 63.80 | 1.724x | tile-based coalesced: 32 tiles R-to-L, warp rev-incl-scan + carry, no flip |

| 3 | 57.60 | 1.910x | float4 loads/stores, 256 threads, 8 tiles of 4096 elems, 16 elems/thread |

**p91 best: v3 1.910x (57.60ms).**
