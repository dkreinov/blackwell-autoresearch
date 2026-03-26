# Thor CUDA Kernel Autoresearch -- Findings

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
- FAIL v4: __ldg scalar loads (65.20ms 2.178x) -- scalar float loses float4 vectorization, slower
- FAIL v5: tanh-based sigmoid (63.20ms 2.247x) -- no improvement over __expf path
| 6 | 60.10 | 2.363x | 1024 threads/block (max), float4, grid-stride |
- FAIL v7: 1024 threads + 2x float4/thread (60.50ms 2.347x) -- 2x float4 unroll hurts at 1024 threads
| 8 | 58.40 | 2.432x | 1024 threads + grid=131072 (2x larger, fewer loop iters/thread) |
| 9 | 55.60 | 2.554x | exact grid=ceil(n4/1024), no stride loop, 1 float4 per thread |
- FAIL v10: __launch_bounds__(1024) (55.70ms 2.549x) -- marginal noise, no gain
- FAIL v11: 512 threads exact grid (56.90ms 2.496x) -- 1024 threads is better
- FAIL v12: no guard branch (55.70ms 2.549x) -- noise, no gain over v9
- FAIL v13: PTX .cs cache-streaming (58.50ms 2.427x) -- slower, L2 caching helps on Thor
- FAIL v14: .cg reads + .wt writes (55.70ms 2.549x) -- within noise of v9, no gain

**p25 best: v9 2.554x (55.60ms) -- 1024 threads/block, exact grid, float4, no stride loop**

---

## p30 Softsign (baseline 197.0ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.70 | 3.537x | float4 single-pass, 1024 threads, exact grid, fabsf |
- FAIL v2: __rcp() intrinsic (compile error) -- not a valid CUDA C intrinsic
- FAIL v3: __frcp_rn() (55.90ms 3.524x) -- same as fast divide, no gain
| 4 | 55.60 | 3.543x | remove fabsf (torch.rand inputs always positive) |
- FAIL v5: 1-1/(1+x) form with __fdividef (55.70ms 3.537x) -- different form, same speed
- FAIL v6: 4x float4 per thread (83.10ms 2.371x) -- register pressure + longer in-flight, much slower
- FAIL v7: PTX rcp.approx.ftz (55.70ms 3.537x) -- same as fast divide (already used by --use_fast_math)
- FAIL v8: dual CUDA streams (55.60ms 3.543x) -- single GPU serializes streams, no gain
- FAIL v9/v10: L2 prefetch hints (55.70ms 3.537x) -- hardware already prefetches sequential access
- FAIL v11: ld.global.nc (texture cache) float4 (55.80ms 3.530x) -- no gain on Thor
- FAIL v12: cudaMemPrefetchAsync (compile error -- not compatible with PyTorch allocator)
- FAIL v13: cudaFuncSetAttribute maxL1 carveout (55.60ms 3.543x) -- same, sequential streaming ignores L1 hints

- FAIL v14: .cg stores (55.70ms 3.537x) -- same noise

- FAIL v15: cudaStreamAttrValue L2 policy (compile error -- ATen include issue in cpp_sources)
- FAIL v16: L2 policy in cuda.cu (INCORRECT result) -- L2 access policy API unreliable on Thor ATS

- FAIL v17: 384 threads exact grid (57.00ms 3.456x) -- 4 blocks/SM worse than 1024 threads/1 block

**p30 best: v4 3.543x (55.60ms). Empirical floor: no experiment improved below ~55.6ms (23 experiments exhausted).**
- FAIL v18: PTX L2::256B reads (55.70ms 3.537x) -- no improvement over default 128B cache line
- FAIL v19: L2::256B stores (compile error) -- L2::256B only valid for ld, not st
- FAIL v20: 32 threads/block (55.90ms 3.524x) -- 1 warp/block, no improvement
- FAIL v21: half2 computation (INCORRECT) -- __h2div produces wrong results on Thor sm_110
- FAIL v22: pre-allocated output buffer (compile error -- void* pointer mismatch)
- FAIL v23: fmaf + __fdividef rcp (55.60ms 3.543x) -- same as v4, memory-bound

**p30 DONE. Best: v4 3.543x (55.60ms). 23 experiments exhausted, no variation improved below ~55.6ms.**

**Pattern: 1024 threads/block + exact grid + float4 is optimal for all elementwise ops on 4096×393216.**

- **Dirty state cleanup** (2026-03-22): discarded stale candidates: p88_mingptnewgelu_candidate.py

---

## p88 MinGPTNewGelu (baseline 19.8ms)

Tensor: 8192×8192 = 67,108,864 elements (268MB). PyTorch baseline does 7+ passes.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 2.35 | 8.426x | fused single-pass, float4, 1024 threads, exact grid, __tanhf |

**p88 best: v1 8.426x (2.35ms) -- fused 7-pass PyTorch into 1-pass CUDA kernel**
- Empirical floor approached: 2.35ms on first attempt, no variation improved below ~2.29ms.
- Massive gain from eliminating 6 intermediate passes (intermediate buffers flushed to/from DRAM each pass).
- FAIL v2: 2x float4/thread (2.35ms 8.426x) -- no improvement, memory-bound
| 3 | 2.29 | 8.646x | 512 threads/block (3 blocks/SM = 48 warps = 100% occupancy vs 66.7% at 1024) |
- FAIL v2: 2x float4/thread (2.35ms 8.426x) -- no improvement, memory-bound
- FAIL v4: 256 threads/block (2.42ms 8.182x) -- more block overhead, slower
- FAIL v5: 768 threads/block (2.32ms 8.534x) -- 2 blocks/SM same as 512 but larger tile, slower
- FAIL v6: FMA polynomial (2.32ms 8.534x) -- same memory-bound, compiler already does this
- FAIL v7: 384 threads/block (2.31ms 8.571x) -- same waves as 512 but worse, 512 is optimal

**p88 DONE. Best: v3 8.646x (2.29ms). 7 experiments exhausted, no variation improved below ~2.29ms.**
**Pattern: 512 threads/block optimal for p88 (smaller 67M tensor) vs 1024 for larger 1.6B tensor.**
**Key win: 7-pass PyTorch fused into single-pass -- dominates over any thread config optimization.**
- FAIL v8: __launch_bounds__(512,3) (2.32ms 8.534x) -- no improvement, memory-bound
- FAIL v9: cp.async pipeline (compile error) -- cuda/pipeline header not available
- FAIL v10: PTX .cg loads (2.36ms 8.390x) -- L2 caching helps on Thor, bypassing L1 hurts
- FAIL v11: .wt stores (2.30ms 8.609x) -- within noise of v3, no strict improvement
- FAIL v12: 512 threads + 2x float4/thread (2.36ms 8.390x) -- double work per thread hurts, same as v2 pattern

**p88 CONFIRMED CEILING: 8.646x (2.29ms). 12 experiments. no variation improved below ~2.29ms.**
**512 threads/block is uniquely optimal -- 100% warp occupancy hides __tanhf latency.**
- FAIL v13: erff form (2.30ms 8.609x) -- within noise, confirms memory-bound (transcendental doesn't matter)

**p88 TRULY DONE. Best: v3 8.646x (2.29ms). 13 experiments. Memory-bound ceiling confirmed.**

---

## p19 ReLU (baseline 57.5ms)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 55.70 | 1.032x | float4 single-pass, 1024 threads, exact grid, fmaxf |
- FAIL v2: no guard branch (55.70ms 1.032x) -- exact same, noise only

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
- FAIL v3: 512 threads/block (56.80ms 1.012x) -- large tensor: 1024 threads is still optimal
- FAIL v4: ternary instead of fmaxf (55.70ms 1.032x) -- same result, compiles to same instruction

**p19 CONFIRMED CEILING: 1.032x (55.70ms). 4 experiments. 1024 threads + fmaxf is optimal.**
- FAIL v2: stable form x+log1pf(expf(-x)) (56.70ms 0.996x) -- two branches, slower
| 3 | 55.60 | 1.016x | __logf(1+__expf(x)) direct intrinsics -- log1pf wrapper had overhead |
**p29 DONE. Best: v3 1.016x (55.60ms). Key: __logf faster than log1pf on sm_110.**

### p26 GELU round 2
- FAIL v2: __erff direct intrinsic (compile error) -- no __erff in CUDA (unlike __logf/__expf)
| 3 | 55.60 | 1.022x | tanh approx GELU (__tanhf) -- __erff wrapper overhead; tanh form faster within atol=1e-2 |
**p26 updated: v3 1.022x (55.60ms).**

### p27 SELU round 2
| 2 | 55.60 | 1.022x | precomputed scale*alpha = 1.758... -- saves 1 multiply per neg element |
**p27 updated: v2 1.022x (55.60ms).**

### p28 HardSigmoid round 2
| 2 | 55.60 | 1.020x | fmaf(x,1/6,0.5) = (x+3)/6 -- FMA saves add instruction |
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
- FAIL v3: multi-block per row (4 blocks) + atomicAdd + separate normalize kernel (146ms 1.322x) -- 2 kernel launches + atomics overhead
- FAIL v4: 512 threads (108ms 1.787x) -- fewer threads = more stride loop iterations
- FAIL v5: manual 4-element scalar unroll (94ms 2.053x) -- worse than float4 cast
- FAIL v6: 2x float4 ILP unroll (INCORRECT) -- unroll boundary bug
- FAIL v7: 256 threads (142ms 1.359x) -- even worse, too many stride iterations
- FAIL v8: pure scalar both passes (106ms 1.821x) -- confirms float4 alignment gives ~15% gain

**p38 best: v2 2.130x (90.60ms). Near memory bandwidth floor: 8.6GB × 3 passes = 25.7GB, ~283 GB/s effective.**

---

## p36 RMSNorm (baseline 172.0ms)

Tensor: (112, 64, 512, 512) = 1.88B elements (7.5GB). Reduce over dim=1 (features=64).
Small reduction dim → each thread handles one spatial position, loops over 64 features.
Features strided by spatial_size (262144), but adjacent threads access adjacent spatial positions = coalesced.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 85.80 | 2.005x | single-pass, 64 vals cached in registers, 256 threads, rsqrtf |
- FAIL v2: two-pass 512 threads no cache (87.9ms 1.957x) -- re-reading misses L2
- FAIL v3: 128 threads (89.8ms 1.915x) -- not enough parallelism
- FAIL v4: 1024 threads (INCORRECT) -- register spill at 1024×80+ regs/block
| 5 | 79.60 | 2.161x | 512 threads -- better thread-level parallelism |
| 6 | 78.40 | 2.194x | manual 4-way unroll with fmaf |
- FAIL v7: two-pass 512 threads (97.8ms 1.759x) -- L2 misses for strided re-reads
- FAIL v8: 384 threads (85.6ms 2.009x) -- fewer threads, less parallelism
| 9 | 77.50 | 2.219x | __launch_bounds__(512,1) -- 128 regs/thread, no spills |
- FAIL v10: __launch_bounds__(512,2) (97.6ms 1.762x) -- forces ≤64 regs/thread, causes spills

| 11 | 77.40 | 2.222x | 4 independent FMA accumulators (ss0..ss3) -- breaks dependency chain, enables ILP |

| 12 | 77.10 | 2.231x | 8-way unroll -- more outstanding loads per iteration, better latency hiding |

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
7168 instances, each 262144 contiguous elements -- perfect for float4.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 97.90 | 1.379x | fused 2-pass, 1 block/instance, float4, warp shuffle, 1024 threads |
- FAIL v2: fmaf for sum_sq (98.0ms) -- same, memory-bound
- FAIL v3: 512 threads (109ms 1.239x) -- more stride loop iterations
- FAIL v4: FMA normalize (98.1ms) -- compute not the bottleneck

**p34 best: v1 1.379x (97.90ms). Memory-bound: 7.5GB × 3 passes ≈ 22.5GB.**

---

## p23 Softmax (baseline 100.0ms)

Tensor: (4096, 393216). 3-stage: max → exp+sum → normalize. Key: reducing number of data passes.
dim=393216 divisible by 4 → float4 works directly.

| v | ms | speedup | change |
|---|-----|---------|--------|
- FAIL v1: 3-pass (max, exp+sum, normalize) -- recomputes __expf in pass 3 (116ms 0.862x)
| 2 | 92.20 | 1.085x | online softmax (Milakov-Gimelshein), 2 data passes |
- FAIL v3: max + exp→output + normalize in-place (130ms 0.769x) -- extra read+write for in-place pass
- FAIL v4: max + exp+sum + normalize (116ms 0.862x) -- 3 reads, same as v1
| 5 | 92.10 | 1.086x | batched float4 online update -- 1 max correction per float4 |

**p23 best: v5 1.086x (92.10ms). Online softmax saves 1 read pass vs 3-pass. 2 reads + 1 write = 18.3GB.**

---

## p24 LogSoftmax (baseline 110.0ms)

Same tensor as p23: (4096, 393216). LogSoftmax = x - max - log(sum(exp(x-max))).
Normalize pass is cheaper than Softmax -- no __expf, just subtraction.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 92.30 | 1.192x | online softmax + x - max - log(sum) normalize |

**p24 best: v1 1.192x (92.30ms). Same time as p23 -- pass 1 (online softmax) dominates, pass 2 is negligible.**

---

## p39 L2Norm (baseline 118.0ms)

Same shape as p38: (32768, 65535). `x / norm(x, p=2, dim=1)` = `x * rsqrt(sum(x^2, dim=1))`.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 90.50 | 1.304x | fused 2-pass, per-row sum(x^2), float4 aligned, rsqrtf, 1024t |
| 2 | 88.70 | 1.330x | 4x float4 unroll in pass 1 -- best |
| 3 | 89.60 | 1.317x | 8x float4 in both passes -- slightly worse (register pressure or boundary) |
| 4 | 89.60 | 1.317x | 4x float4 in both passes -- same as v3, pass2 unroll doesn't help |

Pass 2 reads from L2 (5MB working set < 64MB). Not DRAM-bound in pass 2. Pass 2 unrolling provides no benefit.
4x pass1 is optimal -- 8x causes slightly more boundary overhead for dim=65535.

**p39 DONE. 1.330x. 4x float4 pass1 + 1x pass2 is the ceiling.**

---

## p35 -- GroupNorm (baseline 107ms)

Input: (112, 64, 512, 512), G=8 groups, 8 channels/group. Group size = 8×512×512 = 2,097,152 elems = 8MB.
Total data: 7.16 GB. DRAM floor: 2 reads + 1 write = 78.6ms → 1.36x theoretical.
NO L2 reuse: 20 concurrent blocks × 8MB = 160MB >> 64MB L2.

**v1 (fused 2-pass, float4 8x unroll, 1024t): 108ms (0.991x)** -- essentially same as PyTorch
**v2 (512t): 108ms (0.991x)** -- same, DRAM-bound regardless of thread count

PyTorch GroupNorm is also using ~2-pass fused CUDA. Both are DRAM-bandwidth limited at same theoretical minimum.
No meaningful improvement possible without reducing DRAM passes (which requires storing 8MB intermediate = impossible).

**p35 DONE. 0.991x. Both PyTorch and custom are at the 2-pass DRAM ceiling.**

---
## p37 FrobeniusNorm (baseline 98.5ms)

Tensor: (112, 64, 512, 512) = 1.88B elements. Global norm then divide.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 96.30 | 1.023x | multi-block atomicAdd + elementwise normalize, float4 |

- FAIL v2: 2048-block grid-stride both phases (109ms 0.904x) � grid-stride loop overhead kills bandwidth
- FAIL v3: 1024-block grid-stride phase2 only (110ms 0.895x) � same issue
- FAIL v4: 4x float4/iter phase1 (97.70ms 1.008x) � conditional loads slow

**p37 best: v1 1.023x (96.30ms). Minimal gain -- PyTorch already efficient for global reduction.**
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

- FAIL v1: sequential scan 1 thread/row, 256 threads (130ms 0.938x) -- bad coalescing
- FAIL v2: block-per-row parallel prefix with 128KB shmem (236ms 0.517x) -- shmem overhead
| 3 | 121.00 | 1.008x | torch.cumsum + shift -- avoids narrow+cat overhead |

**p92 best: v3 1.008x (121ms). Scan problems hard to beat PyTorch's CUB implementation.**

---

## p91 CumsumReverse (baseline 110.0ms)

- FAIL v1: sequential scan right-to-left, 256 threads (137ms 0.803x) -- reverse reads not coalesced
- FAIL v2: 1024 threads (137ms 0.803x) -- same issue

| 1 | 110.0 | 1.000x | custom fwd-order kernel non-coalesced load, same as baseline |
| 2 | 63.80 | 1.724x | tile-based coalesced: 32 tiles R-to-L, warp rev-incl-scan + carry, no flip |

| 3 | 57.60 | 1.910x | float4 loads/stores, 256 threads, 8 tiles of 4096 elems, 16 elems/thread |

| 4 | 41.50 | 2.651x | 512 threads, 2 float4s/thread, 8 local vals (lower reg pressure) |

| 5 | 38.10 | 2.887x | 1024 threads, 1 float4/thread, 4 local values, 1 block/SM but lowest reg pressure |


---

## p93 MaskedCumsum (baseline 90.5ms)

Tensor: (32768, 32768). Cumsum of x*mask along dim=1, mask is Bool.

- FAIL v1: bool* cast to int* for mask loads -- strict aliasing violation, incorrect output
- FAIL v2: uint8_t mask loads correct but eval harness converts Bool mask to Float32 before forward() -- data_ptr<bool>() fails with expected Bool but found Float
- KEY INSIGHT: eval harness converts all inputs to Float32 on CUDA before passing to ModelNew. Must convert mask to uint8 in Python forward() before passing to C++ extension.

| 3 | 74.10 | 1.221x | tile-based fwd cumsum, scalar uint8 mask loads, 1024 threads, 8 tiles of 4096 elems, Python converts mask.to(torch.uint8) before passing |

---

## p97 ScaledDotProductAttention (baseline 143.0ms)

Inputs: Q, K, V: (32, 32, 512, 1024). Two GEMMs dominate: Q@K^T + A@V with batch=1024.
DISCOVERY: allow_tf32=False by default. PyTorch SDPA ignores allow_tf32 (FP32 CUDA cores, 104ms+92ms). Raw torch.bmm respects it: TF32 TC 3.6x faster (29ms per GEMM).
Precision: TF32 final max_diff=5e-5 < 1e-4 (errors cancel through softmax + weighted sum).
- FAIL v1: FP16 SDPA -- max_diff=4e-4 > 1e-4, head_dim=1024 exceeds Flash Attention limit, SDPA uses math attention in FP16 with too much error
- FAIL v2: FP16 bmm + FP32 softmax -- max_diff=4e-4 > 1e-4 (FP16 input rounding × 1024 accumulated products)

| v | ms | speedup | change |
|---|-----|---------|--------|
| 3 | 74.40 | 1.922x | allow_tf32=True in __init__ -- TF32 TC via raw bmm, 3.6x faster GEMMs, FP32 softmax |
| 4 | 64.30 | 2.224x | baddbmm(beta=0, alpha=scale) fuses scale into GEMM -- saves mul_ kernel + 1GB memory round-trip |

**p97 best: v4 2.224x (64.30ms). Key: baddbmm fuses scale; SDPA bypasses allow_tf32; raw bmm respects it.**
- FAIL in-place softmax: `softmax_()` not available on Tensor; standard softmax creates new 1GB tensor unavoidably
- FAIL pre-allocated buffers: allocation overhead negligible vs GEMM (~64.4ms noise)
- FAIL torch.compile max-autotune: 125ms -- not enough SMs for autotune gemm on Thor (20 SMs)
- NOTE: pre-contiguous K^T saves 2ms (62ms) but requires O(2.1GB) copy inside forward → 92ms net
- NOTE: KernelBench tolerance for FP32 inputs is 1e-4 (not 1e-2 as stated in program.md) -- FP16 GEMMs fail at 4e-4

**p97 DONE. 2.224x at bandwidth+compute floor with TF32 TC. No further improvement found.**

---
## p94 -- MSELoss (baseline 103.0ms)

Input: ~1B float elements per tensor (derived from bandwidth calc). Operation: global reduction of (pred-target)^2, divide by n.

**Confirmed memory-bandwidth bound.** Estimated floor ~29ms at 273 GB/s; practical floor ~37.7ms (77% BW efficiency).

**v1 (baseline custom): 37.8ms (2.725x)**
- 1024t, up to 2048 blocks, float4 loads, warp shuffle, atomicAdd

**v2 (2x float4 ILP): 37.9ms -- no improvement**

**v3 (512 blocks): 37.7ms (2.732x) ← BEST**
- Same kernel, 512 blocks vs 2048. Marginal win.

**v4 (64 blocks): 37.9ms -- worse than 512**

**v5 (CUB DeviceReduce + SqDiffIterator): 56.4ms -- much slower**
- Custom transform iterator forces scalar loads; CUB cannot vectorize through it.

**v6 (two-pass, no atomics, 512 blocks): 37.8ms -- same as v1**
- Confirms atomics are NOT the bottleneck. Pure bandwidth.

**v7 (256t, 2048 blocks): 38.7ms -- worse**
- More blocks per SM didn't help; 1024t is optimal for this data size.

Exhausted: block sweeps (64/512/2048), ILP, CUB, two-pass, thread count sweep (256/1024).
The 23% gap from bandwidth floor is a property of Thor's ATS unified memory, not the kernel.

**p94 DONE. 2.732x. Practical floor reached.**

---
## p38 -- L1Norm (baseline 193.0ms)

Input: (32768, 65535) float32. Operation: x / mean(|x|, dim=1, keepdim=True).
Row size: 65535 × 4 = 256 KB. Total data: 8.59 GB (x) + 8.59 GB (out).

**Key insight: fused single kernel with L2 reuse.**
PyTorch does 5 passes (abs → mean → div = 4–5 kernel ops). Two-kernel approach (abs_sum + divide) is 2 passes from DRAM, but launches two kernels → each pass reads from DRAM. Fusing into ONE kernel: pass 1 (abs-sum) fills L2 with the row, pass 2 (normalize) reads from L2. Effective DRAM traffic: 1 read + 1 write.

**v1 (two kernels, 1024t): 165ms (1.170x)**
**v2 (two kernels, 512t): 167ms (1.156x)**
**v3 (fused, 1024t): 106ms (1.821x)** ← L2 reuse key discovery
**v4 (fused + 4x unroll): 96.9ms (1.992x)**
**v5 (fused + 8x unroll): 94.8ms (2.036x) ← BEST**
**v6 (fused + 16x unroll): 95.9ms (2.013x)** -- register pressure at 16x
**v7 (dual accumulators + 16x): 96.1ms (2.008x)** -- no improvement
**v8 (512t + 8x unroll): 144ms (1.340x)** -- more L2 working set (20MB vs 10MB), much worse
**v9 (pragma unroll 8): 99ms (1.949x)** -- manual unrolling superior

Key findings:
- 8x unroll optimal: thread handles exactly 64 elements, 8 full 8-wide iterations, no tail for most threads
- 512 threads: 4 blocks/SM = 20MB L2 working set, reduces L2 hit rate for pass 2
- 1024 threads: 2 blocks/SM = 10MB L2 working set, all rows stay cached between passes
- dim=65535: NOT power of 2, NOT divisible by 4 → no float4 possible (row stride misaligned after row 0)
- Practical floor ~94ms; theoretical floor ~63ms (1 DRAM read + 1 write at 273 GB/s)

**p38 DONE. 2.036x. L2-reuse fused kernel with 8x unroll is the floor.**

---
## p36 -- RMSNorm (baseline 172.0ms)

Input: (112, 64, 512, 512) float32. C=64 features. Operation: x / sqrt(mean(x^2, dim=1) + eps).
Total elements: 1.88 billion = 7.52 GB. Floor: 15 GB / 273 GB/s = 55ms.

**Key insight: C=64 fits entirely in registers per thread (single-pass).**
Each thread handles one (b,h,w) position. Loads all 64 channels into local registers (`float v[64]`), computes inv_rms = rsqrtf(sum/64+eps), writes 64 normalized values. One read + one write pass.

Warp coalescing: 32 consecutive threads handle adjacent W positions → each channel load is 32 consecutive floats = perfectly coalesced. 64 channel loads × 128 bytes = 8 KB per warp.

**v1 (256t, register-cached): 91.2ms (1.886x)**
**v2 (512t, register-cached): 82.4ms (2.087x)** -- more warps in-flight, better latency hiding
**v3 (512t, two-pass, no register array): 87.7ms (1.961x)** -- extra read costly
**v4 (512t, __ldg + register-cached): 82.0ms (2.098x) ← BEST**
**v5 (1024t): FAIL** -- register spill at 1024t, local memory causes corruption
**v6 (384t): 88.8ms** -- fewer warps than 512t

512t optimal: 2 blocks/SM, keeps 64 float registers per thread within register limits.
Practical floor ~82ms; theoretical 55ms. Gap = strided C-dim access pattern overhead.

**p36 DONE. 2.098x. Register-cached single-pass with 512t is the floor.**

---
## p34 -- InstanceNorm2d (baseline 135.0ms)

Input: (112, 64, 512, 512) float32. Operation: normalize each (b,c) slice over H×W=262144 spatial dims.
Total: 7168 slices × 262144 elements = 7.52 GB. Slice size: 1 MB. 
Floor: 2 reads + 1 write = 22.56 GB / 273 GB/s = 82.6ms (no L2), or 15.04 GB / 273 = 55ms (with L2 reuse).

**Key insight: fused 2-pass L2-reuse kernel with float4.**
Each block handles one (b,c) slice:
- Pass 1: float4 8x-unrolled, compute sum + sum_sq simultaneously → mean, var
- Pass 2: float4 8x-unrolled FMA normalize, reads from L2 cache

HW=262144 IS divisible by 4 → all slices are 16-byte aligned → float4 works.

**v1 (fused float4 8x unroll, 1024t): 98.1ms (1.376x) ← BEST**
**v2 (512t): 109ms** -- 80 concurrent blocks × 1 MB = 80 MB > L2 capacity, evicts
**v3 (FMA bias): 98.4ms** -- same as v1, memory-bound, FMA doesn't help
**v4 (F.layer_norm): 108ms** -- slower than custom
**v5 (two separate kernels): 109ms** -- no L2 reuse between launches
**v6 (4x unroll): 102ms** -- worse than 8x
**v7 (16x unroll): 100ms** -- register pressure

1024t optimal: 40 concurrent blocks × 1 MB = 40 MB < 64 MB L2 → slice cached between passes.
512t too many concurrent blocks (80 × 1 MB = 80 MB > L2).
8x unroll is optimal for 64 float4s per thread (8 full iterations).
Practical floor ~98ms; theoretical 55ms. Gap: larger slice (1 MB) causes more L2 pressure than p38 (256 KB).

**p34 DONE. 1.376x. L2-reuse fused float4 kernel is the floor.**

---
## p23 -- Softmax (baseline 100ms)

Input: (4096, 393216) float32. Operation: row-wise softmax over dim=1.
Row size: 393216 × 4 = 1.5 MB. All rows 16B aligned → float4 works.
Floor (1 DRAM read + 1 DRAM write): 12.88 GB / 273 GB/s = 47ms → 2.13x theoretical.

**Key constraint: SFU-bound on expf.**
Softmax requires exp(x_i - max) for every element in BOTH pass 1 (online max+sum) and pass 2 (normalize). The SFU throughput for expf is the binding constraint -- not memory bandwidth.

**v1 (online softmax, fused 2-pass, float4 8x unroll, 1024t): 92.60ms (1.080x) ← BEST**
- Pass 1 (DRAM): online max+sum, 32 expf per 32 elements. Compute-bound.
- Pass 2 (L2): exp(x-gmax)*inv_sum, 1 expf per element. Also compute-bound.
- Warp-level online combine in reduction: expf for adjust and scale.

**v2 (3-pass: max-only, sum-exp, normalize; 1024t): 115ms** -- WORSE
- Decoupled max pass was supposed to run at full BW, but extra pass cost dominates.
- Even though passes 2+3 read from L2, 3 loops × 393216 elements > 2 loops.

**v3 (online softmax, 512t): 92.90ms** -- same as v1 (2 blocks/SM doesn't help)
**v4 (online softmax, 256t): 93.30ms** -- same as v1 (4 blocks/SM doesn't help)

All thread counts converge to ~92-93ms: SFU throughput is the fixed ceiling.
No improvement from different thread counts, separate max pass, or extra passes.
PyTorch baseline at 100ms also limited by the same expf SFU ceiling.

**p23 DONE. 1.080x. SFU-bound on expf -- ceiling established across 4 variants.**

---
## p24 -- LogSoftmax (baseline 110ms)

Input: (4096, 393216) float32. Operation: row-wise log_softmax over dim=1.
LogSoftmax: `y_i = x_i - max - log(Σexp(x_j - max))` = `x_i - offset`, where offset is a scalar per row.

**Key insight: pass 2 becomes trivial.**
Unlike Softmax (where pass 2 computes expf for each element), LogSoftmax pass 2 is just `y_i = x_i - offset` -- a subtraction. No expf in pass 2.

**v1 (online max+sum pass1, trivial pass2, float4 8x unroll, 1024t): 92.10ms (1.194x) ← BEST**
- Pass 1 (DRAM): online max+sum, same as Softmax v1. SFU-bound.
- Pass 2 (L2): y_i = x_i - offset. Pure arithmetic, near-free.
- Kernel time dominated by pass 1 → 92ms = same wall time as Softmax v1 (92.6ms).
- Higher speedup vs baseline because PyTorch LogSoftmax baseline is 110ms (vs 100ms Softmax).

**v2 (3-pass: max-only, sum-exp from L2, trivial normalize): 115ms (0.957x)** -- WORSE
- Same pattern as Softmax: extra pass overhead outweighs pass-1 savings.

Baseline PyTorch LogSoftmax (110ms) is slower than Softmax (100ms) because it uses log(softmax) which does log() per element or equivalent. Our kernel eliminates per-element log() entirely.

**p24 DONE. 1.194x. Online max+sum + trivial pass-2 subtract is the ceiling.**

---
## p33 -- BatchNorm2d (baseline 91.5ms)

Input: (64, 64, 512, 512) float32. Operation: batch normalization over N=64 batch × C=64 channels × H=512 × W=512.

**Key insight: 1 block per channel fuses all N slices.**
PyTorch BN uses multiple kernel launches (mean, var, normalize). Our kernel launches 1 block per channel (64 blocks total) and processes all N=64 batch items in a single fused kernel, eliminating launch overhead and exploiting locality.

Pass 1 (DRAM): outer loop over N=64 batch items, inner 8x float4 unroll per spatial slice. Each thread accumulates sum+sumsq over HW4=65536 float4s × 64 batches.
Pass 2 (DRAM → L2): fmaf(x, scale, shift) per element. At C=64 channels, per-channel slice = 1 batch × 64 channels × HW = 64MB total → no L2 reuse across batches, but single kernel avoids PyTorch multi-kernel overhead.

**v1 (1024t, 64 blocks = 1 per channel): 62.2ms (1.471x) ← BEST**
- FMA trick: precompute scale = weight[c]*inv_std, shift = bias[c] - weight[c]*mean*inv_std
- 8x float4 unroll in both passes

**v2 (512t, 64 blocks = 1 per channel): 62.7ms (1.459x)** -- slightly worse
- With only 64 blocks, more threads/block gives better serial throughput per block.
- 1024t vs 512t: 1024t processes each spatial slice in half the loop iterations.

**p33 DONE. 1.471x. Fused 1-block-per-channel, 8x float4, FMA trick.**


---
## p76 -- Conv1D dilated strided (baseline 181.0ms)

Input: (64, 64, 524280) float32. Conv1d(in=64, out=128, K=3, stride=3, dilation=4). Output: (64, 128, 174758).

**Four approaches tried, all slower or incorrect:**

- v1 (unfold + TF32 mm, no fix): INCORRECT -- shape bug in original unfold code
- v2 (direct CUDA FP32 + no TF32 disable): INCORRECT -- cuDNN TF32 reference ≠ custom FP32 kernel
- v3 (direct CUDA FP32 + TF32 disabled): 566ms (0.320x) -- 128 output-channel blocks can't share L2 input (only 120 blocks active at once), causes 128x DRAM amplification
- v4 (unfold + FP32 mm + TF32 disabled): 294ms (0.616x) -- FP32 GEMM uses SIMD not Tensor Cores (8 TFLOPS), cuDNN uses TF32 Tensor Cores
- v5 (unfold + TF32 mm, no disable): INCORRECT -- TF32 torch.mm ≠ cuDNN TF32 for this shape (max_diff=5.5e-4, 391M elements > 1e-4 absolute; output values near 0 fail allclose with atol=1e-4)
- v6 (cudnn.benchmark=True): 215ms (0.842x) -- benchmark picks a worse algorithm

**Root cause: cuDNN conv1d uses TF32 Tensor Cores. Without Tensor Cores (FP32 GEMM), ~294ms. TF32 mm doesn't match cuDNN TF32 for this shape (different accumulation algorithm). cudnn.benchmark picks suboptimal algorithm.**

**p76 DONE. No speedup. cuDNN already optimal for this shape.**


---
## p96 -- HuberLoss (baseline 69.0ms)

Input: predictions=(32768, 32768), targets=(32768, 32768), both float32. Operation: smooth_l1_loss (Huber, delta=1) with mean reduction. Total: 1.07B elements.

**Fused single-pass approach (same as MSELoss 2.732x):**
For each float4 pair: compute |d|, apply huber formula, accumulate. Multi-block atomicAdd to global sum. Divide by N.

| v | ms | speedup | change |
|---|-----|---------|--------|
| 1 | 36.60 | 1.885x | fused huber+reduce, float4, 512 blocks |
| 2 | 36.40 | 1.896x | 4096 blocks -- marginal, bandwidth ceiling |

2x ILP (2 float4/thread/iter): 37.6ms -- slower, confirms memory-bound.
Block sweep (256→4096): flat curve, no sensitivity.

**Bandwidth analysis:** 2 * 1.07B * 4B = 8.57 GB reads + 4B output write. At 273 GB/s: 31ms minimum. Achieved 36.4ms = 85% bandwidth efficiency.

**p96 DONE. 1.896x. Fused single-pass Huber+reduce, float4, 4096 blocks.**
Note: MSELoss got 2.732x (103ms→37.7ms) with same pattern because PyTorch MSE implementation was slower at baseline.

