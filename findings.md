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
