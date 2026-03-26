# Thor fp16 Kernel Autoresearch Program

## 1. Mission

Optimize fp16 CUDA kernels from KernelBench Level 1 on NVIDIA Jetson AGX Thor (Blackwell sm_110).

- **Precision**: fp16 (torch.float16, half precision)
- **Primary metric**: speedup vs PyTorch fp16 baseline (higher = better)
- **Evaluation**: compile with nvcc (sm_110), test correctness (3 trials, atol/rtol=1e-2), benchmark (20 trials, cuda_event)
- **Current category**: HEAVY KERNELS (norms, losses, softmax, scans -- see schedule.json)
- **Iteration**: per-kernel optimization via `eval_kernel.py` -- one problem at a time, round-robin schedule
- **One kernel per experiment**: each KernelBench problem has one PyTorch Model -> write one optimized CUDA replacement (ModelNew)

### Autonomy
Run experiments continuously until externally interrupted. Do not pause for confirmation.

---

## 2. Hardware Reference

See `program.md` Sections 2-4 for Thor hardware specs, compilation settings, and baseline data.

Key hardware facts for fp16 optimization:
- 20 SMs, 1024 max threads/block, 48 warps/SM
- 32 MB L2 cache, 228 KB shared memory/SM
- 273 GB/s theoretical memory bandwidth (LPDDR5X unified)
- 96 Tensor Cores (5th gen, FP8 capable)
- PTX ISA 9.0, sm_110

---

## 3. fp16 Kernel Patterns

### Precision Support

KernelBench `eval_kernel_against_ref` natively supports `precision=torch.float16`:
- Inputs are auto-cast to fp16 via `_process_input_tensor`
- Both reference Model and custom ModelNew are cast via `.to(dtype=precision)`
- Correctness tolerance: 1e-2 (vs 1e-4 for fp32)

### half2 Vectorization

- Use `half2` vectorization (2 halfs per 32-bit operation)
- For bandwidth-bound kernels, load via `float4` (128 bits = 8 halfs) and reinterpret as `half2[4]`
- Dtype check: `TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16")`
- Data pointer: `x.data_ptr<at::Half>()` then reinterpret as `half2*` or `float4*`
- Cast between at::Half* and __half*: use `reinterpret_cast<const __half*>(x.data_ptr<at::Half>())`

### fp16 Intrinsics (sm_110)

| fp32 | fp16 (half2) | Notes |
|------|-------------|-------|
| `__expf(x)` | `h2exp(v)` | Paired exp on half2 |
| `__tanhf(x)` | `h2rcp`, manual | No direct __htanh2, use 1-2/(1+exp(2x)) |
| `1/(1+exp(-x))` | `h2rcp(__hadd2(one, h2exp(neg_v)))` | Sigmoid pattern |
| `fmaxf(x, 0)` | `__hmax2(v, zero)` | ReLU |
| `x * y` | `__hmul2(v, w)` | Paired multiply |
| `x + y` | `__hadd2(v, w)` | Paired add |
| `-x` | `__hneg2(v)` | Paired negate |
| `fabsf(x)` | `__habs2(v)` | Paired abs |
| `float4` load | `float4` -> `half2[4]` | 128-bit load = 8 halfs |
| N/A | `__halves2half2(a,b)` | Pack 2 scalar halfs into half2 (avoids alignment issues) |
| N/A | `__half22float2(v)` | Convert half2 to float2 (for float32 accumulation) |

### Known Issues

- `__h2div` produces INCORRECT results on sm_110 -- do not use
- Odd dimensions (e.g. dim=65535): half2 loads misaligned on alternate rows. Use `__halves2half2()` packing from scalar loads instead of direct half2* cast.
- Accumulate reductions in float32 for numerical safety (half has limited range)

---

## 4. Experiment Loop Protocol

### Current Schedule

`schedule.json` defines the problem order. Current phase: heavy kernels (norms, reductions, losses, softmax, scans).
Each problem has its own independently-evolving kernel file.

Source of truth: `results/Thor_AGX/kernel_results_fp16.json` (per-problem best speedup + version history).

### Files

- `kernels/fp16/p{pid}_{name}.py` -- current best kernel (evolves in place, git tracks history)
- `kernels/fp16/p{pid}_{name}_candidate.py` -- temporary candidate being tested (MUST be deleted after each test)
- `scripts/eval_kernel.py` -- single-problem eval: `python scripts/eval_kernel.py --pid <N> --kernel kernels/fp16/p{N}_*.py --precision fp16`
- `results/Thor_AGX/kernel_results_fp16.json` -- per-problem best speedup + version history
- `findings_fp16.md` -- per-problem experiment results and failures
- `schedule.json` -- round-robin order and timing config

### Loop (run autonomously, do not pause)

**Schedule**: `schedule.json` defines the order and timing. Spend 1 hour per problem in round-robin.
Max 2 minutes per experiment (eval times out at 120s).

**Per-experiment cycle** (repeat until 1hr on current problem, then advance schedule):

1. Read `findings_fp16.md` section for current problem -- what has been tried, what failed
2. Read `results/Thor_AGX/kernel_results_fp16.json` -- current best speedup for this problem
3. Read current kernel file `kernels/fp16/p{pid}_{name}.py` -- the code to improve
4. Invent ONE targeted change based on:
   - Thor hardware specs (Section 2 / program.md)
   - fp16 kernel patterns and intrinsics (Section 3)
   - What failed before (findings_fp16.md for this problem)
   - The fp32 kernel in `kernels/p{pid}_{name}.py` for optimization ideas (read for inspiration, NOT as a starting point -- fp16 kernels use different types and intrinsics)
   - Ideas that haven't been tried yet
5. Write the modified kernel to `kernels/fp16/p{pid}_{name}_candidate.py`
6. Run: `python scripts/eval_kernel.py --pid <N> --kernel kernels/fp16/p{pid}_{name}_candidate.py --precision fp16`
   - Must complete in <120s or it's discarded
7. **If result is correct AND faster than current best**:
   - Copy candidate over `kernels/fp16/p{pid}_{name}.py`
   - **Delete the candidate file immediately**
   - Update `results/Thor_AGX/kernel_results_fp16.json` (best_speedup, best_ms, iterations, history)
   - Update `findings_fp16.md` section: add row to table, note what worked
   - **Commit and push immediately**:
     `git add kernels/fp16/p{pid}_{name}.py results/Thor_AGX/kernel_results_fp16.json findings_fp16.md`
     `git commit -m "p{pid}: {name} v{N} [fp16] -- <change>, {old}x -> {new}x"`
     `git push origin main`
8. **If result is slower, incorrect, or times out**:
   - **Delete the candidate file immediately**
   - Do NOT overwrite current best
   - Note the failure in findings_fp16.md (one line: what was tried and why it failed)
9. Go to step 1 for the same problem (until 1hr elapsed, then advance schedule)

### CRITICAL: Cleanup Discipline

- **No _candidate.py files should survive between experiments.** Delete after EVERY test, pass or fail.
- **Commit each improvement individually**, not in batches. Each commit = one verified speedup.
- **Push after each commit.** Do not accumulate unpushed commits.
- Run `python scripts/eval_kernel.py --clean --precision fp16` to check for stale candidates before starting a session.

### Schedule Enforcement

- Track start time when switching to a new problem
- After 3600s on current problem: advance to next in `schedule.json` order
- After all problems: round-robin back to first
- Only the user decides when to stop the loop entirely

### Commit Format

```
p{pid}: {name} v{N} [fp16] -- {one-line change description}, {old_speedup}x -> {new_speedup}x
```

Only commit improvements. Discards are noted in findings_fp16.md but not committed.

---

## 5. Rules

### Allowed
- Write custom CUDA kernels using any standard CUDA features
- Use shared memory, warp shuffles, cooperative groups, vectorized loads
- Use torch.utils.cpp_extension for compilation
- Math approximations if numerically acceptable (correctness oracle catches mismatches)
- FP16 precision (matching fp16 baseline)
- Use `half2` intrinsics, `__halves2half2()` packing, float32 accumulation for reductions

### NOT Allowed
- Modify the reference Model class, get_inputs(), or get_init_inputs()
- Use cuBLAS/cuDNN library calls that just wrap the same PyTorch operation
- Change the test/evaluation infrastructure
- Use features that require sm_120+ (Thor is sm_110)
- Overwrite a kernel file unless the new version is strictly better (correct + faster)
- Use `__h2div` (produces incorrect results on sm_110)

### Anti-Paralysis Rules
1. First tool call within first response -- no multi-paragraph analysis before acting
2. No cycle estimation in prose -- the benchmark is the oracle
3. One response = one action -- every response must contain a tool call
4. If resuming from context summary: read `kernel_results_fp16.json` + schedule.json, pick current problem, execute immediately

---

## 6. Output Format

After each experiment (one problem, one change):

```
=== p{pid} {name} -- attempt {N} [fp16] ===
Idea: <one-line description of the change>
Result: OK {ms}ms {speedup}x | FAIL {reason}
vs current best: {old_speedup}x -> {new_speedup}x (+{pct}%) | no improvement
Action: committed v{N} | discarded
findings_fp16.md: updated | noted failure
Next: <what to try next for this problem>
===
```
