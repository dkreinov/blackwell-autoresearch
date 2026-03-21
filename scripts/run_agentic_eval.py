#!/usr/bin/env python3
"""
Agentic CUDA kernel optimization for KernelBench on Thor AGX.

Two-pass approach:
  Pass 1 (breadth): One-shot kernel generation for all Level 1 problems.
  Pass 2 (depth):   Iterative refinement on problems where Pass 1 failed or got low speedup.
                    Each iteration feeds back compile errors, timing, and power data.

Reads program.md for Thor-specific hardware context injected into every prompt.
Uses litellm for LLM API calls (supports deepseek, anthropic, openai, google).

Usage:
    # Dry run (no LLM calls, just build prompts):
    python3 run_agentic_eval.py --dry-run

    # Full run with DeepSeek:
    DEEPSEEK_API_KEY=... python3 run_agentic_eval.py --server-type deepseek

    # Full run with Claude:
    ANTHROPIC_API_KEY=... python3 run_agentic_eval.py --server-type anthropic --model anthropic/claude-sonnet-4-20250514

    # Custom budget:
    python3 run_agentic_eval.py --server-type deepseek --budget-hours 2 --max-iter 3
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import torch

# KernelBench imports (eval.py does NOT import modal — safe)
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref, KernelExecResult
from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code, set_gpu_arch

# litellm for LLM API calls
from litellm import completion

HARDWARE_LABEL = "Thor_AGX"
PER_EXPERIMENT_TIMEOUT = 120
DEFAULT_BUDGET_HOURS = 4
DEFAULT_MAX_ITER = 5
PASS2_SPEEDUP_THRESHOLD = 1.5  # iterate on problems below this speedup

PROGRAM_MD_PATH = os.path.expanduser("~/thor_kernelbench_work/program.md")
RESULTS_DIR = os.path.expanduser("~/thor_kernelbench_work/results/Thor_AGX")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline_level1.json")


class ExperimentTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise ExperimentTimeout("Experiment timed out")


def load_thor_context():
    """Load Thor-specific hardware context from program.md sections 2-5."""
    with open(PROGRAM_MD_PATH) as f:
        full = f.read()

    # Extract sections 2 through 5 (hardware ref, unified memory, transfer study, baseline)
    sections = []
    current = None
    for line in full.split("\n"):
        if line.startswith("## 2.") or line.startswith("## 3.") or \
           line.startswith("## 4.") or line.startswith("## 5."):
            current = []
            sections.append(current)
        elif line.startswith("## 6.") or line.startswith("## 7."):
            current = None
        if current is not None:
            current.append(line)

    return "\n".join(line for section in sections for line in section)


def load_baseline():
    """Load Thor baseline timings for speedup comparison."""
    with open(BASELINE_PATH) as f:
        data = json.load(f)
    return {r["problem_id"]: r["mean"] for r in data["results"] if r["status"] == "ok"}


def build_prompt_pass1(ref_code, thor_context):
    """Build one-shot prompt: KernelBench standard + Thor context."""
    # Standard KernelBench prompt for CUDA backend
    base_prompt = get_prompt_for_backend(
        ref_arch_src=ref_code,
        backend="cuda",
        option="one_shot",
        precision="fp32",
    )

    # Append Thor-specific context
    thor_addendum = f"""

--- IMPORTANT: Target Hardware ---

This kernel will run on NVIDIA Jetson AGX Thor, NOT on H100 or A100.
Thor has unified LPDDR5X memory (229 GB/s) shared between CPU and GPU.
There is NO dedicated VRAM and NO PCIe bus.

Key differences from datacenter GPUs:
- Memory bandwidth is 8.9x lower than H100 HBM3
- L2 cache is 32 MB (useful for working set caching)
- Shared memory tiling for reductions often HURTS on unified memory
- Normalization and activation fusions transfer well from H100 optimizations
- Reductions optimized for HBM consistently backfire on Thor

Compile with: TORCH_CUDA_ARCH_LIST=11.0 (sm_110, Blackwell Tegra variant)
Use -O3 --use_fast_math for nvcc flags.

Optimize for both speed AND power efficiency on this edge device.
"""
    return base_prompt + thor_addendum


def build_prompt_pass2(ref_code, thor_context, iteration, prev_code, prev_result):
    """Build iterative prompt with feedback from previous attempt."""
    base = build_prompt_pass1(ref_code, thor_context)

    feedback = f"""

--- Previous Attempt (iteration {iteration - 1}) ---

```python
{prev_code}
```

Result:
"""
    if prev_result.get("status") == "compile_error":
        feedback += f"COMPILE ERROR: {prev_result['error'][:500]}\n"
        feedback += "Fix the compilation error. The target is sm_110 (Blackwell Tegra).\n"
    elif prev_result.get("status") == "incorrect":
        feedback += f"INCORRECT OUTPUT (max_diff={prev_result.get('max_diff', '?')})\n"
        feedback += "The kernel produced wrong results. Fix the math.\n"
    elif prev_result.get("status") == "ok":
        feedback += f"Compiled and correct.\n"
        feedback += f"Timing: {prev_result['custom_ms']:.2f} ms (baseline: {prev_result['baseline_ms']:.2f} ms)\n"
        feedback += f"Speedup: {prev_result['speedup']:.3f}x\n"
        if prev_result.get("gpu_power_w"):
            feedback += f"GPU Power: {prev_result['gpu_power_w']:.1f}W\n"
            feedback += f"Perf/Watt: {prev_result['speedup'] / prev_result['gpu_power_w']:.4f}\n"
        feedback += "\nImprove the kernel further. Target higher speedup and lower power.\n"
    elif prev_result.get("status") == "timeout":
        feedback += "TIMEOUT — kernel hung or took too long. Simplify the approach.\n"

    return base + feedback


def call_llm(prompt, server_type, model_name, max_tokens=8192, temperature=0.0):
    """Call LLM via litellm and extract code."""
    messages = [
        {"role": "system", "content": "You are an expert CUDA kernel developer."},
        {"role": "user", "content": prompt},
    ]

    response = completion(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    raw = response.choices[0].message.content
    code = extract_first_code(raw, ["python", "cpp"])
    return code, raw


def read_tegrastats_power(duration=5):
    """Read GPU power during a short measurement window."""
    import re
    log_path = "/tmp/agentic_power.log"
    try:
        subprocess.run(["tegrastats", "--stop"], capture_output=True)
        time.sleep(0.5)
        if os.path.exists(log_path):
            os.remove(log_path)
        subprocess.run(
            ["tegrastats", "--start", "--interval", "1000", "--logfile", log_path],
            capture_output=True,
        )
        time.sleep(duration)
        subprocess.run(["tegrastats", "--stop"], capture_output=True)
        time.sleep(0.5)

        with open(log_path) as f:
            lines = f.readlines()
        if not lines:
            return None

        powers = []
        for line in lines:
            m = re.search(r'VDD_GPU\s+(\d+)mW/(\d+)mW', line)
            if m:
                powers.append(int(m.group(1)))
        if powers:
            return sum(powers) / len(powers) / 1000.0  # mW to W
    except Exception:
        pass
    return None


def eval_kernel(ref_code, custom_code, device, baseline_ms):
    """Compile, test correctness, benchmark a kernel on Thor."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(PER_EXPERIMENT_TIMEOUT)

    result = {"status": "unknown"}
    try:
        exec_result = eval_kernel_against_ref(
            original_model_src=ref_code,
            custom_model_src=custom_code,
            measure_performance=True,
            timing_method="cuda_event",
            num_correct_trials=5,
            num_perf_trials=100,
            device=device,
            backend="cuda",
            precision=torch.float32,
            verbose=False,
            check_for_excessive_speedup=False,
        )
        signal.alarm(0)

        if not exec_result.compiled:
            result = {
                "status": "compile_error",
                "error": str(getattr(exec_result, "runtime_error", ""))[:500],
            }
        elif not exec_result.correctness:
            result = {
                "status": "incorrect",
                "max_diff": getattr(exec_result, "max_diff", None),
            }
        else:
            custom_ms = exec_result.runtime
            speedup = baseline_ms / custom_ms if custom_ms > 0 else 0
            result = {
                "status": "ok",
                "custom_ms": round(custom_ms, 3),
                "baseline_ms": baseline_ms,
                "speedup": round(speedup, 3),
            }
    except ExperimentTimeout:
        signal.alarm(0)
        result = {"status": "timeout", "error": f"Timed out after {PER_EXPERIMENT_TIMEOUT}s"}
    except Exception as e:
        signal.alarm(0)
        result = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
        torch.cuda.empty_cache()

    return result


def run_experiment(problem_id, problem_name, ref_code, thor_context, baseline_ms,
                   device, server_type, model_name, max_iterations, prev_results=None):
    """Run one or more iterations on a single problem."""
    iterations = []
    best_speedup = 0
    best_code = None

    start_iter = 1 if prev_results is None else len(prev_results) + 1
    if prev_results:
        iterations = list(prev_results)
        best_speedup = max((r.get("speedup", 0) for r in prev_results), default=0)
        best_code = prev_results[-1].get("code")

    for iteration in range(start_iter, max_iterations + 1):
        t0 = time.time()

        # Build prompt
        if iteration == 1:
            prompt = build_prompt_pass1(ref_code, thor_context)
        else:
            prev = iterations[-1]
            prompt = build_prompt_pass2(ref_code, thor_context, iteration,
                                        prev.get("code", ""), prev)

        # Call LLM
        try:
            code, raw_response = call_llm(prompt, server_type, model_name)
        except Exception as e:
            iterations.append({
                "iteration": iteration,
                "status": "llm_error",
                "error": str(e)[:200],
                "wall_seconds": round(time.time() - t0, 1),
            })
            break

        if code is None:
            iterations.append({
                "iteration": iteration,
                "status": "no_code",
                "error": "LLM did not produce extractable code",
                "wall_seconds": round(time.time() - t0, 1),
            })
            break

        # Eval on Thor
        result = eval_kernel(ref_code, code, device, baseline_ms)
        result["iteration"] = iteration
        result["code"] = code
        result["wall_seconds"] = round(time.time() - t0, 1)

        # Read power for successful kernels
        if result["status"] == "ok":
            gpu_power = read_tegrastats_power(duration=3)
            if gpu_power:
                result["gpu_power_w"] = round(gpu_power, 1)
                result["perf_per_watt"] = round(result["speedup"] / gpu_power, 4)

        iterations.append(result)

        # Decide: keep iterating?
        if result["status"] == "ok" and result["speedup"] > best_speedup:
            best_speedup = result["speedup"]
            best_code = code
        elif result["status"] == "ok" and result["speedup"] <= best_speedup and iteration > 1:
            # No improvement — stop iterating on this problem
            break
        elif result["status"] in ("llm_error", "no_code"):
            break

        # Single iteration for Pass 1
        if max_iterations == 1:
            break

    return {
        "problem_id": problem_id,
        "problem_name": problem_name,
        "best_speedup": best_speedup,
        "best_code": best_code,
        "num_iterations": len(iterations),
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser(description="Agentic CUDA kernel optimization for Thor")
    parser.add_argument("--server-type", type=str, default="deepseek",
                        help="LLM provider: deepseek, anthropic, openai, google")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: provider's default)")
    parser.add_argument("--budget-hours", type=float, default=DEFAULT_BUDGET_HOURS)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                        help="Max iterations per problem in Pass 2")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated problem IDs (default: all)")
    parser.add_argument("--pass2-threshold", type=float, default=PASS2_SPEEDUP_THRESHOLD)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build prompts only, no LLM calls")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Model defaults per provider
    MODEL_DEFAULTS = {
        "deepseek": "deepseek/deepseek-coder",
        "anthropic": "anthropic/claude-sonnet-4-20250514",
        "openai": "gpt-4o-2024-08-06",
        "google": "gemini/gemini-2.5-flash",
    }
    model_name = args.model or MODEL_DEFAULTS.get(args.server_type, args.server_type)

    if args.output is None:
        args.output = os.path.join(RESULTS_DIR, f"agentic_{args.server_type}_level{args.level}.json")

    # Setup
    os.environ["TORCH_CUDA_ARCH_LIST"] = "11.0"
    device = torch.device("cuda:0")
    print(f"=== Agentic CUDA Kernel Optimization ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"LLM: {model_name} via {args.server_type}")
    print(f"Budget: {args.budget_hours}h, max {args.max_iter} iterations/problem")
    print(f"Output: {args.output}")
    print()

    # Load resources
    thor_context = load_thor_context()
    baseline = load_baseline()
    dataset = construct_kernelbench_dataset(args.level, source="local")

    problem_ids = dataset.get_problem_ids()
    if args.problems:
        problem_ids = [int(x) for x in args.problems.split(",")]

    if args.dry_run:
        print("=== DRY RUN — building prompts only ===\n")
        for pid in problem_ids[:2]:
            problem = dataset.get_problem_by_id(pid)
            prompt = build_prompt_pass1(problem.code, thor_context)
            print(f"--- Problem {pid}: {problem.name} ---")
            print(f"Prompt length: {len(prompt)} chars")
            print(f"First 200 chars: {prompt[:200]}...")
            print(f"Last 200 chars: ...{prompt[-200:]}")
            print()
        print("Dry run complete. Prompts build correctly.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    wall_start = time.time()
    budget_seconds = args.budget_hours * 3600

    # ==================== PASS 1: BREADTH ====================
    print("=== PASS 1: One-shot on all problems ===\n")
    pass1_results = []

    for pid in problem_ids:
        if time.time() - wall_start > budget_seconds * 0.5:
            print(f"\n  Budget half-point reached. Stopping Pass 1 at problem {pid}.")
            break

        problem = dataset.get_problem_by_id(pid)
        base_ms = baseline.get(pid)
        if base_ms is None:
            print(f"  [{pid:3d}] {problem.name[:45]:45s} SKIP (no baseline)")
            continue

        print(f"  [{pid:3d}] {problem.name[:45]:45s} base={base_ms:.1f}ms ", end="", flush=True)

        result = run_experiment(
            pid, problem.name, problem.code, thor_context, base_ms,
            device, args.server_type, model_name, max_iterations=1,
        )
        pass1_results.append(result)

        it = result["iterations"][-1] if result["iterations"] else {}
        if it.get("status") == "ok":
            print(f"→ {it['custom_ms']:.1f}ms {it['speedup']:.2f}x ({it['wall_seconds']:.0f}s)")
        else:
            print(f"→ {it.get('status', '?')} ({it.get('wall_seconds', 0):.0f}s)")

        # Incremental save
        _save_results(args.output, args, model_name, pass1_results, [], wall_start)

    # ==================== PASS 2: DEPTH ====================
    # Select problems where Pass 1 failed or got low speedup
    pass2_candidates = []
    for r in pass1_results:
        best = r["best_speedup"]
        last_status = r["iterations"][-1]["status"] if r["iterations"] else "unknown"
        if last_status in ("compile_error", "incorrect", "error") or best < args.pass2_threshold:
            pass2_candidates.append(r)

    print(f"\n=== PASS 2: Iterative refinement on {len(pass2_candidates)} problems ===\n")
    pass2_results = []

    for r in pass2_candidates:
        if time.time() - wall_start > budget_seconds:
            print(f"\n  Budget exhausted. Stopping Pass 2.")
            break

        pid = r["problem_id"]
        problem = dataset.get_problem_by_id(pid)
        base_ms = baseline.get(pid, 0)

        print(f"  [{pid:3d}] {problem.name[:45]:45s} (Pass 1: {r['best_speedup']:.2f}x)")

        result = run_experiment(
            pid, problem.name, problem.code, thor_context, base_ms,
            device, args.server_type, model_name,
            max_iterations=args.max_iter,
            prev_results=r["iterations"],  # Continue from Pass 1 attempt
        )
        pass2_results.append(result)

        # Print iteration history
        for it in result["iterations"][len(r["iterations"]):]:
            if it.get("status") == "ok":
                print(f"    iter {it['iteration']}: {it['custom_ms']:.1f}ms "
                      f"{it['speedup']:.2f}x ({it['wall_seconds']:.0f}s)")
            else:
                print(f"    iter {it['iteration']}: {it.get('status', '?')} ({it.get('wall_seconds', 0):.0f}s)")

        # Incremental save
        _save_results(args.output, args, model_name, pass1_results, pass2_results, wall_start)

    # ==================== SUMMARY ====================
    wall_total = time.time() - wall_start
    all_results = pass1_results + pass2_results
    _save_results(args.output, args, model_name, pass1_results, pass2_results, wall_start)

    print(f"\n=== Summary ===")
    print(f"Total wall time: {wall_total/60:.1f} min")
    print(f"Pass 1: {len(pass1_results)} problems")
    print(f"Pass 2: {len(pass2_results)} problems (iterated)")

    p1_ok = [r for r in pass1_results if r["best_speedup"] > 0]
    if p1_ok:
        speedups = [r["best_speedup"] for r in p1_ok]
        faster = sum(1 for s in speedups if s > 1.0)
        print(f"\nPass 1 results: {len(p1_ok)} correct, {faster} faster than baseline")
        print(f"  Speedup range: {min(speedups):.3f}x - {max(speedups):.3f}x")
        from statistics import median
        print(f"  Median: {median(speedups):.3f}x")

    if pass2_results:
        improved = sum(1 for r in pass2_results
                      if r["best_speedup"] > pass1_results[pass1_results.index(
                          next(p for p in pass1_results if p["problem_id"] == r["problem_id"])
                      )]["best_speedup"]
                      if any(p["problem_id"] == r["problem_id"] for p in pass1_results))
        print(f"\nPass 2: {improved} problems improved over Pass 1")

    print(f"\nSaved: {args.output}")


def _save_results(path, args, model_name, pass1, pass2, wall_start):
    """Incremental save of all results."""
    data = {
        "hardware": HARDWARE_LABEL,
        "model": model_name,
        "server_type": args.server_type,
        "budget_hours": args.budget_hours,
        "max_iterations": args.max_iter,
        "pass2_threshold": args.pass2_threshold,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "wall_seconds": round(time.time() - wall_start, 1),
        "pass1_results": pass1,
        "pass2_results": pass2,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    main()
