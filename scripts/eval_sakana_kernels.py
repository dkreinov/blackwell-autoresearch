#!/usr/bin/env python3
"""
Evaluate Sakana AI's best CUDA kernels on Thor AGX.
Downloads kernels from HuggingFace, compiles on sm_110, tests correctness, benchmarks.

The Sakana archive stores raw C++ CUDA code with PYBIND11 forward() that matches
the functional interface (PyTorch_Code_Functional). We compile directly, map args
from Model's stored params + forward inputs, and benchmark.

Usage:
    python3 eval_sakana_kernels.py                    # All tasks with speedup > 1.0
    python3 eval_sakana_kernels.py --tasks 1,12,34    # Specific tasks
    python3 eval_sakana_kernels.py --min-speedup 2.0  # Only high-speedup tasks
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import inspect

import torch
from datasets import load_dataset
from kernelbench.utils import set_gpu_arch

HARDWARE_LABEL = "Thor_AGX"
TIMEOUT_SECONDS = 180
WARMUP_TRIALS = 5
NUM_TRIALS = 100


class KernelTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise KernelTimeout(f"Timed out after {TIMEOUT_SECONDS}s")


def compile_sakana_cuda(cuda_code, task_id):
    """Compile Sakana CUDA code on Thor with sm_110 support."""
    from torch.utils.cpp_extension import load

    code_hash = hashlib.md5(cuda_code.encode()).hexdigest()[:8]
    module_name = f"sakana_t{task_id}_{code_hash}"
    build_dir = os.path.expanduser(f"~/.cache/torch_extensions/sakana/{module_name}")
    os.makedirs(build_dir, exist_ok=True)
    cu_path = os.path.join(build_dir, f"{module_name}.cu")
    with open(cu_path, "w") as f:
        f.write(cuda_code)

    module = load(
        name=module_name,
        sources=[cu_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return module


def benchmark_cuda_event(fn, warmup=WARMUP_TRIALS, trials=NUM_TRIALS):
    """Benchmark a function using CUDA events. Returns mean ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def get_cuda_forward_args(model, inputs, cuda_module):
    """Figure out the right args for the CUDA forward function.

    Sakana kernels export forward() matching the functional interface:
    - Simple ops (matmul): forward(A, B) — same as Model.forward
    - Parameterized ops (LayerNorm): forward(x, weight, bias, eps) — includes model params

    Strategy: try Model.forward args first, then try appending model params.
    """
    # Attempt 1: direct passthrough (works for matmul, relu, etc.)
    try:
        with torch.no_grad():
            result = cuda_module.forward(*inputs)
        return inputs, result
    except TypeError:
        pass

    # Attempt 2: inputs + all nn.Parameters + scalar attributes
    params = list(model.parameters())
    # Also collect common scalar attributes
    scalar_attrs = []
    for attr_name in ['eps', 'negative_slope', 'margin', 'p', 'delta', 'lambd',
                      'alpha', 'beta', 'threshold', 'dim', 'kernel_size',
                      'stride', 'padding', 'dilation', 'output_padding', 'groups']:
        if hasattr(model, attr_name):
            scalar_attrs.append(getattr(model, attr_name))

    # Try: inputs + params
    try:
        all_args = list(inputs) + params
        with torch.no_grad():
            result = cuda_module.forward(*all_args)
        return all_args, result
    except TypeError:
        pass

    # Try: inputs + params + scalars
    try:
        all_args = list(inputs) + params + scalar_attrs
        with torch.no_grad():
            result = cuda_module.forward(*all_args)
        return all_args, result
    except TypeError:
        pass

    # Attempt 3: inspect the functional code to get the exact signature
    # This is a fallback — most cases should be caught above
    raise TypeError(f"Cannot determine CUDA forward args for this kernel")


def eval_single_task(task_info, device):
    """Compile, test correctness, and benchmark a single Sakana kernel."""
    # 1. Load PyTorch reference model
    ns = {}
    exec(task_info["pytorch_code"], ns)
    ModelClass = ns["Model"]
    get_inputs = ns["get_inputs"]
    get_init_inputs = ns["get_init_inputs"]

    init_inputs = get_init_inputs()
    model = ModelClass(*init_inputs).to(device).eval()

    # 2. Compile CUDA kernel
    cuda_module = compile_sakana_cuda(task_info["cuda_code"], task_info["task_id"])

    # 3. Test correctness (5 trials with different random inputs)
    for trial in range(5):
        torch.manual_seed(42 + trial)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

        with torch.no_grad():
            ref_output = model(*inputs)

        cuda_args, custom_output = get_cuda_forward_args(model, inputs, cuda_module)

        if not torch.allclose(ref_output, custom_output, atol=1e-2, rtol=1e-2):
            max_diff = (ref_output - custom_output).abs().max().item()
            return {"correct": False, "max_diff": max_diff, "compiled": True}

    # 4. Benchmark
    torch.manual_seed(42)
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    cuda_args, _ = get_cuda_forward_args(model, inputs, cuda_module)

    ref_time = benchmark_cuda_event(lambda: model(*inputs))
    custom_time = benchmark_cuda_event(lambda: cuda_module.forward(*cuda_args))

    return {
        "correct": True,
        "compiled": True,
        "ref_time_ms": round(ref_time, 3),
        "custom_time_ms": round(custom_time, 3),
        "thor_speedup": round(ref_time / custom_time, 3),
    }


def load_best_sakana_kernels(min_speedup=1.0, task_filter=None):
    """Stream Sakana archive, keep best correct kernel per task."""
    print("Loading Sakana AI CUDA Engineer Archive (streaming)...")
    ds = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive", split="level_1", streaming=True)
    best = {}
    total = 0
    for row in ds:
        total += 1
        tid = row["Task_ID"]
        if task_filter and tid not in task_filter:
            continue
        if not row["Correct"]:
            continue
        sp = row.get("CUDA_Speedup_Native")
        if sp is None:
            continue
        if tid not in best or sp > best[tid]["h100_speedup"]:
            best[tid] = {
                "task_id": tid,
                "op_name": row["Op_Name"],
                "h100_speedup": sp,
                "h100_cuda_runtime": row.get("CUDA_Runtime"),
                "cuda_code": row["CUDA_Code"],
                "pytorch_code": row["PyTorch_Code_Module"],
            }
    print(f"  Scanned {total} rows, found best kernels for {len(best)} tasks")
    filtered = {k: v for k, v in best.items() if v["h100_speedup"] >= min_speedup}
    print(f"  After min_speedup={min_speedup}: {len(filtered)} tasks")
    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--output", type=str,
                        default=os.path.expanduser("~/thor_kernelbench_work/results/Thor_AGX/sakana_transfer_level1.json"))
    args = parser.parse_args()

    task_filter = set(int(x) for x in args.tasks.split(",")) if args.tasks else None

    # CRITICAL: set arch to 11.0 for Thor sm_110
    os.environ["TORCH_CUDA_ARCH_LIST"] = "11.0"
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")

    # Load Thor baseline
    thor_baseline_path = os.path.expanduser(
        "~/thor_kernelbench_work/results/Thor_AGX/baseline_level1.json"
    )
    with open(thor_baseline_path) as f:
        thor_baseline = {r["problem_id"]: r for r in json.load(f)["results"] if r["status"] == "ok"}

    kernels = load_best_sakana_kernels(min_speedup=args.min_speedup, task_filter=task_filter)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = []
    wall_start = time.time()

    for tid in sorted(kernels.keys()):
        k = kernels[tid]
        t0 = time.time()
        print(f"  Task {tid:3d} {k['op_name'][:45]:45s} H100={k['h100_speedup']:7.2f}x ... ", end="", flush=True)

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
        try:
            result = eval_single_task(k, device)
            signal.alarm(0)
        except KernelTimeout:
            signal.alarm(0)
            result = {"status": "timeout"}
            print(f"TIMEOUT ({time.time()-t0:.0f}s)")
            results.append({"task_id": tid, "op_name": k["op_name"],
                           "h100_speedup": k["h100_speedup"], "status": "timeout"})
            continue
        except Exception as e:
            signal.alarm(0)
            result = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"ERROR: {str(e)[:80]} ({time.time()-t0:.0f}s)")
            results.append({"task_id": tid, "op_name": k["op_name"],
                           "h100_speedup": k["h100_speedup"], **result})
            # Clear CUDA state after error
            torch.cuda.empty_cache()
            continue

        elapsed = time.time() - t0
        entry = {
            "task_id": tid,
            "op_name": k["op_name"],
            "h100_speedup": k["h100_speedup"],
            "wall_seconds": round(elapsed, 1),
            **result,
        }

        if result.get("correct"):
            entry["status"] = "ok"
            # Add Thor baseline comparison
            if tid in thor_baseline:
                entry["thor_baseline_ms"] = thor_baseline[tid]["mean"]
                entry["thor_speedup_vs_baseline"] = round(
                    thor_baseline[tid]["mean"] / result["custom_time_ms"], 3
                )
            print(f"custom={result['custom_time_ms']:.2f}ms "
                  f"speedup={result['thor_speedup']:.2f}x ({elapsed:.0f}s)")
        elif result.get("compiled") and not result.get("correct"):
            entry["status"] = "incorrect"
            print(f"INCORRECT max_diff={result.get('max_diff', '?')} ({elapsed:.0f}s)")
        else:
            entry["status"] = "compile_fail"
            print(f"COMPILE FAIL ({elapsed:.0f}s)")

        results.append(entry)

        # Incremental save
        with open(args.output, "w") as f:
            json.dump({
                "hardware": HARDWARE_LABEL,
                "source": "SakanaAI/AI-CUDA-Engineer-Archive",
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
                "results": results,
            }, f, indent=2)

    wall_total = time.time() - wall_start
    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\n=== Summary ===")
    print(f"Total: {len(results)}, OK: {len(ok)}, "
          f"Errors: {sum(1 for r in results if r.get('status') in ('error','timeout','compile_fail','incorrect'))}")
    print(f"Wall time: {wall_total:.0f}s")
    if ok:
        speedups = [r["thor_speedup"] for r in ok]
        faster = sum(1 for s in speedups if s > 1.0)
        print(f"\nOf {len(ok)} correct kernels:")
        print(f"  Faster than PyTorch on Thor: {faster}/{len(ok)}")
        print(f"  Speedup range: {min(speedups):.3f}x - {max(speedups):.3f}x")
        print(f"  Median: {sorted(speedups)[len(speedups)//2]:.3f}x")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
