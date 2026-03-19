#!/usr/bin/env python3
"""
Baseline timing script for KernelBench on NVIDIA Thor.
Runs reference PyTorch programs and records wall-clock timing statistics.
Handles per-problem timeouts, OOMs, and other failures gracefully.
"""

import json
import os
import signal
import sys
import time
import traceback

import torch
import numpy as np

from kernelbench.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_dataset
from kernelbench.timing import measure_ref_program_time
from kernelbench.utils import set_gpu_arch

HARDWARE_LABEL = "Thor_AGX"
TIMEOUT_SECONDS = 120  # per-problem timeout


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError(f"Timed out after {TIMEOUT_SECONDS}s")


def time_single_problem(dataset, problem_id, device, precision="fp32"):
    """Time a single problem with timeout and error handling."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIMEOUT_SECONDS)
    try:
        _, name, src = fetch_ref_arch_from_dataset(dataset, problem_id)
        result = measure_ref_program_time(
            ref_arch_name=name,
            ref_arch_src=src,
            device=device,
            timing_method="cuda_event",
            precision=precision,
        )
        signal.alarm(0)
        return {"problem_id": problem_id, "name": name, "status": "ok", **result}
    except TimeoutError:
        signal.alarm(0)
        return {"problem_id": problem_id, "name": name, "status": "timeout",
                "error": f"Timed out after {TIMEOUT_SECONDS}s"}
    except torch.cuda.OutOfMemoryError as e:
        signal.alarm(0)
        torch.cuda.empty_cache()
        return {"problem_id": problem_id, "name": name, "status": "oom",
                "error": str(e)}
    except Exception as e:
        signal.alarm(0)
        return {"problem_id": problem_id, "name": name if 'name' in dir() else f"problem_{problem_id}",
                "status": "error", "error": f"{type(e).__name__}: {e}"}


def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    start_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end_id = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    precision = sys.argv[4] if len(sys.argv) > 4 else "fp32"

    set_gpu_arch(["Blackwell"])
    device = torch.device("cuda:0")
    dataset = construct_kernelbench_dataset(level)
    num_problems = len(dataset)

    print(f"Level {level}: {num_problems} problems, running {start_id}-{end_id}")
    print(f"Hardware: {HARDWARE_LABEL}, Precision: {precision}")
    print(f"Timeout: {TIMEOUT_SECONDS}s per problem")
    print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    results_dir = os.path.expanduser(f"~/thor_kernelbench_work/results/{HARDWARE_LABEL}")
    os.makedirs(results_dir, exist_ok=True)

    results = []
    wall_start = time.time()

    problem_ids = [pid for pid in dataset.get_problem_ids() if start_id <= pid <= end_id]

    for pid in problem_ids:
        t0 = time.time()
        result = time_single_problem(dataset, pid, device, precision)
        elapsed = time.time() - t0
        result["wall_seconds"] = round(elapsed, 2)
        results.append(result)

        status_icon = {"ok": "✓", "timeout": "⏰", "oom": "💥", "error": "✗"}
        icon = status_icon.get(result["status"], "?")
        mean_str = f'{result["mean"]:.2f}ms' if result["status"] == "ok" else result.get("error", "")[:60]
        print(f"  [{icon}] {pid:3d} {result.get('name', '?')[:50]:50s} {mean_str} ({elapsed:.1f}s)")

        # Save incrementally
        out_path = os.path.join(results_dir, f"baseline_level{level}_{start_id}-{end_id}.json")
        with open(out_path, "w") as f:
            json.dump({"hardware": HARDWARE_LABEL, "level": level,
                        "precision": precision, "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "gpu_name": torch.cuda.get_device_name(0),
                        "results": results}, f, indent=2)

    wall_total = time.time() - wall_start
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = len(results) - ok
    print(f"\nDone: {ok}/{len(results)} succeeded, {fail} failed, wall time {wall_total:.1f}s")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
