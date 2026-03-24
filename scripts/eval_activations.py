#!/usr/bin/env python3
"""
Evaluate an optimization strategy across all activation kernels.

Usage:
    python scripts/eval_activations.py --opt <optimization_name>
    python scripts/eval_activations.py --opt float4       # run float4 template on all 13
    python scripts/eval_activations.py --opt baseline      # run PyTorch baseline for reference
    python scripts/eval_activations.py --list              # show all results so far
    python scripts/eval_activations.py --matrix            # show optimization × problem matrix

Results are appended to results/Thor_AGX/activation_matrix.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
RESULTS_FILE = WORK / "results" / "Thor_AGX" / "activation_matrix.json"
TEMPLATES_DIR = WORK / "templates" / "activations"
CATEGORIES_FILE = WORK / "categories.json"

SSH = "ssh -o StrictHostKeyChecking=no nvidia@nvidia-thor-01"
AGENT = "bash ~/thor_kernelbench_work/thor_agent.sh"
REMOTE_KERNELS = "~/thor_kernelbench_work/kernels"

# Activation definitions: problem_id -> (name, cuda_math_expr)
ACTIVATIONS = {
    19: ("ReLU",         "fmaxf(val, 0.0f)"),
    20: ("LeakyReLU",    "val > 0.0f ? val : 0.01f * val"),
    21: ("Sigmoid",      "1.0f / (1.0f + expf(-val))"),
    22: ("Tanh",         "tanhf(val)"),
    25: ("Swish",        "val / (1.0f + expf(-val))"),
    26: ("GELU",         "0.5f * val * (1.0f + erff(val * 0.7071067811865476f))"),
    27: ("SELU",         "val > 0.0f ? 1.0507f * val : 1.0507f * 1.6733f * (expf(val) - 1.0f)"),
    28: ("HardSigmoid",  "fminf(fmaxf(val / 6.0f + 0.5f, 0.0f), 1.0f)"),
    29: ("Softplus",     "val > 20.0f ? val : logf(1.0f + expf(val))"),
    30: ("Softsign",     "val / (1.0f + fabsf(val))"),
    31: ("ELU",          "val > 0.0f ? val : 1.0f * (expf(val) - 1.0f)"),
    32: ("HardTanh",     "fminf(fmaxf(val, -1.0f), 1.0f)"),
    88: ("MinGPTNewGelu","0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)))"),
}


def load_template(opt_name: str) -> str:
    """Load a CUDA kernel template. Template must have {ACTIVATION_FUNC} placeholder."""
    path = TEMPLATES_DIR / f"{opt_name}.py.template"
    if not path.exists():
        print(f"ERROR: Template not found: {path}")
        print(f"Available templates: {[f.stem for f in TEMPLATES_DIR.glob('*.py.template')]}")
        sys.exit(1)
    return path.read_text()


def generate_kernel(template: str, pid: int, name: str, math_expr: str) -> str:
    """Generate a kernel file from template by substituting the activation math."""
    return template.replace("{ACTIVATION_FUNC}", math_expr).replace("{ACTIVATION_NAME}", name.lower())


def scp_to_thor(local_path: str, remote_name: str):
    """Copy a kernel file to Thor."""
    cmd = f'scp -o StrictHostKeyChecking=no "{local_path}" nvidia@nvidia-thor-01:{REMOTE_KERNELS}/{remote_name}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")


def eval_kernel(remote_name: str, pid: int) -> dict:
    """Evaluate a kernel on Thor. Returns dict with status, custom_ms, speedup."""
    cmd = f'{SSH} "{AGENT} eval-kernel kernels/{remote_name} {pid}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
    output = result.stdout + result.stderr

    info = {"output": output.strip()}

    if "[OK]" in output:
        info["status"] = "ok"
        for line in output.splitlines():
            if "Custom:" in line:
                info["custom_ms"] = float(line.split(":")[1].strip().replace("ms", "").strip())
            if "Speedup:" in line:
                info["speedup"] = float(line.split(":")[1].strip().replace("x", "").strip())
    elif "[COMPILE_ERROR]" in output:
        info["status"] = "compile_error"
    elif "[INCORRECT]" in output:
        info["status"] = "incorrect"
    else:
        info["status"] = "error"

    return info


def load_results() -> dict:
    """Load existing results matrix."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"optimizations": {}}


def save_results(results: dict):
    """Save results matrix."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))


def run_optimization(opt_name: str):
    """Run an optimization template across all activations."""
    template = load_template(opt_name)
    results = load_results()

    print(f"\n{'='*70}")
    print(f"  Optimization: {opt_name}")
    print(f"  Evaluating {len(ACTIVATIONS)} activation kernels on Thor")
    print(f"{'='*70}\n")

    opt_results = {}
    t0 = time.time()

    for pid, (name, math_expr) in sorted(ACTIVATIONS.items()):
        print(f"[{pid:3d}] {name:15s} ... ", end="", flush=True)

        # Generate kernel from template
        kernel_code = generate_kernel(template, pid, name, math_expr)

        # Write to temp file, SCP to Thor
        remote_name = f"act_{name.lower()}_{opt_name}.py"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(kernel_code)
            tmp_path = f.name

        try:
            scp_to_thor(tmp_path, remote_name)
            info = eval_kernel(remote_name, pid)

            if info["status"] == "ok":
                print(f"{info['custom_ms']:8.2f}ms  {info['speedup']:.3f}x")
            else:
                print(f"{info['status']}")

            opt_results[str(pid)] = {
                "name": name,
                "status": info["status"],
                "custom_ms": info.get("custom_ms"),
                "speedup": info.get("speedup"),
            }
        except Exception as e:
            print(f"ERROR: {e}")
            opt_results[str(pid)] = {"name": name, "status": "error", "error": str(e)}
        finally:
            os.unlink(tmp_path)

    elapsed = time.time() - t0

    # Save
    results["optimizations"][opt_name] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(elapsed, 1),
        "results": opt_results,
    }
    save_results(results)

    # Summary
    ok = [r for r in opt_results.values() if r["status"] == "ok"]
    print(f"\n{'='*70}")
    print(f"  Done: {len(ok)}/{len(ACTIVATIONS)} correct")
    if ok:
        speedups = [r["speedup"] for r in ok]
        print(f"  Speedup: min={min(speedups):.3f}x  median={sorted(speedups)[len(speedups)//2]:.3f}x  max={max(speedups):.3f}x")
    print(f"  Elapsed: {elapsed:.0f}s")
    print(f"  Saved to: {RESULTS_FILE}")
    print(f"{'='*70}\n")


def show_matrix():
    """Display the optimization × problem matrix."""
    results = load_results()
    if not results["optimizations"]:
        print("No results yet.")
        return

    opts = list(results["optimizations"].keys())
    pids = sorted(ACTIVATIONS.keys())

    # Header
    header = f"{'Problem':>20s}"
    for opt in opts:
        header += f" {opt:>12s}"
    print(header)
    print("-" * len(header))

    # Rows
    for pid in pids:
        name = ACTIVATIONS[pid][0]
        row = f"{f'{pid} {name}':>20s}"
        for opt in opts:
            r = results["optimizations"][opt]["results"].get(str(pid), {})
            if r.get("status") == "ok":
                row += f" {r['speedup']:11.3f}x"
            elif r.get("status"):
                row += f" {r['status']:>12s}"
            else:
                row += f" {'—':>12s}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate activation optimizations")
    parser.add_argument("--opt", type=str, help="Optimization template name to evaluate")
    parser.add_argument("--list", action="store_true", help="List all results")
    parser.add_argument("--matrix", action="store_true", help="Show optimization × problem matrix")
    args = parser.parse_args()

    if args.matrix or args.list:
        show_matrix()
    elif args.opt:
        run_optimization(args.opt)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
