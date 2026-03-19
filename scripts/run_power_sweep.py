#!/usr/bin/env python3
"""
Power-mode sweep script for KernelBench on NVIDIA Thor.
Sets a power mode, locks clocks, samples tegrastats during run, saves results with power metadata.

Usage:
    python3 run_power_sweep.py <mode_id> [level] [start_id] [end_id]
    e.g.  python3 run_power_sweep.py 1        # 120W mode, Level 1, all problems
          python3 run_power_sweep.py 0 1 1 5  # MAXN mode, problems 1-5 (dry run)

Power modes on Thor:
    0 = MAXN   (no limits, GPU 1575MHz, CPU 2601MHz)
    1 = 120W   (GPU 1386MHz, CPU 2601MHz)
    2 = 90W    (GPU 1530MHz but heavy PG, 12 CPU cores)
    3 = 70W    (GPU 1530MHz, CPU 1998MHz, 12 cores)
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback

import torch

from kernelbench.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_dataset
from kernelbench.timing import measure_ref_program_time
from kernelbench.utils import set_gpu_arch

HARDWARE_LABEL = "Thor_AGX"
TIMEOUT_SECONDS = 120
SUDO_PASS = os.environ.get("SUDO_PASS", "")
THERMAL_SETTLE_SECS = 60
TEGRASTATS_INTERVAL_MS = 2000  # sample every 2 seconds

MODE_NAMES = {
    0: "MAXN",
    1: "120W",
    2: "90W",
    3: "70W",
}


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError(f"Timed out after {TIMEOUT_SECONDS}s")


def sudo(cmd, check=True):
    """Run a sudo command with password piped in."""
    full = f"echo {SUDO_PASS} | sudo -S {cmd}"
    result = subprocess.run(full, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"sudo command failed: {cmd}\nstderr: {result.stderr}")
    return result


def set_power_mode(mode_id):
    print(f"  Setting power mode {mode_id} ({MODE_NAMES.get(mode_id, '?')})...")
    sudo(f"nvpmodel -m {mode_id}")
    time.sleep(5)


def lock_clocks():
    print("  Locking clocks at mode maximum (jetson_clocks)...")
    sudo("jetson_clocks")
    time.sleep(3)


def get_current_mode():
    result = sudo("nvpmodel -q", check=False)
    for line in result.stdout.splitlines():
        if "NV Power Mode" in line:
            return line.strip()
    return result.stdout.strip()


def get_clock_state():
    """Snapshot current GPU/CPU/MEM frequencies."""
    clocks = {}
    # GPU GPC freq (compute)
    try:
        with open("/sys/class/devfreq/gpu-gpc-0/cur_freq") as f:
            clocks["gpu_gpc_freq_mhz"] = int(f.read().strip()) // 1_000_000
    except Exception:
        clocks["gpu_gpc_freq_mhz"] = None
    # GPU NVD freq (video)
    try:
        with open("/sys/class/devfreq/gpu-nvd-0/cur_freq") as f:
            clocks["gpu_nvd_freq_mhz"] = int(f.read().strip()) // 1_000_000
    except Exception:
        clocks["gpu_nvd_freq_mhz"] = None
    # CPU freq (first core)
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            clocks["cpu0_freq_mhz"] = int(f.read().strip()) // 1_000
    except Exception:
        clocks["cpu0_freq_mhz"] = None
    # tegrastats single sample: run for 2s and grab first non-empty line
    try:
        result = subprocess.run(
            "timeout 2 tegrastats 2>/dev/null | head -1",
            shell=True, capture_output=True, text=True, timeout=5,
        )
        clocks["tegrastats_once"] = result.stdout.strip()
    except Exception as e:
        clocks["tegrastats_once"] = f"error: {e}"
    return clocks


def parse_tegrastats_line(line):
    """Extract power readings from a tegrastats line. Returns dict of {sensor: mW}.
    Format: VDD_GPU 3960mW/3960mW  (instantaneous/average)
    Also extracts temperatures: cpu@33.5C gpu@34.8C tj@34.6C
    """
    powers = {}
    # Power: SENSOR_NAME <inst>mW/<avg>mW
    for m in re.finditer(r'(VDD\w+|VIN\w+)\s+(\d+)mW/(\d+)mW', line):
        powers[m.group(1) + "_inst_mW"] = int(m.group(2))
        powers[m.group(1) + "_avg_mW"] = int(m.group(3))
    # Temperatures: label@value C
    for m in re.finditer(r'(\w+)@([\d.]+)C', line):
        powers[m.group(1) + "_temp_C"] = float(m.group(2))
    return powers


def start_tegrastats(log_path):
    """Start tegrastats daemon logging to file."""
    # Stop any existing instance first
    subprocess.run(["tegrastats", "--stop"], capture_output=True)
    time.sleep(1)
    result = subprocess.run(
        ["tegrastats", "--start", "--interval", str(TEGRASTATS_INTERVAL_MS),
         "--logfile", log_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Warning: tegrastats --start returned {result.returncode}: {result.stderr}")
    return None  # daemon, no proc handle needed


def stop_tegrastats(proc):
    """Stop tegrastats daemon."""
    subprocess.run(["tegrastats", "--stop"], capture_output=True)


def summarize_power_log(log_path):
    """Parse tegrastats log and return power summary dict."""
    if not os.path.exists(log_path):
        return {"error": "no tegrastats log"}
    lines = []
    try:
        with open(log_path) as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception as e:
        return {"error": str(e)}

    if not lines:
        return {"error": "empty tegrastats log"}

    all_powers = []
    for line in lines:
        p = parse_tegrastats_line(line)
        if p:
            all_powers.append(p)

    if not all_powers:
        return {"raw_lines": lines[:3], "error": "could not parse power readings"}

    # Aggregate per sensor
    summary = {"num_samples": len(all_powers)}
    all_keys = set()
    for p in all_powers:
        all_keys.update(p.keys())
    for key in sorted(all_keys):
        vals = [p[key] for p in all_powers if key in p]
        if vals:
            is_temp = key.endswith("_temp_C")
            unit = "C" if is_temp else "mW"
            summary[key] = {
                f"mean_{unit}": round(sum(vals) / len(vals), 2) if is_temp else round(sum(vals) / len(vals)),
                f"min_{unit}": min(vals),
                f"max_{unit}": max(vals),
            }
    return summary


def time_single_problem(dataset, problem_id, device, precision="fp32"):
    """Time a single problem with timeout and error handling."""
    name = f"problem_{problem_id}"
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
        return {"problem_id": problem_id, "name": name, "status": "oom", "error": str(e)}
    except Exception as e:
        signal.alarm(0)
        return {"problem_id": problem_id, "name": name, "status": "error",
                "error": f"{type(e).__name__}: {e}"}


def main():
    if len(sys.argv) < 2:
        print("Usage: run_power_sweep.py <mode_id> [level] [start_id] [end_id]")
        sys.exit(1)

    mode_id = int(sys.argv[1])
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    start_id = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    end_id = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    precision = sys.argv[5] if len(sys.argv) > 5 else "fp32"
    mode_name = MODE_NAMES.get(mode_id, f"mode{mode_id}")

    results_dir = os.path.expanduser(f"~/thor_kernelbench_work/results/{HARDWARE_LABEL}")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"power_{mode_name}_level{level}_{start_id}-{end_id}.json")
    tegrastats_log = os.path.join(results_dir, f"tegrastats_{mode_name}.log")

    print(f"=== Power Sweep: {mode_name} (mode {mode_id}) ===")
    print(f"Level {level}, problems {start_id}-{end_id}, precision={precision}")
    print(f"Output: {out_path}")
    print()

    # 1. Set power mode
    set_power_mode(mode_id)
    current_mode_str = get_current_mode()
    print(f"  Confirmed: {current_mode_str}")

    # 2. Lock clocks
    lock_clocks()

    # 3. Thermal settle
    print(f"  Waiting {THERMAL_SETTLE_SECS}s for thermals to settle...")
    time.sleep(THERMAL_SETTLE_SECS)

    # 4. Snapshot clocks/thermals before run
    clock_state = get_clock_state()
    print(f"  GPU GPC: {clock_state.get('gpu_gpc_freq_mhz')}MHz  CPU0: {clock_state.get('cpu0_freq_mhz')}MHz")
    print(f"  tegrastats: {clock_state.get('tegrastats_once', 'N/A')[:120]}")
    print()

    # 5. Setup CUDA
    set_gpu_arch(["Blackwell"])
    device = torch.device("cuda:0")
    dataset = construct_kernelbench_dataset(level)
    num_problems = len(dataset)
    print(f"torch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Level {level}: {num_problems} problems, running {start_id}-{end_id}")
    print()

    # 6. Start tegrastats logging
    print(f"  Starting tegrastats (interval {TEGRASTATS_INTERVAL_MS}ms) → {tegrastats_log}")
    if os.path.exists(tegrastats_log):
        os.remove(tegrastats_log)
    tegrastats_proc = start_tegrastats(tegrastats_log)
    time.sleep(2)  # Let it write a few samples before benchmark starts

    # 7. Run timing
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

        # Incremental save
        snapshot = {
            "hardware": HARDWARE_LABEL,
            "power_mode_id": mode_id,
            "power_mode_name": mode_name,
            "level": level,
            "precision": precision,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "clock_state_before_run": clock_state,
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(snapshot, f, indent=2)

    wall_total = time.time() - wall_start

    # 8. Stop tegrastats and parse power
    time.sleep(2)  # capture a few samples after last kernel
    stop_tegrastats(tegrastats_proc)
    power_summary = summarize_power_log(tegrastats_log)
    print()
    print("=== Power Summary ===")
    for k, v in power_summary.items():
        print(f"  {k}: {v}")

    # 9. Final save with power data
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = len(results) - ok
    final = {
        "hardware": HARDWARE_LABEL,
        "power_mode_id": mode_id,
        "power_mode_name": mode_name,
        "level": level,
        "precision": precision,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "clock_state_before_run": clock_state,
        "power_summary": power_summary,
        "timing_summary": {
            "total": len(results),
            "ok": ok,
            "failed": fail,
            "wall_seconds": round(wall_total, 1),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    print()
    print(f"Done: {ok}/{len(results)} succeeded, {fail} failed, wall time {wall_total:.1f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
