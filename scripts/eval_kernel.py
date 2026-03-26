#!/usr/bin/env python3
"""
Per-problem kernel evaluator with round-robin schedule tracking.

Usage:
    python scripts/eval_kernel.py --pid 25 --kernel kernels/p25_swish.py
    python scripts/eval_kernel.py --pid 25 --kernel kernels/fp16/p25_swish.py --precision fp16
    python scripts/eval_kernel.py --status       # show all problems' current best + schedule state
    python scripts/eval_kernel.py --next         # print which pid to work on now (handles switching)
    python scripts/eval_kernel.py --clean        # discard any stale _candidate.py files + log them
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
RESULTS_DIR = WORK / "results" / "Thor_AGX"
RESULTS_FILE = RESULTS_DIR / "kernel_results.json"
SCHEDULE_FILE = WORK / "schedule.json"
STATE_FILE = WORK / "loop_state.json"
KERNELS_DIR = WORK / "kernels"
FINDINGS_FILE = WORK / "findings.md"

THOR_HOST = os.environ.get("THOR_HOST", "nvidia-thor-01")
THOR_USER = os.environ.get("THOR_USER", "nvidia")
SSH = f"ssh -o StrictHostKeyChecking=no {THOR_USER}@{THOR_HOST}"
AGENT = "bash ~/thor_kernelbench_work/thor_agent.sh"
REMOTE_KERNELS = "~/thor_kernelbench_work/kernels"

MAX_EXPERIMENT_S = 120


def load_schedule() -> dict:
    return json.loads(SCHEDULE_FILE.read_text())


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {}


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    # Bootstrap: start at first problem in schedule
    schedule = load_schedule()
    first_pid = schedule["order"][0]
    return {"current_pid": first_pid, "started_at": now_iso(), "experiment_count": 0}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_s(started_at_iso: str) -> float:
    started = datetime.fromisoformat(started_at_iso)
    return (datetime.now(timezone.utc) - started).total_seconds()


def next_pid(current_pid: int, schedule: dict) -> int:
    order = schedule["order"]
    idx = order.index(current_pid)
    return order[(idx + 1) % len(order)]


def scp_to_thor(local_path: str, remote_name: str):
    cmd = f'scp -o StrictHostKeyChecking=no "{local_path}" {THOR_USER}@{THOR_HOST}:{REMOTE_KERNELS}/{remote_name}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr.strip()}")


def results_file_for(precision: str) -> Path:
    if precision == "fp32":
        return RESULTS_DIR / "kernel_results.json"
    return RESULTS_DIR / f"kernel_results_{precision}.json"


def run_eval(remote_name: str, pid: int, precision: str = "fp32") -> dict:
    cmd = f'{SSH} "{AGENT} eval-kernel kernels/{remote_name} {pid} {precision}"'
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=MAX_EXPERIMENT_S
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "custom_ms": None, "speedup": None, "output": "TIMEOUT"}

    info = {"status": "error", "custom_ms": None, "speedup": None, "output": output.strip()}

    if "[OK]" in output:
        info["status"] = "ok"
        for line in output.splitlines():
            if "Custom:" in line:
                try:
                    info["custom_ms"] = float(line.split(":")[1].strip().replace("ms", "").strip())
                except ValueError:
                    pass
            if "Speedup:" in line:
                try:
                    info["speedup"] = float(line.split(":")[1].strip().replace("x", "").strip())
                except ValueError:
                    pass
    elif "[COMPILE_ERROR]" in output:
        info["status"] = "compile_error"
    elif "[INCORRECT]" in output:
        info["status"] = "incorrect"

    return info


def clean_candidates(log_to_findings: bool = True) -> list[str]:
    """Delete any leftover _candidate.py files. Returns list of deleted files."""
    candidates = list(KERNELS_DIR.glob("*_candidate.py"))
    if not candidates:
        return []

    deleted = []
    for f in candidates:
        f.unlink()
        deleted.append(f.name)
        print(f"  CLEANED: {f.name} (stale candidate discarded)")

    if log_to_findings and deleted:
        note = f"\n- **Dirty state cleanup** ({now_iso()[:10]}): discarded stale candidates: {', '.join(deleted)}\n"
        findings = FINDINGS_FILE.read_text()
        FINDINGS_FILE.write_text(findings + note)

    return deleted


def cmd_clean():
    deleted = clean_candidates(log_to_findings=True)
    if deleted:
        print(f"Cleaned {len(deleted)} stale candidate(s): {deleted}")
    else:
        print("No stale candidates found.")


def cmd_next():
    """Print current pid, handle switching if time elapsed."""
    schedule = load_schedule()
    state = load_state()

    current_pid = state["current_pid"]
    time_limit = schedule["time_per_activation_s"]
    elapsed = elapsed_s(state["started_at"])
    remaining = time_limit - elapsed

    if remaining <= 0:
        # Time's up — switch
        old_pid = current_pid
        current_pid = next_pid(old_pid, schedule)
        names = schedule["names"]

        print(f"SWITCH: {names[str(old_pid)]} (pid={old_pid}) time elapsed ({elapsed:.0f}s >= {time_limit}s)")

        # Clean up any dirty candidates before switching
        deleted = clean_candidates(log_to_findings=True)
        if deleted:
            print(f"  Cleaned dirty candidates: {deleted}")

        # Update state
        state["current_pid"] = current_pid
        state["started_at"] = now_iso()
        state["experiment_count"] = 0
        save_state(state)

        print(f"NOW ON: {names[str(current_pid)]} (pid={current_pid}) — 0s elapsed")
    else:
        names = schedule["names"]
        print(f"CURRENT: {names[str(current_pid)]} (pid={current_pid}) — {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")

    return current_pid


def cmd_eval(pid: int, kernel_path: str, force_pid: bool = False, precision: str = "fp32"):
    # Schedule enforcement: only eval the currently scheduled pid
    state = load_state()
    if pid != state["current_pid"] and not force_pid:
        names = load_schedule().get("names", {})
        print(f"BLOCKED: pid={pid} ({names.get(str(pid), '?')}) is not the scheduled pid={state['current_pid']} ({names.get(str(state['current_pid']), '?')})")
        print(f"  Use --force-pid to override (requires explicit user intent)")
        sys.exit(1)

    kernel_path = Path(kernel_path)
    if not kernel_path.exists():
        print(f"ERROR: kernel file not found: {kernel_path}")
        sys.exit(1)

    remote_name = f"eval_pid{pid}_{kernel_path.stem}.py"

    print(f"Evaling pid={pid} kernel={kernel_path.name} precision={precision} ...", flush=True)
    scp_to_thor(str(kernel_path), remote_name)
    result = run_eval(remote_name, pid, precision)

    if result["status"] == "ok":
        print(f"OK  {result['custom_ms']:.2f}ms  {result['speedup']:.3f}x")
    else:
        print(f"FAIL  {result['status']}")
        if result.get("output"):
            for line in result["output"].splitlines()[-5:]:
                print(f"  {line}")

    # Increment experiment count in state
    state = load_state()
    state["experiment_count"] = state.get("experiment_count", 0) + 1
    save_state(state)

    return result


def cmd_status():
    results = load_results()
    schedule = load_schedule()
    names = schedule.get("names", {})
    state = load_state()

    # Check for stale candidates
    candidates = list(KERNELS_DIR.glob("*_candidate.py"))
    if candidates:
        print(f"\n*** DIRTY STATE: {len(candidates)} stale candidate(s) found:")
        for c in candidates:
            print(f"   {c.name}  — run --clean to discard")

    elapsed = elapsed_s(state["started_at"])
    time_limit = schedule["time_per_activation_s"]
    current_pid = state["current_pid"]
    print(f"\nSchedule: {time_limit}s per activation | Current: pid={current_pid} ({names.get(str(current_pid),'?')}) | Elapsed: {elapsed:.0f}s | Remaining: {max(0, time_limit-elapsed):.0f}s")
    print(f"Experiments this slot: {state.get('experiment_count', 0)}\n")

    if not results:
        print("No results yet.")
        return

    print(f"{'PID':<6} {'Name':<18} {'Baseline':>10} {'Best':>10} {'Speedup':>10} {'Iters':>6}")
    print("-" * 65)
    for pid_str in sorted(results.keys(), key=int):
        r = results[pid_str]
        name = names.get(pid_str, r.get("name", "?"))
        marker = " <--" if int(pid_str) == current_pid else ""
        print(f"{pid_str:<6} {name:<18} {r.get('baseline_ms',0):>9.1f}ms {r.get('best_ms',0):>9.2f}ms {r.get('best_speedup',0):>9.3f}x {r.get('iterations',0):>6}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Per-problem kernel evaluator")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--kernel", type=str)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--next", action="store_true", help="Print current pid, switch if time elapsed")
    parser.add_argument("--clean", action="store_true", help="Discard stale candidate files")
    parser.add_argument("--force-pid", action="store_true", help="Override schedule enforcement (user-directed only)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                        help="Precision for eval (default: fp32)")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.next:
        cmd_next()
    elif args.clean:
        cmd_clean()
    elif args.pid and args.kernel:
        cmd_eval(args.pid, args.kernel, force_pid=args.force_pid, precision=args.precision)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
