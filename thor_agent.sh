#!/bin/bash
# Thor Agent — remote helper for LLM-driven CUDA kernel optimization
# Usage: bash thor_agent.sh <command> [args...]
#
# Deploy: scp to nvidia@nvidia-thor-01:~/thor_kernelbench_work/thor_agent.sh
# Invoke: ssh nvidia@nvidia-thor-01 "bash ~/thor_kernelbench_work/thor_agent.sh <cmd>"

set -euo pipefail

WORK="$HOME/thor_kernelbench_work"
VENV="$WORK/venv/bin"
RESULTS="$WORK/results/Thor_AGX"
BASELINE="$RESULTS/baseline_level1.json"
KERNELS_DIR="$WORK/kernels"  # agent writes kernels here

export PATH="$VENV:/usr/local/cuda-13.0/bin:/usr/bin:/bin"
export TORCH_CUDA_ARCH_LIST="11.0"

# ── helpers ──────────────────────────────────────────────
ok()   { echo "[OK]   $*"; }
fail() { echo "[FAIL] $*"; }
info() { echo "[INFO] $*"; }

# ── commands ─────────────────────────────────────────────
cmd_health() {
    echo "=== Thor Agent Health Check ==="
    echo ""
    echo "--- GPU ---"
    "$VENV/python3" -c "
import torch
p = torch.cuda.get_device_properties(0)
print(f'  Name: {p.name}')
print(f'  Compute: {p.major}.{p.minor} (sm_{p.major}{p.minor}0)')
print(f'  SMs: {p.multi_processor_count}')
print(f'  L2 Cache: {p.L2_cache_size // 1024 // 1024} MB')
print(f'  Shared Mem/Block: {p.shared_memory_per_block // 1024} KB')
print(f'  torch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
" 2>&1
    echo ""
    echo "--- Files ---"
    [ -f "$WORK/program.md" ] && ok "program.md ($(wc -l < "$WORK/program.md") lines)" || fail "program.md missing"
    [ -f "$BASELINE" ] && ok "baseline_level1.json exists" || fail "baseline missing"
    [ -f "$WORK/findings.md" ] && ok "findings.md ($(wc -l < "$WORK/findings.md") lines)" || info "findings.md not yet created"
    echo ""
    echo "--- nvcc ---"
    nvcc --version 2>&1 | grep "release" || fail "nvcc not found"
    echo ""
    echo "--- Power Mode ---"
    sudo nvpmodel -q 2>/dev/null || info "nvpmodel unavailable (configure passwordless sudo)"
    echo ""
    echo "=== Health check complete ==="
}

cmd_read_problem() {
    # Read a KernelBench Level 1 problem by ID
    local pid="${1:?Usage: thor_agent.sh read-problem <problem_id>}"
    "$VENV/python3" -c "
import json, sys
from kernelbench.dataset import construct_kernelbench_dataset
ds = construct_kernelbench_dataset(1, source='local')
problem = ds.get_problem_by_id($pid)
print(f'Problem {$pid}: {problem.name}')
print(f'---CODE---')
print(problem.code)
" 2>&1
}

cmd_baseline() {
    # Show baseline timing for a problem
    local pid="${1:?Usage: thor_agent.sh baseline <problem_id>}"
    "$VENV/python3" -c "
import json
with open('$BASELINE') as f:
    data = json.load(f)
for r in data['results']:
    if r['problem_id'] == $pid:
        print(f'Problem {$pid}: {r[\"name\"]}')
        print(f'  Status: {r[\"status\"]}')
        if r['status'] == 'ok':
            print(f'  Mean: {r[\"mean\"]:.2f} ms')
            print(f'  Std:  {r[\"std\"]:.3f} ms')
        break
else:
    print(f'Problem {$pid} not found in baseline')
" 2>&1
}

cmd_eval_kernel() {
    # Evaluate a kernel file against the reference
    # Kernel file must be a Python file with ModelNew class
    local kernel_file="${1:?Usage: thor_agent.sh eval-kernel <kernel_file.py> <problem_id>}"
    local pid="${2:?Usage: thor_agent.sh eval-kernel <kernel_file.py> <problem_id>}"
    local run_log="$RESULTS/eval_run.log"

    if [ ! -f "$kernel_file" ]; then
        # Try relative to work dir
        if [ -f "$WORK/$kernel_file" ]; then
            kernel_file="$WORK/$kernel_file"
        elif [ -f "$KERNELS_DIR/$(basename "$kernel_file")" ]; then
            kernel_file="$KERNELS_DIR/$(basename "$kernel_file")"
        fi
    fi
    [ -f "$kernel_file" ] || { fail "Kernel file not found: $kernel_file"; return 1; }

    "$VENV/python3" -c "
import json, sys, time, torch
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref
from kernelbench.utils import set_gpu_arch

torch.cuda.set_device(0)

# Load problem
ds = construct_kernelbench_dataset(1, source='local')
problem = ds.get_problem_by_id($pid)
ref_code = problem.code

# Load kernel
with open('$kernel_file') as f:
    custom_code = f.read()

# Load baseline
with open('$BASELINE') as f:
    base = json.load(f)
base_ms = next((r['mean'] for r in base['results'] if r['problem_id'] == $pid and r['status'] == 'ok'), None)

print(f'Problem: {$pid} ({problem.name})')
print(f'Baseline: {base_ms:.2f} ms' if base_ms else 'Baseline: N/A')
print(f'Kernel: $kernel_file')
print()

t0 = time.time()
try:
    result = eval_kernel_against_ref(
        original_model_src=ref_code,
        custom_model_src=custom_code,
        measure_performance=True,
        timing_method='cuda_event',
        num_correct_trials=5,
        num_perf_trials=100,
        device=torch.device('cuda:0'),
        backend='cuda',
        precision=torch.float32,
        verbose=False,
        check_for_excessive_speedup=False,
    )
    elapsed = time.time() - t0

    if not result.compiled:
        print(f'[COMPILE_ERROR] {getattr(result, \"runtime_error\", \"unknown\")}'[:500])
    elif not result.correctness:
        print(f'[INCORRECT] max_diff={getattr(result, \"max_diff\", \"?\")}')
    else:
        custom_ms = result.runtime
        speedup = base_ms / custom_ms if base_ms and custom_ms > 0 else 0
        print(f'[OK]')
        print(f'  Custom:  {custom_ms:.2f} ms')
        print(f'  Speedup: {speedup:.3f}x')
        print(f'  Time:    {elapsed:.1f}s')
except Exception as e:
    print(f'[ERROR] {type(e).__name__}: {e}'[:500])
" 2>&1
}

cmd_power() {
    # Read current power state
    local duration="${1:-3}"
    local log="/tmp/thor_agent_power.log"
    tegrastats --stop 2>/dev/null || true
    sleep 0.5
    rm -f "$log"
    tegrastats --start --interval 1000 --logfile "$log"
    sleep "$duration"
    tegrastats --stop 2>/dev/null || true
    sleep 0.5
    if [ -f "$log" ]; then
        echo "=== Power (${duration}s sample) ==="
        # Extract VDD_GPU readings
        grep -oP 'VDD_GPU \K\d+mW/\d+mW' "$log" | tail -3
        echo "---"
        # Extract temps
        grep -oP 'gpu@[\d.]+C' "$log" | tail -1
        grep -oP 'cpu@[\d.]+C' "$log" | tail -1
    else
        fail "No tegrastats data"
    fi
}

cmd_list_problems() {
    # List all Level 1 problems with baseline timings
    "$VENV/python3" -c "
import json
from kernelbench.dataset import construct_kernelbench_dataset
ds = construct_kernelbench_dataset(1, source='local')
with open('$BASELINE') as f:
    base = {r['problem_id']: r for r in json.load(f)['results']}
print(f'{'ID':>3}  {'Name':<55} {'Baseline':>10} {'Status':>8}')
print('-' * 82)
for pid in ds.get_problem_ids():
    p = ds.get_problem_by_id(pid)
    b = base.get(pid, {})
    ms = f\"{b['mean']:.1f}ms\" if b.get('status') == 'ok' else 'N/A'
    st = b.get('status', '?')
    print(f'{pid:3d}  {p.name[:55]:<55} {ms:>10} {st:>8}')
" 2>&1
}

cmd_list_kernels() {
    # List saved kernel files
    mkdir -p "$KERNELS_DIR"
    if ls "$KERNELS_DIR"/*.py 1>/dev/null 2>&1; then
        for f in "$KERNELS_DIR"/*.py; do
            echo "  $(basename "$f") ($(wc -l < "$f") lines)"
        done
    else
        info "No kernels saved yet. Dir: $KERNELS_DIR"
    fi
}

cmd_help() {
    cat <<EOF
Thor Agent — remote helper for CUDA kernel autoresearch

DIAGNOSTIC:
  health                       GPU info, file check, nvcc, power mode
  list-problems                All Level 1 problems with baseline timings
  list-kernels                 Saved kernel files

READ:
  read-problem <id>            Show KernelBench problem code
  baseline <id>                Show baseline timing for a problem

EVALUATE:
  eval-kernel <file> <id>      Compile + correctness + benchmark a kernel

POWER:
  power [duration_sec]         Sample tegrastats (default 3s)

EXAMPLES:
  bash thor_agent.sh health
  bash thor_agent.sh read-problem 40
  bash thor_agent.sh baseline 40
  bash thor_agent.sh eval-kernel kernels/p40_layernorm_v1.py 40
  bash thor_agent.sh power 5
EOF
}

# ── dispatch ─────────────────────────────────────────────
case "${1:-help}" in
    health)          cmd_health ;;
    read-problem)    cmd_read_problem "${2:-}" ;;
    baseline)        cmd_baseline "${2:-}" ;;
    eval-kernel)     cmd_eval_kernel "${2:-}" "${3:-}" ;;
    power)           cmd_power "${2:-3}" ;;
    list-problems)   cmd_list_problems ;;
    list-kernels)    cmd_list_kernels ;;
    help|--help|-h)  cmd_help ;;
    *)               fail "Unknown command: $1"; echo ""; cmd_help; exit 1 ;;
esac
