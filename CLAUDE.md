# KernelBench Project

## SSH Access

### nvidia-thor-01
- **Host:** nvidia-thor-01
- **User:** nvidia
- **Password:** `<device-default>`
- **OS:** Ubuntu 24.04.3 LTS (Linux 6.8.12-tegra aarch64) — NVIDIA Jetson/Tegra platform

```bash
ssh nvidia@nvidia-thor-01
# password: <device-default>
```

Working directory on the remote host: `~/thor_kernelbench_work`

## Thor Agent

A helper script is deployed on Thor for all CUDA kernel operations. **Use the agent instead of raw commands.**

```bash
ssh -o StrictHostKeyChecking=no nvidia@nvidia-thor-01 \
  "bash ~/thor_kernelbench_work/thor_agent.sh <command>"
```

| Command | What it does |
|---------|-------------|
| `health` | GPU info, file check, nvcc, power mode |
| `read-problem <id>` | Show KernelBench problem code |
| `baseline <id>` | Show baseline timing |
| `eval-kernel <file> <id>` | Compile + correctness + benchmark a kernel |
| `power [sec]` | Sample tegrastats power readings |
| `list-problems` | All Level 1 problems with baselines |
| `list-kernels` | Saved kernel files |

## Data Integrity

NEVER add assumptions, speculation, interpretations, or unverified calculations to program.md, findings.md, or any project file. This is engineering, not research.

- `program.md` contains ONLY: verified hardware specs (with source command or doc reference), measured data (with methodology), and protocol instructions
- `findings.md` contains ONLY: measured experiment results (kernel version, timing, speedup) and observed failures with their error output
- If a fact cannot be verified by running a command on Thor or citing an official document, it does not belong in any file
- Do not calculate theoretical limits (bandwidth floors, roofline estimates) and present them as facts
- Do not write "room for optimization", "bandwidth-bound", "at the floor", or similar conclusions — the benchmark is the only oracle

## Autoresearch

To run the autonomous CUDA kernel optimization loop: `/thor-autoresearch`

Key files:
- `program.md` — Thor hardware facts, unified memory behavior, transfer study data, loop protocol
- `findings.md` — Agent-writable research log (experiments, dead ends, observations)
- `thor_agent.sh` — Remote helper script (deployed on Thor)
- `.claude/commands/thor-autoresearch.md` — Slash command definition
