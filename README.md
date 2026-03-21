# KernelBench on NVIDIA Jetson AGX Thor (Blackwell)

**First KernelBench baseline and power characterization on a Jetson/Tegra SoC with Blackwell architecture (sm_110).**

| | |
|---|---|
| **Hardware** | NVIDIA Jetson AGX Thor -- Blackwell, sm_110, compute capability 11.0 |
| **Memory** | 125 GB unified (LPDDR5X, ATS -- shared CPU/GPU, no dedicated VRAM) |
| **Software** | PyTorch 2.9.1+cu130, CUDA 13.0, Python 3.12.3 |
| **Benchmark** | [KernelBench](https://github.com/ScalingIntelligence/KernelBench) v0.2.0 -- Level 1 (100 single-kernel operators) |
| **Precision** | fp32, cuda_event timing (5 warmup + 100 trials per problem) |

---

## Key Results

### Baseline Timing (Level 1 -- 100 problems)

| Mode | GPU Clock | Pass Rate | Median (ms) | Mean (ms) | Wall Time |
|------|-----------|-----------|-------------|-----------|-----------|
| **MAXN** | 1575 MHz | **99/100** | 51.5 | 59.0 | 920.9 s |
| **120W** | 1386 MHz | **99/100** | 54.9 | 61.6 | 948.6 s |

One failure in both modes: problem #95 (CrossEntropyLoss) -- `nll_loss` kernel not compiled for sm_110 in PyTorch 2.9.1+cu130. This is a PyTorch issue, not a Thor issue.

### Power Characterization (MAXN vs 120W)

| Metric | MAXN | 120W | Delta |
|--------|------|------|-------|
| GPU power (avg) | 22.1 W | 20.8 W | -5.7% |
| GPU temp (peak) | 66.6 C | 63.2 C | -3.4 C |
| Median latency | 51.5 ms | 54.9 ms | +6.6% |
| **Perf / Watt** | **4.88** | **5.02** | **+2.9%** |

**120W mode is 2.9% more efficient per watt** despite being 6.6% slower in median latency. Memory-bandwidth-bound operations (activations, reductions) are unaffected by the clock reduction; compute-bound operations (matmul, attention) slow 12-13%.

### Performance by Operator Category

| Category | n | MAXN median | 120W median | Slowdown | Bottleneck |
|----------|---|-------------|-------------|----------|------------|
| matmul | 18 | 23.1 ms | 26.0 ms | +12.6% | Compute |
| conv | 35 | 29.2 ms | 32.3 ms | +10.6% | Mixed |
| activation | 13 | 56.7 ms | 56.7 ms | 0.0% | Bandwidth |
| normalization | 8 | 112.5 ms | 115.0 ms | +2.2% | Bandwidth |
| pooling | 6 | 72.6 ms | 77.4 ms | +6.5% | Mixed |
| reduction | 11 | 65.0 ms | 64.7 ms | -0.5% | Bandwidth |
| softmax | 2 | 105.0 ms | 104.6 ms | -0.4% | Bandwidth |
| loss | 5 | 69.1 ms | 70.1 ms | +1.4% | Bandwidth |
| attention | 1 | 143.0 ms | 162.0 ms | +13.3% | Compute |

---


---

## Phase 6 — LLM Kernel Generation Research

### Transfer Study: Sakana AI H100-Optimized Kernels on Thor

We evaluated 63 kernels from the [Sakana AI CUDA Engineer Archive](https://huggingface.co/SakanaAI) — kernels optimized for H100 (HBM, PCIe) — on Thor's unified LPDDR5X memory to understand cross-architecture transfer.

| Metric | Count | % |
|--------|-------|---|
| Compiled + correct | 32 | 51% |
| Compile / runtime fail | 26 | 41% |
| Incorrect output | 5 | 8% |

Of the 32 correct kernels:

| Outcome | Count | % | Median speedup |
|---------|-------|---|----------------|
| Transfers well (≥1.0x faster on Thor) | 23 | 72% | — |
| Backfires (slower on Thor) | 9 | 28% | — |
| **All correct** | **32** | — | **1.195x** |

**Per-category breakdown:**

| Category | n | Faster | Median speedup | Key finding |
|----------|---|--------|----------------|-------------|
| Normalization | 4 | 4/4 | **2.35x** | Best transfer — bandwidth-bound, L2 locality helps |
| Activation | 10 | 9/10 | **1.23x** | Near-perfect transfer — element-wise, no memory pattern dependency |
| Reduction | 9 | 2/9 | **0.75x** | Consistently backfires — HBM tiling hurts on unified LPDDR5X |

**Notable results:**
- Problem 12 (Diagonal Matmul): 39.95x speedup on Thor — diagonal structure exploits unified memory better than HBM
- Problem 40 (LayerNorm): 2.57x on Thor vs 8.60x on H100 — still 2.57x, normalization transfers well
- Problem 47 (Sum Reduction): 0.38x on Thor — shared-memory warp reduction adds overhead on unified memory

**Key insight:** H100 reduction kernels optimize for HBM coalescing and warp-level synchronization — these patterns add overhead on Thor's unified LPDDR5X where the cost model is different.

### Autoresearch Infrastructure

An LLM kernel autoresearch loop is set up following the [VMP/Mobileye pattern](https://x.com/karpathy/status/2015883857489522876):

| File | Purpose |
|------|---------|
| `program.md` | Facts-only hardware reference for the agent (Thor specs, memory model, empirical data) |
| `thor_agent.sh` | Remote helper — wraps `eval_kernel_against_ref`, tegrastats, dataset access |
| `findings.md` | Agent-writable research log — priority problems, completed experiments, dead ends |
| `.claude/commands/thor-autoresearch.md` | Slash command to invoke the research loop in Claude Code |

The loop runs directly in Claude Code (no separate API script) — the agent reads `program.md`, picks a problem, writes a CUDA kernel, SCPs it to Thor, evals via `thor_agent.sh`, and iterates.

**Priority targets** (highest baseline time = most room to improve):

| ID | Problem | Baseline (ms) | Transfer Study |
|----|---------|--------------|----------------|
| 30 | Softsign | 197.0 | Sakana 1.78x ✓ |
| 38 | L1Norm | 193.0 | Sakana 2.42x ✓ |
| 76 | conv_standard_1D_dilated_strided | 181.0 | Compile fail |
| 36 | RMSNorm | 172.0 | Sakana 2.28x ✓ |
| 97 | ScaledDotProductAttention | 143.0 | Not in Sakana |

---

## Why This Matters

1. **First KernelBench results on Jetson/Blackwell.** No prior published baselines exist for sm_110 on a Tegra SoC. This data establishes the performance floor for LLM-generated kernel optimization on edge Blackwell hardware.

2. **Unified memory changes the game.** Thor's ATS architecture eliminates OOMs and PCIe transfer overhead. Timing variance is remarkably low (35 of 99 problems have std < 1.0 ms), making Thor well-suited for reproducible benchmarking.

3. **Power efficiency is measurable.** The 120W mode's 2.9% efficiency gain with only 6.6% latency cost shows that for bandwidth-bound workloads, running at max clocks wastes power. This is directly relevant for edge deployment optimization.

---

## Repository Structure

```
.
|-- README.md                           # This file
|-- LICENSE                             # MIT License
|-- program.md                          # Facts-only hardware reference for autoresearch agent
|-- findings.md                         # Agent research log (updated per-experiment)
|-- thor_agent.sh                       # Remote helper script for kernel eval on Thor
|-- .claude/commands/thor-autoresearch.md  # Slash command for the autoresearch loop
|-- reports/
|   |-- 00_thor_env_report.md           # Raw environment inventory (nvidia-smi, lscpu, etc.)
|   |-- 01_env_and_repo_audit.md        # Phase 1: compatibility analysis and patch plan
|   |-- 02_thor_compatibility_patches.md # Patches needed (pyproject.toml, cu130 wheel)
|   |-- 03_smoke_test.md                # Phase 2: CUDA + KernelBench smoke tests
|   |-- 04_baseline_pilot.md            # Phase 3: full Level 1 baseline (99/100 pass)
|   |-- 05_power_characterization.md    # Phase 4b: MAXN vs 120W power/timing analysis
|-- scripts/
|   |-- run_baseline_timing.py          # Baseline timing with timeout/OOM handling
|   |-- run_power_sweep.py              # Power mode sweep with tegrastats monitoring
|   |-- analyze_baseline.py             # Stats and per-category analysis
|   |-- compare_thor_h100.py            # Thor vs H100 PCIe baseline comparison
|   |-- eval_sakana_kernels.py          # Transfer study: Sakana H100 kernels on Thor sm_110
|   |-- analyze_transfer.py             # Transfer analysis — per-category breakdown
|   |-- run_agentic_eval.py             # Two-pass agentic kernel eval (reproducible benchmark)
|-- results/
|   |-- Thor_AGX/
|       |-- baseline_level1.json        # MAXN baseline (99 problems, no power data)
|       |-- power_MAXN_level1_1-100.json  # MAXN with tegrastats power monitoring
|       |-- power_120W_level1_1-100.json  # 120W with tegrastats power monitoring
|       |-- sakana_transfer_level1.json # Transfer study: 63 Sakana kernels on Thor
|-- kernels/                            # Agent-written CUDA kernels (populated by autoresearch loop)
```

---

## How to Reproduce

### Prerequisites

- NVIDIA Jetson AGX Thor (or any Blackwell GPU with CUDA 13.0+)
- Python 3.10+ with venv

### Setup

```bash
# 1. Create isolated environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with sm_110 support (cu130 required for Blackwell)
pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130

# 3. Install KernelBench (with relaxed Python pin)
git clone https://github.com/ScalingIntelligence/KernelBench.git
# Edit KernelBench/pyproject.toml: change requires-python = "==3.10.*" to ">=3.10"
pip install -e KernelBench --no-deps
pip install numpy tqdm packaging ninja tomli einops python-dotenv \
  pydra-config tabulate datasets transformers openai litellm

# 4. Add CUDA to PATH
export PATH=$PATH:/usr/local/cuda-13.0/bin
```

### Run baseline timing

```bash
# Single problem smoke test
python scripts/run_baseline_timing.py 1 1 1  # Level 1, problem 1 only

# Full Level 1 baseline (100 problems, ~15 min)
python scripts/run_baseline_timing.py 1 1 100

# Analyze results
python scripts/analyze_baseline.py results/Thor_AGX/baseline_level1.json
```

### Run power sweep

```bash
# Requires sudo for nvpmodel/jetson_clocks
python scripts/run_power_sweep.py 0  # MAXN mode, full Level 1
python scripts/run_power_sweep.py 1  # 120W mode, full Level 1
# Modes 2 (90W) and 3 (70W) require a reboot -- see reports/05_power_characterization.md
```

---

## Thor-Specific Observations

**Unified Memory (ATS)**
- No OOMs: GPU shares the full 125 GB system pool. All 100 problems run without memory issues.
- No PCIe jitter: data is not copied between host and device. Timing variance is exceptionally low.
- Lower bandwidth than HBM: memory-bound kernels (activations, normalization) are slower relative to compute-bound ones compared to discrete GPUs.

**Blackwell sm_110 on PyTorch 2.9.1+cu130**
- 99/100 Level 1 kernels work. The one failure (CrossEntropyLoss `nll_loss`) is a missing kernel dispatch, not an architecture limitation.
- The cu126 wheel does NOT support sm_110 -- cu130 is required.
- No aarch64-specific code changes were needed in KernelBench itself.

**Power Modes**
- MAXN and 120W switch without reboot (same GPU power-gating mask).
- 90W and 70W require a reboot (different GPU partition configuration).
- `jetson_clocks` locks frequencies at the mode's maximum for stable benchmarking.

---

## Compatibility Patches

Only two changes were needed to run KernelBench on Thor:

1. **`pyproject.toml`**: `requires-python = "==3.10.*"` changed to `">=3.10"` (Thor ships Python 3.12.3; no 3.10-specific syntax in codebase).
2. **PyTorch wheel**: must use `cu130` index (not `cu126`) to get sm_110 compiled kernels.

See [`reports/02_thor_compatibility_patches.md`](reports/02_thor_compatibility_patches.md) for full details.

---

## Roadmap

- [ ] **Level 2 & 3 baselines** -- Run Level 2 (operator fusion, 100 problems) and Level 3 (full architectures, 50 problems)
- [x] **LLM kernel generation infrastructure** -- thor_agent.sh + program.md + findings.md + autoresearch slash command ready; transfer study (72% of H100 kernels work on Thor) complete
- [ ] **LLM autoresearch loop** -- Run `/thor-autoresearch` to iteratively optimize high-baseline-time problems (Softsign 197ms, L1Norm 193ms, ...)
- [ ] **torch.compile baselines** -- Compare eager vs Inductor-compiled performance
- [ ] **H100/L40S comparison** -- Cross-reference with published datacenter baselines
- [ ] **GPU clock sweep** -- Fine-grained clock-vs-performance curves within MAXN mode
- [ ] **Publish findings** -- Blog post: "First KernelBench Results on Jetson AGX Thor"

---

## Acknowledgments

- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) by the Scaling Intelligence team at Stanford
- NVIDIA for the Jetson AGX Thor developer kit

---

## License

[MIT](LICENSE)
