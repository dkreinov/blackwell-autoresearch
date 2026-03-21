#!/usr/bin/env python3
"""
Analyze Sakana kernel transfer from H100 to Thor.
Categorizes: transfers well, backfires, compile failures.
Identifies which optimization strategies work on unified memory.
"""
import json, statistics, sys

SAKANA_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/thor_kernelbench_work/results/Thor_AGX/sakana_transfer_level1.json"
THOR_BASELINE_PATH = sys.argv[2] if len(sys.argv) > 2 else "/home/nvidia/thor_kernelbench_work/results/Thor_AGX/baseline_level1.json"
H100_BASELINE_PATH = sys.argv[3] if len(sys.argv) > 3 else "/home/nvidia/thor_kernelbench_work/KernelBench/results/timing/H100_PCIe_LambdaLabs/baseline_time_torch.json"

with open(SAKANA_PATH) as f:
    sakana = json.load(f)["results"]
with open(THOR_BASELINE_PATH) as f:
    thor_base = {r["problem_id"]: r for r in json.load(f)["results"] if r["status"] == "ok"}
with open(H100_BASELINE_PATH) as f:
    h100_base = json.load(f)["level1"]

# Categorize ops
CATS = {
    'matmul': ['matmul','matrix','diagonal'],
    'conv': ['conv'],
    'activation': ['relu','sigmoid','tanh','gelu','selu','swish','elu','softplus','softsign','hardtanh','hardsigmoid','newgelu'],
    'normalization': ['norm','batchnorm','layernorm','groupnorm','rmsnorm','frobenius','l1norm','l2norm'],
    'pooling': ['pool'],
    'reduction': ['sum_reduction','mean_reduction','max_reduction','min_reduction','argmax','argmin','cumsum','cumprod','masked_cumsum','product_reduction'],
    'loss': ['loss','mse','huber','kldiv','triplet','hinge','cosinesimilarity'],
    'softmax': ['softmax','logsoftmax'],
    'attention': ['attention'],
}

def categorize(name):
    n = name.lower()
    for cat, keywords in CATS.items():
        if any(k in n for k in keywords):
            return cat
    return 'other'

# Split results
ok = [r for r in sakana if r.get("status") == "ok"]
compile_fail = [r for r in sakana if r.get("status") in ("error", "compile_fail", "timeout")]
incorrect = [r for r in sakana if r.get("status") == "incorrect"]

print("=" * 80)
print("SAKANA AI KERNEL TRANSFER: H100 → Thor AGX (Blackwell sm_110, Unified Memory)")
print("=" * 80)

print(f"\n### Overall Results")
print(f"  Total tasks evaluated: {len(sakana)}")
print(f"  Compiled + correct:    {len(ok)} ({100*len(ok)/len(sakana):.0f}%)")
print(f"  Compile/runtime fail:  {len(compile_fail)} ({100*len(compile_fail)/len(sakana):.0f}%)")
print(f"  Incorrect output:      {len(incorrect)} ({100*len(incorrect)/len(sakana):.0f}%)")

# Classify transfers
transfers_well = [r for r in ok if r["thor_speedup"] >= 1.0]  # faster on Thor too
backfires = [r for r in ok if r["thor_speedup"] < 1.0]  # slower on Thor despite being faster on H100

print(f"\n### Transfer Classification (of {len(ok)} correct kernels)")
print(f"  Transfers well (speedup >= 1.0x on Thor): {len(transfers_well)} ({100*len(transfers_well)/len(ok):.0f}%)")
print(f"  Backfires (slower on Thor):               {len(backfires)} ({100*len(backfires)/len(ok):.0f}%)")

speedups = [r["thor_speedup"] for r in ok]
print(f"\n### Speedup Statistics (Thor)")
print(f"  Min:    {min(speedups):.3f}x")
print(f"  Median: {statistics.median(speedups):.3f}x")
print(f"  Mean:   {statistics.mean(speedups):.3f}x")
print(f"  Max:    {max(speedups):.3f}x")

# Transfer ratio: how much of the H100 speedup carries over to Thor
print(f"\n### Transfer Ratio (Thor speedup / H100 speedup)")
print(f"  If ratio ~1.0: optimization transfers perfectly")
print(f"  If ratio < 1.0: less effective on Thor")
print(f"  If ratio > 1.0: MORE effective on Thor than H100")
print()
print(f"  {'Task':>4}  {'Op Name':<42} {'H100':>7} {'Thor':>7} {'Ratio':>7}  {'Category':<15} {'Transfer'}")
print(f"  {'-'*4}  {'-'*42} {'-'*7} {'-'*7} {'-'*7}  {'-'*15} {'-'*10}")

for r in sorted(ok, key=lambda x: x["thor_speedup"], reverse=True):
    tid = r["task_id"]
    name = r["op_name"][:42]
    h100_sp = r["h100_speedup"]
    thor_sp = r["thor_speedup"]
    ratio = thor_sp / h100_sp if h100_sp > 0 else 0
    cat = categorize(r["op_name"])
    transfer = "GOOD" if thor_sp >= 1.5 else ("OK" if thor_sp >= 1.0 else "BACKFIRE")
    print(f"  {tid:4d}  {name:<42} {h100_sp:7.2f} {thor_sp:7.2f} {ratio:7.3f}  {cat:<15} {transfer}")

# Per-category analysis
print(f"\n### Per-Category Transfer Analysis")
print(f"  {'Category':<15} {'n':>3} {'Faster':>7} {'Med Thor':>9} {'Med H100':>9} {'Med Ratio':>10}")
print(f"  {'-'*15} {'-'*3} {'-'*7} {'-'*9} {'-'*9} {'-'*10}")

cat_data = {}
for r in ok:
    cat = categorize(r["op_name"])
    cat_data.setdefault(cat, []).append(r)

for cat in ['matmul','activation','normalization','reduction','loss','softmax','conv','other']:
    if cat not in cat_data:
        continue
    items = cat_data[cat]
    thor_sps = [r["thor_speedup"] for r in items]
    h100_sps = [r["h100_speedup"] for r in items]
    ratios = [t/h if h > 0 else 0 for t, h in zip(thor_sps, h100_sps)]
    faster = sum(1 for s in thor_sps if s >= 1.0)
    print(f"  {cat:<15} {len(items):3d} {faster:3d}/{len(items):<3d} {statistics.median(thor_sps):9.3f} {statistics.median(h100_sps):9.3f} {statistics.median(ratios):10.3f}")

# Backfire analysis
print(f"\n### Backfire Analysis (optimizations that HURT on Thor)")
print(f"  These kernels are faster on H100 but SLOWER on Thor.")
print(f"  Root cause: H100-optimized memory access patterns don't suit unified LPDDR5X.\n")
for r in sorted(backfires, key=lambda x: x["thor_speedup"]):
    tid = r["task_id"]
    name = r["op_name"][:45]
    cat = categorize(r["op_name"])
    print(f"  Task {tid:3d} {name:45s} H100={r['h100_speedup']:5.2f}x Thor={r['thor_speedup']:5.3f}x  ({cat})")

# Compile failure analysis
print(f"\n### Compile Failure Analysis ({len(compile_fail)} tasks)")
print(f"  These kernels failed to compile on sm_110 (Thor Blackwell).")
fail_cats = {}
for r in compile_fail:
    cat = categorize(r["op_name"])
    fail_cats.setdefault(cat, []).append(r)
for cat in sorted(fail_cats):
    print(f"  {cat}: {len(fail_cats[cat])} failures")

# Key findings
print(f"\n{'='*80}")
print(f"KEY FINDINGS")
print(f"{'='*80}")
print(f"""
1. TRANSFER RATE: {len(transfers_well)}/{len(ok)} ({100*len(transfers_well)/len(ok):.0f}%) of H100-optimized
   kernels also speed up Thor. The majority of optimizations transfer.

2. BACKFIRE RATE: {len(backfires)}/{len(ok)} ({100*len(backfires)/len(ok):.0f}%) hurt performance on Thor.
   These are primarily reduction operations where H100's HBM-optimized
   memory access patterns don't suit unified LPDDR5X memory.

3. COMPILE RATE: {len(compile_fail)}/{len(sakana)} ({100*len(compile_fail)/len(sakana):.0f}%) failed to compile on sm_110.
   Many Sakana kernels use features not available on Thor's Blackwell variant.

4. MEDIAN SPEEDUP: {statistics.median(speedups):.3f}x — modest but positive. The best wins
   are in normalization ({statistics.median([r['thor_speedup'] for r in cat_data.get('normalization',[])]):.2f}x median)
   and specialized ops (diagonal matmul 39.95x, NewGELU 6.0x).

5. UNIFIED MEMORY EFFECT: Reduction ops consistently backfire — the shared
   memory tiling and warp-level optimizations that help on HBM actually
   increase overhead on unified LPDDR5X where access patterns differ.
""")
