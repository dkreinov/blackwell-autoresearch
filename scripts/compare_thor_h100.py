#!/usr/bin/env python3
"""Compare Thor AGX vs H100 PCIe baseline timings from KernelBench Level 1."""
import json, statistics, sys

# Paths
H100_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/thor_kernelbench_work/KernelBench/results/timing/H100_PCIe_LambdaLabs/baseline_time_torch.json"
THOR_PATH = sys.argv[2] if len(sys.argv) > 2 else "/home/nvidia/thor_kernelbench_work/results/Thor_AGX/baseline_level1.json"

# Load H100 (format: {"level1": {"filename": {"mean": X, ...}}})
with open(H100_PATH) as f:
    h100_data = json.load(f)["level1"]

# Load Thor (format: {"results": [{"problem_id": N, "name": "filename", "mean": X, ...}]})
with open(THOR_PATH) as f:
    thor_results = json.load(f)["results"]
thor_by_name = {r["name"]: r for r in thor_results if r["status"] == "ok"}

# Categorize
CATS = {
    'matmul': ['matmul','matrix'],
    'conv': ['conv'],
    'activation': ['relu','sigmoid','tanh','gelu','selu','swish','elu','softplus','softsign','hardtanh','hardsigmoid','newgelu'],
    'normalization': ['norm','batchnorm','layernorm','groupnorm','rmsnorm','frobenius','l1norm','l2norm'],
    'pooling': ['pool'],
    'reduction': ['sum_reduction','mean_reduction','max_reduction','min_reduction','argmax','argmin','cumsum','cumprod','masked_cumsum'],
    'loss': ['loss','mse','huber','kldiv','triplet','hinge'],
    'softmax': ['softmax','logsoftmax'],
    'attention': ['attention'],
}

def categorize(name):
    n = name.lower()
    for cat, keywords in CATS.items():
        if any(k in n for k in keywords):
            return cat
    return 'other'

# Match and compare
matched = []
for h100_name, h100_stats in sorted(h100_data.items()):
    if h100_name in thor_by_name:
        thor = thor_by_name[h100_name]
        ratio = thor["mean"] / h100_stats["mean"]
        matched.append({
            "name": h100_name,
            "problem_id": thor["problem_id"],
            "h100_ms": h100_stats["mean"],
            "thor_ms": thor["mean"],
            "ratio": ratio,
            "category": categorize(h100_name),
        })

print(f"Matched: {len(matched)} problems\n")

# Per-problem table (top 10 closest to H100, top 10 furthest)
print("=== All Problems (sorted by ratio) ===")
print(f"{'#':>3}  {'Problem':<55} {'H100':>8} {'Thor':>8} {'Ratio':>7}  {'Category'}")
print("-" * 100)
for m in sorted(matched, key=lambda x: x["ratio"]):
    print(f"{m['problem_id']:3d}  {m['name'][:55]:<55} {m['h100_ms']:8.2f} {m['thor_ms']:8.2f} {m['ratio']:7.2f}x  {m['category']}")

# Summary stats
ratios = [m["ratio"] for m in matched]
print(f"\n=== Summary ===")
print(f"Matched problems: {len(matched)}")
print(f"Ratio (Thor/H100): min={min(ratios):.2f}x  median={statistics.median(ratios):.2f}x  mean={statistics.mean(ratios):.2f}x  max={max(ratios):.2f}x")
print(f"Thor is {statistics.median(ratios):.1f}x slower than H100 at the median")

# Per-category breakdown
print(f"\n=== By Category ===")
print(f"{'Category':<15} {'n':>3}  {'Median ratio':>13}  {'Min ratio':>10}  {'Max ratio':>10}  {'H100 med ms':>12}  {'Thor med ms':>12}")
print("-" * 85)
cat_data = {}
for m in matched:
    cat_data.setdefault(m["category"], []).append(m)

for cat in ['matmul','conv','activation','normalization','pooling','reduction','softmax','loss','attention','other']:
    if cat not in cat_data:
        continue
    items = cat_data[cat]
    cat_ratios = [i["ratio"] for i in items]
    h100_meds = statistics.median([i["h100_ms"] for i in items])
    thor_meds = statistics.median([i["thor_ms"] for i in items])
    print(f"{cat:<15} {len(items):3d}  {statistics.median(cat_ratios):13.2f}x  {min(cat_ratios):10.2f}x  {max(cat_ratios):10.2f}x  {h100_meds:12.2f}  {thor_meds:12.2f}")

# Top 5 closest to H100 (where Thor is most competitive)
print(f"\n=== Top 5 Most Competitive (Thor closest to H100) ===")
for m in sorted(matched, key=lambda x: x["ratio"])[:5]:
    print(f"  #{m['problem_id']:3d} {m['name'][:50]:50s}  ratio={m['ratio']:.2f}x  ({m['category']})")

# Top 5 worst (where Thor is furthest from H100)
print(f"\n=== Top 5 Least Competitive (Thor furthest from H100) ===")
for m in sorted(matched, key=lambda x: x["ratio"], reverse=True)[:5]:
    print(f"  #{m['problem_id']:3d} {m['name'][:50]:50s}  ratio={m['ratio']:.2f}x  ({m['category']})")
