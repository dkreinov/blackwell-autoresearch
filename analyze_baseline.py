#!/usr/bin/env python3
"""Analyze merged baseline timing results."""
import json, statistics, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/nvidia/thor_kernelbench_work/results/Thor_AGX/baseline_level1.json"
with open(path) as f:
    data = json.load(f)

results = data["results"]
ok = [r for r in results if r["status"] == "ok"]
fail = [r for r in results if r["status"] != "ok"]
means = [r["mean"] for r in ok]
wall = sum(r.get("wall_seconds", 0) for r in results)

print(f"Total: {len(results)}, OK: {len(ok)}, Failed: {len(fail)}")
print(f"Min: {min(means):.2f}ms, Max: {max(means):.2f}ms, Median: {statistics.median(means):.2f}ms, Mean: {statistics.mean(means):.2f}ms")
print(f"Total wall time: {wall:.1f}s ({wall/60:.1f}min)")
print()

print("=== Failures ===")
for r in fail:
    print(f"  #{r['problem_id']}: {r['name']} -- {r['status']}: {r.get('error','')}")
print()

print("=== Top 10 Slowest ===")
for r in sorted(ok, key=lambda x: x["mean"], reverse=True)[:10]:
    print(f"  #{r['problem_id']:3d}  {r['name'][:55]:55s}  {r['mean']:8.2f}ms  std={r['std']:.3f}")
print()

print("=== Top 10 Fastest ===")
for r in sorted(ok, key=lambda x: x["mean"])[:10]:
    print(f"  #{r['problem_id']:3d}  {r['name'][:55]:55s}  {r['mean']:8.2f}ms  std={r['std']:.3f}")
print()

# Categorize
categories = {}
for r in ok:
    name = r["name"].lower()
    if "matmul" in name or "matrix" in name:
        cat = "matmul"
    elif "conv" in name:
        cat = "conv"
    elif any(x in name for x in ["relu","sigmoid","tanh","gelu","selu","swish","elu","softplus","softsign","hardtanh","hardsigmoid"]):
        cat = "activation"
    elif any(x in name for x in ["norm","batchnorm","layernorm","groupnorm","rmsnorm","frobenius","l1norm","l2norm"]):
        cat = "normalization"
    elif "pool" in name:
        cat = "pooling"
    elif any(x in name for x in ["sum_reduction","mean_reduction","max_reduction","min_reduction","argmax","argmin","cumsum","cumprod","masked_cumsum"]):
        cat = "reduction"
    elif any(x in name for x in ["loss","mse","huber","kldiv","triplet","hinge"]):
        cat = "loss"
    elif "softmax" in name or "logsoftmax" in name:
        cat = "softmax"
    elif "attention" in name:
        cat = "attention"
    elif "newgelu" in name:
        cat = "activation"
    else:
        cat = "other"
    categories.setdefault(cat, []).append(r["mean"])

print("=== By Operator Category ===")
for cat in sorted(categories):
    vals = categories[cat]
    med = statistics.median(vals)
    print(f"  {cat:15s}  n={len(vals):2d}  median={med:7.2f}ms  min={min(vals):7.2f}ms  max={max(vals):7.2f}ms")
