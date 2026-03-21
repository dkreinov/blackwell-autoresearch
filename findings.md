# Thor CUDA Kernel Autoresearch — Findings

## Completed Experiments

(none yet)

## Observations

### From Transfer Study (Sakana H100 → Thor, 63 kernels)
- Reduction ops consistently backfire on Thor (7/9 slower): HBM-optimized shared-memory tiling adds overhead on unified LPDDR5X
- Normalization kernels transfer best (4/4 faster, median 2.35x): bandwidth-bound, L2 locality applies equally
- Activations have near-unity transfer ratio (~1.05x median): element-wise, no memory access pattern dependency
- Problem 12 (diagonal matmul): Thor 39.95x speedup — unified memory eliminates the penalty that hurts H100's sparse access pattern

## Dead Ends

(none yet)
