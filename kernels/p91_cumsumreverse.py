import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// v2: Tile-based coalesced reverse cumsum.
// Process 32 tiles of 1024 elements each, right-to-left, accumulating carry.
// COALESCED: thread t loads tile_base+t each round (adjacent threads = adjacent addresses).
// Within each tile: warp-level reverse inclusive scan + block-level suffix combination.
// Out[pos] = local_rev_scan[t] + carry_from_right.

__global__ void cumsum_reverse_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    const float* row_in  = x   + (long long)row * n;
          float* row_out = out + (long long)row * n;

    __shared__ float ws[32];   // warp partial sums
    __shared__ float tile_total_smem;

    float carry = 0.f;  // sum of elements processed so far (from the right)

    // Process tiles right-to-left
    for (int tile = 31; tile >= 0; tile--) {
        int pos = tile * 1024 + tid;
        float v = row_in[pos];  // coalesced: threads 0..1023 load consecutive elements

        // Warp-level reverse inclusive scan using: rev[l] = warp_total - fwd_excl[l]
        float fwd = v;
        for (int d = 1; d < 32; d <<= 1) {
            float up = __shfl_up_sync(0xffffffff, fwd, d);
            if (lane >= d) fwd += up;
        }
        float warp_total = __shfl_sync(0xffffffff, fwd, 31);
        // rev_warp[l] = warp_total - (fwd - v) = warp_total - fwd + v = rev inclusive within warp
        float rev_warp = warp_total - fwd + v;

        // Store warp totals for block-level scan
        if (lane == 0) ws[wid] = warp_total;
        __syncthreads();

        // Warp 0: compute block total + reverse exclusive for each warp
        float warp_suffix = 0.f;
        if (wid == 0) {
            float w = ws[lane];
            float fw = w;
            for (int d = 1; d < 32; d <<= 1) {
                float up = __shfl_up_sync(0xffffffff, fw, d);
                if (lane >= d) fw += up;
            }
            float bt = __shfl_sync(0xffffffff, fw, 31);  // block_total for this tile
            // reverse exclusive for warp wid: bt - fw[wid]
            ws[lane] = bt - fw;
            if (lane == 0) tile_total_smem = bt;
        }
        __syncthreads();

        warp_suffix = ws[wid];
        float out_val = rev_warp + warp_suffix + carry;
        row_out[pos] = out_val;

        carry += tile_total_smem;
        __syncthreads();  // ensure ws[] is ready for next iteration
    }
}

torch::Tensor cumsum_reverse_cuda(torch::Tensor x, int dim) {
    auto out = torch::empty_like(x);
    int rows = x.size(0);
    int n    = x.size(1);
    cumsum_reverse_kernel<<<rows, 1024>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

cpp_source = "torch::Tensor cumsum_reverse_cuda(torch::Tensor x, int dim);"

mod = load_inline(
    name='cumsum_reverse_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['cumsum_reverse_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mod.cumsum_reverse_cuda(x.cuda(), self.dim)
