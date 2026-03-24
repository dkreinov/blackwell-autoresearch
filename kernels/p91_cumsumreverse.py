import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// v4: 512 threads/block, 2 float4s per thread per tile (vs 4 in v3).
// 512 threads x 2 float4s = 1024 float4s per tile = 4096 elems per tile.
// 8 tiles of 4096 elems. 512t: 1536/512=3 blocks/SM = 100% warp occupancy.
// Try: larger block covers more warp shuffles → better ILP.

__global__ void cumsum_reverse_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;  // 0..15 (16 warps for 512 threads)

    const float4* row_in4  = reinterpret_cast<const float4*>(x   + (long long)row * n);
          float4* row_out4 = reinterpret_cast<float4*>(out + (long long)row * n);

    __shared__ float ws[16];
    __shared__ float tile_total_smem;
    float carry = 0.f;

    for (int tile = 7; tile >= 0; tile--) {
        int base4 = tile * 1024 + tid * 2;  // 512 * 2 = 1024 float4s per tile
        float4 a = row_in4[base4 + 0];
        float4 b = row_in4[base4 + 1];

        // Local rev prefix for 8 elements: s7..s0
        float s7 = b.w;
        float s6 = b.z + s7;
        float s5 = b.y + s6;
        float s4 = b.x + s5;
        float s3 = a.w + s4;
        float s2 = a.z + s3;
        float s1 = a.y + s2;
        float s0 = a.x + s1;
        float local_sum = s0;

        // Warp-level rev incl scan
        float fwd = local_sum;
        for (int dd = 1; dd < 32; dd <<= 1) {
            float up = __shfl_up_sync(0xffffffff, fwd, dd);
            if (lane >= dd) fwd += up;
        }
        float warp_total = __shfl_sync(0xffffffff, fwd, 31);
        float warp_rev_excl = warp_total - fwd;

        if (lane == 0) ws[wid] = warp_total;
        __syncthreads();

        float warp_suffix = 0.f;
        if (wid == 0) {
            int num_warps = blockDim.x >> 5;  // 16
            float w = (lane < num_warps) ? ws[lane] : 0.f;
            float fw = w;
            for (int dd = 1; dd < 32; dd <<= 1) {
                float up = __shfl_up_sync(0xffffffff, fw, dd);
                if (lane >= dd) fw += up;
            }
            float bt = __shfl_sync(0xffffffff, fw, num_warps - 1);
            ws[lane] = (lane < num_warps) ? bt - fw : 0.f;
            if (lane == 0) tile_total_smem = bt;
        }
        __syncthreads();

        float suffix = ws[wid] + warp_rev_excl + carry;
        float4 oa = {s0+suffix, s1+suffix, s2+suffix, s3+suffix};
        float4 ob = {s4+suffix, s5+suffix, s6+suffix, s7+suffix};
        row_out4[base4+0] = oa;
        row_out4[base4+1] = ob;

        carry += tile_total_smem;
        __syncthreads();
    }
}

torch::Tensor cumsum_reverse_cuda(torch::Tensor x, int dim) {
    auto out = torch::empty_like(x);
    int rows = x.size(0);
    int n    = x.size(1);
    cumsum_reverse_kernel<<<rows, 512>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

cpp_source = "torch::Tensor cumsum_reverse_cuda(torch::Tensor x, int dim);"

mod = load_inline(
    name='cumsum_reverse_v4',
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
