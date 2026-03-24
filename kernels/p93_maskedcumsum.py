import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>

// v3: same tile-based forward cumsum as v2, but mask accepted as uint8_t directly.
// Handles eval harness converting Bool mask to Float: Python side converts to uint8 first.
// 1024 threads, 8 tiles of 4096 elements, 1 float4 per thread per tile (x).

__global__ void masked_cumsum_kernel(const float* __restrict__ x,
                                      const uint8_t* __restrict__ mask,
                                      float* __restrict__ out, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    const float4*   x4   = reinterpret_cast<const float4*>(x    + (long long)row * n);
    const uint8_t*  msk  = mask + (long long)row * n;
          float4*   out4 = reinterpret_cast<float4*>(out + (long long)row * n);

    __shared__ float ws[32];
    __shared__ float tile_total_smem;
    float carry = 0.f;

    for (int tile = 0; tile < 8; tile++) {
        int base4  = tile * 1024 + tid;
        int base   = base4 * 4;

        float4 xv = x4[base4];

        // Scalar mask loads — safe uint8 reads, no aliasing
        float v0 = msk[base+0] ? xv.x : 0.f;
        float v1 = msk[base+1] ? xv.y : 0.f;
        float v2 = msk[base+2] ? xv.z : 0.f;
        float v3 = msk[base+3] ? xv.w : 0.f;

        // Local forward inclusive prefix
        float s0 = v0;
        float s1 = s0 + v1;
        float s2 = s1 + v2;
        float s3 = s2 + v3;
        float local_sum = s3;

        // Warp-level inclusive forward scan
        float fwd = local_sum;
        for (int dd = 1; dd < 32; dd <<= 1) {
            float up = __shfl_up_sync(0xffffffff, fwd, dd);
            if (lane >= dd) fwd += up;
        }
        float warp_total = __shfl_sync(0xffffffff, fwd, 31);
        float warp_excl  = fwd - local_sum;

        if (lane == 0) ws[wid] = warp_total;
        __syncthreads();

        if (wid == 0) {
            float w  = ws[lane];
            float fw = w;
            for (int dd = 1; dd < 32; dd <<= 1) {
                float up = __shfl_up_sync(0xffffffff, fw, dd);
                if (lane >= dd) fw += up;
            }
            float bt = __shfl_sync(0xffffffff, fw, 31);
            ws[lane] = fw - w;  // exclusive warp prefix
            if (lane == 0) tile_total_smem = bt;
        }
        __syncthreads();

        float prefix = ws[wid] + warp_excl + carry;
        out4[base4] = {s0+prefix, s1+prefix, s2+prefix, s3+prefix};

        carry += tile_total_smem;
        __syncthreads();
    }
}

torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask_u8, int dim) {
    auto out = torch::empty_like(x);
    int rows = x.size(0);
    int n    = x.size(1);
    masked_cumsum_kernel<<<rows, 1024>>>(x.data_ptr<float>(),
                                         mask_u8.data_ptr<uint8_t>(),
                                         out.data_ptr<float>(), n);
    return out;
}
"""

cpp_source = "torch::Tensor masked_cumsum_cuda(torch::Tensor x, torch::Tensor mask_u8, int dim);"

mod = load_inline(
    name='masked_cumsum_v3',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['masked_cumsum_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Convert mask to uint8 regardless of incoming dtype (Bool, Float, etc.)
        mask_u8 = mask.to(torch.uint8)
        return mod.masked_cumsum_cuda(x, mask_u8, self.dim)
