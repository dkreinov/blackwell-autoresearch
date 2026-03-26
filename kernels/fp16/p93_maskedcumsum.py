import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// v1: fp16 MaskedCumsum. x (32768,32768) fp16, mask (32768,32768) uint8.
// Adapted from fp32 v3. float4 (8 halfs). 4 tiles x 1024t x 1 float4/thread.
// Forward inclusive prefix scan within warp, cross-warp, with carry across tiles.
// Float32 accumulation for precision.

__global__ void masked_cumsum_fp16_v1(
    const __half* __restrict__ x,
    const uint8_t* __restrict__ mask,
    __half* __restrict__ out,
    int n
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31, wid = tid >> 5;

    const float4* x4 = reinterpret_cast<const float4*>(x + (int64_t)row * n);
    const uint8_t* msk = mask + (int64_t)row * n;
    float4* out4 = reinterpret_cast<float4*>(out + (int64_t)row * n);

    int n4    = n >> 3;         // 4096 float4 chunks
    int tiles = n4 / 1024;     // 4 tiles

    __shared__ float ws[32];
    __shared__ float tile_total;
    float carry = 0.0f;

    for (int tile = 0; tile < tiles; tile++) {
        int base4 = tile * 1024 + tid;
        int base  = base4 * 8;  // first half element index

        float4 xv = x4[base4];
        const half2* h = (const half2*)&xv;
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);

        // Apply mask (scalar uint8 reads, hot in L1 since mask is 32KB/row... wait 32768 bytes = 32KB)
        float v0 = msk[base+0] ? f0.x : 0.0f;
        float v1 = msk[base+1] ? f0.y : 0.0f;
        float v2 = msk[base+2] ? f1.x : 0.0f;
        float v3 = msk[base+3] ? f1.y : 0.0f;
        float v4 = msk[base+4] ? f2.x : 0.0f;
        float v5 = msk[base+5] ? f2.y : 0.0f;
        float v6 = msk[base+6] ? f3.x : 0.0f;
        float v7 = msk[base+7] ? f3.y : 0.0f;

        // Local forward inclusive prefix within float4 (8 elements)
        float s0 = v0;
        float s1 = s0 + v1;
        float s2 = s1 + v2;
        float s3 = s2 + v3;
        float s4 = s3 + v4;
        float s5 = s4 + v5;
        float s6 = s5 + v6;
        float s7 = s6 + v7;
        float local_sum = s7;

        // Warp-level inclusive forward scan
        float fwd = local_sum;
        for (int dd = 1; dd < 32; dd <<= 1) {
            float up = __shfl_up_sync(0xffffffff, fwd, dd);
            if (lane >= dd) fwd += up;
        }
        float warp_total = __shfl_sync(0xffffffff, fwd, 31);
        float warp_excl  = fwd - local_sum;  // exclusive warp prefix for this thread

        if (lane == 0) ws[wid] = warp_total;
        __syncthreads();

        // Cross-warp exclusive prefix in warp 0
        if (wid == 0) {
            int nw = blockDim.x >> 5;
            float w = (lane < nw) ? ws[lane] : 0.0f;
            float fw = w;
            for (int dd = 1; dd < 32; dd <<= 1) {
                float up = __shfl_up_sync(0xffffffff, fw, dd);
                if (lane >= dd) fw += up;
            }
            float bt = __shfl_sync(0xffffffff, fw, 31);
            ws[lane] = fw - w;  // exclusive warp prefix
            if (lane == 0) tile_total = bt;
        }
        __syncthreads();

        float prefix = ws[wid] + warp_excl + carry;

        // Write: output[k] = s[k] + prefix
        float4 ov;
        half2* oh = (half2*)&ov;
        oh[0] = __floats2half2_rn(s0+prefix, s1+prefix);
        oh[1] = __floats2half2_rn(s2+prefix, s3+prefix);
        oh[2] = __floats2half2_rn(s4+prefix, s5+prefix);
        oh[3] = __floats2half2_rn(s6+prefix, s7+prefix);
        out4[base4] = ov;

        carry += tile_total;
        __syncthreads();
    }
}

torch::Tensor masked_cumsum_fp16_cuda(torch::Tensor x, torch::Tensor mask, int dim) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    auto out = torch::empty_like(x);
    int rows = x.size(0), n = x.size(1);
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const uint8_t* mp = reinterpret_cast<const uint8_t*>(mask.data_ptr<bool>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    masked_cumsum_fp16_v1<<<rows, 1024>>>(xp, mp, op, n);
    return out;
}
"""

cpp_source = "torch::Tensor masked_cumsum_fp16_cuda(torch::Tensor x, torch::Tensor mask, int dim);"

module = load_inline(
    name='masked_cumsum_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['masked_cumsum_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return module.masked_cumsum_fp16_cuda(x.contiguous(), mask.contiguous(), self.dim)
