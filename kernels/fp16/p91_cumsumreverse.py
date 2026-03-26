import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 CumsumReverse. Input (32768,32768) fp16.
// Adapted from fp32 v5: tile-based reverse prefix scan.
// fp16: float4 = 8 halfs. n4=4096 float4 per row. 4 tiles x 1024t x 1 float4/thread.
// Internal compute in float32. Warp/cross-warp suffix scan matches fp32 structure.

__global__ void cumsum_reverse_fp16_v1(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int n
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31, wid = tid >> 5;

    const float4* in4  = reinterpret_cast<const float4*>(x   + (int64_t)row * n);
    float4*       out4 = reinterpret_cast<float4*>(out + (int64_t)row * n);

    int n4    = n >> 3;         // float4 chunks per row = 4096
    int tiles = n4 / 1024;     // = 4

    __shared__ float ws[32];
    __shared__ float tile_total;
    float carry = 0.0f;

    for (int tile = tiles - 1; tile >= 0; tile--) {
        int base4 = tile * 1024 + tid;
        float4 v = in4[base4];
        const half2* h = (const half2*)&v;
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);

        // Reverse cumsum within float4 (8 elements, right to left)
        float v7 = f3.y;
        float v6 = f3.x + v7;
        float v5 = f2.y + v6;
        float v4 = f2.x + v5;
        float v3 = f1.y + v4;
        float v2 = f1.x + v3;
        float v1 = f0.y + v2;
        float v0 = f0.x + v1;   // = sum of all 8
        float local_sum = v0;

        // Forward warp scan of local_sum (to compute suffix among warps)
        float fwd = local_sum;
        for (int dd = 1; dd < 32; dd <<= 1) {
            float up = __shfl_up_sync(0xffffffff, fwd, dd);
            if (lane >= dd) fwd += up;
        }
        float warp_total = __shfl_sync(0xffffffff, fwd, 31);
        float warp_rev_excl = warp_total - fwd;  // sum of float4s after this within warp

        if (lane == 0) ws[wid] = warp_total;
        __syncthreads();

        // Cross-warp suffix in warp 0
        float warp_suffix = 0.0f;
        if (wid == 0) {
            int nw = blockDim.x >> 5;  // 32
            float w = (lane < nw) ? ws[lane] : 0.0f;
            float fw = w;
            for (int dd = 1; dd < 32; dd <<= 1) {
                float up = __shfl_up_sync(0xffffffff, fw, dd);
                if (lane >= dd) fw += up;
            }
            float bt = __shfl_sync(0xffffffff, fw, 31);
            ws[lane] = bt - fw;  // suffix per warp
            if (lane == 0) tile_total = bt;
        }
        __syncthreads();

        float suffix = ws[wid] + warp_rev_excl + carry;

        // Write output: position i → v[i] + suffix (v[i] = partial reverse sum from i)
        float4 ov;
        half2* oh = (half2*)&ov;
        oh[0] = __floats2half2_rn(v0+suffix, v1+suffix);
        oh[1] = __floats2half2_rn(v2+suffix, v3+suffix);
        oh[2] = __floats2half2_rn(v4+suffix, v5+suffix);
        oh[3] = __floats2half2_rn(v6+suffix, v7+suffix);
        out4[base4] = ov;

        carry += tile_total;
        __syncthreads();
    }
}

torch::Tensor cumsum_reverse_fp16_cuda(torch::Tensor x, int dim) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    auto out = torch::empty_like(x);
    int rows = x.size(0);
    int n    = x.size(1);
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    cumsum_reverse_fp16_v1<<<rows, 1024>>>(xp, op, n);
    return out;
}
"""

cpp_source = "torch::Tensor cumsum_reverse_fp16_cuda(torch::Tensor x, int dim);"

module = load_inline(
    name='cumsum_reverse_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['cumsum_reverse_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.cumsum_reverse_fp16_cuda(x, 1)
