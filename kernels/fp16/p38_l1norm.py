import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v2: float4 loads (128-bit = 8 halfs per load) where row is 8-byte aligned.
// dim=65535: row r at byte offset r*131070. r*131070 % 16 != 0 in general.
// For float4 (16-byte): row r aligned if r*65535 % 8 == 0.
// 65535 is odd -> row r aligned iff r is even.
// Handle even rows with float4, odd rows with 2x float2 (8 bytes each).
// Actually: use float4 for all even rows, scalar fallback for odd rows.
// Simpler: align all to float2 (4 bytes = 2 halfs) always -- still 2x improvement over scalar.

__global__ void l1norm_fp16_v2(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int64_t dim,
    int row_aligned  // 1 if all rows are 16-byte aligned
) {
    int row = blockIdx.x;
    const __half* rp = x + (int64_t)row * dim;
    __half* op = out + (int64_t)row * dim;
    int stride = blockDim.x;

    float s = 0.0f;

    // Check if this row's pointer is 16-byte aligned for float4
    // row r: byte offset = r * dim * 2. For dim=65535: offset = r*131070.
    // 131070 % 16 = 131070 - 16*8191 = 131070 - 131056 = 14. So never 16B aligned after row 0.
    // Use float4 for row 0, float2 for others -- or just use float2 always.
    // float2 = 4 bytes = 2 halfs: rows aligned at 4B boundary: r*131070 % 4 = r*2 % 4 (since 131070 % 4 = 2)
    // Row 0: 0 % 4 = 0 (aligned). Row 1: 2 % 4 = 2 (not). Row 2: 4 % 4 = 0 (aligned). Alternating.
    // Easiest: just use scalar half loads but process 8 per thread step (same as v1).
    // Upgrade: use half2 via __halves2half2 packing -- avoids alignment, still gets half2 compute.

    // Use half2 packing from scalar loads: pack pairs, use __habs2 + float accumulation
    int n2 = (int)(dim / 2);
    int tail = (int)(dim & 1);
    int stride2 = stride;  // half2 stride = same thread stride (each thread reads 1 half2 = 2 halfs)
    int stride8 = 4 * stride2;  // 4x half2 unroll

    // Pass 1: abs-sum using half2 packed from scalar loads
    int j = threadIdx.x;
    for (; j + 3 * stride2 < n2; j += stride8) {
        // Load 8 halfs as 4 half2 pairs
        half2 v0 = __halves2half2(rp[2*j],   rp[2*j+1]);
        half2 v1 = __halves2half2(rp[2*(j+stride2)],   rp[2*(j+stride2)+1]);
        half2 v2 = __halves2half2(rp[2*(j+2*stride2)], rp[2*(j+2*stride2)+1]);
        half2 v3 = __halves2half2(rp[2*(j+3*stride2)], rp[2*(j+3*stride2)+1]);
        float2 f0 = __half22float2(__habs2(v0));
        float2 f1 = __half22float2(__habs2(v1));
        float2 f2 = __half22float2(__habs2(v2));
        float2 f3 = __half22float2(__habs2(v3));
        s += f0.x + f0.y + f1.x + f1.y + f2.x + f2.y + f3.x + f3.y;
    }
    for (; j < n2; j += stride2) {
        half2 v = __halves2half2(rp[2*j], rp[2*j+1]);
        float2 f = __half22float2(__habs2(v));
        s += f.x + f.y;
    }
    if (tail && threadIdx.x == 0)
        s += fabsf(__half2float(rp[dim-1]));

    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);

    __shared__ float ws[32];
    __shared__ float inv_mean;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = s;
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        s = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) inv_mean = (float)dim / s;
    }
    __syncthreads();

    // Pass 2: normalize with half2 packed from scalar loads
    float im = inv_mean;
    half2 im2 = __float2half2_rn(im);
    j = threadIdx.x;
    for (; j + 3 * stride2 < n2; j += stride8) {
        half2 v0 = __halves2half2(rp[2*j],   rp[2*j+1]);
        half2 v1 = __halves2half2(rp[2*(j+stride2)],   rp[2*(j+stride2)+1]);
        half2 v2 = __halves2half2(rp[2*(j+2*stride2)], rp[2*(j+2*stride2)+1]);
        half2 v3 = __halves2half2(rp[2*(j+3*stride2)], rp[2*(j+3*stride2)+1]);
        half2 r0 = __hmul2(v0, im2);
        half2 r1 = __hmul2(v1, im2);
        half2 r2 = __hmul2(v2, im2);
        half2 r3 = __hmul2(v3, im2);
        op[2*j]   = __low2half(r0);  op[2*j+1]   = __high2half(r0);
        op[2*(j+stride2)]   = __low2half(r1);  op[2*(j+stride2)+1]   = __high2half(r1);
        op[2*(j+2*stride2)] = __low2half(r2);  op[2*(j+2*stride2)+1] = __high2half(r2);
        op[2*(j+3*stride2)] = __low2half(r3);  op[2*(j+3*stride2)+1] = __high2half(r3);
    }
    for (; j < n2; j += stride2) {
        half2 v = __halves2half2(rp[2*j], rp[2*j+1]);
        half2 r = __hmul2(v, im2);
        op[2*j] = __low2half(r);  op[2*j+1] = __high2half(r);
    }
    if (tail && threadIdx.x == 0)
        op[dim-1] = __float2half(__half2float(rp[dim-1]) * im);
}

torch::Tensor l1norm_fp16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    int rows = x.size(0);
    int64_t dim = x.size(1);
    auto out = torch::empty_like(x);
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    l1norm_fp16_v2<<<rows, 1024>>>(xp, op, dim, 0);
    return out;
}
"""

cpp_source = "torch::Tensor l1norm_fp16_cuda(torch::Tensor x);"

module = load_inline(
    name='l1norm_fp16_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l1norm_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return module.l1norm_fp16_cuda(x)
