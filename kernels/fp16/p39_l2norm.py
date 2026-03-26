import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 L2Norm. dim=65535 (odd), rows=32768.
// Use __halves2half2 packing from scalar loads (bypasses alignment issues).
// Pass 1: sum of squares via 4x half2 unroll, float32 accumulation.
// Pass 2: multiply by inv_l2 = rsqrtf(sq_sum) via half2 multiply.
// No __ldcg/__ldlu cache hints -- data (65535*2B=128KB) per row > L1 (256KB per SM),
// with 32768 rows and only 20 SMs the L2 cannot cache all rows simultaneously.

__global__ void l2norm_fp16_v1(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int64_t dim
) {
    int row = blockIdx.x;
    const __half* rp = x   + (int64_t)row * dim;
    __half*       op = out + (int64_t)row * dim;
    int stride = blockDim.x;

    int n2 = (int)(dim / 2);
    int tail = (int)(dim & 1);
    int stride4 = 4 * stride;  // 4x half2 unroll

    // Pass 1: sum of squares using half2 from scalar loads
    float s2 = 0.0f;
    int j = threadIdx.x;
    for (; j + 3 * stride < n2; j += stride4) {
        __half a0 = rp[2*j],   b0 = rp[2*j+1];
        __half a1 = rp[2*(j+stride)],   b1 = rp[2*(j+stride)+1];
        __half a2 = rp[2*(j+2*stride)], b2 = rp[2*(j+2*stride)+1];
        __half a3 = rp[2*(j+3*stride)], b3 = rp[2*(j+3*stride)+1];
        float2 f0 = __half22float2(__halves2half2(a0, b0));
        float2 f1 = __half22float2(__halves2half2(a1, b1));
        float2 f2 = __half22float2(__halves2half2(a2, b2));
        float2 f3 = __half22float2(__halves2half2(a3, b3));
        s2 += f0.x*f0.x + f0.y*f0.y + f1.x*f1.x + f1.y*f1.y
            + f2.x*f2.x + f2.y*f2.y + f3.x*f3.x + f3.y*f3.y;
    }
    for (; j < n2; j += stride) {
        float2 f = __half22float2(__halves2half2(rp[2*j], rp[2*j+1]));
        s2 += f.x*f.x + f.y*f.y;
    }
    if (tail && threadIdx.x == 0) {
        float v = __half2float(rp[dim-1]);
        s2 += v * v;
    }

    // Block reduce
    for (int off = 16; off > 0; off >>= 1)
        s2 += __shfl_down_sync(0xffffffff, s2, off);

    __shared__ float ws[32];
    __shared__ float inv_l2;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = s2;
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        s2 = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s2 += __shfl_down_sync(0xffffffff, s2, off);
        if (lane == 0) inv_l2 = rsqrtf(s2);
    }
    __syncthreads();

    // Pass 2: normalize via half2 multiply
    float il = inv_l2;
    half2 il2 = __float2half2_rn(il);
    j = threadIdx.x;
    for (; j + 3 * stride < n2; j += stride4) {
        half2 v0 = __halves2half2(rp[2*j],   rp[2*j+1]);
        half2 v1 = __halves2half2(rp[2*(j+stride)],   rp[2*(j+stride)+1]);
        half2 v2 = __halves2half2(rp[2*(j+2*stride)], rp[2*(j+2*stride)+1]);
        half2 v3 = __halves2half2(rp[2*(j+3*stride)], rp[2*(j+3*stride)+1]);
        half2 r0 = __hmul2(v0, il2);
        half2 r1 = __hmul2(v1, il2);
        half2 r2 = __hmul2(v2, il2);
        half2 r3 = __hmul2(v3, il2);
        op[2*j]   = __low2half(r0);  op[2*j+1]   = __high2half(r0);
        op[2*(j+stride)]   = __low2half(r1);  op[2*(j+stride)+1]   = __high2half(r1);
        op[2*(j+2*stride)] = __low2half(r2);  op[2*(j+2*stride)+1] = __high2half(r2);
        op[2*(j+3*stride)] = __low2half(r3);  op[2*(j+3*stride)+1] = __high2half(r3);
    }
    for (; j < n2; j += stride) {
        half2 v = __halves2half2(rp[2*j], rp[2*j+1]);
        half2 r = __hmul2(v, il2);
        op[2*j] = __low2half(r);  op[2*j+1] = __high2half(r);
    }
    if (tail && threadIdx.x == 0)
        op[dim-1] = __float2half(__half2float(rp[dim-1]) * il);
}

torch::Tensor l2norm_fp16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    int rows = x.size(0);
    int64_t dim = x.size(1);
    auto out = torch::empty_like(x);
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    l2norm_fp16_v1<<<rows, 1024>>>(xp, op, dim);
    return out;
}
"""

cpp_source = "torch::Tensor l2norm_fp16_cuda(torch::Tensor x);"

module = load_inline(
    name='l2norm_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l2norm_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return module.l2norm_fp16_cuda(x)
