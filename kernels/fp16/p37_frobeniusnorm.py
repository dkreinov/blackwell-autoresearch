import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 FrobeniusNorm. Tensor (112,64,512,512) = 1,879,048,192 halfs = 234,881,024 float4.
// 2 kernels: pass1 reduces sum-of-squares to partial_sums[1024]; pass2 normalizes.
// __ldcg pass1 (L2-only, keeps for pass2 L2 reuse), __ldlu+__hmul2 pass2 (evict+bypass write).
// Pass1: 1024 blocks x 1024t, each covering n4/1024 = 229376 float4 chunks (28.7 8x-unroll iters).
// Pass2: same 1024 blocks x 1024t, simple multiply by inv_norm via half2.

__global__ void frobnorm_sq_fp16(
    const __half* __restrict__ x,
    float* __restrict__ partial_sums,
    int64_t n4
) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;  // 1024*1024 = 1M
    int64_t stride8 = 8 * stride;
    const float4* xf4 = reinterpret_cast<const float4*>(x);

    float s = 0.0f;
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    #define SQ8(V) {                                                              \
        const half2* h = (const half2*)&(V);                                      \
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);                  \
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);                  \
        s += f0.x*f0.x+f0.y*f0.y + f1.x*f1.x+f1.y*f1.y                         \
           + f2.x*f2.x+f2.y*f2.y + f3.x*f3.x+f3.y*f3.y; }

    for (; j + 7 * stride < n4; j += stride8) {
        float4 A=__ldcg(&xf4[j]),          B=__ldcg(&xf4[j+stride]);
        float4 C=__ldcg(&xf4[j+2*stride]), D=__ldcg(&xf4[j+3*stride]);
        float4 E=__ldcg(&xf4[j+4*stride]), F=__ldcg(&xf4[j+5*stride]);
        float4 G=__ldcg(&xf4[j+6*stride]), H=__ldcg(&xf4[j+7*stride]);
        SQ8(A); SQ8(B); SQ8(C); SQ8(D);
        SQ8(E); SQ8(F); SQ8(G); SQ8(H);
    }
    for (; j < n4; j += stride) {
        float4 v = __ldcg(&xf4[j]);
        SQ8(v);
    }
    #undef SQ8

    // Block reduce
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    __shared__ float ws[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = s;
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        s = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) partial_sums[blockIdx.x] = s;
    }
}

__global__ void frobnorm_normalize_fp16(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    float inv_norm,
    int64_t n4
) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t stride8 = 8 * stride;
    const float4* xf4 = reinterpret_cast<const float4*>(x);
    float4* of4 = reinterpret_cast<float4*>(out);
    half2 inv2 = __float2half2_rn(inv_norm);

    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 A=__ldlu(&xf4[j]),          B=__ldlu(&xf4[j+stride]);
        float4 C=__ldlu(&xf4[j+2*stride]), D=__ldlu(&xf4[j+3*stride]);
        float4 E=__ldlu(&xf4[j+4*stride]), F=__ldlu(&xf4[j+5*stride]);
        float4 G=__ldlu(&xf4[j+6*stride]), H=__ldlu(&xf4[j+7*stride]);
        #define MUL4(V) {                                                         \
            half2* h = (half2*)&(V);                                              \
            h[0]=__hmul2(h[0],inv2); h[1]=__hmul2(h[1],inv2);                   \
            h[2]=__hmul2(h[2],inv2); h[3]=__hmul2(h[3],inv2); }
        MUL4(A); MUL4(B); MUL4(C); MUL4(D);
        MUL4(E); MUL4(F); MUL4(G); MUL4(H);
        #undef MUL4
        __stwt(&of4[j],         A); __stwt(&of4[j+stride],   B);
        __stwt(&of4[j+2*stride],C); __stwt(&of4[j+3*stride], D);
        __stwt(&of4[j+4*stride],E); __stwt(&of4[j+5*stride], F);
        __stwt(&of4[j+6*stride],G); __stwt(&of4[j+7*stride], H);
    }
    for (; j < n4; j += stride) {
        float4 v = __ldlu(&xf4[j]);
        half2* h = (half2*)&v;
        h[0]=__hmul2(h[0],inv2); h[1]=__hmul2(h[1],inv2);
        h[2]=__hmul2(h[2],inv2); h[3]=__hmul2(h[3],inv2);
        __stwt(&of4[j], v);
    }
}
"""

cpp_source = """
torch::Tensor frobnorm_sq_cuda(torch::Tensor x, torch::Tensor partial_sums, int64_t n4);
torch::Tensor frobnorm_normalize_cuda(torch::Tensor x, torch::Tensor out, float inv_norm, int64_t n4);
"""

# Load both kernels
_cuda_src_full = cuda_source + """
torch::Tensor frobnorm_sq_cuda(torch::Tensor x, torch::Tensor partial_sums, int64_t n4) {
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    float* pp = partial_sums.data_ptr<float>();
    frobnorm_sq_fp16<<<1024, 1024>>>(xp, pp, n4);
    return partial_sums;
}
torch::Tensor frobnorm_normalize_cuda(torch::Tensor x, torch::Tensor out, float inv_norm, int64_t n4) {
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    frobnorm_normalize_fp16<<<1024, 1024>>>(xp, op, inv_norm, n4);
    return out;
}
"""

module = load_inline(
    name='frobeniusnorm_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=_cuda_src_full,
    functions=['frobnorm_sq_cuda', 'frobnorm_normalize_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        n4 = n >> 3  # n / 8
        out = torch.empty_like(x)
        partial = torch.empty(1024, dtype=torch.float32, device=x.device)
        module.frobnorm_sq_cuda(x, partial, n4)
        sq_sum = partial.sum().item()
        inv_norm = 1.0 / math.sqrt(sq_sum)
        module.frobnorm_normalize_cuda(x, out, inv_norm, n4)
        return out
