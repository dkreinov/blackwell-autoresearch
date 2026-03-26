import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

// v1: fp16 online softmax.
// Pass 1: online max+sum via __ldcg (L2-only), float32 accumulation, 8x unroll.
// Pass 2: normalize via __ldlu+__stwt (evict-after-read + bypass-write caches).
// 393216 elements/row = 49152 float4 chunks. 1024 threads: 48 float4/thread.
// 8x unroll: exactly 6 iterations, no tail.

__global__ void softmax_fp16_v1(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int dim
) {
    int row = blockIdx.x;
    const __half* row_in = x   + (long long)row * dim;
    __half*       row_out = out + (long long)row * dim;

    int n4 = dim >> 3;  // float4 chunks (8 halfs each)
    const float4* row4  = reinterpret_cast<const float4*>(row_in);
    float4*       out4  = reinterpret_cast<float4*>(row_out);
    int stride  = blockDim.x;
    int stride8 = 8 * stride;

    // Pass 1: online softmax (max + sum) via __ldcg, float32 throughout
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    int j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 A = __ldcg(&row4[j]),          B = __ldcg(&row4[j+stride]);
        float4 C = __ldcg(&row4[j+2*stride]), D = __ldcg(&row4[j+3*stride]);
        float4 E = __ldcg(&row4[j+4*stride]), F = __ldcg(&row4[j+5*stride]);
        float4 G = __ldcg(&row4[j+6*stride]), H = __ldcg(&row4[j+7*stride]);

        #define UPDATE8(V) {                                                          \
            const half2* h = (const half2*)&(V);                                     \
            float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);                 \
            float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);                 \
            float cm = fmaxf(fmaxf(fmaxf(f0.x,f0.y),fmaxf(f1.x,f1.y)),             \
                             fmaxf(fmaxf(f2.x,f2.y),fmaxf(f3.x,f3.y)));             \
            float nm = fmaxf(local_max, cm);                                          \
            local_sum = local_sum * __expf(local_max - nm)                            \
                      + __expf(f0.x-nm)+__expf(f0.y-nm)                              \
                      + __expf(f1.x-nm)+__expf(f1.y-nm)                              \
                      + __expf(f2.x-nm)+__expf(f2.y-nm)                              \
                      + __expf(f3.x-nm)+__expf(f3.y-nm);                             \
            local_max = nm; }

        UPDATE8(A); UPDATE8(B); UPDATE8(C); UPDATE8(D);
        UPDATE8(E); UPDATE8(F); UPDATE8(G); UPDATE8(H);
        #undef UPDATE8
    }
    for (; j < n4; j += stride) {
        float4 v4 = __ldcg(&row4[j]);
        const half2* h = (const half2*)&v4;
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
        float cm = fmaxf(fmaxf(fmaxf(f0.x,f0.y),fmaxf(f1.x,f1.y)),
                         fmaxf(fmaxf(f2.x,f2.y),fmaxf(f3.x,f3.y)));
        float nm = fmaxf(local_max, cm);
        local_sum = local_sum * __expf(local_max - nm)
                  + __expf(f0.x-nm)+__expf(f0.y-nm)
                  + __expf(f1.x-nm)+__expf(f1.y-nm)
                  + __expf(f2.x-nm)+__expf(f2.y-nm)
                  + __expf(f3.x-nm)+__expf(f3.y-nm);
        local_max = nm;
    }

    // Warp-level reduction: online combine (max, sum) pairs
    for (int off = 16; off > 0; off >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, off);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, off);
        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * __expf(local_max - new_max)
                  + other_sum * __expf(other_max - new_max);
        local_max = new_max;
    }

    __shared__ float warp_max[32], warp_sum[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { warp_max[wid] = local_max; warp_sum[wid] = local_sum; }
    __syncthreads();

    // Cross-warp reduction in warp 0
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        local_max = (lane < nw) ? warp_max[lane] : -FLT_MAX;
        local_sum = (lane < nw) ? warp_sum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, off);
            float other_sum = __shfl_down_sync(0xffffffff, local_sum, off);
            float new_max = fmaxf(local_max, other_max);
            local_sum = local_sum * __expf(local_max - new_max)
                      + other_sum * __expf(other_max - new_max);
            local_max = new_max;
        }
    }

    __shared__ float s_max, s_inv_sum;
    if (threadIdx.x == 0) {
        s_max = local_max;
        s_inv_sum = 1.0f / local_sum;
    }
    __syncthreads();
    float row_max = s_max, inv_sum = s_inv_sum;

    // Pass 2: normalize via __ldlu (L2 hit) + __stwt (bypass write cache)
    j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 A = __ldlu(&row4[j]),          B = __ldlu(&row4[j+stride]);
        float4 C = __ldlu(&row4[j+2*stride]), D = __ldlu(&row4[j+3*stride]);
        float4 E = __ldlu(&row4[j+4*stride]), F = __ldlu(&row4[j+5*stride]);
        float4 G = __ldlu(&row4[j+6*stride]), H = __ldlu(&row4[j+7*stride]);

        #define NORM8(V) ({                                                          \
            const half2* h = (const half2*)&(V);                                     \
            float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);                 \
            float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);                 \
            half2 r0=__floats2half2_rn(__expf(f0.x-row_max)*inv_sum,                 \
                                       __expf(f0.y-row_max)*inv_sum);                 \
            half2 r1=__floats2half2_rn(__expf(f1.x-row_max)*inv_sum,                 \
                                       __expf(f1.y-row_max)*inv_sum);                 \
            half2 r2=__floats2half2_rn(__expf(f2.x-row_max)*inv_sum,                 \
                                       __expf(f2.y-row_max)*inv_sum);                 \
            half2 r3=__floats2half2_rn(__expf(f3.x-row_max)*inv_sum,                 \
                                       __expf(f3.y-row_max)*inv_sum);                 \
            float4 res; half2* hr=(half2*)&res;                                      \
            hr[0]=r0; hr[1]=r1; hr[2]=r2; hr[3]=r3; res; })

        float4 rA=NORM8(A), rB=NORM8(B), rC=NORM8(C), rD=NORM8(D);
        float4 rE=NORM8(E), rF=NORM8(F), rG=NORM8(G), rH=NORM8(H);
        #undef NORM8
        __stwt(&out4[j],         rA); __stwt(&out4[j+stride],   rB);
        __stwt(&out4[j+2*stride],rC); __stwt(&out4[j+3*stride], rD);
        __stwt(&out4[j+4*stride],rE); __stwt(&out4[j+5*stride], rF);
        __stwt(&out4[j+6*stride],rG); __stwt(&out4[j+7*stride], rH);
    }
    for (; j < n4; j += stride) {
        float4 v4 = __ldlu(&row4[j]);
        const half2* h = (const half2*)&v4;
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
        half2 r0=__floats2half2_rn(__expf(f0.x-row_max)*inv_sum,
                                   __expf(f0.y-row_max)*inv_sum);
        half2 r1=__floats2half2_rn(__expf(f1.x-row_max)*inv_sum,
                                   __expf(f1.y-row_max)*inv_sum);
        half2 r2=__floats2half2_rn(__expf(f2.x-row_max)*inv_sum,
                                   __expf(f2.y-row_max)*inv_sum);
        half2 r3=__floats2half2_rn(__expf(f3.x-row_max)*inv_sum,
                                   __expf(f3.y-row_max)*inv_sum);
        float4 res; half2* hr=(half2*)&res;
        hr[0]=r0; hr[1]=r1; hr[2]=r2; hr[3]=r3;
        __stwt(&out4[j], res);
    }
}

torch::Tensor softmax_fp16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    TORCH_CHECK(x.dim() == 2, "expected 2D input");
    auto xc = x.contiguous();
    int batch = xc.size(0), dim = xc.size(1);
    auto out = torch::empty_like(xc);
    const __half* xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    softmax_fp16_v1<<<batch, 1024>>>(xp, op, dim);
    return out;
}
"""

cpp_source = "torch::Tensor softmax_fp16_cuda(torch::Tensor x);"

module = load_inline(
    name='softmax_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['softmax_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.softmax_fp16_cuda(x)
