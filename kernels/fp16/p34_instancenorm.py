import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v3: pass 1 uses __ldcg (L2-only, cache for pass 2 re-read) + pass 2 uses __ldlu+__stwt.
// Optimal 2-pass cache strategy: keep data in L2 during pass 1, evict in pass 2.

__global__ void instancenorm_fp16_v3(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int HW, float eps
) {
    const __half* row = x   + (int64_t)blockIdx.x * HW;
    __half*        op = out + (int64_t)blockIdx.x * HW;
    int n4 = HW >> 3;
    const float4* row4 = reinterpret_cast<const float4*>(row);
    float4*       out4 = reinterpret_cast<float4*>(op);
    int stride = blockDim.x;
    int stride8 = 8 * stride;

    // Pass 1: __ldcg (L2-only) -- data stays in L2 for pass 2
    float s1 = 0.0f, s2 = 0.0f;
    int j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 A = __ldcg(&row4[j]),          B = __ldcg(&row4[j+stride]);
        float4 C = __ldcg(&row4[j+2*stride]), D = __ldcg(&row4[j+3*stride]);
        float4 E = __ldcg(&row4[j+4*stride]), F = __ldcg(&row4[j+5*stride]);
        float4 G = __ldcg(&row4[j+6*stride]), H = __ldcg(&row4[j+7*stride]);
        #define ACC8(V) {                                                            \
            const half2* h = (const half2*)&(V);                                    \
            float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);                \
            float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);                \
            s1 += f0.x+f0.y+f1.x+f1.y+f2.x+f2.y+f3.x+f3.y;                       \
            s2 += f0.x*f0.x+f0.y*f0.y+f1.x*f1.x+f1.y*f1.y                         \
                + f2.x*f2.x+f2.y*f2.y+f3.x*f3.x+f3.y*f3.y; }
        ACC8(A); ACC8(B); ACC8(C); ACC8(D);
        ACC8(E); ACC8(F); ACC8(G); ACC8(H);
        #undef ACC8
    }
    for (; j < n4; j += stride) {
        float4 v = __ldcg(&row4[j]);
        const half2* h = (const half2*)&v;
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
        s1 += f0.x+f0.y+f1.x+f1.y+f2.x+f2.y+f3.x+f3.y;
        s2 += f0.x*f0.x+f0.y*f0.y+f1.x*f1.x+f1.y*f1.y
            + f2.x*f2.x+f2.y*f2.y+f3.x*f3.x+f3.y*f3.y;
    }

    __shared__ float ws1[32], ws2[32];
    __shared__ float bias_sh, is_sh;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int off=16; off>0; off>>=1) {
        s1 += __shfl_down_sync(0xffffffff, s1, off);
        s2 += __shfl_down_sync(0xffffffff, s2, off);
    }
    if (lane==0) { ws1[wid]=s1; ws2[wid]=s2; }
    __syncthreads();
    if (wid==0) {
        int nw = blockDim.x >> 5;
        s1 = (lane<nw) ? ws1[lane] : 0.0f;
        s2 = (lane<nw) ? ws2[lane] : 0.0f;
        for (int off=16; off>0; off>>=1) {
            s1 += __shfl_down_sync(0xffffffff, s1, off);
            s2 += __shfl_down_sync(0xffffffff, s2, off);
        }
        if (lane==0) {
            float mu = s1 / (float)HW;
            float var = s2 / (float)HW - mu*mu;
            float is = rsqrtf(var + eps);
            is_sh = is;  bias_sh = -mu*is;
        }
    }
    __syncthreads();

    // Pass 2: __ldlu (evict-after-use) + __stwt (write-through)
    float is = is_sh, bias = bias_sh;
    j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 A = __ldlu(&row4[j]),          B = __ldlu(&row4[j+stride]);
        float4 C = __ldlu(&row4[j+2*stride]), D = __ldlu(&row4[j+3*stride]);
        float4 E = __ldlu(&row4[j+4*stride]), F = __ldlu(&row4[j+5*stride]);
        float4 G = __ldlu(&row4[j+6*stride]), H = __ldlu(&row4[j+7*stride]);
        #define NORM8(V) ({                                                          \
            const half2* h = (const half2*)&(V);                                    \
            float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);                \
            float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);                \
            half2 r0=__floats2half2_rn(fmaf(f0.x,is,bias),fmaf(f0.y,is,bias));     \
            half2 r1=__floats2half2_rn(fmaf(f1.x,is,bias),fmaf(f1.y,is,bias));     \
            half2 r2=__floats2half2_rn(fmaf(f2.x,is,bias),fmaf(f2.y,is,bias));     \
            half2 r3=__floats2half2_rn(fmaf(f3.x,is,bias),fmaf(f3.y,is,bias));     \
            float4 res; half2* hr=(half2*)&res;                                     \
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
        float4 v = __ldlu(&row4[j]);
        const half2* h = (const half2*)&v;
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
        half2 r0=__floats2half2_rn(fmaf(f0.x,is,bias),fmaf(f0.y,is,bias));
        half2 r1=__floats2half2_rn(fmaf(f1.x,is,bias),fmaf(f1.y,is,bias));
        half2 r2=__floats2half2_rn(fmaf(f2.x,is,bias),fmaf(f2.y,is,bias));
        half2 r3=__floats2half2_rn(fmaf(f3.x,is,bias),fmaf(f3.y,is,bias));
        float4 res; half2* hr=(half2*)&res;
        hr[0]=r0; hr[1]=r1; hr[2]=r2; hr[3]=r3;
        __stwt(&out4[j], res);
    }
}

torch::Tensor instancenorm_fp16_cuda(torch::Tensor x, float eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    auto xc = x.contiguous();
    int B = xc.size(0), C = xc.size(1);
    int HW = xc.size(2) * xc.size(3);
    auto out = torch::empty_like(xc);
    const __half* xp = reinterpret_cast<const __half*>(xc.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    instancenorm_fp16_v3<<<B * C, 1024>>>(xp, op, HW, eps);
    return out;
}
"""

cpp_source = "torch::Tensor instancenorm_fp16_cuda(torch::Tensor x, float eps);"

module = load_inline(
    name='instancenorm_fp16_v3',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['instancenorm_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.eps = 1e-5

    def forward(self, x):
        return module.instancenorm_fp16_cuda(x, self.eps)
