import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 BatchNorm2d. N=64, C=64, H=W=512.
// 1 block per channel (64 blocks). Pass1: sum+sumsq over N*HW=16.7M halfs per channel.
// float4 (8 halfs) loads, 8x unroll per N-slice (32768 float4s/slice, exactly 4 iters).
// Pass1: __ldcg (L2-only). Pass2: __ldlu (evict-after-use) + fmaf in float32 → __stwt.
// N*HW data (32MB) > L2 so cross-pass reuse minimal, but __ldcg helps prefetching.
// Weight/bias in float32 (BN standard). Scale/shift precomputed in shared mem.

__global__ __launch_bounds__(1024)
void batchnorm_fp16_v1(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    const float* __restrict__ weight,  // [C] float32
    const float* __restrict__ bias,    // [C] float32
    int N, int C, int HW, float eps
) {
    int c = blockIdx.x;
    int HW4 = HW >> 3;  // HW/8: float4 chunks per slice = 32768

    // Pass 1: sum + sumsq via __ldcg
    float s1 = 0.0f, s2 = 0.0f;
    int stride  = blockDim.x;   // 1024
    int stride8 = stride << 3;  // 8192

    #define ACC8(V) {                                                              \
        const half2* h = (const half2*)&(V);                                      \
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);                  \
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);                  \
        s1 += f0.x+f0.y + f1.x+f1.y + f2.x+f2.y + f3.x+f3.y;                   \
        s2 += f0.x*f0.x+f0.y*f0.y + f1.x*f1.x+f1.y*f1.y                        \
            + f2.x*f2.x+f2.y*f2.y + f3.x*f3.x+f3.y*f3.y; }

    for (int n = 0; n < N; n++) {
        const float4* rp = reinterpret_cast<const float4*>(x)
                           + (int64_t)(n * C + c) * HW4;
        int j = threadIdx.x;
        for (; j + 7 * stride < HW4; j += stride8) {
            float4 A=__ldcg(&rp[j]),          B=__ldcg(&rp[j+stride]);
            float4 C_=__ldcg(&rp[j+2*stride]),D=__ldcg(&rp[j+3*stride]);
            float4 E=__ldcg(&rp[j+4*stride]), F=__ldcg(&rp[j+5*stride]);
            float4 G=__ldcg(&rp[j+6*stride]), H=__ldcg(&rp[j+7*stride]);
            ACC8(A); ACC8(B); ACC8(C_); ACC8(D);
            ACC8(E); ACC8(F); ACC8(G); ACC8(H);
        }
        for (; j < HW4; j += stride) {
            float4 v = __ldcg(&rp[j]);
            ACC8(v);
        }
    }
    #undef ACC8

    // Block reduce s1, s2
    for (int off = 16; off > 0; off >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, off);
        s2 += __shfl_down_sync(0xffffffff, s2, off);
    }
    __shared__ float ws1[32], ws2[32];
    __shared__ float s_scale, s_shift;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { ws1[wid] = s1; ws2[wid] = s2; }
    __syncthreads();

    if (wid == 0) {
        int nw = blockDim.x >> 5;
        s1 = (lane < nw) ? ws1[lane] : 0.0f;
        s2 = (lane < nw) ? ws2[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            s1 += __shfl_down_sync(0xffffffff, s1, off);
            s2 += __shfl_down_sync(0xffffffff, s2, off);
        }
        if (lane == 0) {
            float count = (float)(N * HW);
            float mean  = s1 / count;
            float var   = s2 / count - mean * mean;
            float inv_s = rsqrtf(var + eps);
            float w_c   = weight[c];
            float b_c   = bias[c];
            s_scale = w_c * inv_s;
            s_shift = b_c - mean * s_scale;
        }
    }
    __syncthreads();
    float sc = s_scale, sh = s_shift;

    // Pass 2: normalize via __ldlu + fmaf + __stwt
    for (int n = 0; n < N; n++) {
        const float4* rp = reinterpret_cast<const float4*>(x)
                           + (int64_t)(n * C + c) * HW4;
        float4* op = reinterpret_cast<float4*>(out)
                     + (int64_t)(n * C + c) * HW4;
        int j = threadIdx.x;
        for (; j + 7 * stride < HW4; j += stride8) {
            float4 A=__ldlu(&rp[j]),          B=__ldlu(&rp[j+stride]);
            float4 C_=__ldlu(&rp[j+2*stride]),D=__ldlu(&rp[j+3*stride]);
            float4 E=__ldlu(&rp[j+4*stride]), F=__ldlu(&rp[j+5*stride]);
            float4 G=__ldlu(&rp[j+6*stride]), H=__ldlu(&rp[j+7*stride]);

            #define NORM8(V) {                                                     \
                const half2* h = (const half2*)&(V);                               \
                float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);           \
                float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);           \
                half2* hr = (half2*)&(V);                                           \
                hr[0]=__floats2half2_rn(fmaf(f0.x,sc,sh), fmaf(f0.y,sc,sh));      \
                hr[1]=__floats2half2_rn(fmaf(f1.x,sc,sh), fmaf(f1.y,sc,sh));      \
                hr[2]=__floats2half2_rn(fmaf(f2.x,sc,sh), fmaf(f2.y,sc,sh));      \
                hr[3]=__floats2half2_rn(fmaf(f3.x,sc,sh), fmaf(f3.y,sc,sh)); }

            NORM8(A); NORM8(B); NORM8(C_); NORM8(D);
            NORM8(E); NORM8(F); NORM8(G); NORM8(H);
            #undef NORM8

            __stwt(&op[j],         A); __stwt(&op[j+stride],   B);
            __stwt(&op[j+2*stride],C_);__stwt(&op[j+3*stride], D);
            __stwt(&op[j+4*stride],E); __stwt(&op[j+5*stride], F);
            __stwt(&op[j+6*stride],G); __stwt(&op[j+7*stride], H);
        }
        for (; j < HW4; j += stride) {
            float4 v = __ldlu(&rp[j]);
            const half2* h = (const half2*)&v;
            float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
            float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
            half2* hr = (half2*)&v;
            hr[0]=__floats2half2_rn(fmaf(f0.x,sc,sh), fmaf(f0.y,sc,sh));
            hr[1]=__floats2half2_rn(fmaf(f1.x,sc,sh), fmaf(f1.y,sc,sh));
            hr[2]=__floats2half2_rn(fmaf(f2.x,sc,sh), fmaf(f2.y,sc,sh));
            hr[3]=__floats2half2_rn(fmaf(f3.x,sc,sh), fmaf(f3.y,sc,sh));
            __stwt(&op[j], v);
        }
    }
}

torch::Tensor batchnorm_fp16_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                   float eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    TORCH_CHECK(x.dim() == 4, "expected 4D input (N,C,H,W)");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int HW = H * W;
    TORCH_CHECK(HW % 8 == 0, "H*W must be divisible by 8");
    auto out = torch::empty_like(x);
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    batchnorm_fp16_v1<<<C, 1024>>>(xp, op, weight.data_ptr<float>(),
                                    bias.data_ptr<float>(), N, C, HW, eps);
    return out;
}
"""

cpp_source = "torch::Tensor batchnorm_fp16_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps);"

module = load_inline(
    name='batchnorm_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batchnorm_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))   # float32
        self.bias   = nn.Parameter(torch.zeros(num_features))  # float32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure float32 for CUDA kernel (weight/bias may be cast to fp16 by .half())
        w = self.weight.float() if self.weight.dtype != torch.float32 else self.weight
        b = self.bias.float() if self.bias.dtype != torch.float32 else self.bias
        return module.batchnorm_fp16_cuda(x, w, b, 1e-5)
