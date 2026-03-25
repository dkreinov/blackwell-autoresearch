import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused BatchNorm: 1 block per channel, 2 passes.
// Pass 1 (DRAM): sum + sumsq over all N*HW = 16.7M elements per channel
//   Inner loop: float4 8x unroll within each (n,c) slice
// Pass 2 (L2 reuse): y = x * scale + shift (no expf, trivial FMA)
//
// N=64, C=64, H=512, W=512. HW=262144=65536 float4s.
// NCHW layout: channel c elements are N strided slices.
// weight=1, bias=0 (default init) → scale = inv_std, shift = -mean*inv_std.
// 1024 threads, 64 blocks total (1 per channel).

__global__ __launch_bounds__(1024)
void batchnorm_fused(const float* __restrict__ x, float* __restrict__ out,
                     const float* __restrict__ weight, const float* __restrict__ bias,
                     int N, int C, int HW, float eps) {
    int c = blockIdx.x;  // channel 0..C-1
    int HW4 = HW >> 2;  // 65536 float4s per slice

    // ---- Pass 1: sum + sumsq over all N slices of channel c ----
    float s1 = 0.f, s2 = 0.f;
    int stride  = blockDim.x;   // 1024
    int stride8 = stride << 3;  // 8192

    for (int n = 0; n < N; n++) {
        const float4* rp4 = reinterpret_cast<const float4*>(x) + (int64_t)(n * C + c) * HW4;
        int j = threadIdx.x;
        for (; j + 7 * stride < HW4; j += stride8) {
            float4 v0=rp4[j],         v1=rp4[j+stride],   v2=rp4[j+2*stride], v3=rp4[j+3*stride];
            float4 v4=rp4[j+4*stride],v5=rp4[j+5*stride], v6=rp4[j+6*stride], v7=rp4[j+7*stride];
            s1 += v0.x+v0.y+v0.z+v0.w + v1.x+v1.y+v1.z+v1.w + v2.x+v2.y+v2.z+v2.w + v3.x+v3.y+v3.z+v3.w
                + v4.x+v4.y+v4.z+v4.w + v5.x+v5.y+v5.z+v5.w + v6.x+v6.y+v6.z+v6.w + v7.x+v7.y+v7.z+v7.w;
            s2 += v0.x*v0.x+v0.y*v0.y+v0.z*v0.z+v0.w*v0.w
                + v1.x*v1.x+v1.y*v1.y+v1.z*v1.z+v1.w*v1.w
                + v2.x*v2.x+v2.y*v2.y+v2.z*v2.z+v2.w*v2.w
                + v3.x*v3.x+v3.y*v3.y+v3.z*v3.z+v3.w*v3.w
                + v4.x*v4.x+v4.y*v4.y+v4.z*v4.z+v4.w*v4.w
                + v5.x*v5.x+v5.y*v5.y+v5.z*v5.z+v5.w*v5.w
                + v6.x*v6.x+v6.y*v6.y+v6.z*v6.z+v6.w*v6.w
                + v7.x*v7.x+v7.y*v7.y+v7.z*v7.z+v7.w*v7.w;
        }
        for (; j < HW4; j += stride) {
            float4 v = rp4[j];
            s1 += v.x+v.y+v.z+v.w;
            s2 += v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w;
        }
    }

    for (int off = 16; off > 0; off >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, off);
        s2 += __shfl_down_sync(0xffffffff, s2, off);
    }

    __shared__ float ws1[16], ws2[16];
    __shared__ float final_scale, final_shift;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { ws1[wid] = s1; ws2[wid] = s2; }
    __syncthreads();

    if (wid == 0) {
        int nw = blockDim.x >> 5;
        s1 = (lane < nw) ? ws1[lane] : 0.f;
        s2 = (lane < nw) ? ws2[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            s1 += __shfl_down_sync(0xffffffff, s1, off);
            s2 += __shfl_down_sync(0xffffffff, s2, off);
        }
        if (lane == 0) {
            float count = (float)(N * HW);
            float mean  = s1 / count;
            float var   = s2 / count - mean * mean;
            float w_c   = weight[c];
            float b_c   = bias[c];
            float inv_s = rsqrtf(var + eps);
            final_scale = w_c * inv_s;
            final_shift = b_c - w_c * mean * inv_s;
        }
    }
    __syncthreads();

    // ---- Pass 2: y = x * scale + shift, float4 8x unroll ----
    float scale = final_scale, shift = final_shift;
    for (int n = 0; n < N; n++) {
        const float4* rp4 = reinterpret_cast<const float4*>(x)   + (int64_t)(n * C + c) * HW4;
        float4*       op4 = reinterpret_cast<float4*>(out) + (int64_t)(n * C + c) * HW4;
        int j = threadIdx.x;
        for (; j + 7 * stride < HW4; j += stride8) {
            float4 v0=rp4[j],         v1=rp4[j+stride],   v2=rp4[j+2*stride], v3=rp4[j+3*stride];
            float4 v4=rp4[j+4*stride],v5=rp4[j+5*stride], v6=rp4[j+6*stride], v7=rp4[j+7*stride];
            op4[j]          = {fmaf(v0.x,scale,shift), fmaf(v0.y,scale,shift), fmaf(v0.z,scale,shift), fmaf(v0.w,scale,shift)};
            op4[j+stride]   = {fmaf(v1.x,scale,shift), fmaf(v1.y,scale,shift), fmaf(v1.z,scale,shift), fmaf(v1.w,scale,shift)};
            op4[j+2*stride] = {fmaf(v2.x,scale,shift), fmaf(v2.y,scale,shift), fmaf(v2.z,scale,shift), fmaf(v2.w,scale,shift)};
            op4[j+3*stride] = {fmaf(v3.x,scale,shift), fmaf(v3.y,scale,shift), fmaf(v3.z,scale,shift), fmaf(v3.w,scale,shift)};
            op4[j+4*stride] = {fmaf(v4.x,scale,shift), fmaf(v4.y,scale,shift), fmaf(v4.z,scale,shift), fmaf(v4.w,scale,shift)};
            op4[j+5*stride] = {fmaf(v5.x,scale,shift), fmaf(v5.y,scale,shift), fmaf(v5.z,scale,shift), fmaf(v5.w,scale,shift)};
            op4[j+6*stride] = {fmaf(v6.x,scale,shift), fmaf(v6.y,scale,shift), fmaf(v6.z,scale,shift), fmaf(v6.w,scale,shift)};
            op4[j+7*stride] = {fmaf(v7.x,scale,shift), fmaf(v7.y,scale,shift), fmaf(v7.z,scale,shift), fmaf(v7.w,scale,shift)};
        }
        for (; j < HW4; j += stride) {
            float4 v = rp4[j];
            op4[j] = {fmaf(v.x,scale,shift), fmaf(v.y,scale,shift), fmaf(v.z,scale,shift), fmaf(v.w,scale,shift)};
        }
    }
}

torch::Tensor batchnorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps) {
    auto xc = x.contiguous();
    int N  = xc.size(0);
    int C  = xc.size(1);
    int HW = xc.size(2) * xc.size(3);
    auto out = torch::empty_like(xc);
    batchnorm_fused<<<C, 1024>>>(
        xc.data_ptr<float>(), out.data_ptr<float>(),
        weight.data_ptr<float>(), bias.data_ptr<float>(),
        N, C, HW, eps);
    return out;
}
"""

cpp_source = "torch::Tensor batchnorm_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps);"

batchnorm_module = load_inline(
    name='batchnorm_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batchnorm_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return batchnorm_module.batchnorm_cuda(x.cuda(), self.weight.cuda(), self.bias.cuda(), 1e-5)
