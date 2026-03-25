import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void instancenorm_v3(const float* __restrict__ x, float* __restrict__ out,
                                  int HW, float eps) {
    const float* row = x   + (int64_t)blockIdx.x * HW;
    float*       op  = out + (int64_t)blockIdx.x * HW;
    int n4 = HW >> 2;
    const float4* row4 = reinterpret_cast<const float4*>(row);
    float4* out4 = reinterpret_cast<float4*>(op);

    int stride = blockDim.x;
    int stride8 = 8 * stride;

    // Pass 1: float4, 8x unrolled, dual accumulators
    float s1 = 0.0f, s2 = 0.0f;
    int j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 a = row4[j],          b = row4[j+stride];
        float4 c = row4[j+2*stride], d = row4[j+3*stride];
        float4 e = row4[j+4*stride], f = row4[j+5*stride];
        float4 g = row4[j+6*stride], h = row4[j+7*stride];
        s1 += a.x+a.y+a.z+a.w + b.x+b.y+b.z+b.w + c.x+c.y+c.z+c.w + d.x+d.y+d.z+d.w
            + e.x+e.y+e.z+e.w + f.x+f.y+f.z+f.w + g.x+g.y+g.z+g.w + h.x+h.y+h.z+h.w;
        s2 += a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w + b.x*b.x+b.y*b.y+b.z*b.z+b.w*b.w
            + c.x*c.x+c.y*c.y+c.z*c.z+c.w*c.w + d.x*d.x+d.y*d.y+d.z*d.z+d.w*d.w
            + e.x*e.x+e.y*e.y+e.z*e.z+e.w*e.w + f.x*f.x+f.y*f.y+f.z*f.z+f.w*f.w
            + g.x*g.x+g.y*g.y+g.z*g.z+g.w*g.w + h.x*h.x+h.y*h.y+h.z*h.z+h.w*h.w;
    }
    for (; j < n4; j += stride) {
        float4 v = row4[j];
        s1 += v.x+v.y+v.z+v.w;
        s2 += v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w;
    }

    __shared__ float ws1[32], ws2[32];
    __shared__ float bias_sh, is_sh;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;

    for (int off = 16; off > 0; off >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, off);
        s2 += __shfl_down_sync(0xffffffff, s2, off);
    }
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
            float mu = s1 / (float)HW;
            float var = s2 / (float)HW - mu * mu;
            float is = rsqrtf(var + eps);
            is_sh = is;
            bias_sh = -mu * is;  // FMA: out = x*is + bias
        }
    }
    __syncthreads();

    // Pass 2: float4, 8x unrolled, FMA normalize
    float is = is_sh, bias = bias_sh;
    j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 a = row4[j],          b = row4[j+stride];
        float4 c = row4[j+2*stride], d = row4[j+3*stride];
        float4 e = row4[j+4*stride], f = row4[j+5*stride];
        float4 g = row4[j+6*stride], h = row4[j+7*stride];
        out4[j]          = {fmaf(a.x,is,bias),fmaf(a.y,is,bias),fmaf(a.z,is,bias),fmaf(a.w,is,bias)};
        out4[j+stride]   = {fmaf(b.x,is,bias),fmaf(b.y,is,bias),fmaf(b.z,is,bias),fmaf(b.w,is,bias)};
        out4[j+2*stride] = {fmaf(c.x,is,bias),fmaf(c.y,is,bias),fmaf(c.z,is,bias),fmaf(c.w,is,bias)};
        out4[j+3*stride] = {fmaf(d.x,is,bias),fmaf(d.y,is,bias),fmaf(d.z,is,bias),fmaf(d.w,is,bias)};
        out4[j+4*stride] = {fmaf(e.x,is,bias),fmaf(e.y,is,bias),fmaf(e.z,is,bias),fmaf(e.w,is,bias)};
        out4[j+5*stride] = {fmaf(f.x,is,bias),fmaf(f.y,is,bias),fmaf(f.z,is,bias),fmaf(f.w,is,bias)};
        out4[j+6*stride] = {fmaf(g.x,is,bias),fmaf(g.y,is,bias),fmaf(g.z,is,bias),fmaf(g.w,is,bias)};
        out4[j+7*stride] = {fmaf(h.x,is,bias),fmaf(h.y,is,bias),fmaf(h.z,is,bias),fmaf(h.w,is,bias)};
    }
    for (; j < n4; j += stride) {
        float4 v = row4[j];
        out4[j] = {fmaf(v.x,is,bias),fmaf(v.y,is,bias),fmaf(v.z,is,bias),fmaf(v.w,is,bias)};
    }
}

torch::Tensor instancenorm_cuda(torch::Tensor x, float eps) {
    auto xc = x.contiguous();
    int B = xc.size(0), C = xc.size(1);
    int HW = xc.size(2) * xc.size(3);
    auto out = torch::empty_like(xc);
    instancenorm_v3<<<B * C, 1024>>>(xc.data_ptr<float>(), out.data_ptr<float>(), HW, eps);
    return out;
}
"""

cpp_source = "torch::Tensor instancenorm_cuda(torch::Tensor x, float eps);"

instancenorm_module = load_inline(
    name='instancenorm_v3',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['instancenorm_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.eps = 1e-5

    def forward(self, x):
        return instancenorm_module.instancenorm_cuda(x.cuda(), self.eps)
