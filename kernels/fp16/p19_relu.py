import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 ReLU. float4 (8 halfs), float32 max(0,x), default loads/stores, 1024t.
__global__ void relu_fp16_v1(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i];
    const half2* h = (const half2*)&v;
    float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
    float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
    float4 r; half2* hr = (half2*)&r;
    hr[0]=__floats2half2_rn(fmaxf(0.0f,f0.x),fmaxf(0.0f,f0.y));
    hr[1]=__floats2half2_rn(fmaxf(0.0f,f1.x),fmaxf(0.0f,f1.y));
    hr[2]=__floats2half2_rn(fmaxf(0.0f,f2.x),fmaxf(0.0f,f2.y));
    hr[3]=__floats2half2_rn(fmaxf(0.0f,f3.x),fmaxf(0.0f,f3.y));
    out[i] = r;
}
torch::Tensor relu_fp16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kHalf);
    int64_t n4 = x.numel() >> 3;
    auto out = torch::empty_like(x);
    relu_fp16_v1<<<(n4+1023)/1024, 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<at::Half>()),
        reinterpret_cast<float4*>(out.data_ptr<at::Half>()), n4);
    return out;
}
"""
cpp_source = "torch::Tensor relu_fp16_cuda(torch::Tensor x);"
module = load_inline(name='relu_fp16_v1', cpp_sources=cpp_source, cuda_sources=cuda_source,
    functions=['relu_fp16_cuda'], extra_cuda_cflags=['-O3', '--use_fast_math'], verbose=False)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.relu_fp16_cuda(x)
