import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 MinGPT NewGELU. Input (8192,8192) fp16. Output same shape.
// GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// 512 threads/block, exact grid (no stride loop). float4 (8 halfs) loads.
// float32 intermediates for tanhf accuracy (half precision tanh is inaccurate).

static __device__ __forceinline__ float new_gelu_f(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + __tanhf(inner));
}

__global__ void gelu_fp16_v1(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    int64_t n4
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i];
    const half2* h = (const half2*)&v;
    float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
    float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
    float4 r;
    half2* hr = (half2*)&r;
    hr[0] = __floats2half2_rn(new_gelu_f(f0.x), new_gelu_f(f0.y));
    hr[1] = __floats2half2_rn(new_gelu_f(f1.x), new_gelu_f(f1.y));
    hr[2] = __floats2half2_rn(new_gelu_f(f2.x), new_gelu_f(f2.y));
    hr[3] = __floats2half2_rn(new_gelu_f(f3.x), new_gelu_f(f3.y));
    out[i] = r;
}

torch::Tensor gelu_fp16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    int64_t n4 = x.numel() >> 3;  // float4 chunks (8 halfs each)
    auto out = torch::empty_like(x);
    const float4* xp = reinterpret_cast<const float4*>(x.data_ptr<at::Half>());
    float4* op = reinterpret_cast<float4*>(out.data_ptr<at::Half>());
    int block = 512;
    int grid = (int)((n4 + block - 1) / block);
    gelu_fp16_v1<<<grid, block>>>(xp, op, n4);
    return out;
}
"""

cpp_source = "torch::Tensor gelu_fp16_cuda(torch::Tensor x);"

module = load_inline(
    name='gelu_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['gelu_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.gelu_fp16_cuda(x)
