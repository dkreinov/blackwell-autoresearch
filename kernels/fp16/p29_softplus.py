import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 Softplus. Input (4096,393216) fp16.
// softplus(x) = log(1 + exp(x)). Numerically stable: x if x > 20, else __logf(1+__expf(x)).
// float4 (8 halfs), float32 compute, __ldlu+__stwt, 1024t exact-grid.

__global__ void softplus_fp16_v1(
    const float4* __restrict__ x,
    float4*       __restrict__ out,
    int64_t n4
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = __ldlu(&x[i]);
    const half2* h = (const half2*)&v;
    float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
    float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
    #define SP(a) ((a) > 20.0f ? (a) : __logf(1.0f + __expf(a)))
    float4 r;
    half2* hr = (half2*)&r;
    hr[0] = __floats2half2_rn(SP(f0.x), SP(f0.y));
    hr[1] = __floats2half2_rn(SP(f1.x), SP(f1.y));
    hr[2] = __floats2half2_rn(SP(f2.x), SP(f2.y));
    hr[3] = __floats2half2_rn(SP(f3.x), SP(f3.y));
    #undef SP
    __stwt(&out[i], r);
}

torch::Tensor softplus_fp16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    int64_t n4 = x.numel() >> 3;
    auto out = torch::empty_like(x);
    const float4* xp = reinterpret_cast<const float4*>(x.data_ptr<at::Half>());
    float4* op = reinterpret_cast<float4*>(out.data_ptr<at::Half>());
    int64_t block = 1024, grid = (n4 + block - 1) / block;
    softplus_fp16_v1<<<grid, block>>>(xp, op, n4);
    return out;
}
"""

cpp_source = "torch::Tensor softplus_fp16_cuda(torch::Tensor x);"

module = load_inline(
    name='softplus_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['softplus_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.softplus_fp16_cuda(x)
