import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# v3: __logf(1+__expf(x)) — explicit fast intrinsics, no log1pf wrapper overhead
sp_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__device__ __forceinline__ float softplus(float x) {
    return x > 20.0f ? x : __logf(1.0f + __expf(x));
}

__global__ void sp_kernel(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i]; float4 r;
    r.x = softplus(v.x); r.y = softplus(v.y); r.z = softplus(v.z); r.w = softplus(v.w);
    out[i] = r;
}

torch::Tensor sp_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat);
    auto out = torch::empty_like(x); int64_t n4 = x.numel() / 4;
    sp_kernel<<<(int)((n4+1023)/1024), 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
    return out;
}
"""
module = load_inline(name="sp_v3", cpp_sources="torch::Tensor sp_forward(torch::Tensor x);",
    cuda_sources=sp_cuda_src, functions=["sp_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return module.sp_forward(x)
