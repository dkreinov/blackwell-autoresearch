import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# v1: float4 single-pass, 1024 threads, exact grid, __tanhf fast math
tanh_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void tanh_kernel(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i];
    float4 r;
    r.x = __tanhf(v.x); r.y = __tanhf(v.y);
    r.z = __tanhf(v.z); r.w = __tanhf(v.w);
    out[i] = r;
}

torch::Tensor tanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat);
    auto out = torch::empty_like(x);
    int64_t n4 = x.numel() / 4;
    tanh_kernel<<<(int)((n4+1023)/1024), 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
    return out;
}
"""
module = load_inline(name="tanh_v1", cpp_sources="torch::Tensor tanh_forward(torch::Tensor x);",
    cuda_sources=tanh_cuda_src, functions=["tanh_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return module.tanh_forward(x)
