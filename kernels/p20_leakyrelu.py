import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# v1: float4 single-pass, 1024 threads, exact grid, slope=0.01
leaky_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void leaky_kernel(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i];
    float4 r;
    r.x = v.x > 0.0f ? v.x : 0.01f * v.x;
    r.y = v.y > 0.0f ? v.y : 0.01f * v.y;
    r.z = v.z > 0.0f ? v.z : 0.01f * v.z;
    r.w = v.w > 0.0f ? v.w : 0.01f * v.w;
    out[i] = r;
}

torch::Tensor leaky_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat);
    auto out = torch::empty_like(x);
    int64_t n4 = x.numel() / 4;
    leaky_kernel<<<(int)((n4+1023)/1024), 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
    return out;
}
"""
module = load_inline(name="leaky_v1", cpp_sources="torch::Tensor leaky_forward(torch::Tensor x);",
    cuda_sources=leaky_cuda_src, functions=["leaky_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return module.leaky_forward(x)
