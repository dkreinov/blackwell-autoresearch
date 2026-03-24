import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# v1: float4 single-pass, 1024 threads, exact grid, ELU alpha=1.0
elu_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void elu_kernel(const float4* __restrict__ x, float4* __restrict__ o, int64_t n4, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i]; float4 r;
    r.x = v.x > 0.0f ? v.x : alpha * (__expf(v.x) - 1.0f);
    r.y = v.y > 0.0f ? v.y : alpha * (__expf(v.y) - 1.0f);
    r.z = v.z > 0.0f ? v.z : alpha * (__expf(v.z) - 1.0f);
    r.w = v.w > 0.0f ? v.w : alpha * (__expf(v.w) - 1.0f);
    o[i] = r;
}

torch::Tensor elu_forward(torch::Tensor x, float alpha) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat);
    auto o = torch::empty_like(x); int64_t n4 = x.numel() / 4;
    elu_kernel<<<(int)((n4+1023)/1024), 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(o.data_ptr<float>()), n4, alpha);
    return o;
}
"""
module = load_inline(name="elu_v1", cpp_sources="torch::Tensor elu_forward(torch::Tensor x, float alpha);",
    cuda_sources=elu_cuda_src, functions=["elu_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x): return module.elu_forward(x, self.alpha)
