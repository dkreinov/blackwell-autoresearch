import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# v2: precompute 1/6 as constant, use __fmaf_rn for (x+3)*inv6
hs_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

static constexpr float INV6 = 1.0f / 6.0f;

__global__ void hs_kernel(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i]; float4 r;
    r.x = fminf(1.0f, fmaxf(0.0f, __fmaf_rn(v.x, INV6, 0.5f)));
    r.y = fminf(1.0f, fmaxf(0.0f, __fmaf_rn(v.y, INV6, 0.5f)));
    r.z = fminf(1.0f, fmaxf(0.0f, __fmaf_rn(v.z, INV6, 0.5f)));
    r.w = fminf(1.0f, fmaxf(0.0f, __fmaf_rn(v.w, INV6, 0.5f)));
    out[i] = r;
}

torch::Tensor hs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat);
    auto out = torch::empty_like(x); int64_t n4 = x.numel() / 4;
    hs_kernel<<<(int)((n4+1023)/1024), 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
    return out;
}
"""
module = load_inline(name="hs_v2", cpp_sources="torch::Tensor hs_forward(torch::Tensor x);",
    cuda_sources=hs_cuda_src, functions=["hs_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return module.hs_forward(x)
