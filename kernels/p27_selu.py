import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# v2: precompute SCALE*ALPHA as one constant, reduce multiply count
selu_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

static constexpr float SELU_SCALE = 1.0507009873554805f;
static constexpr float SELU_ALPHA = 1.6732631550316224f;
static constexpr float SELU_SA = 1.0507009873554805f * 1.6732631550316224f;  // scale*alpha

__global__ void selu_kernel(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i]; float4 r;
    r.x = v.x > 0.0f ? SELU_SCALE * v.x : SELU_SA * (__expf(v.x) - 1.0f);
    r.y = v.y > 0.0f ? SELU_SCALE * v.y : SELU_SA * (__expf(v.y) - 1.0f);
    r.z = v.z > 0.0f ? SELU_SCALE * v.z : SELU_SA * (__expf(v.z) - 1.0f);
    r.w = v.w > 0.0f ? SELU_SCALE * v.w : SELU_SA * (__expf(v.w) - 1.0f);
    out[i] = r;
}

torch::Tensor selu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat);
    auto out = torch::empty_like(x); int64_t n4 = x.numel() / 4;
    selu_kernel<<<(int)((n4+1023)/1024), 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
    return out;
}
"""
module = load_inline(name="selu_v2", cpp_sources="torch::Tensor selu_forward(torch::Tensor x);",
    cuda_sources=selu_cuda_src, functions=["selu_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return module.selu_forward(x)
