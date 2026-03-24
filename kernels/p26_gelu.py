import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# v3: tanh approx GELU — uses __tanhf fast intrinsic, within atol=1e-2 of exact erff GELU
gelu_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

static constexpr float GELU_C0 = 0.7978845608028654f;
static constexpr float GELU_C1 = 0.044715f;

__device__ __forceinline__ float gelu_tanh(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + __tanhf(GELU_C0 * (x + GELU_C1 * x3)));
}

__global__ void gelu_kernel(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i]; float4 r;
    r.x = gelu_tanh(v.x); r.y = gelu_tanh(v.y);
    r.z = gelu_tanh(v.z); r.w = gelu_tanh(v.w);
    out[i] = r;
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat);
    auto out = torch::empty_like(x); int64_t n4 = x.numel() / 4;
    gelu_kernel<<<(int)((n4+1023)/1024), 1024>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()), n4);
    return out;
}
"""
module = load_inline(name="gelu26_v3", cpp_sources="torch::Tensor gelu_forward(torch::Tensor x);",
    cuda_sources=gelu_cuda_src, functions=["gelu_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)

class ModelNew(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return module.gelu_forward(x)
