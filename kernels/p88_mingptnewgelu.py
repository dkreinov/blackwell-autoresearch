import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

# v3: 512 threads/block — 3 blocks/SM = 48 warps = 100% occupancy (vs 1 block/SM at 1024)
gelu_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

static constexpr float GELU_C0 = 0.7978845608028654f;
static constexpr float GELU_C1 = 0.044715f;

__device__ __forceinline__ float new_gelu(float x) {
    float x3 = x * x * x;
    float inner = GELU_C0 * (x + GELU_C1 * x3);
    return 0.5f * x * (1.0f + __tanhf(inner));
}

__global__ void gelu_512_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    int64_t n4
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 v = x[i];
    float4 r;
    r.x = new_gelu(v.x);
    r.y = new_gelu(v.y);
    r.z = new_gelu(v.z);
    r.w = new_gelu(v.w);
    out[i] = r;
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");
    auto out = torch::empty_like(x);
    int64_t n4 = x.numel() / 4;

    const float4* xp = reinterpret_cast<const float4*>(x.data_ptr<float>());
    float4* op = reinterpret_cast<float4*>(out.data_ptr<float>());

    int block = 512;
    int grid = (int)((n4 + block - 1) / block);

    gelu_512_kernel<<<grid, block>>>(xp, op, n4);
    return out;
}
"""

gelu_cpp_src = """
torch::Tensor gelu_forward(torch::Tensor x);
"""

module = load_inline(
    name="gelu_v3",
    cpp_sources=gelu_cpp_src,
    cuda_sources=gelu_cuda_src,
    functions=["gelu_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.gelu_forward(x)
