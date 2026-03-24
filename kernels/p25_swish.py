import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# v9: exact grid = ceil(n4 / block), no stride loop
# n4 = 402,653,184, block = 1024 → grid = 393,216 blocks
# Every thread does exactly 1 float4 (or 0 for tail), no loop overhead
swish_cuda_src = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void swish_exact_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    int n4
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;

    float4 v = x[i];
    float4 r;
    r.x = v.x / (1.0f + __expf(-v.x));
    r.y = v.y / (1.0f + __expf(-v.y));
    r.z = v.z / (1.0f + __expf(-v.z));
    r.w = v.w / (1.0f + __expf(-v.w));
    out[i] = r;
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "x must be float32");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    int64_t n4 = n / 4;

    const float4* xp = reinterpret_cast<const float4*>(x.data_ptr<float>());
    float4* op = reinterpret_cast<float4*>(out.data_ptr<float>());

    int block = 1024;
    int grid = (int)((n4 + block - 1) / block);  // exact coverage, no loop

    swish_exact_kernel<<<grid, block>>>(xp, op, (int)n4);
    return out;
}
"""

swish_cpp_src = """
torch::Tensor swish_forward(torch::Tensor x);
"""

module = load_inline(
    name="swish_exact_v9",
    cpp_sources=swish_cpp_src,
    cuda_sources=swish_cuda_src,
    functions=["swish_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.swish_forward(x)
