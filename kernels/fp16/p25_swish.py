import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# fp16 Swish: half2 vectorized, 128-bit loads (8 halfs per float4)
# Same exact-grid strategy as fp32 but with half2 compute
swish_fp16_cuda_src = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ void swish_fp16_kernel(
    const half2* __restrict__ x,
    half2* __restrict__ out,
    int n2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n2) return;

    half2 v = x[i];
    half2 one = __float2half2_rn(1.0f);
    half2 neg_v = __hneg2(v);
    half2 sig = h2rcp(__hadd2(one, h2exp(neg_v)));
    out[i] = __hmul2(v, sig);
}

torch::Tensor swish_fp16_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    int64_t n2 = n / 2;  // half2 packing: 2 halfs per element

    const half2* xp = reinterpret_cast<const half2*>(x.data_ptr<at::Half>());
    half2* op = reinterpret_cast<half2*>(out.data_ptr<at::Half>());

    int block = 1024;
    int grid = (int)((n2 + block - 1) / block);

    swish_fp16_kernel<<<grid, block>>>(xp, op, (int)n2);
    return out;
}
"""

swish_fp16_cpp_src = """
torch::Tensor swish_fp16_forward(torch::Tensor x);
"""

module = load_inline(
    name="swish_fp16_v1",
    cpp_sources=swish_fp16_cpp_src,
    cuda_sources=swish_fp16_cuda_src,
    functions=["swish_fp16_forward"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.swish_fp16_forward(x)
