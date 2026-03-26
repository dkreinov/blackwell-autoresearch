import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused fp16 L1 norm: 2-pass L2-reuse kernel.
// dim=65535 is odd -> use scalar __half loads to avoid half2 alignment issues on odd rows.
// Accumulate abs-sum in float32 for numerical safety. 8x unroll.
__global__ void l1norm_fp16_fused(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    int64_t dim
) {
    int row = blockIdx.x;
    const __half* rp = x + (int64_t)row * dim;
    __half* op = out + (int64_t)row * dim;
    int stride = blockDim.x;
    int stride8 = 8 * stride;

    // Pass 1: abs-sum in float32, 8x unroll
    float s = 0.0f;
    int j = threadIdx.x;
    for (; j + 7 * stride < (int)dim; j += stride8) {
        s += fabsf(__half2float(rp[j]));
        s += fabsf(__half2float(rp[j +   stride]));
        s += fabsf(__half2float(rp[j + 2*stride]));
        s += fabsf(__half2float(rp[j + 3*stride]));
        s += fabsf(__half2float(rp[j + 4*stride]));
        s += fabsf(__half2float(rp[j + 5*stride]));
        s += fabsf(__half2float(rp[j + 6*stride]));
        s += fabsf(__half2float(rp[j + 7*stride]));
    }
    for (; j < (int)dim; j += stride)
        s += fabsf(__half2float(rp[j]));

    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);

    __shared__ float ws[32];
    __shared__ float inv_mean;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = s;
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        s = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) inv_mean = (float)dim / s;
    }
    __syncthreads();

    // Pass 2: normalize, reads from L2, 8x unroll
    float im = inv_mean;
    j = threadIdx.x;
    for (; j + 7 * stride < (int)dim; j += stride8) {
        op[j]          = __float2half(__half2float(rp[j])          * im);
        op[j +   stride] = __float2half(__half2float(rp[j +   stride]) * im);
        op[j + 2*stride] = __float2half(__half2float(rp[j + 2*stride]) * im);
        op[j + 3*stride] = __float2half(__half2float(rp[j + 3*stride]) * im);
        op[j + 4*stride] = __float2half(__half2float(rp[j + 4*stride]) * im);
        op[j + 5*stride] = __float2half(__half2float(rp[j + 5*stride]) * im);
        op[j + 6*stride] = __float2half(__half2float(rp[j + 6*stride]) * im);
        op[j + 7*stride] = __float2half(__half2float(rp[j + 7*stride]) * im);
    }
    for (; j < (int)dim; j += stride)
        op[j] = __float2half(__half2float(rp[j]) * im);
}

torch::Tensor l1norm_fp16_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    int rows = x.size(0);
    int64_t dim = x.size(1);
    auto out = torch::empty_like(x);
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    l1norm_fp16_fused<<<rows, 1024>>>(xp, op, dim);
    return out;
}
"""

cpp_source = "torch::Tensor l1norm_fp16_cuda(torch::Tensor x);"

module = load_inline(
    name='l1norm_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l1norm_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return module.l1norm_fp16_cuda(x)
