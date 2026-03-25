import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused L1 norm: two passes within same block.
// Pass 1: abs-sum (fills L2). Pass 2: normalize (reads L2, writes DRAM).
// 8x manual unroll to hide memory latency and reduce loop overhead.
__global__ void l1norm_fused(const float* __restrict__ x, float* __restrict__ out, int64_t dim) {
    int row = blockIdx.x;
    const float* rp = x + (int64_t)row * dim;
    float*       op = out + (int64_t)row * dim;
    int stride = blockDim.x;  // 1024
    int stride8 = 8 * stride;  // 8192

    // Pass 1: abs-sum with 8x unroll
    float s = 0.0f;
    int j = threadIdx.x;
    for (; j + 7 * stride < (int)dim; j += stride8) {
        s += fabsf(rp[j]);         s += fabsf(rp[j+stride]);
        s += fabsf(rp[j+2*stride]);s += fabsf(rp[j+3*stride]);
        s += fabsf(rp[j+4*stride]);s += fabsf(rp[j+5*stride]);
        s += fabsf(rp[j+6*stride]);s += fabsf(rp[j+7*stride]);
    }
    for (; j < (int)dim; j += stride)
        s += fabsf(rp[j]);

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

    // Pass 2: normalize 8x unrolled (reads L2, writes DRAM)
    float im = inv_mean;
    j = threadIdx.x;
    for (; j + 7 * stride < (int)dim; j += stride8) {
        op[j]          = rp[j]          * im; op[j+stride]   = rp[j+stride]   * im;
        op[j+2*stride] = rp[j+2*stride] * im; op[j+3*stride] = rp[j+3*stride] * im;
        op[j+4*stride] = rp[j+4*stride] * im; op[j+5*stride] = rp[j+5*stride] * im;
        op[j+6*stride] = rp[j+6*stride] * im; op[j+7*stride] = rp[j+7*stride] * im;
    }
    for (; j < (int)dim; j += stride)
        op[j] = rp[j] * im;
}

torch::Tensor l1norm_cuda(torch::Tensor x) {
    auto xc = x.contiguous();
    int rows = xc.size(0);
    int64_t dim = xc.size(1);
    auto out = torch::empty_like(xc);
    l1norm_fused<<<rows, 1024>>>(xc.data_ptr<float>(), out.data_ptr<float>(), dim);
    return out;
}
"""

cpp_source = "torch::Tensor l1norm_cuda(torch::Tensor x);"

l1norm_module = load_inline(
    name='l1norm_fused',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l1norm_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return l1norm_module.l1norm_cuda(x.cuda())
