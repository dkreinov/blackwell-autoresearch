import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// v4: Online softmax, 256 threads → 4 blocks/SM for latency hiding.
// Pass 1 (DRAM): online max+sum with 8x float4 unroll.
// Pass 2 (L2):   exp(x-gmax)*inv_sum → write output.
// dim=393216 = 98304 float4, all rows 16B aligned.

__global__ void softmax_v3(const float* __restrict__ x, float* __restrict__ out, int dim) {  // 256t
    int row = blockIdx.x;
    const float* rp = x   + (int64_t)row * dim;
    float*       op = out + (int64_t)row * dim;

    int n4 = dim >> 2;
    const float4* rp4 = reinterpret_cast<const float4*>(rp);
    float4*       op4 = reinterpret_cast<float4*>(op);

    int stride = blockDim.x;       // 512
    int stride8 = 8 * stride;      // 4096

    // ---- Pass 1: Online max+sum ----
    float max_val = -1e30f, sum_val = 0.0f;
    int j = threadIdx.x;

    for (; j + 7 * stride < n4; j += stride8) {
        float4 v0=rp4[j], v1=rp4[j+stride], v2=rp4[j+2*stride], v3=rp4[j+3*stride];
        float4 v4=rp4[j+4*stride], v5=rp4[j+5*stride], v6=rp4[j+6*stride], v7=rp4[j+7*stride];

        float m = max(max(max(v0.x,v0.y),max(v0.z,v0.w)), max(max(v1.x,v1.y),max(v1.z,v1.w)));
        m = max(m, max(max(max(v2.x,v2.y),max(v2.z,v2.w)), max(max(v3.x,v3.y),max(v3.z,v3.w))));
        m = max(m, max(max(max(v4.x,v4.y),max(v4.z,v4.w)), max(max(v5.x,v5.y),max(v5.z,v5.w))));
        m = max(m, max(max(max(v6.x,v6.y),max(v6.z,v6.w)), max(max(v7.x,v7.y),max(v7.z,v7.w))));

        float new_max = max(max_val, m);
        float adjust = __expf(max_val - new_max);
        float partial = __expf(v0.x-m)+__expf(v0.y-m)+__expf(v0.z-m)+__expf(v0.w-m)
                      + __expf(v1.x-m)+__expf(v1.y-m)+__expf(v1.z-m)+__expf(v1.w-m)
                      + __expf(v2.x-m)+__expf(v2.y-m)+__expf(v2.z-m)+__expf(v2.w-m)
                      + __expf(v3.x-m)+__expf(v3.y-m)+__expf(v3.z-m)+__expf(v3.w-m)
                      + __expf(v4.x-m)+__expf(v4.y-m)+__expf(v4.z-m)+__expf(v4.w-m)
                      + __expf(v5.x-m)+__expf(v5.y-m)+__expf(v5.z-m)+__expf(v5.w-m)
                      + __expf(v6.x-m)+__expf(v6.y-m)+__expf(v6.z-m)+__expf(v6.w-m)
                      + __expf(v7.x-m)+__expf(v7.y-m)+__expf(v7.z-m)+__expf(v7.w-m);
        sum_val = sum_val * adjust + partial * __expf(m - new_max);
        max_val = new_max;
    }
    for (; j < n4; j += stride) {
        float4 v = rp4[j];
        float m = max(max(v.x,v.y),max(v.z,v.w));
        float new_max = max(max_val, m);
        float partial = __expf(v.x-m)+__expf(v.y-m)+__expf(v.z-m)+__expf(v.w-m);
        sum_val = sum_val * __expf(max_val - new_max) + partial * __expf(m - new_max);
        max_val = new_max;
    }

    // Warp reduction of (max_val, sum_val)
    for (int off = 16; off > 0; off >>= 1) {
        float om = __shfl_down_sync(0xffffffff, max_val, off);
        float os = __shfl_down_sync(0xffffffff, sum_val, off);
        float nm = max(max_val, om);
        sum_val = sum_val * __expf(max_val - nm) + os * __expf(om - nm);
        max_val = nm;
    }

    __shared__ float wmaxv[8], wsumv[8];     // 256t = 8 warps
    __shared__ float final_max, final_inv_sum;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { wmaxv[wid] = max_val; wsumv[wid] = sum_val; }
    __syncthreads();

    if (wid == 0) {
        int nw = blockDim.x >> 5;  // 8
        max_val = (lane < nw) ? wmaxv[lane] : -1e30f;
        sum_val = (lane < nw) ? wsumv[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            float om = __shfl_down_sync(0xffffffff, max_val, off);
            float os = __shfl_down_sync(0xffffffff, sum_val, off);
            float nm = max(max_val, om);
            sum_val = sum_val * __expf(max_val - nm) + os * __expf(om - nm);
            max_val = nm;
        }
        if (lane == 0) {
            final_max = max_val;
            final_inv_sum = 1.0f / sum_val;
        }
    }
    __syncthreads();

    // ---- Pass 2: normalize, read from L2 ----
    float gmax = final_max, inv_sum = final_inv_sum;
    j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 v0=rp4[j], v1=rp4[j+stride], v2=rp4[j+2*stride], v3=rp4[j+3*stride];
        float4 v4=rp4[j+4*stride], v5=rp4[j+5*stride], v6=rp4[j+6*stride], v7=rp4[j+7*stride];
        op4[j]          = {__expf(v0.x-gmax)*inv_sum, __expf(v0.y-gmax)*inv_sum, __expf(v0.z-gmax)*inv_sum, __expf(v0.w-gmax)*inv_sum};
        op4[j+stride]   = {__expf(v1.x-gmax)*inv_sum, __expf(v1.y-gmax)*inv_sum, __expf(v1.z-gmax)*inv_sum, __expf(v1.w-gmax)*inv_sum};
        op4[j+2*stride] = {__expf(v2.x-gmax)*inv_sum, __expf(v2.y-gmax)*inv_sum, __expf(v2.z-gmax)*inv_sum, __expf(v2.w-gmax)*inv_sum};
        op4[j+3*stride] = {__expf(v3.x-gmax)*inv_sum, __expf(v3.y-gmax)*inv_sum, __expf(v3.z-gmax)*inv_sum, __expf(v3.w-gmax)*inv_sum};
        op4[j+4*stride] = {__expf(v4.x-gmax)*inv_sum, __expf(v4.y-gmax)*inv_sum, __expf(v4.z-gmax)*inv_sum, __expf(v4.w-gmax)*inv_sum};
        op4[j+5*stride] = {__expf(v5.x-gmax)*inv_sum, __expf(v5.y-gmax)*inv_sum, __expf(v5.z-gmax)*inv_sum, __expf(v5.w-gmax)*inv_sum};
        op4[j+6*stride] = {__expf(v6.x-gmax)*inv_sum, __expf(v6.y-gmax)*inv_sum, __expf(v6.z-gmax)*inv_sum, __expf(v6.w-gmax)*inv_sum};
        op4[j+7*stride] = {__expf(v7.x-gmax)*inv_sum, __expf(v7.y-gmax)*inv_sum, __expf(v7.z-gmax)*inv_sum, __expf(v7.w-gmax)*inv_sum};
    }
    for (; j < n4; j += stride) {
        float4 v = rp4[j];
        op4[j] = {__expf(v.x-gmax)*inv_sum, __expf(v.y-gmax)*inv_sum, __expf(v.z-gmax)*inv_sum, __expf(v.w-gmax)*inv_sum};
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    auto xc = x.contiguous();
    int rows = xc.size(0);
    int dim  = xc.size(1);
    auto out = torch::empty_like(xc);
    softmax_v3<<<rows, 256>>>(xc.data_ptr<float>(), out.data_ptr<float>(), dim);
    return out;
}
"""

cpp_source = "torch::Tensor softmax_cuda(torch::Tensor x);"

softmax_module = load_inline(
    name='softmax_v4',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['softmax_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return softmax_module.softmax_cuda(x.cuda())
