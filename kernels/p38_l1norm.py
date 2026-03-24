import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// One block per row. Two-pass fused: reduce |x|, then divide by mean.
// Uses float4 with alignment handling.
__global__ void l1norm_kernel(const float* __restrict__ x, float* __restrict__ out, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    long long row_offset = (long long)row * dim;
    const float* row_in = x + row_offset;
    float* row_out = out + row_offset;

    // Compute alignment: how many scalar elements before we hit a 16-byte boundary
    uintptr_t addr = (uintptr_t)row_in;
    int misalign = (addr & 15) >> 2;  // number of floats to skip (0-3)
    int head = misalign ? (4 - misalign) : 0;
    if (head > dim) head = dim;

    int body_start = head;
    int body_elems = dim - head;
    int body4 = body_elems >> 2;
    int tail_start = body_start + (body4 << 2);

    // Pass 1: accumulate |x|
    float local_sum = 0.0f;

    // Head scalars
    for (int i = tid; i < head; i += blockDim.x) {
        local_sum += fabsf(row_in[i]);
    }

    // Aligned float4 body
    const float4* body_ptr = reinterpret_cast<const float4*>(row_in + body_start);
    for (int i = tid; i < body4; i += blockDim.x) {
        float4 v = body_ptr[i];
        local_sum += fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }

    // Tail scalars
    for (int i = tail_start + tid; i < dim; i += blockDim.x) {
        local_sum += fabsf(row_in[i]);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Block-level reduction
    __shared__ float warp_sums[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }

    __shared__ float s_inv_mean;
    if (tid == 0) {
        s_inv_mean = (float)dim / block_sum;
    }
    __syncthreads();
    float inv_mean = s_inv_mean;

    // Pass 2: scale with float4
    for (int i = tid; i < head; i += blockDim.x) {
        row_out[i] = row_in[i] * inv_mean;
    }

    float4* out_body = reinterpret_cast<float4*>(row_out + body_start);
    for (int i = tid; i < body4; i += blockDim.x) {
        float4 v = body_ptr[i];
        float4 r;
        r.x = v.x * inv_mean;
        r.y = v.y * inv_mean;
        r.z = v.z * inv_mean;
        r.w = v.w * inv_mean;
        out_body[i] = r;
    }

    for (int i = tail_start + tid; i < dim; i += blockDim.x) {
        row_out[i] = row_in[i] * inv_mean;
    }
}

torch::Tensor l1norm_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int batch = x.size(0);
    int dim = x.size(1);
    l1norm_kernel<<<batch, 1024>>>(x.data_ptr<float>(), out.data_ptr<float>(), dim);
    return out;
}
"""

cpp_source = "torch::Tensor l1norm_cuda(torch::Tensor x);"

l1norm_module = load_inline(
    name='l1norm_v2',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l1norm_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l1norm_module.l1norm_cuda(x.cuda())
