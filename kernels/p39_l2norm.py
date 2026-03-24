import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Same pattern as p38 L1Norm but: sum(x^2) → rsqrtf → multiply.
// One block per row, float4 with alignment handling.
__global__ __launch_bounds__(1024)
void l2norm_kernel(const float* __restrict__ x, float* __restrict__ out, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    long long row_offset = (long long)row * dim;
    const float* row_in = x + row_offset;
    float* row_out = out + row_offset;

    uintptr_t addr = (uintptr_t)row_in;
    int misalign = (addr & 15) >> 2;
    int head = misalign ? (4 - misalign) : 0;
    if (head > dim) head = dim;
    int body_start = head;
    int body4 = (dim - head) >> 2;
    int tail_start = body_start + (body4 << 2);

    const float4* body_ptr = reinterpret_cast<const float4*>(row_in + body_start);

    // Pass 1: sum of squares
    float local_sum = 0.0f;
    for (int i = tid; i < head; i += blockDim.x) {
        float v = row_in[i];
        local_sum += v * v;
    }
    for (int i = tid; i < body4; i += blockDim.x) {
        float4 v = body_ptr[i];
        local_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (int i = tail_start + tid; i < dim; i += blockDim.x) {
        float v = row_in[i];
        local_sum += v * v;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    __shared__ float warp_sums[32];
    int lane = tid & 31, warp_id = tid >> 5;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    __shared__ float s_inv_norm;
    if (tid == 0) s_inv_norm = rsqrtf(local_sum);
    __syncthreads();
    float inv_norm = s_inv_norm;

    // Pass 2: normalize
    for (int i = tid; i < head; i += blockDim.x)
        row_out[i] = row_in[i] * inv_norm;

    float4* out_body = reinterpret_cast<float4*>(row_out + body_start);
    for (int i = tid; i < body4; i += blockDim.x) {
        float4 v = body_ptr[i];
        float4 r;
        r.x = v.x * inv_norm;
        r.y = v.y * inv_norm;
        r.z = v.z * inv_norm;
        r.w = v.w * inv_norm;
        out_body[i] = r;
    }
    for (int i = tail_start + tid; i < dim; i += blockDim.x)
        row_out[i] = row_in[i] * inv_norm;
}

torch::Tensor l2norm_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int batch = x.size(0);
    int dim = x.size(1);
    l2norm_kernel<<<batch, 1024>>>(x.data_ptr<float>(), out.data_ptr<float>(), dim);
    return out;
}
"""

cpp_source = "torch::Tensor l2norm_cuda(torch::Tensor x);"

l2norm_module = load_inline(
    name='l2norm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['l2norm_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm_module.l2norm_cuda(x.cuda())
