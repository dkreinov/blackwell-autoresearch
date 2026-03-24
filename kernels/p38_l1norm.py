import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// v3: 4x float4 per iteration in pass 1 body loop (same trick as p39 v2).
// body4 ≈ 16383, 1024 threads → ~16 iters → ~4 iters with 4x stride.
__global__ void l1norm_kernel(const float* __restrict__ x, float* __restrict__ out, int dim) {
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

    float local_sum = 0.0f;
    for (int i = tid; i < head; i += blockDim.x)
        local_sum += fabsf(row_in[i]);

    // 4x unrolled float4 body
    for (int i = tid; i < body4; i += blockDim.x * 4) {
        float4 a = body_ptr[i];
        float4 b = (i + blockDim.x   < body4) ? body_ptr[i + blockDim.x  ] : make_float4(0,0,0,0);
        float4 c = (i + blockDim.x*2 < body4) ? body_ptr[i + blockDim.x*2] : make_float4(0,0,0,0);
        float4 d = (i + blockDim.x*3 < body4) ? body_ptr[i + blockDim.x*3] : make_float4(0,0,0,0);
        local_sum += fabsf(a.x)+fabsf(a.y)+fabsf(a.z)+fabsf(a.w)
                   + fabsf(b.x)+fabsf(b.y)+fabsf(b.z)+fabsf(b.w)
                   + fabsf(c.x)+fabsf(c.y)+fabsf(c.z)+fabsf(c.w)
                   + fabsf(d.x)+fabsf(d.y)+fabsf(d.z)+fabsf(d.w);
    }
    for (int i = tail_start + tid; i < dim; i += blockDim.x)
        local_sum += fabsf(row_in[i]);

    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    __shared__ float warp_sums[32];
    int lane = tid & 31, warp_id = tid >> 5;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }

    __shared__ float s_inv_mean;
    if (tid == 0) s_inv_mean = (float)dim / block_sum;
    __syncthreads();
    float inv_mean = s_inv_mean;

    for (int i = tid; i < head; i += blockDim.x)
        row_out[i] = row_in[i] * inv_mean;

    float4* out_body = reinterpret_cast<float4*>(row_out + body_start);
    for (int i = tid; i < body4; i += blockDim.x) {
        float4 v = body_ptr[i];
        float4 r = {v.x*inv_mean, v.y*inv_mean, v.z*inv_mean, v.w*inv_mean};
        out_body[i] = r;
    }
    for (int i = tail_start + tid; i < dim; i += blockDim.x)
        row_out[i] = row_in[i] * inv_mean;
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
    name='l1norm_v3',
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
