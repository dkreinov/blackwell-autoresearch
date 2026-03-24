import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// Online softmax pass 1 (max + sum), then normalize: out = x - max - log(sum).
// No __expf in normalize pass — just subtraction.
__global__ __launch_bounds__(1024)
void logsoftmax_kernel(const float* __restrict__ x, float* __restrict__ out, int dim) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    long long row_offset = (long long)row * dim;
    const float* row_in = x + row_offset;
    float* row_out = out + row_offset;

    int dim4 = dim >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(row_in);

    // Pass 1: online softmax — compute max and sum(exp(x-max))
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;

    for (int i = tid; i < dim4; i += blockDim.x) {
        float4 v = in4[i];
        float chunk_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        float new_max = fmaxf(local_max, chunk_max);
        float correction = __expf(local_max - new_max);
        local_sum = local_sum * correction
                  + __expf(v.x - new_max) + __expf(v.y - new_max)
                  + __expf(v.z - new_max) + __expf(v.w - new_max);
        local_max = new_max;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
        float new_max = fmaxf(local_max, other_max);
        local_sum = local_sum * __expf(local_max - new_max)
                  + other_sum * __expf(other_max - new_max);
        local_max = new_max;
    }

    __shared__ float warp_max[32];
    __shared__ float warp_sum[32];
    int lane = tid & 31, warp_id = tid >> 5;
    if (lane == 0) {
        warp_max[warp_id] = local_max;
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < 32) ? warp_max[lane] : -FLT_MAX;
        local_sum = (lane < 32) ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
            float new_max = fmaxf(local_max, other_max);
            local_sum = local_sum * __expf(local_max - new_max)
                      + other_sum * __expf(other_max - new_max);
            local_max = new_max;
        }
    }

    __shared__ float s_max, s_log_sum;
    if (tid == 0) {
        s_max = local_max;
        s_log_sum = __logf(local_sum);
    }
    __syncthreads();
    float row_max = s_max;
    float log_sum = s_log_sum;

    // Pass 2: out = x - max - log(sum) — no __expf needed!
    float bias = -(row_max + log_sum);
    float4* out4 = reinterpret_cast<float4*>(row_out);
    for (int i = tid; i < dim4; i += blockDim.x) {
        float4 v = in4[i];
        float4 r;
        r.x = v.x + bias;
        r.y = v.y + bias;
        r.z = v.z + bias;
        r.w = v.w + bias;
        out4[i] = r;
    }
}

torch::Tensor logsoftmax_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int batch = x.size(0);
    int dim = x.size(1);
    logsoftmax_kernel<<<batch, 1024>>>(x.data_ptr<float>(), out.data_ptr<float>(), dim);
    return out;
}
"""

cpp_source = "torch::Tensor logsoftmax_cuda(torch::Tensor x);"

logsoftmax_module = load_inline(
    name='logsoftmax',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['logsoftmax_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim: int = 1):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return logsoftmax_module.logsoftmax_cuda(x.cuda())
