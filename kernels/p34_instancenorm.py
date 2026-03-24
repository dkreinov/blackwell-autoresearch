import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// One block per (batch, feature) instance. 262144 spatial elements per instance, contiguous.
// Pass 1: compute mean and variance via warp shuffle reduction.
// Pass 2: normalize (x - mean) / sqrt(var + eps).
__global__ __launch_bounds__(1024)
void instancenorm_kernel(const float* __restrict__ x, float* __restrict__ out,
                          int instances, int spatial, float eps) {
    int inst = blockIdx.x;
    if (inst >= instances) return;
    int tid = threadIdx.x;
    long long base = (long long)inst * spatial;

    // spatial is 262144, divisible by 4 — float4 aligned since base is contiguous
    int spatial4 = spatial >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(x + base);

    // Pass 1: accumulate sum and sum_sq with float4
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < spatial4; i += blockDim.x) {
        float4 v = in4[i];
        local_sum += v.x + v.y + v.z + v.w;
        local_sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    __shared__ float warp_sum[32];
    __shared__ float warp_sum_sq[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        warp_sum[warp_id] = local_sum;
        warp_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    float block_sum = 0.0f, block_sum_sq = 0.0f;
    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        block_sum = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        block_sum_sq = (lane < num_warps) ? warp_sum_sq[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            block_sum_sq += __shfl_down_sync(0xffffffff, block_sum_sq, offset);
        }
    }

    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        float mean = block_sum / (float)spatial;
        float var = block_sum_sq / (float)spatial - mean * mean;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Pass 2: normalize with float4
    float4* out4 = reinterpret_cast<float4*>(out + base);
    for (int i = tid; i < spatial4; i += blockDim.x) {
        float4 v = in4[i];
        float4 r;
        r.x = (v.x - mean) * inv_std;
        r.y = (v.y - mean) * inv_std;
        r.z = (v.z - mean) * inv_std;
        r.w = (v.w - mean) * inv_std;
        out4[i] = r;
    }
}

torch::Tensor instancenorm_cuda(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int batch = x.size(0);
    int features = x.size(1);
    int spatial = x.size(2) * x.size(3);
    int instances = batch * features;

    instancenorm_kernel<<<instances, 1024>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                              instances, spatial, eps);
    return out;
}
"""

cpp_source = "torch::Tensor instancenorm_cuda(torch::Tensor x, float eps);"

instancenorm_module = load_inline(
    name='instancenorm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['instancenorm_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int):
        super(ModelNew, self).__init__()
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return instancenorm_module.instancenorm_cuda(x.cuda(), self.eps)
