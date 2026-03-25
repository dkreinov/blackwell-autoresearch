import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused MSE: single pass reads both inputs, computes (pred-target)^2, reduces to scalar.
// Uses multi-block reduction with atomicAdd.
__global__ void mse_sum_kernel(const float* __restrict__ pred, const float* __restrict__ target,
                                float* __restrict__ partial_sum, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float local_sum = 0.0f;
    // Float4 vectorized
    int n4 = n >> 2;
    const float4* pred4 = reinterpret_cast<const float4*>(pred);
    const float4* targ4 = reinterpret_cast<const float4*>(target);

    for (int i = idx; i < n4; i += stride) {
        float4 p = pred4[i];
        float4 t = targ4[i];
        float d0 = p.x - t.x, d1 = p.y - t.y, d2 = p.z - t.z, d3 = p.w - t.w;
        local_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }

    // Remainder
    for (int i = n4 * 4 + tid; i < n; i += blockDim.x) {
        float d = pred[i] - target[i];
        local_sum += d * d;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    __shared__ float warp_sums[32];
    int lane = tid & 31, warp_id = tid >> 5;
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        int nw = blockDim.x >> 5;
        local_sum = (lane < nw) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        if (lane == 0) atomicAdd(partial_sum, local_sum);
    }
}

torch::Tensor mse_cuda(torch::Tensor pred, torch::Tensor target) {
    auto pred_flat = pred.contiguous().view({-1});
    auto targ_flat = target.contiguous().view({-1});
    int n = pred_flat.numel();
    auto sum_tensor = torch::zeros({1}, pred.options());

    int threads = 1024;
    int blocks = 512;  // 512 blocks optimal: fewer atomic events, same float4 throughput
    mse_sum_kernel<<<blocks, threads>>>(pred_flat.data_ptr<float>(), targ_flat.data_ptr<float>(),
                                         sum_tensor.data_ptr<float>(), n);

    return (sum_tensor / (float)n).squeeze();
}
"""

cpp_source = "torch::Tensor mse_cuda(torch::Tensor pred, torch::Tensor target);"

mse_module = load_inline(
    name='mseloss',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['mse_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return mse_module.mse_cuda(predictions.cuda(), targets.cuda())
