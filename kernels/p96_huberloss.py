import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused Huber (smooth_l1) loss: single pass reads both inputs, computes huber element,
// reduces to scalar via multi-block atomicAdd.
// Huber: |d| < 1 ? 0.5*d^2 : |d| - 0.5  (default delta=1)
// Same pattern as MSELoss which got 2.732x.

__global__ __launch_bounds__(1024)
void huber_sum_kernel(const float* __restrict__ pred, const float* __restrict__ targ,
                       float* __restrict__ partial_sum, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float local_sum = 0.f;
    int n4 = n >> 2;
    const float4* pred4 = reinterpret_cast<const float4*>(pred);
    const float4* targ4 = reinterpret_cast<const float4*>(targ);

    for (int i = idx; i < n4; i += stride) {
        float4 p = pred4[i];
        float4 t = targ4[i];

        float d0 = p.x - t.x, d1 = p.y - t.y, d2 = p.z - t.z, d3 = p.w - t.w;
        float a0 = fabsf(d0), a1 = fabsf(d1), a2 = fabsf(d2), a3 = fabsf(d3);
        local_sum += (a0 < 1.f ? 0.5f * d0 * d0 : a0 - 0.5f)
                   + (a1 < 1.f ? 0.5f * d1 * d1 : a1 - 0.5f)
                   + (a2 < 1.f ? 0.5f * d2 * d2 : a2 - 0.5f)
                   + (a3 < 1.f ? 0.5f * d3 * d3 : a3 - 0.5f);
    }

    // Remainder (n not multiple of 4)
    for (int i = n4 * 4 + tid; i < n; i += blockDim.x) {
        float d = pred[i] - targ[i];
        float a = fabsf(d);
        local_sum += a < 1.f ? 0.5f * d * d : a - 0.5f;
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
        local_sum = (lane < nw) ? warp_sums[lane] : 0.f;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        if (lane == 0) atomicAdd(partial_sum, local_sum);
    }
}

torch::Tensor huber_cuda(torch::Tensor pred, torch::Tensor target) {
    auto p_flat = pred.contiguous().view({-1});
    auto t_flat = target.contiguous().view({-1});
    int n = p_flat.numel();
    auto sum_t = torch::zeros({1}, pred.options());

    int threads = 1024;
    int blocks = 4096;
    huber_sum_kernel<<<blocks, threads>>>(p_flat.data_ptr<float>(), t_flat.data_ptr<float>(),
                                           sum_t.data_ptr<float>(), n);
    return (sum_t / (float)n).squeeze();
}
"""

cpp_source = "torch::Tensor huber_cuda(torch::Tensor pred, torch::Tensor target);"

huber_module = load_inline(
    name="huberloss_v2",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['huber_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        return huber_module.huber_cuda(predictions.cuda(), targets.cuda())
