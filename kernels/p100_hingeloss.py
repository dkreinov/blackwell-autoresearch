import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused hinge loss: mean(clamp(1 - pred * target, min=0))
// pred: (B, D), target: (B,) broadcast to (B, D) via target[i % D_target]
// Actually: pred (32768, 32768), target (32768,) → broadcast target[j] for pred[i,j]
// Flatten: for flat index k, j = k % dim, target_val = target[j]
__global__ void hinge_sum_kernel(const float* __restrict__ pred, const float* __restrict__ target,
                                  float* __restrict__ partial_sum, int n, int dim) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        int j = i % dim;
        float d = 1.0f - pred[i] * target[j];
        local_sum += fmaxf(d, 0.0f);
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

torch::Tensor hinge_cuda(torch::Tensor pred, torch::Tensor target) {
    auto pred_flat = pred.contiguous().view({-1});
    int n = pred_flat.numel();
    int dim = pred.size(-1);  // last dimension for broadcast
    auto sum_tensor = torch::zeros({1}, pred.options());

    int threads = 1024;
    int blocks = min(2048, (n + threads - 1) / threads);
    hinge_sum_kernel<<<blocks, threads>>>(pred_flat.data_ptr<float>(), target.contiguous().data_ptr<float>(),
                                           sum_tensor.data_ptr<float>(), n, dim);

    return (sum_tensor / (float)n).squeeze();
}
"""

cpp_source = "torch::Tensor hinge_cuda(torch::Tensor pred, torch::Tensor target);"

hinge_module = load_inline(
    name='hingeloss',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['hinge_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return hinge_module.hinge_cuda(predictions.cuda(), targets.cuda())
