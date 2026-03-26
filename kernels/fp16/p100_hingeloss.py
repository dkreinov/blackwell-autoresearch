import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 HingeLoss. pred (32768,32768) fp16, target (32768,) fp16.
// mean(clamp(1 - pred[i,j]*target[j], 0))
// target[j] = target[k % 32768] for flat index k.
// target is 64KB (32768 halfs) -> fits in L1 cache.
// 2048 blocks x 1024 threads. Flat loop with stride, float4 pred loads (8 halfs).
// float32 accumulation for numerical stability.

__global__ void hinge_fp16_v1(
    const __half* __restrict__ pred,
    const __half* __restrict__ target,
    float* __restrict__ partial_sum,
    int64_t n,     // total elements = 32768*32768 = 1073741824
    int dim        // 32768 (target length = last dim of pred)
) {
    const float4* pred4 = reinterpret_cast<const float4*>(pred);
    int64_t n4 = n >> 3;  // total float4 chunks
    int64_t stride4 = (int64_t)blockDim.x * gridDim.x;
    int64_t j4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0f;
    // Each float4 contains 8 consecutive (i,j) pairs. j = k%dim.
    // For 8 halfs at float4 base j4*8: col = (j4*8 + offset) % dim
    // Since dim=32768, and float4 chunk boundaries: j4*8 is always aligned to 8
    // (since n4 = n/8 and dim=32768 is divisible by 8).
    // So all 8 elements in a float4 chunk are in the same row (same i).
    // col of first element: (j4*8) % dim = (j4 % (dim>>3)) * 8

    for (; j4 < n4; j4 += stride4) {
        float4 v = __ldcg(&pred4[j4]);
        const half2* h = (const half2*)&v;
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);

        // Target column indices for this float4: col = (j4 * 8) % dim
        // All 8 elements: col, col+1, ..., col+7 (no row wrap since dim%8==0)
        int col = (int)((j4 % (dim >> 3)) << 3);  // = (j4 % 4096) * 8
        float t0 = __half2float(target[col]);
        float t1 = __half2float(target[col+1]);
        float t2 = __half2float(target[col+2]);
        float t3 = __half2float(target[col+3]);
        float t4 = __half2float(target[col+4]);
        float t5 = __half2float(target[col+5]);
        float t6 = __half2float(target[col+6]);
        float t7 = __half2float(target[col+7]);

        s += fmaxf(1.0f - f0.x * t0, 0.0f);
        s += fmaxf(1.0f - f0.y * t1, 0.0f);
        s += fmaxf(1.0f - f1.x * t2, 0.0f);
        s += fmaxf(1.0f - f1.y * t3, 0.0f);
        s += fmaxf(1.0f - f2.x * t4, 0.0f);
        s += fmaxf(1.0f - f2.y * t5, 0.0f);
        s += fmaxf(1.0f - f3.x * t6, 0.0f);
        s += fmaxf(1.0f - f3.y * t7, 0.0f);
    }

    // Block reduce
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
    __shared__ float ws[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) ws[wid] = s;
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        s = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffff, s, off);
        if (lane == 0) atomicAdd(partial_sum, s);
    }
}

torch::Tensor hinge_fp16_cuda(torch::Tensor pred, torch::Tensor target) {
    TORCH_CHECK(pred.is_cuda() && pred.is_contiguous());
    TORCH_CHECK(pred.scalar_type() == torch::kHalf, "pred must be float16");
    TORCH_CHECK(target.scalar_type() == torch::kHalf, "target must be float16");
    int64_t n = pred.numel();
    int dim = pred.size(-1);
    TORCH_CHECK(n % 8 == 0, "numel must be divisible by 8");
    auto sum_t = torch::zeros({1}, pred.options().dtype(torch::kFloat32));
    const __half* pp = reinterpret_cast<const __half*>(pred.data_ptr<at::Half>());
    const __half* tp = reinterpret_cast<const __half*>(target.data_ptr<at::Half>());
    hinge_fp16_v1<<<2048, 1024>>>(pp, tp, sum_t.data_ptr<float>(), n, dim);
    return (sum_t / (float)n).to(torch::kHalf).squeeze();
}
"""

cpp_source = "torch::Tensor hinge_fp16_cuda(torch::Tensor pred, torch::Tensor target);"

module = load_inline(
    name='hingeloss_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['hinge_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return module.hinge_fp16_cuda(predictions.contiguous(), targets.contiguous())
