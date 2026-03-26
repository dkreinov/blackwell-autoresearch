import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 MSELoss. pred,target (32768,32768) fp16. Output: scalar.
// mean((pred-target)^2). Single pass, float4 (8 halfs) vectorized.
// 2048 blocks x 1024 threads. __ldcg for both inputs.
// float32 accumulation.

__global__ void mse_fp16_v1(
    const __half* __restrict__ pred,
    const __half* __restrict__ target,
    float* __restrict__ partial_sum,
    int64_t n4
) {
    const float4* p4 = reinterpret_cast<const float4*>(pred);
    const float4* t4 = reinterpret_cast<const float4*>(target);
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride8 = stride * 8;

    float s = 0.0f;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 P0=__ldcg(&p4[j]),          P1=__ldcg(&p4[j+stride]);
        float4 P2=__ldcg(&p4[j+2*stride]), P3=__ldcg(&p4[j+3*stride]);
        float4 P4=__ldcg(&p4[j+4*stride]), P5=__ldcg(&p4[j+5*stride]);
        float4 P6=__ldcg(&p4[j+6*stride]), P7=__ldcg(&p4[j+7*stride]);
        float4 T0=__ldcg(&t4[j]),          T1=__ldcg(&t4[j+stride]);
        float4 T2=__ldcg(&t4[j+2*stride]), T3=__ldcg(&t4[j+3*stride]);
        float4 T4=__ldcg(&t4[j+4*stride]), T5=__ldcg(&t4[j+5*stride]);
        float4 T6=__ldcg(&t4[j+6*stride]), T7=__ldcg(&t4[j+7*stride]);

        #define MSE8(P, T) {                                                       \
            const half2* ph=(const half2*)&(P); const half2* th=(const half2*)&(T);\
            float2 p0=__half22float2(ph[0]), p1=__half22float2(ph[1]);             \
            float2 p2=__half22float2(ph[2]), p3=__half22float2(ph[3]);             \
            float2 t0=__half22float2(th[0]), t1=__half22float2(th[1]);             \
            float2 t2=__half22float2(th[2]), t3=__half22float2(th[3]);             \
            float d0=p0.x-t0.x, d1=p0.y-t0.y, d2=p1.x-t1.x, d3=p1.y-t1.y;      \
            float d4=p2.x-t2.x, d5=p2.y-t2.y, d6=p3.x-t3.x, d7=p3.y-t3.y;      \
            s += d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7; }

        MSE8(P0,T0); MSE8(P1,T1); MSE8(P2,T2); MSE8(P3,T3);
        MSE8(P4,T4); MSE8(P5,T5); MSE8(P6,T6); MSE8(P7,T7);
        #undef MSE8
    }
    for (; j < n4; j += stride) {
        float4 P=__ldcg(&p4[j]), T=__ldcg(&t4[j]);
        const half2* ph=(const half2*)&P; const half2* th=(const half2*)&T;
        float2 p0=__half22float2(ph[0]), p1=__half22float2(ph[1]);
        float2 p2=__half22float2(ph[2]), p3=__half22float2(ph[3]);
        float2 t0=__half22float2(th[0]), t1=__half22float2(th[1]);
        float2 t2=__half22float2(th[2]), t3=__half22float2(th[3]);
        float d0=p0.x-t0.x, d1=p0.y-t0.y, d2=p1.x-t1.x, d3=p1.y-t1.y;
        float d4=p2.x-t2.x, d5=p2.y-t2.y, d6=p3.x-t3.x, d7=p3.y-t3.y;
        s += d0*d0+d1*d1+d2*d2+d3*d3+d4*d4+d5*d5+d6*d6+d7*d7;
    }

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

torch::Tensor mse_fp16_cuda(torch::Tensor pred, torch::Tensor target) {
    TORCH_CHECK(pred.is_cuda() && pred.is_contiguous() && target.is_contiguous());
    TORCH_CHECK(pred.scalar_type() == torch::kHalf, "pred must be float16");
    int64_t n = pred.numel();
    int64_t n4 = n >> 3;
    auto sum_t = torch::zeros({1}, pred.options().dtype(torch::kFloat32));
    const __half* pp = reinterpret_cast<const __half*>(pred.data_ptr<at::Half>());
    const __half* tp = reinterpret_cast<const __half*>(target.data_ptr<at::Half>());
    mse_fp16_v1<<<2048, 1024>>>(pp, tp, sum_t.data_ptr<float>(), n4);
    return (sum_t / (float)n).to(torch::kHalf).squeeze();
}
"""

cpp_source = "torch::Tensor mse_fp16_cuda(torch::Tensor pred, torch::Tensor target);"

module = load_inline(
    name='mseloss_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['mse_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return module.mse_fp16_cuda(predictions.contiguous(), targets.contiguous())
