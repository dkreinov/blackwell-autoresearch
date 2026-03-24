import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// v6 with __launch_bounds__ to tell compiler max 512 threads, min 1 block.
// This allows compiler to use up to 128 registers/thread (65536/512).
__global__ __launch_bounds__(512, 1)
void rmsnorm_kernel(const float* __restrict__ x, float* __restrict__ out,
                    int batch, int spatial, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * spatial;
    if (idx >= total) return;

    int b = idx / spatial;
    int s = idx % spatial;
    long long base = (long long)b * 64 * spatial + s;
    long long stride = (long long)spatial;

    float sum_sq = 0.0f;
    float vals[64];

    for (int f = 0; f < 64; f += 4) {
        float v0 = x[base + (long long)(f) * stride];
        float v1 = x[base + (long long)(f+1) * stride];
        float v2 = x[base + (long long)(f+2) * stride];
        float v3 = x[base + (long long)(f+3) * stride];
        vals[f]   = v0;
        vals[f+1] = v1;
        vals[f+2] = v2;
        vals[f+3] = v3;
        sum_sq = fmaf(v0, v0, sum_sq);
        sum_sq = fmaf(v1, v1, sum_sq);
        sum_sq = fmaf(v2, v2, sum_sq);
        sum_sq = fmaf(v3, v3, sum_sq);
    }

    float inv_rms = rsqrtf(fmaf(sum_sq, 0.015625f, eps));

    for (int f = 0; f < 64; f += 4) {
        out[base + (long long)(f) * stride]   = vals[f]   * inv_rms;
        out[base + (long long)(f+1) * stride] = vals[f+1] * inv_rms;
        out[base + (long long)(f+2) * stride] = vals[f+2] * inv_rms;
        out[base + (long long)(f+3) * stride] = vals[f+3] * inv_rms;
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int batch = x.size(0);
    int spatial = x.size(2) * x.size(3);

    int total = batch * spatial;
    int threads = 512;
    int blocks = (total + threads - 1) / threads;

    rmsnorm_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                         batch, spatial, eps);
    return out;
}
"""

cpp_source = "torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps);"

rmsnorm_module = load_inline(
    name='rmsnorm_v9',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['rmsnorm_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_module.rmsnorm_cuda(x.cuda(), self.eps)
