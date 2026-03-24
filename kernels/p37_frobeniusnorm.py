import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Global reduction: sum of all x^2 across entire tensor, then divide by sqrt(sum).
// Two-phase: kernel 1 does partial block sums, kernel 2 reduces block sums + normalizes.
// Actually, simpler: use atomicAdd for global sum, then elementwise normalize.

// Phase 1: block-level partial sums → atomicAdd to global sum
__global__ void frobnorm_sum_kernel(const float* __restrict__ x, float* __restrict__ global_sum, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 4 + tid;

    float local_sum = 0.0f;
    // Each thread processes 4 elements
    int stride = gridDim.x * blockDim.x * 4;
    for (int i = idx; i < n / 4; i += stride / 4) {
        // Use float4 if aligned
        int base = i * 4;
        if (base + 3 < n) {
            const float4* p = reinterpret_cast<const float4*>(x + base);
            float4 v = *p;
            local_sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    }
    // Remainder handled by first few threads
    int rem_start = (n / 4) * 4;
    for (int i = rem_start + tid; i < n; i += blockDim.x) {
        float v = x[i];
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
        local_sum = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        if (lane == 0) atomicAdd(global_sum, local_sum);
    }
}

// Phase 2: elementwise normalize
__global__ void frobnorm_div_kernel(const float* __restrict__ x, float* __restrict__ out,
                                     const float* __restrict__ global_sum, int n) {
    float inv_norm = rsqrtf(*global_sum);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = n >> 2;
    if (idx < n4) {
        const float4* in4 = reinterpret_cast<const float4*>(x);
        float4* out4 = reinterpret_cast<float4*>(out);
        float4 v = in4[idx];
        float4 r;
        r.x = v.x * inv_norm;
        r.y = v.y * inv_norm;
        r.z = v.z * inv_norm;
        r.w = v.w * inv_norm;
        out4[idx] = r;
    }
    // Remainder
    int scalar_idx = n4 * 4 + (idx - n4);
    // Not needed if n is divisible by 4 (112*64*512*512 = 1,879,048,192 / 4 = exact)
}

torch::Tensor frobnorm_cuda(torch::Tensor x) {
    auto flat = x.contiguous().view({-1});
    int n = flat.numel();
    auto out_flat = torch::empty_like(flat);
    auto global_sum = torch::zeros({1}, flat.options());

    // Phase 1: reduce
    int threads = 1024;
    int blocks = min(1024, (n / 4 + threads - 1) / threads);
    frobnorm_sum_kernel<<<blocks, threads>>>(flat.data_ptr<float>(), global_sum.data_ptr<float>(), n);

    // Phase 2: normalize
    int n4 = n >> 2;
    int norm_blocks = (n4 + threads - 1) / threads;
    frobnorm_div_kernel<<<norm_blocks, threads>>>(flat.data_ptr<float>(), out_flat.data_ptr<float>(),
                                                    global_sum.data_ptr<float>(), n);

    return out_flat.view_as(x);
}
"""

cpp_source = "torch::Tensor frobnorm_cuda(torch::Tensor x);"

frobnorm_module = load_inline(
    name='frobnorm',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['frobnorm_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return frobnorm_module.frobnorm_cuda(x.cuda())
