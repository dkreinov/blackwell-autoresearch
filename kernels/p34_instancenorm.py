import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// v5: 4x float4 per thread per stride loop iteration (64 floats vs 16 floats).
// 65536 float4s / 1024 threads = 64 iters → 16 iters with 4x unroll.
// More outstanding loads per issued instruction, better latency hiding.
__global__ __launch_bounds__(1024)
void instancenorm_kernel(const float* __restrict__ x, float* __restrict__ out,
                          int instances, int spatial, float eps) {
    int inst = blockIdx.x;
    if (inst >= instances) return;
    int tid = threadIdx.x;
    long long base = (long long)inst * spatial;

    int spatial4 = spatial >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(x + base);

    float local_sum = 0.0f, local_sq = 0.0f;
    // 4x unrolled stride loop: 4 float4s per thread per iteration
    int stride4 = blockDim.x;
    for (int i = tid; i < spatial4; i += stride4 * 4) {
        float4 a = in4[i];
        float4 b = (i + stride4   < spatial4) ? in4[i + stride4  ] : make_float4(0,0,0,0);
        float4 c = (i + stride4*2 < spatial4) ? in4[i + stride4*2] : make_float4(0,0,0,0);
        float4 d = (i + stride4*3 < spatial4) ? in4[i + stride4*3] : make_float4(0,0,0,0);
        local_sum += a.x+a.y+a.z+a.w + b.x+b.y+b.z+b.w + c.x+c.y+c.z+c.w + d.x+d.y+d.z+d.w;
        local_sq  += a.x*a.x+a.y*a.y+a.z*a.z+a.w*a.w
                   + b.x*b.x+b.y*b.y+b.z*b.z+b.w*b.w
                   + c.x*c.x+c.y*c.y+c.z*c.z+c.w*c.w
                   + d.x*d.x+d.y*d.y+d.z*d.z+d.w*d.w;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq  += __shfl_down_sync(0xffffffff, local_sq,  offset);
    }

    __shared__ float warp_sum[32], warp_sq[32];
    int lane = tid & 31, warp_id = tid >> 5;
    if (lane == 0) { warp_sum[warp_id] = local_sum; warp_sq[warp_id] = local_sq; }
    __syncthreads();

    float block_sum = 0.0f, block_sq = 0.0f;
    if (warp_id == 0) {
        block_sum = (lane < 32) ? warp_sum[lane] : 0.0f;
        block_sq  = (lane < 32) ? warp_sq[lane]  : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            block_sq  += __shfl_down_sync(0xffffffff, block_sq,  offset);
        }
    }

    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        float mean = block_sum / (float)spatial;
        float var  = block_sq  / (float)spatial - mean * mean;
        s_mean    = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = s_mean, inv_std = s_inv_std;

    // Pass 2: normalize
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
    name='instancenorm_v5',
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
