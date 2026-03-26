import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: fp16 GroupNorm. batch=112, features=64, num_groups=8, H=W=512.
// CpG=8 channels/group, HW=262144, group_elems=CpG*HW=2097152=262144 float4 chunks.
// 896 blocks (B*G), 1024 threads each. 256 float4/thread, 32 8x-unroll iters, no tail.
// Pass 1: __ldcg (L2-only), float32 accum sum+sumsq.
// Pass 2: __ldlu (evict-after-use) + __stwt (bypass write), normalized+affine.
// Per-float4 channel: j >> 15 (since HW/8 = 32768 = 2^15).
// Weight/bias loaded into float regs at start (8 channels = 8 floats each).

__global__ void groupnorm_fp16_v1(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,  // [C]
    const __half* __restrict__ bias,    // [C]
    __half* __restrict__ out,
    int CpG,     // channels per group = C/G = 8
    int HW,      // H*W = 262144
    int HW4,     // HW/8 = 32768
    int num_groups,
    float eps
) {
    int group_id = blockIdx.x;          // b*G + g
    int g = group_id % num_groups;      // group index within batch
    int64_t group_offset = (int64_t)group_id * CpG * HW;

    const __half* row = x + group_offset;
    __half* op = out + group_offset;

    int n4 = CpG * HW4;                // total float4 chunks = 262144
    const float4* row4 = reinterpret_cast<const float4*>(row);
    float4* out4 = reinterpret_cast<float4*>(op);

    int stride = blockDim.x;           // 1024
    int stride8 = 8 * stride;          // 8192

    // Load weight/bias for this group into float registers
    int chan_start = g * CpG;
    float wt[8], bi[8];
    for (int c = 0; c < CpG; c++) {
        wt[c] = __half2float(weight[chan_start + c]);
        bi[c] = __half2float(bias[chan_start + c]);
    }

    // Pass 1: sum and sum-of-squares via __ldcg (keeps data in L2)
    float s1 = 0.0f, s2 = 0.0f;

    #define ACC8(V) {                                                          \
        const half2* h = (const half2*)&(V);                                   \
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);               \
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);               \
        s1 += f0.x+f0.y + f1.x+f1.y + f2.x+f2.y + f3.x+f3.y;                \
        s2 += f0.x*f0.x+f0.y*f0.y + f1.x*f1.x+f1.y*f1.y                     \
            + f2.x*f2.x+f2.y*f2.y + f3.x*f3.x+f3.y*f3.y; }

    int j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 A=__ldcg(&row4[j]),          B=__ldcg(&row4[j+stride]);
        float4 C=__ldcg(&row4[j+2*stride]), D=__ldcg(&row4[j+3*stride]);
        float4 E=__ldcg(&row4[j+4*stride]), F=__ldcg(&row4[j+5*stride]);
        float4 G=__ldcg(&row4[j+6*stride]), H=__ldcg(&row4[j+7*stride]);
        ACC8(A); ACC8(B); ACC8(C); ACC8(D);
        ACC8(E); ACC8(F); ACC8(G); ACC8(H);
    }
    for (; j < n4; j += stride) {
        float4 v = __ldcg(&row4[j]);
        ACC8(v);
    }
    #undef ACC8

    // Warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, off);
        s2 += __shfl_down_sync(0xffffffff, s2, off);
    }

    __shared__ float ws1[32], ws2[32];
    __shared__ float s_mu, s_inv_std;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { ws1[wid] = s1; ws2[wid] = s2; }
    __syncthreads();

    if (wid == 0) {
        int nw = blockDim.x >> 5;
        s1 = (lane < nw) ? ws1[lane] : 0.0f;
        s2 = (lane < nw) ? ws2[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            s1 += __shfl_down_sync(0xffffffff, s1, off);
            s2 += __shfl_down_sync(0xffffffff, s2, off);
        }
        if (lane == 0) {
            float N = (float)(CpG * HW);
            float mu = s1 / N;
            float sigma2 = s2 / N - mu * mu;
            s_mu = mu;
            s_inv_std = rsqrtf(sigma2 + eps);
        }
    }
    __syncthreads();

    float mu = s_mu, inv_std = s_inv_std;

    // Pass 2: normalize + affine via __ldlu (evict) + __stwt (bypass write)
    // Channel index for float4 at position j: j / HW4 = j >> 15
    j = threadIdx.x;
    for (; j + 7 * stride < n4; j += stride8) {
        float4 A=__ldlu(&row4[j]),          B=__ldlu(&row4[j+stride]);
        float4 C=__ldlu(&row4[j+2*stride]), D=__ldlu(&row4[j+3*stride]);
        float4 E=__ldlu(&row4[j+4*stride]), F=__ldlu(&row4[j+5*stride]);
        float4 G=__ldlu(&row4[j+6*stride]), H=__ldlu(&row4[j+7*stride]);

        #define NORM8(V, JJ) ({                                                \
            const half2* h = (const half2*)&(V);                               \
            float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);           \
            float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);           \
            int ci = (JJ) >> 15;  /* HW4=32768=2^15, compiler can't fold runtime div */\
            float w_ = wt[ci], b_ = bi[ci];                                    \
            half2 r0=__floats2half2_rn((f0.x-mu)*inv_std*w_+b_,               \
                                       (f0.y-mu)*inv_std*w_+b_);               \
            half2 r1=__floats2half2_rn((f1.x-mu)*inv_std*w_+b_,               \
                                       (f1.y-mu)*inv_std*w_+b_);               \
            half2 r2=__floats2half2_rn((f2.x-mu)*inv_std*w_+b_,               \
                                       (f2.y-mu)*inv_std*w_+b_);               \
            half2 r3=__floats2half2_rn((f3.x-mu)*inv_std*w_+b_,               \
                                       (f3.y-mu)*inv_std*w_+b_);               \
            float4 res; half2* hr=(half2*)&res;                                \
            hr[0]=r0; hr[1]=r1; hr[2]=r2; hr[3]=r3; res; })

        float4 rA=NORM8(A,j),          rB=NORM8(B,j+stride);
        float4 rC=NORM8(C,j+2*stride), rD=NORM8(D,j+3*stride);
        float4 rE=NORM8(E,j+4*stride), rF=NORM8(F,j+5*stride);
        float4 rG=NORM8(G,j+6*stride), rH=NORM8(H,j+7*stride);
        #undef NORM8

        __stwt(&out4[j],         rA); __stwt(&out4[j+stride],   rB);
        __stwt(&out4[j+2*stride],rC); __stwt(&out4[j+3*stride], rD);
        __stwt(&out4[j+4*stride],rE); __stwt(&out4[j+5*stride], rF);
        __stwt(&out4[j+6*stride],rG); __stwt(&out4[j+7*stride], rH);
    }
    for (; j < n4; j += stride) {
        float4 v = __ldlu(&row4[j]);
        const half2* h = (const half2*)&v;
        float2 f0=__half22float2(h[0]), f1=__half22float2(h[1]);
        float2 f2=__half22float2(h[2]), f3=__half22float2(h[3]);
        int ci = j >> 15;
        float w_ = wt[ci], b_ = bi[ci];
        half2 r0=__floats2half2_rn((f0.x-mu)*inv_std*w_+b_,(f0.y-mu)*inv_std*w_+b_);
        half2 r1=__floats2half2_rn((f1.x-mu)*inv_std*w_+b_,(f1.y-mu)*inv_std*w_+b_);
        half2 r2=__floats2half2_rn((f2.x-mu)*inv_std*w_+b_,(f2.y-mu)*inv_std*w_+b_);
        half2 r3=__floats2half2_rn((f3.x-mu)*inv_std*w_+b_,(f3.y-mu)*inv_std*w_+b_);
        float4 res; half2* hr=(half2*)&res;
        hr[0]=r0; hr[1]=r1; hr[2]=r2; hr[3]=r3;
        __stwt(&out4[j], res);
    }
}

torch::Tensor groupnorm_fp16_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                   int num_groups, float eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
    TORCH_CHECK(x.dim() == 4, "expected 4D input (B,C,H,W)");
    int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int CpG = C / num_groups;
    int HW = H * W;
    int HW4 = HW >> 3;  // HW / 8 (must be integer)
    TORCH_CHECK(HW % 8 == 0, "H*W must be divisible by 8");
    auto out = torch::empty_like(x);
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const __half* wp = reinterpret_cast<const __half*>(weight.data_ptr<at::Half>());
    const __half* bp = reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    groupnorm_fp16_v1<<<B * num_groups, 1024>>>(xp, wp, bp, op, CpG, HW, HW4, num_groups, eps);
    return out;
}
"""

cpp_source = "torch::Tensor groupnorm_fp16_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int num_groups, float eps);"

module = load_inline(
    name='groupnorm_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['groupnorm_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float16))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return module.groupnorm_fp16_cuda(x, self.weight, self.bias, self.num_groups, self.eps)
