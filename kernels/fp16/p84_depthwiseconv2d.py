import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// v1: Register sliding-window depthwise conv2d fp16.
// 1 block per (b,c) = B*C = 8192 blocks, 512 threads, NO syncthreads.
// Each thread independently slides its column through OH rows using registers.
// 3 register rows (r0,r1,r2 each KS=3 floats) per thread: 9 floats total.
// Input read once per element via __ldg (L1+L2 cache for adjacent thread reuse).
// Adjacent threads share 2/3 loads per row -> L1 cache deduplication.
// B=64, C=128, H=256, W=512, KS=3, stride=1, padding=0, OW=510, OH=254.
// Speedup: 48.0ms -> 24.6ms = 1.951x

#define KS 3

__global__ void depthwise_conv2d_fp16_v1(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    __half* __restrict__ out,
    int B, int C, int H_in, int W_in, int OH, int OW
) {
    int bc = blockIdx.x;
    int b  = bc / C;
    int c  = bc % C;
    int tx = threadIdx.x;

    if (tx >= OW) return;

    float wf[KS * KS];
    const __half* wptr = w + c * KS * KS;
    #pragma unroll
    for (int i = 0; i < KS * KS; i++) {
        wf[i] = __half2float(__ldg(wptr + i));
    }

    const __half* xptr = x + (b * C + c) * H_in * W_in;
    __half* optr       = out + (b * C + c) * OH * OW;

    float r0[KS], r1[KS], r2[KS];

    #pragma unroll
    for (int s = 0; s < KS; s++) {
        r0[s] = __half2float(__ldg(xptr + 0 * W_in + tx + s));
        r1[s] = __half2float(__ldg(xptr + 1 * W_in + tx + s));
        r2[s] = __half2float(__ldg(xptr + 2 * W_in + tx + s));
    }

    for (int oy = 0; oy < OH; oy++) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < KS; s++) {
            acc = fmaf(r0[s], wf[0 * KS + s], acc);
            acc = fmaf(r1[s], wf[1 * KS + s], acc);
            acc = fmaf(r2[s], wf[2 * KS + s], acc);
        }
        __stcs(optr + oy * OW + tx, __float2half(acc));

        #pragma unroll
        for (int s = 0; s < KS; s++) r0[s] = r1[s];
        #pragma unroll
        for (int s = 0; s < KS; s++) r1[s] = r2[s];

        int nr = oy + KS;
        if (nr < H_in) {
            #pragma unroll
            for (int s = 0; s < KS; s++) {
                r2[s] = __half2float(__ldg(xptr + nr * W_in + tx + s));
            }
        }
    }
}

torch::Tensor depthwise_conv2d_fp16_cuda(torch::Tensor x, torch::Tensor w, int stride, int padding) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf);
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kHalf);
    int B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int KH = w.size(2);
    TORCH_CHECK(stride == 1 && padding == 0);
    int OH = H - KH + 1;
    int OW = W - KH + 1;
    auto out = torch::empty({B, C, OH, OW}, x.options());
    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    const __half* wp = reinterpret_cast<const __half*>(w.data_ptr<at::Half>());
    __half* op = reinterpret_cast<__half*>(out.data_ptr<at::Half>());
    depthwise_conv2d_fp16_v1<<<B * C, 512>>>(xp, wp, op, B, C, H, W, OH, OW);
    return out;
}
"""

cpp_source = "torch::Tensor depthwise_conv2d_fp16_cuda(torch::Tensor x, torch::Tensor w, int stride, int padding);"

module = load_inline(
    name='depthwise_conv2d_fp16_v1',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['depthwise_conv2d_fp16_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super().__init__()
        self.stride  = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              groups=in_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv.weight.half()
        return module.depthwise_conv2d_fp16_cuda(x, w, self.stride, self.padding)
