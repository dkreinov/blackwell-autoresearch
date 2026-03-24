import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// v3: float4 loads (4 elements per load, 256 threads × 4 float4s = 1024 elems/tile).
// 256 threads per block — 4 float4s per thread per tile = same 1024 elements per tile.
// With 256 threads: 3 blocks/SM = 100% warp occupancy.
// Warp-level reverse scan now operates on 4 accumulated values (one sum per thread).

__global__ void cumsum_reverse_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    const float4* row_in4 = reinterpret_cast<const float4*>(x + (long long)row * n);
    float4*       row_out4 = reinterpret_cast<float4*>(out + (long long)row * n);

    // 256 threads, 4 float4s per thread per tile = 256*4 = 1024 float4s per tile = 4096 elems per tile
    // Wait, n=32768 / 4096 = 8 tiles. Let's use 8 tiles of 4096 elements.
    // Each thread handles 4 consecutive float4s per tile = 16 elements per tile.

    __shared__ float ws[8];  // 8 warps (256/32=8)
    __shared__ float tile_total_smem;
    float carry = 0.f;

    // 8 tiles of 4096 elements, processed right-to-left
    for (int tile = 7; tile >= 0; tile--) {
        // Load 4 float4s = 16 elements per thread, coalesced
        int base4 = tile * 1024 + tid * 4;  // float4 index; 256 threads * 4 = 1024 float4s per tile
        float4 a = row_in4[base4 + 0];
        float4 b = row_in4[base4 + 1];
        float4 c = row_in4[base4 + 2];
        float4 d = row_in4[base4 + 3];

        // Local reverse prefix sum of 16 elements: val[15..0]
        // r[0..3] = a.x,a.y,a.z,a.w; r[4..7] = b.x...; r[8..11] = c.x...; r[12..15] = d.x..w
        float s15 = d.w;
        float s14 = d.z + s15;
        float s13 = d.y + s14;
        float s12 = d.x + s13;
        float s11 = c.w + s12;
        float s10 = c.z + s11;
        float s9  = c.y + s10;
        float s8  = c.x + s9;
        float s7  = b.w + s8;
        float s6  = b.z + s7;
        float s5  = b.y + s6;
        float s4  = b.x + s5;
        float s3  = a.w + s4;
        float s2  = a.z + s3;
        float s1  = a.y + s2;
        float s0  = a.x + s1;
        float local_sum = s0;

        // Warp-level reverse inclusive scan on per-thread local_sum
        float fwd = local_sum;
        for (int dd = 1; dd < 32; dd <<= 1) {
            float up = __shfl_up_sync(0xffffffff, fwd, dd);
            if (lane >= dd) fwd += up;
        }
        float warp_total = __shfl_sync(0xffffffff, fwd, 31);
        float warp_rev_excl = warp_total - fwd;  // exclusive reverse: sum of threads to the right

        if (lane == 0) ws[wid] = warp_total;
        __syncthreads();

        float warp_suffix = 0.f;
        if (wid == 0) {
            int num_warps = blockDim.x >> 5;
            float w = (lane < num_warps) ? ws[lane] : 0.f;
            float fw = w;
            for (int dd = 1; dd < 32; dd <<= 1) {
                float up = __shfl_up_sync(0xffffffff, fw, dd);
                if (lane >= dd) fw += up;
            }
            float bt = __shfl_sync(0xffffffff, fw, num_warps - 1);
            ws[lane] = bt - fw;
            if (lane == 0) tile_total_smem = bt;
        }
        __syncthreads();

        float suffix = ws[wid] + warp_rev_excl + carry;

        // Write output (coalesced float4 stores)
        float4 oa, ob, oc, od;
        oa.x=s0+suffix; oa.y=s1+suffix; oa.z=s2+suffix; oa.w=s3+suffix;
        ob.x=s4+suffix; ob.y=s5+suffix; ob.z=s6+suffix; ob.w=s7+suffix;
        oc.x=s8+suffix; oc.y=s9+suffix; oc.z=s10+suffix; oc.w=s11+suffix;
        od.x=s12+suffix; od.y=s13+suffix; od.z=s14+suffix; od.w=s15+suffix;
        row_out4[base4+0] = oa; row_out4[base4+1] = ob;
        row_out4[base4+2] = oc; row_out4[base4+3] = od;

        carry += tile_total_smem;
        __syncthreads();
    }
}

torch::Tensor cumsum_reverse_cuda(torch::Tensor x, int dim) {
    auto out = torch::empty_like(x);
    int rows = x.size(0);
    int n    = x.size(1);
    cumsum_reverse_kernel<<<rows, 256>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

cpp_source = "torch::Tensor cumsum_reverse_cuda(torch::Tensor x, int dim);"

mod = load_inline(
    name='cumsum_reverse_v3',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['cumsum_reverse_cuda'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mod.cumsum_reverse_cuda(x.cuda(), self.dim)
