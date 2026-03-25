import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """
    p97 ScaledDotProductAttention v2
    Same TF32 approach as v1 but use baddbmm for scale-fused Q@K^T,
    and softmax(..., dtype=torch.float32) to avoid internal casts.
    Also try contiguous() on K transpose to improve memory layout for bmm.
    """

    def __init__(self):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, S, D = Q.shape
        scale = 1.0 / math.sqrt(D)

        Qf = Q.reshape(B * H, S, D)
        Kf = K.reshape(B * H, S, D)
        Vf = V.reshape(B * H, S, D)

        # baddbmm fuses scale into the GEMM (one kernel instead of bmm + mul_)
        S_mat = torch.baddbmm(
            torch.empty(B * H, S, S, device=Q.device, dtype=Q.dtype),
            Qf, Kf.transpose(1, 2),
            beta=0.0, alpha=scale
        )
        A = F.softmax(S_mat, dim=-1)
        out = torch.bmm(A, Vf)

        return out.reshape(B, H, S, D)
