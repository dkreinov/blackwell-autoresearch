import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Use torch.cumsum (fast CUB implementation) then shift
        inclusive = torch.cumsum(x, dim=self.dim)
        # Shift right by 1: exclusive[0]=0, exclusive[i]=inclusive[i-1]
        out = torch.zeros_like(x)
        out.narrow(self.dim, 1, x.size(self.dim) - 1).copy_(
            inclusive.narrow(self.dim, 0, x.size(self.dim) - 1)
        )
        return out
