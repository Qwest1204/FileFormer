import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MiSSLinear(nn.Module):
    def __init__(self, original: nn.Linear, shard_size: int = 16):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        out_f, in_f = original.weight.shape
        self.shard_size = shard_size
        self.num_shards = (in_f + shard_size - 1) // shard_size
        self.shard = nn.Parameter(torch.zeros(out_f, shard_size))
        nn.init.kaiming_uniform_(self.shard, a=math.sqrt(5))
        self.register_buffer('shard_scale', torch.tensor(1.0 / self.num_shards))

    def forward(self, x):
        result = self.original(x)
        in_f = self.original.in_features
        idx = torch.arange(in_f, device=x.device) % self.shard_size  # (in_f,)
        W_hat = self.shard[:, idx] * self.shard_scale                # (out_f, in_f)
        return result + F.linear(x, W_hat)