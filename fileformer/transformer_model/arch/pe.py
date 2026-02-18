import torch.nn as nn
import torch


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device):
        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(device)
        seq_idx = torch.arange(seq_len, device=device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos().to(device)  # Shape: (seq_len, d)
        self.sin_cached = idx_theta2.sin().to(device)  # Shape: (seq_len, d)

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)  # Shape: (batch_size, seq_len, d)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape
        self._build_cache(seq_len, x.device)

        # Broadcast cos and sin to match batch dimension
        cos_vals = self.cos_cached[:seq_len].unsqueeze(0)  # Shape: (1, seq_len, d)
        sin_vals = self.sin_cached[:seq_len].unsqueeze(0)  # Shape: (1, seq_len, d)

        neg_half_x = self._neg_half(x)
        x_rope = (x * cos_vals) + (neg_half_x * sin_vals)
        return x_rope