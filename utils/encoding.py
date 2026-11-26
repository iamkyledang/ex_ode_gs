"""Sin/cos positional encoding gamma(x) = (x, sin(2^k x), cos(2^k x)), used by the optical correction h_psi."""

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, in_dim: int, num_freqs: int):
        super().__init__()
        self.register_buffer('freqs', 2.0 ** torch.arange(num_freqs, dtype=torch.float32), persistent=False)
        self.out_dim = in_dim * (1 + 2 * num_freqs)

    def forward(self, x):
        scaled = x[..., None] * self.freqs
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(-2)
        return torch.cat([x, encoded], dim=-1)
