"""
The causal autoregressive refiner f_theta: a small residual U-Net.

    I_hat_t = I_t^base + f_theta(I_t^base, W_{t<-t-1}(I_hat_{t-1}), ..., W_{t<-t-K}(I_hat_{t-K}), V_{t,1}, ..., V_{t,K})

I_t^base is the composited render of frame t, I_hat_{t-k} the refined output of an earlier frame, W_{t<-t-k} the
backward warp along the optical flow between the base renders of frames t and t-k (refiner/flow.py), and
V_{t,k} in {0, 1} marks the pixels where that warp is valid (inside the image and not occluded). A missing frame
(the first K frames of the sequence) is given as zeros with V = 0.

Architecture (an encoder-decoder with skip connections, as U-Net): three resolutions with 32 / 64 / 128 channels,
two 3x3 convolutions per block with LeakyReLU, strided convolutions down, bilinear upsampling up. The last layer
is initialised to zero, so before training the refiner returns I_t^base unchanged; it can only learn a correction.
"""

import torch
import torch.nn.functional as F
from torch import nn


def block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
    )


class CausalRefiner(nn.Module):
    def __init__(self, history=3, width=32):
        super().__init__()
        self.history = history
        in_ch = 3 + 4 * history  # I_base + K x (warped previous output + validity)
        w1, w2, w3 = width, 2 * width, 4 * width
        self.enc1 = block(in_ch, w1)
        self.down1 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.enc2 = block(w2, w2)
        self.down2 = nn.Conv2d(w2, w3, 3, stride=2, padding=1)
        self.bottleneck = block(w3, w3)
        self.dec2 = block(w3 + w2, w2)
        self.dec1 = block(w2 + w1, w1)
        self.out = nn.Conv2d(w1, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, base, warped, valid):
        """
        base: (B, 3, H, W) in [0, 1] (sRGB); warped: (B, K, 3, H, W); valid: (B, K, 1, H, W).
        Returns (refined, residual), refined = base + residual (not clamped, so the loss sees the raw output).
        """
        b, k, _, h, w = warped.shape
        x = torch.cat([base, (warped * valid).reshape(b, 3 * k, h, w), valid.reshape(b, k, h, w)], dim=1)
        # Pad to a multiple of 4 for the two downsamplings, crop back at the end.
        ph, pw = (-h) % 4, (-w) % 4
        x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        e1 = self.enc1(x)
        e2 = self.enc2(F.leaky_relu(self.down1(e1), 0.1))
        m = self.bottleneck(F.leaky_relu(self.down2(e2), 0.1))
        d2 = self.dec2(torch.cat([F.interpolate(m, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1], dim=1))
        residual = self.out(d1)[..., :h, :w]
        return base + residual, residual
