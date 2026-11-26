"""
Motion alignment of the refiner's history: optical flow between *base renders* (never ground truth).

For frame t and an earlier frame t-k, RAFT-small (Teed and Deng, ECCV 2020; the pretrained torchvision model)
estimates the backward flow F_{t->t-k} from I_t^base to I_{t-k}^base. A refined frame of the past is brought to
frame t by sampling it at x + F_{t->t-k}(x):

    W_{t<-t-k}(I_hat_{t-k})(x) = I_hat_{t-k}(x + F_{t->t-k}(x))                      (bilinear, grid_sample)

The forward-backward check (Sundaram et al., ECCV 2010) marks a pixel invalid when the two flows disagree, which is
where frame t sees something that frame t-k did not (occlusion), or where the warp leaves the image:

    |F_{t->t-k}(x) + F_{t-k->t}(x + F_{t->t-k}(x))|^2 > a (|F_{t->t-k}|^2 + |F_{t-k->t}(.)|^2) + b,   a = 0.01, b = 0.5

Flows depend only on the frozen renders, so they are computed once per scene and reused for training and inference.
"""

import torch
import torch.nn.functional as F


class FlowEstimator:
    """Pretrained RAFT-small. Inputs are (B, 3, H, W) sRGB images in [0, 1]."""

    def __init__(self, device='cuda', iterations=12):
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device).eval()
        self.iterations = iterations

    @torch.no_grad()
    def __call__(self, image1, image2):
        """Flow from image1 to image2: image1(x) ~ image2(x + flow(x)). Returns (B, 2, H, W) in pixels (dx, dy)."""
        h, w = image1.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8  # RAFT needs sizes divisible by 8
        a = F.pad(image1 * 2 - 1, (0, pw, 0, ph), mode='replicate')
        b = F.pad(image2 * 2 - 1, (0, pw, 0, ph), mode='replicate')
        flow = self.model(a, b, num_flow_updates=self.iterations)[-1]
        return flow[..., :h, :w]


def pixel_grid(h, w, device):
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
    return torch.stack([xs, ys], dim=0)[None]  # (1, 2, H, W)


def backward_warp(image, flow):
    """Sample `image` (B, C, H, W) at x + flow(x). Returns (warped, inside) with inside (B, 1, H, W) in {0, 1}."""
    _, _, h, w = image.shape
    coords = pixel_grid(h, w, image.device) + flow
    gx = 2.0 * coords[:, 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * coords[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)
    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    inside = ((gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1)).float()[:, None]
    return warped, inside


def consistency_mask(flow_back, flow_fwd, alpha=0.01, beta=0.5):
    """Forward-backward check of flow_back (t -> t-k) against flow_fwd (t-k -> t). Returns (B, 1, H, W) in {0, 1}."""
    fwd_at, inside = backward_warp(flow_fwd, flow_back)
    diff = (flow_back + fwd_at).pow(2).sum(1, keepdim=True)
    bound = alpha * (flow_back.pow(2).sum(1, keepdim=True) + fwd_at.pow(2).sum(1, keepdim=True)) + beta
    return (diff <= bound).float() * inside


@torch.no_grad()
def history_flows(estimator, bases, history, device='cuda'):
    """
    For every frame t and k = 1..K: the backward flow F_{t->t-k} and its validity, from the base renders only.

    bases: list of (3, H, W) tensors in time order (CPU is fine). Returns two lists indexed by t of
    (K, 2, H, W) float16 flows and (K, 1, H, W) uint8 masks, on the CPU. Missing frames (t - k < 0) get zero flow
    and an all-zero mask.
    """
    flows, masks = [], []
    for t in range(len(bases)):
        cur = bases[t].to(device).float()[None]
        h, w = cur.shape[-2:]
        f_t = torch.zeros((history, 2, h, w), dtype=torch.float16)
        m_t = torch.zeros((history, 1, h, w), dtype=torch.uint8)
        for k in range(1, history + 1):
            if t - k < 0:
                continue
            past = bases[t - k].to(device).float()[None]
            back = estimator(cur, past)
            fwd = estimator(past, cur)
            f_t[k - 1] = back[0].half().cpu()
            m_t[k - 1] = consistency_mask(back, fwd)[0].to(torch.uint8).cpu()
        flows.append(f_t)
        masks.append(m_t)
    return flows, masks
