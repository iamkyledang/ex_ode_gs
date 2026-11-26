"""The causal autoregressive refiner (refiner/): identity at initialisation, motion alignment, causality."""

from types import SimpleNamespace

import pytest
import torch

from refiner.flow import backward_warp, consistency_mask
from refiner.network import CausalRefiner
from refiner.sequence import Frame, run_sequence

pytestmark = pytest.mark.gpu


def test_fresh_refiner_returns_the_base_render():
    net = CausalRefiner(history=3).cuda()
    base = torch.rand(1, 3, 37, 50, device='cuda')  # odd size: padded inside, cropped back
    refined, residual = net(base, torch.rand(1, 3, 3, 37, 50, device='cuda'), torch.ones(1, 3, 1, 37, 50, device='cuda'))
    assert refined.shape == base.shape and torch.equal(refined, base) and residual.abs().max() == 0


def test_backward_warp_follows_the_flow():
    image = torch.rand(1, 3, 20, 30, device='cuda')
    flow = torch.zeros(1, 2, 20, 30, device='cuda')
    flow[:, 0] = 2.0  # every pixel reads the pixel two columns to its right
    warped, inside = backward_warp(image, flow)
    assert torch.allclose(warped[..., :-2], image[..., 2:], atol=1e-5)
    assert inside[..., -2:].sum() == 0 and inside[..., :-2].min() == 1


def test_consistent_flows_are_valid_and_inconsistent_ones_are_not():
    back = torch.zeros(1, 2, 16, 16, device='cuda')
    back[:, 0] = 1.0
    assert consistency_mask(back, -back)[..., :, :-2].min() == 1  # forward = -backward: consistent
    assert consistency_mask(back, back).max() == 0  # the two flows disagree everywhere: occluded


def sequence(n=6, h=24, w=32, history=2):
    frames, flows, masks = [], [], []
    for t in range(n):
        split = 'test' if t % 3 == 2 else 'train'
        frames.append(Frame(f'{t:03d}', split, t / n, SimpleNamespace(), torch.rand(3, h, w).half(), torch.rand(3, h, w).half() if split == 'train' else None))
        flows.append(torch.zeros(history, 2, h, w).half())
        masks.append(torch.ones(history, 1, h, w, dtype=torch.uint8))
    return frames, flows, masks


def test_refinement_is_causal():
    """Changing a later frame never changes the refined output of an earlier frame."""
    torch.manual_seed(0)
    net = CausalRefiner(history=2).cuda()
    torch.nn.init.normal_(net.out.weight, std=0.05)  # a non-trivial refiner
    frames, flows, masks = sequence()
    before = run_sequence(net, frames, flows, masks)
    frames[4].base = torch.rand_like(frames[4].base.float()).half()
    after = run_sequence(net, frames, flows, masks)
    for t in range(4):
        assert torch.equal(before[t], after[t]), t
    assert not torch.equal(before[4], after[4])


def test_training_never_reads_test_ground_truth():
    from train_refiner import train

    torch.manual_seed(0)
    net = CausalRefiner(history=2, width=8).cuda()
    frames, flows, masks = sequence()
    assert all(f.gt is None for f in frames if f.split == 'test')  # any access would fail on None
    log = train(net, frames, flows, masks, SimpleNamespace(iterations=100, clip=3, lr=1e-3, lambda_dssim=0.2))
    assert log and torch.isfinite(torch.tensor(log[-1]['loss']))
