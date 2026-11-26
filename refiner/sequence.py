"""
The frame sequence the refiner runs over, and one causal refinement step.

Every frame of the scene (training and test cameras) is ordered by its timestamp. The base render I_t^base of each
frame comes from the frozen reconstruction, in display sRGB in [0, 1], exactly as render_vigor.py saves it. The
refiner visits the frames in time order, so the history of frame t only ever holds refined outputs of frames that
come before t, and never a ground-truth image.

Ground truth is read for the training frames only (`gt`); a test frame carries `gt = None`. Training losses are
computed on training frames only (train_refiner.py). The test images are read separately, after refinement, to be
written next to the renders for metrics.py.
"""

from dataclasses import dataclass

import torch
from tqdm import tqdm

from gaussian_renderer.vigor import render_composite, render_direct
from refiner.flow import backward_warp
from utils.color_utils import linear2srgb


@dataclass
class Frame:
    name: str
    split: str  # 'train' or 'test'
    time: float
    cam: object
    base: torch.Tensor  # (3, H, W) float16, CPU, sRGB
    gt: torch.Tensor = None  # (3, H, W) float16, CPU, sRGB; None for test frames


def to_srgb(image, linear_rgb):
    image = image.clamp(0.0, 1.0)
    return linear2srgb(image).clamp(0.0, 1.0) if linear_rgb else image


@torch.no_grad()
def base_render(cam, model, pipe, background, linear_rgb):
    """I_t^base of one camera in display sRGB: the composite of render_vigor.py (or B alone without a Phase II)."""
    direct_out = render_direct(cam, model['interface'], model['interface_motion'], pipe, background)
    if model['reflection'] is None:
        return to_srgb(direct_out['direct'], linear_rgb)
    out = render_composite(
        cam,
        direct_out['direct'],
        direct_out['glass_map'],
        model['reflection'],
        pipe,
        model['normal'],
        model['offset'],
        reflection_geometry=model['reflection_motion'].geometry(cam.time),
        blend=model['blend'],
        use_virtual_camera=model['use_virtual_camera'],
    )
    return to_srgb(out['composite'], linear_rgb)


def ordered_cameras(scene):
    """All cameras with their split, sorted by time (ties keep the image-name order)."""
    cams = [(cam, 'train') for cam in scene.getTrainCameras()] + [(cam, 'test') for cam in scene.getTestCameras()]
    return sorted(cams, key=lambda item: (item[0].time, item[0].image_name))


@torch.no_grad()
def build_sequence(scene, model, pipe, background, linear_rgb):
    frames = []
    for cam, split in tqdm(ordered_cameras(scene), desc='Base renders'):
        base = base_render(cam, model, pipe, background, linear_rgb).half().cpu()
        gt = to_srgb(cam.original_image[:3], linear_rgb).half().cpu() if split == 'train' else None
        frames.append(Frame(cam.image_name, split, cam.time, cam, base, gt))
    return frames


def history_inputs(outputs, t, flows_t, masks_t, history, device='cuda'):
    """
    The K warped previous outputs of frame t and their validity: (1, K, 3, H, W), (1, K, 1, H, W).

    outputs[j] is the refined output of frame j (j < t), on any device; a frame before the sequence start is zero
    with validity 0.
    """
    warped, valid = [], []
    for k in range(1, history + 1):
        flow = flows_t[k - 1].to(device).float()[None]
        if t - k >= 0:
            image, _ = backward_warp(outputs[t - k].to(device).float()[None], flow)
            mask = masks_t[k - 1].to(device).float()[None]
        else:
            image = torch.zeros((1, 3) + flow.shape[-2:], device=device)
            mask = torch.zeros((1, 1) + flow.shape[-2:], device=device)
        warped.append(image)
        valid.append(mask)
    return torch.stack(warped, dim=1), torch.stack(valid, dim=1)


def refine_step(net, base, outputs, t, flows_t, masks_t, device='cuda', window=None):
    """
    One causal step: (refined (1, 3, h, w), residual). `outputs` must hold the (full) outputs of frames 0..t-1.

    window = (y0, y1, x0, x1) refines only that crop (training): the history is warped at full resolution, so a crop
    sees the same past content as at inference, and then cropped. None refines the whole frame.
    """
    warped, valid = history_inputs(outputs, t, flows_t, masks_t, net.history, device)
    base = base.to(device).float()[None]
    if window is not None:
        y0, y1, x0, x1 = window
        warped, valid, base = warped[..., y0:y1, x0:x1], valid[..., y0:y1, x0:x1], base[..., y0:y1, x0:x1]
    return net(base, warped, valid)


@torch.no_grad()
def run_sequence(net, frames, flows, masks, device='cuda'):
    """Refine the whole sequence in time order (inference). Returns the refined frames, clamped, on the CPU."""
    net.eval()
    outputs = []
    for t, frame in enumerate(tqdm(frames, desc='Causal refinement')):
        refined, _ = refine_step(net, frame.base, outputs, t, flows[t], masks[t], device)
        outputs.append(refined[0].clamp(0.0, 1.0).half().cpu())
    return outputs
