"""
Post-refinement: train the causal autoregressive refiner f_theta on a frozen reconstruction, then refine the sequence.

    I_hat_t = I_t^base + f_theta(I_t^base, W(I_hat_{t-1}), ..., W(I_hat_{t-K}))          (refiner/network.py)

Nothing of the reconstruction is trained here: G_I, G_R, their ExODE-GS motion, the plane and the blend are loaded
and only rendered. Training uses the training frames only:
  - every frame (training and test cameras) is base-rendered and ordered by time (refiner/sequence.py);
  - the flows F_{t->t-k} between base renders, k = 1..K, are computed once with RAFT-small (refiner/flow.py);
  - each step refines a clip of `clip` consecutive frames in time order, on one random crop (--refiner_crop, 512) shared
    by the clip's frames (the history is warped at full resolution, then cropped), feeding each frame its own previous
    refined outputs (backpropagation through the clip); the loss is (1 - l) L1 + l (1 - SSIM) against the ground
    truth on the clip's training frames;
  - the history of the clip's first frames comes from a bank holding the latest refined output of every frame
    (initialised with the base renders), so the refiner always sees outputs of its own, as at inference.
Test frames can sit inside a clip (they are refined and become history) but never receive a loss, and their ground
truth is not read until the final images are written for evaluation.

Writes <model>/refiner/{refiner.pth, config.json, training.json} and, unless --skip_render,
<model>/test/refined_<iterations>/{renders, gt, transparent_masks, vis} (and train/ with --render_train).

    python train_refiner.py -m output/<scene>

Every --checkpoint_interval iterations (default 1000) the network, optimizer, scheduler and RNG states go to
<model>/refiner/checkpoint.pth; a rerun continues from it (the history bank is rebuilt by running the restored network
over the sequence, instead of storing every frame).

--base_from LABEL takes the base renders from <model>/{train,test}/LABEL/renders/*.png instead of rendering them,
e.g. the output of another reconstruction (such as the deformation-MLP VIGOR-GS) rendered with its own
render_vigor.py for both splits. Only the cameras are loaded from the dataset; the PNGs must follow the camera order
of the splits, which is checked against the gt/ images written next to them.

    python train_refiner.py -m <model> -s <scene> --base_from ours_30000 --data_device cpu
"""

import json
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from refiner.flow import FlowEstimator, history_flows
from refiner.network import CausalRefiner
from refiner.sequence import Frame, build_sequence, ordered_cameras, refine_step, run_sequence, to_srgb
from render_vigor import load_model
from scene import Scene
from utils.checkpoint_utils import load_checkpoint, remove_checkpoint, save_checkpoint, set_rng_state
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.render_utils import save_img_u8
from utils.vigor_utils import REFLECTION_DIR

REFINER_DIR = 'refiner'


class History:
    """
    outputs[j] for the causal step: the bank before `start`; from `start` on, the clip's own outputs (with gradients).

    With a crop `window`, the clip's outputs cover the crop only: a full frame is the bank's frame with the crop
    replaced by the new output, so warping still reads the whole image while gradients flow through the crop.
    """

    def __init__(self, bank, start, window=None, device='cuda'):
        self.bank, self.start, self.window, self.device, self.clip = bank, start, window, device, {}

    def __getitem__(self, j):
        if j < self.start:
            return self.bank[j]
        if self.window is None:
            return self.clip[j]
        y0, y1, x0, x1 = self.window
        full = self.bank[j].to(self.device).float().clone()
        full[:, y0:y1, x0:x1] = self.clip[j]
        return full

    def full(self, j):
        """The detached full frame of a clip output, for the bank."""
        return self[j].detach().clamp(0.0, 1.0).half().cpu()


def crop_window(height, width, size):
    """A random size x size window, or None (whole frame) when size <= 0 or the frame is not larger."""
    if size <= 0 or (height <= size and width <= size):
        return None
    ch, cw = min(size, height), min(size, width)
    y0, x0 = random.randint(0, height - ch), random.randint(0, width - cw)
    return (y0, y0 + ch, x0, x0 + cw)


def refiner_loss(refined, gt, lambda_dssim):
    return (1.0 - lambda_dssim) * l1_loss(refined, gt) + lambda_dssim * (1.0 - ssim(refined, gt))


def train(net, frames, flows, masks, args, device='cuda'):
    """Truncated backpropagation through time over random clips. Returns the training log."""
    bank = [frame.base.clone() for frame in frames]
    clip = min(args.clip, len(frames))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.lr * 0.01)
    starts = [s for s in range(len(frames) - clip + 1) if any(frames[t].split == 'train' for t in range(s, s + clip))]
    log, ema, first = [], None, 1
    folder, interval = getattr(args, 'checkpoint_folder', None), getattr(args, 'checkpoint_interval', 0)
    checkpoint = load_checkpoint(folder, args.iterations) if folder else None
    if checkpoint is not None:
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        log, ema = checkpoint['log'], checkpoint['ema']
        bank = run_sequence(net, frames, flows, masks, device)  # the latest refined output of every frame, as the bank holds it
        set_rng_state(checkpoint['rng'])
        first = checkpoint['iteration'] + 1
        del checkpoint
    net.train()
    progress = tqdm(range(first, args.iterations + 1), desc='Refiner training', initial=first - 1, total=args.iterations)
    for iteration in progress:
        start = random.choice(starts)
        window = crop_window(*frames[start].base.shape[-2:], getattr(args, 'crop', 0))  # one window for the whole clip
        history = History(bank, start, window, device)
        losses = []
        for t in range(start, start + clip):
            refined, _ = refine_step(net, frames[t].base, history, t, flows[t], masks[t], device, window)
            history.clip[t] = refined[0]
            if frames[t].split == 'train':  # test frames never get a loss (and frames[t].gt is None for them)
                gt = frames[t].gt.to(device).float()[None]
                if window is not None:
                    gt = gt[..., window[0]:window[1], window[2]:window[3]]
                losses.append(refiner_loss(refined, gt, args.lambda_dssim))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for t in history.clip:
                bank[t] = history.full(t)
        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
        if iteration % 10 == 0:
            progress.set_postfix({'loss': f'{ema:.4f}'})
        if iteration % 100 == 0:
            log.append({'iteration': iteration, 'loss': ema, 'lr': scheduler.get_last_lr()[0]})
        if folder and interval > 0 and iteration % interval == 0 and iteration < args.iterations:
            save_checkpoint(folder, iteration, args.iterations, {'net': net.state_dict(), 'optimizer': optimizer.state_dict(),
                                                                 'scheduler': scheduler.state_dict(), 'log': log, 'ema': ema})
    return log


def write_split(model_path, split, label, frames, refined, scene, linear_rgb):
    """Save the refined frames of one split in the layout of render_vigor.py (same file names per camera)."""
    cameras = scene.getTrainCameras() if split == 'train' else scene.getTestCameras()
    index = {cam.image_name: i for i, cam in enumerate(cameras)}
    out_dir = os.path.join(model_path, split, label)
    for sub in ('renders', 'gt', 'transparent_masks', 'vis'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    for frame, image in zip(frames, refined):
        if frame.split != split:
            continue
        name = '{0:05d}.png'.format(index[frame.name])
        cam = frame.cam
        save_img_u8(image.float().permute(1, 2, 0).numpy(), os.path.join(out_dir, 'renders', name))
        save_img_u8(to_srgb(cam.original_image[:3], linear_rgb).permute(1, 2, 0).cpu().numpy(), os.path.join(out_dir, 'gt', name))
        save_img_u8(cam.gt_transparent_mask.squeeze(0).float().cpu().numpy(), os.path.join(out_dir, 'transparent_masks', name))
        residual = (image.float() - frame.base.float()).abs().mean(0).mul(4.0).clamp(0, 1).numpy()  # |I_hat - I_base| x 4
        save_img_u8(frame.base.float().permute(1, 2, 0).numpy(), os.path.join(out_dir, 'vis', f'base_{name}'))
        save_img_u8(residual, os.path.join(out_dir, 'vis', f'residual_{name}'))
    return out_dir


class _CamerasOnly:
    """Stands in for the Gaussians when only the cameras of a scene are needed (Scene then loads nothing else)."""

    def load_ply(self, path):
        pass


def read_png(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)


def sequence_from_pngs(dataset, label):
    """
    The sequence with base renders read from <model>/<split>/<label>/renders/<index>.png (render_vigor.py layout).

    The index of a PNG is the camera's position in its split. Every frame's saved gt/ image is compared with the
    dataset image of that camera, so a different camera order (or resolution) stops the run instead of training the
    refiner on mismatched frames.
    """
    scene = Scene(dataset, _CamerasOnly(), load_iteration=1, shuffle=False)  # load_iteration: write nothing to the model
    index = {split: {cam.image_name: i for i, cam in enumerate(cams)} for split, cams in (('train', scene.getTrainCameras()), ('test', scene.getTestCameras()))}
    frames, worst = [], 0.0
    for cam, split in tqdm(ordered_cameras(scene), desc=f'Base renders from {label}'):
        folder = os.path.join(dataset.model_path, split, label)
        name = '{0:05d}.png'.format(index[split][cam.image_name])
        base = read_png(os.path.join(folder, 'renders', name))
        gt = to_srgb(cam.original_image[:3], dataset.linear_rgb).float().cpu()
        if base.shape != gt.shape:
            raise SystemExit(f'{folder}/renders/{name} is {tuple(base.shape)}, the camera {cam.image_name} is {tuple(gt.shape)}: use the same -r')
        diff = (read_png(os.path.join(folder, 'gt', name)) - gt).abs().mean().item()
        worst = max(worst, diff)
        if diff > 2.0 / 255.0:
            raise SystemExit(f'{folder}/gt/{name} does not match camera {cam.image_name} (mean |diff| {diff:.4f}): different camera order or data')
        frames.append(Frame(cam.image_name, split, cam.time, cam, base.half(), gt.half() if split == 'train' else None))
    print(f'[ INFO ] {len(frames)} base renders from {label}; saved gt vs dataset: max mean |diff| {worst * 255:.2f}/255')
    return scene, frames


def prepare(dataset, pipe, history, flow_iterations, iteration=-1, flow_device='cuda', base_from=''):
    """Load the frozen reconstruction, render the sequence and its history flows (RAFT on `flow_device`)."""
    if base_from:
        model, label = None, base_from
        scene, frames = sequence_from_pngs(dataset, base_from)
    else:
        scene, model, label = load_model(dataset, iteration)
        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device='cuda')
        frames = build_sequence(scene, model, pipe, background, dataset.linear_rgb)
    estimator = FlowEstimator(device=flow_device, iterations=flow_iterations)
    start = time.time()
    flows, masks = history_flows(estimator, [f.base for f in frames], history, device=flow_device)
    print(f'[ INFO ] flows for {len(frames)} frames x {history} past frames in {time.time() - start:.0f} s')
    del estimator
    torch.cuda.empty_cache()
    if flow_device != 'cuda':
        with torch.cuda.device(flow_device):
            torch.cuda.empty_cache()
    return scene, model, label, frames, flows, masks


def main():
    parser = ArgumentParser(description='Train the causal autoregressive refiner on a frozen reconstruction')
    model_params = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int, help='Phase II iteration to load (default: the latest)')
    parser.add_argument('--history', default=3, type=int, help='K: number of previous refined frames')
    parser.add_argument('--width', default=32, type=int, help='channels of the first U-Net level')
    parser.add_argument('--refiner_iterations', default=4000, type=int)
    parser.add_argument('--clip', default=4, type=int, help='frames per training clip (backpropagation through time)')
    parser.add_argument('--refiner_crop', default=512, type=int, help='training crop size (0: whole frames); inference always uses whole frames')
    parser.add_argument('--refiner_lr', default=2e-4, type=float)
    parser.add_argument('--refiner_dssim', default=0.2, type=float)
    parser.add_argument('--flow_iterations', default=12, type=int, help='RAFT refinement iterations')
    parser.add_argument('--base_from', default='', help='read the base renders from <model>/{train,test}/<this label>/renders instead of rendering')
    parser.add_argument('--flow_device', default='cuda', help='GPU for RAFT, e.g. cuda:1 (its full-resolution correlation volume needs ~9 GiB at 1080p)')
    parser.add_argument('--refiner_seed', default=0, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int, help='save the training state every N iterations to resume from (0: never)')
    parser.add_argument('--skip_render', action='store_true')
    parser.add_argument('--render_train', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser, cfg_subdir=REFLECTION_DIR)
    safe_state(args.quiet)
    random.seed(args.refiner_seed)
    np.random.seed(args.refiner_seed)
    torch.manual_seed(args.refiner_seed)

    dataset, pipe = model_params.extract(args), pipeline.extract(args)
    scene, _, label, frames, flows, masks = prepare(dataset, pipe, args.history, args.flow_iterations, args.iteration, args.flow_device, args.base_from)
    print(f'[ INFO ] sequence: {len(frames)} frames ({sum(f.split == "train" for f in frames)} train), reconstruction {label}')

    net = CausalRefiner(args.history, args.width).cuda()
    print(f'[ INFO ] refiner: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, K = {args.history}')
    train_args = type('A', (), {'iterations': args.refiner_iterations, 'clip': args.clip, 'lr': args.refiner_lr, 'lambda_dssim': args.refiner_dssim,
                                'crop': args.refiner_crop, 'checkpoint_folder': os.path.join(dataset.model_path, REFINER_DIR),
                                'checkpoint_interval': args.checkpoint_interval})
    start = time.time()
    log = train(net, frames, flows, masks, train_args)
    minutes = (time.time() - start) / 60

    folder = os.path.join(dataset.model_path, REFINER_DIR)
    os.makedirs(folder, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(folder, 'refiner.pth'))
    config = {'history': args.history, 'width': args.width, 'iterations': args.refiner_iterations, 'clip': args.clip, 'crop': args.refiner_crop, 'lr': args.refiner_lr,
              'lambda_dssim': args.refiner_dssim, 'flow_iterations': args.flow_iterations, 'reconstruction': label, 'base_from': args.base_from or None, 'train_minutes': minutes}
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(folder, 'training.json'), 'w') as f:
        json.dump(log, f, indent=1)
    remove_checkpoint(folder)  # refiner.pth holds the final network
    print(f'[ INFO ] refiner trained in {minutes:.1f} min')

    if not args.skip_render:
        refined = run_sequence(net, frames, flows, masks)
        for split in ('test', 'train') if args.render_train else ('test',):
            print('Refined renders:', write_split(dataset.model_path, split, f'refined_{args.refiner_iterations}', frames, refined, scene, dataset.linear_rgb))


if __name__ == '__main__':
    main()
