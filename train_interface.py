"""
Phase I: dynamic interface reconstruction.

Fits the interface field G_I (2D Gaussian disks) and its ExODE-GS motion Theta_I = {A, b, omega, kappa} to the clean,
reflection-removed images, with the glass masks supervising where the interface is transparent:

    G_I(t) = ExODE-GS(G_I^0, Theta_I, t)                               (scene/exode.py)
    B_t    = R_2D(G_I(t), C_t)                                         (direct image, volume pass with p * o)
    L_I    = L_rgb(B_t, I_t^clean) + lambda_s * L_surf + L_mask + L_occupancy + L_ode + lambda_scale * L_scale

The first `warm_up` iterations train the canonical field alone; then every render goes through the motion.

Every --checkpoint_interval iterations (default 1000) the full training state goes to <model_path>/interface/checkpoint.pth;
a rerun of the same command continues from it (utils/checkpoint_utils.py). Everything is written under <model_path>/interface/. The glass plane is extracted from the result afterwards
(extract_plane.py), and Phase II (train_reflection.py) keeps all of it frozen.
"""

import os
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from arguments import InterfaceOptimizationParams, ModelParams, PipelineParams
from gaussian_renderer.interface import render_interface
from scene import Scene
from scene.exode import ExODEMotion
from utils.checkpoint_utils import load_checkpoint, remove_checkpoint, save_checkpoint, set_rng_state, to_cuda
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import binary_cross_entropy, l1_loss, ssim
from utils.vigor_utils import INTERFACE_DIR, new_interface, phase_args

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def time_noise_std(opt, iteration, num_frames):
    """Std of the timestamp jitter: a fraction of one frame interval, annealed linearly to zero."""
    if opt.time_noise_scale <= 0 or opt.time_noise_until_iter <= 0:
        return 0.0
    return opt.time_noise_scale / num_frames * max(0.0, 1.0 - iteration / opt.time_noise_until_iter)


def interface_losses(out, viewpoint_cam, gaussians, opt, iteration, motion_losses=None):
    """The terms of L_I for one view, as a dict of scalar tensors."""
    gt_image = viewpoint_cam.clean_image.cuda()
    gt_mask = viewpoint_cam.gt_transparent_mask.cuda()
    losses = {}

    direct = out['direct']
    losses['rgb'] = (1.0 - opt.lambda_dssim) * l1_loss(direct, gt_image) + opt.lambda_dssim * (1.0 - ssim(direct, gt_image))

    if opt.norm_loss_weight > 0 and iteration >= opt.norm_loss_from_iter:
        error = 1 - (out['surface_normal'] * out['surface_depth_normal']).sum(dim=0, keepdim=True)
        losses['norm'] = opt.norm_loss_weight * (error * out['surface_alpha'].detach()).mean()

    if opt.dist_loss_weight > 0 and iteration >= opt.dist_loss_from_iter:
        losses['dist'] = opt.dist_loss_weight * out['volume_dist'].mean()

    if iteration >= opt.reg_from_iter:
        if opt.mask_loss_weight > 0:
            losses['mask'] = opt.mask_loss_weight * binary_cross_entropy(out['glass_map'], gt_mask.float())
        if opt.occupancy_decay_weight > 0 and gaussians.factorized:
            losses['occupancy'] = opt.occupancy_decay_weight * gaussians.get_occupancy[out['visibility_filter']].mean()

    losses.update(motion_losses or {})  # L_ode and L_scale of the ExODE-GS motion (empty during the warm-up)

    return losses


@torch.no_grad()
def evaluate_interface(cameras, gaussians, motion, pipe, background):
    """Mean PSNR of the direct image against the clean target and mean IoU of the glass map against the mask."""
    psnrs, ious = [], []
    for cam in cameras:
        out = render_interface(cam, gaussians, pipe, background, geometry=motion.geometry(cam.time))
        psnrs.append(psnr(out['direct'].clamp(0.0, 1.0), cam.clean_image.cuda()).mean().item())
        pred_mask = out['glass_map'] > 0.5
        gt_mask = cam.gt_transparent_mask.cuda()
        union = (pred_mask | gt_mask).sum().item()
        ious.append((pred_mask & gt_mask).sum().item() / union if union > 0 else 1.0)
    return {'psnr': float(np.mean(psnrs)), 'glass_iou': float(np.mean(ious))}


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join('./output', os.path.basename(args.source_path))
    interface_path = os.path.join(args.model_path, INTERFACE_DIR)
    print('Output folder: {}'.format(interface_path))
    os.makedirs(interface_path, exist_ok=True)
    with open(os.path.join(interface_path, 'cfg_args'), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(interface_path) if TENSORBOARD_FOUND else None
    if tb_writer is None:
        print('Tensorboard not available: not logging progress')
    return interface_path, tb_writer


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_interval=1000):
    interface_path, tb_writer = prepare_output_and_logger(dataset)

    # The Scene reads and writes the point cloud of this phase under <model_path>/interface.
    gaussians = new_interface(dataset)
    scene = Scene(phase_args(dataset, INTERFACE_DIR), gaussians, resolution_scales=[1.0])
    gaussians.training_setup(opt)
    motion = ExODEMotion(gaussians, dataset.ode_chunk)
    print(f'ExODE-GS motion in G_I: {"on" if motion.dynamic else "off (static interface)"}')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    num_frames = len(viewpoint_stack)
    print('Training set length', num_frames)
    if not any(cam.has_clean_image for cam in viewpoint_stack):
        print('[ WARN ] No clean images: Phase I is fitting the original images, reflections included.')

    history = []
    ema_loss_dict = {}
    last_reset_iter = -100000
    first = 1
    checkpoint = load_checkpoint(interface_path, opt.iterations)
    if checkpoint is not None:
        gaussians.restore(to_cuda(checkpoint['field']), opt)
        history, ema_loss_dict, last_reset_iter = checkpoint['history'], checkpoint['ema_loss_dict'], checkpoint['last_reset_iter']
        set_rng_state(checkpoint['rng'])
        first = checkpoint['iteration'] + 1
        del checkpoint
    progress_bar = tqdm(range(0, opt.iterations), desc='Interface training', initial=first - 1)

    for iteration in range(first, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam = viewpoint_stack[np.random.randint(num_frames)]

        # Static warm-up first; afterwards every render goes through the ExODE-GS motion.
        moving = motion.dynamic and iteration > opt.warm_up
        geometry = motion.geometry(viewpoint_cam.time, enabled=moving, time_noise=time_noise_std(opt, iteration, num_frames))

        out = render_interface(viewpoint_cam, gaussians, pipe, background, geometry=geometry)
        motion_losses = motion.regularizers(geometry[2], opt, scene.cameras_extent) if moving else {}
        loss_dict = interface_losses(out, viewpoint_cam, gaussians, opt, iteration, motion_losses)
        loss = sum(loss_dict.values())
        loss.backward()

        with torch.no_grad():
            for key, value in loss_dict.items():
                ema_loss_dict[key] = 0.4 * value.item() + 0.6 * ema_loss_dict.get(key, value.item())
            if iteration % 10 == 0:
                progress_bar.set_postfix({**{k: f'{v:.4f}' for k, v in ema_loss_dict.items()}, 'Points': gaussians.num_points})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss/total', loss.item(), iteration)
                for key, value in loss_dict.items():
                    tb_writer.add_scalar(f'train_loss/{key}', value.item(), iteration)
                tb_writer.add_scalar('total_points', gaussians.num_points, iteration)

            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                train_subset = [viewpoint_stack[idx % num_frames] for idx in range(5, 30, 5)]
                for name, cameras in (('test', scene.getTestCameras()), ('train', train_subset)):
                    if not cameras:
                        continue
                    metrics = evaluate_interface(cameras, gaussians, motion, pipe, background)
                    history.append({'iteration': iteration, 'split': name, **metrics})
                    print(f'\n[ITER {iteration}] Evaluating {name}: PSNR {metrics["psnr"]:.3f}, glass IoU {metrics["glass_iou"]:.3f}')
                    if tb_writer:
                        tb_writer.add_scalar(f'{name}/psnr', metrics['psnr'], iteration)
                        tb_writer.add_scalar(f'{name}/glass_iou', metrics['glass_iou'], iteration)
                torch.cuda.empty_cache()

            if iteration in saving_iterations:
                print('\n[ITER {}] Saving interface'.format(iteration))
                scene.save(iteration)

            # Densification, driven by the occupancy
            if iteration < opt.densify_until_iter:
                visibility_filter, radii = out['visibility_filter'], out['radii']
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(out['viewspace_points'], visibility_filter, iteration)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.occupancy_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.occupancy_cull, scene.cameras_extent, size_threshold, last_reset_iter)

                if iteration % opt.occupancy_reset_interval == 0:
                    gaussians.reset_occupancy()
                    last_reset_iter = iteration

                if iteration >= opt.occupancy_reset_interval and iteration % opt.occupancy_reset_interval == opt.occupancy_reset_interval // 2:
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                    save_checkpoint(interface_path, iteration, opt.iterations,
                                    {'field': gaussians.capture(), 'history': history, 'ema_loss_dict': ema_loss_dict, 'last_reset_iter': last_reset_iter})

    remove_checkpoint(interface_path)  # the final model is saved; the checkpoint is no longer needed
    if tb_writer:
        tb_writer.close()
    return history


if __name__ == '__main__':
    parser = ArgumentParser(description='Phase I: interface reconstruction with ExODE-GS motion')
    lp = ModelParams(parser)
    op = InterfaceOptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--srgb', action='store_true', help='Train on sRGB targets instead of linear RGB')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='save the training state every N iterations to resume from (0: never)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.linear_rgb = not args.srgb  # the reflection blend of Phase II is linear, so linear RGB is the default

    print('Optimizing ' + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_interval)

    print('\nInterface training complete.')
