"""
Phase II: dynamic reflection reconstruction.

Everything from Phase I is frozen: the interface field G_I, its ExODE-GS motion Theta_I and the glass plane Pi*.
The reflection field G_R (3D Gaussians), its ExODE-GS motion Theta_R, the Schlick F0 and, optionally, the
reflectance correction h_psi are fitted to the original images:

    R_t        = R_3D(G_R(t), V(C_t, Pi*))                           reflection through the virtual camera
    I_t^base   = (1 - alpha_t) B_t + alpha_t R_t                     alpha_t: Fresnel weight on the glass
    L_R        = L_rgb(I_t^base, I_t) + L_ode + lambda_scale * L_scale + lambda_o * L_opt

Reads <model_path>/interface and <model_path>/glass_plane.json, writes <model_path>/reflection. Every --checkpoint_interval
iterations (default 1000) the training state goes to <model_path>/reflection/checkpoint.pth, and a rerun continues from it.

    python train_reflection.py -m output/<scene>

Ablations (model parameters, see arguments/__init__.py): --ablate_virtual_camera, --ablate_fresnel and
--shared_reflection_field, which has no G_R at all and renders the frozen interface through the virtual camera.
"""

import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, ReflectionOptimizationParams, get_combined_args
from gaussian_renderer.vigor import render_composite, render_direct
from scene.exode import ExODEMotion
from train_interface import time_noise_std
from utils.checkpoint_utils import load_checkpoint, remove_checkpoint, save_checkpoint, set_rng_state, to_cuda
from utils.fresnel import BlendModel, optical_loss
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.memory_utils import get_placer
from utils.plane_utils import load_plane
from utils.vigor_utils import INTERFACE_DIR, PLANE_FILE, REFLECTION_DIR, load_interface, new_reflection, save_reflection

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def reflection_losses(out, viewpoint_cam, opt, motion_losses=None):
    """The terms of L_R for one view, as a dict of scalar tensors."""
    gt_image = viewpoint_cam.original_image[:3].cuda()
    prediction = out['composite']
    losses = {'rgb': (1.0 - opt.lambda_dssim) * l1_loss(prediction, gt_image) + opt.lambda_dssim * (1.0 - ssim(prediction, gt_image))}
    losses.update(motion_losses or {})  # L_ode and L_scale of G_R's ExODE-GS motion
    if out['h'] is not None and opt.lambda_opt > 0:
        losses['opt'] = opt.lambda_opt * optical_loss(out['h'])
    return losses


class DirectCache:
    """
    The frozen interface's direct image B_t and glass map for each training view, rendered once.

    Phase I never changes during Phase II, so there is no reason to rasterize it again every iteration. The cache goes
    through the cameras' placer (utils/memory_utils.py), so with --data_device auto it shares their VRAM budget and
    whatever does not fit waits in pinned host RAM.
    """

    def __init__(self, cameras, interface, interface_motion, pipe, background, use_gt_mask=False):
        self.entries = {}
        for cam in tqdm(cameras, desc='Rendering the frozen interface'):
            out = render_direct(cam, interface, interface_motion, pipe, background)
            glass_map = cam.gt_transparent_mask.float() if use_gt_mask else out['glass_map']
            placer = self._placer(cam)
            self.entries[cam.uid] = (placer.place(out['direct'].detach()), placer.place(glass_map.detach()))
        if cameras:
            print('[ INFO ] with the cache of direct images,', self._placer(cameras[0]).summary())

    @staticmethod
    def _placer(cam):
        """A scene camera's placer; bare cameras (e.g. MiniCam in tests) fall back to their data_device."""
        return getattr(cam, 'placer', None) or get_placer(getattr(cam, 'data_device', 'cuda'))

    def __call__(self, cam):
        direct, glass_map = self.entries[cam.uid]
        return direct.cuda(), glass_map.cuda()


@torch.no_grad()
def evaluate_composite(cameras, interface, interface_motion, reflection, reflection_motion, pipe, background, normal, offset, blend, use_gt_mask=False, use_virtual_camera=True):
    """
    Mean PSNR against the original images of the composite and of the direct image alone, over the whole image
    and over the glass region. The gap between the two is what the reflection explains.
    """
    stats = {'psnr': [], 'psnr_direct': [], 'glass_psnr': [], 'glass_psnr_direct': []}
    for cam in cameras:
        direct_out = render_direct(cam, interface, interface_motion, pipe, background)
        glass_map = cam.gt_transparent_mask.cuda().float() if use_gt_mask else direct_out['glass_map']
        geometry = reflection_motion.geometry(cam.time)
        out = render_composite(cam, direct_out['direct'], glass_map, reflection, pipe, normal, offset, reflection_geometry=geometry, blend=blend, use_virtual_camera=use_virtual_camera)
        gt_image = cam.original_image[:3].cuda()
        composite, direct = out['composite'].clamp(0.0, 1.0), out['direct'].clamp(0.0, 1.0)
        stats['psnr'].append(psnr(composite, gt_image).mean().item())
        stats['psnr_direct'].append(psnr(direct, gt_image).mean().item())
        mask = cam.gt_transparent_mask.cuda()
        if mask.any():
            stats['glass_psnr'].append(psnr(composite * mask, gt_image * mask).mean().item())
            stats['glass_psnr_direct'].append(psnr(direct * mask, gt_image * mask).mean().item())
    return {key: float(np.mean(values)) if values else float('nan') for key, values in stats.items()}


def prepare_output_and_logger(args):
    reflection_path = os.path.join(args.model_path, REFLECTION_DIR)
    print('Output folder: {}'.format(reflection_path))
    os.makedirs(reflection_path, exist_ok=True)
    with open(os.path.join(reflection_path, 'cfg_args'), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(reflection_path) if TENSORBOARD_FOUND else None
    if tb_writer is None:
        print('Tensorboard not available: not logging progress')
    return reflection_path, tb_writer


def training(dataset, opt, pipe, testing_iterations, saving_iterations, interface_iteration=-1, checkpoint_interval=1000):
    reflection_path, tb_writer = prepare_output_and_logger(dataset)

    # ---- Phase I results, frozen for the whole of Phase II
    scene, interface, interface_motion = load_interface(dataset, interface_iteration, shuffle=True)
    interface.freeze()  # its motion parameters are part of the field, so they freeze with it
    normal, offset, plane_record = load_plane(os.path.join(dataset.model_path, PLANE_FILE))
    print(f'Glass plane ({plane_record.get("source", "?")}): n = {[round(v, 4) for v in normal.tolist()]}, d = {offset:.4f}')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    use_virtual_camera = not dataset.ablate_virtual_camera

    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    num_frames = len(viewpoint_stack)
    camera_centers = torch.stack([cam.camera_center for cam in viewpoint_stack])
    direct_cache = DirectCache(viewpoint_stack, interface, interface_motion, pipe, background, use_gt_mask=opt.gt_glass_mask)

    # ---- What Phase II trains
    blend = BlendModel.from_args(dataset).cuda()
    blend_groups = [{'params': [p], 'lr': opt.f0_lr if name == 'f0_logit' else opt.optical_lr} for name, p in blend.named_parameters()]
    blend_optimizer = torch.optim.Adam(blend_groups) if blend_groups else None

    own_field = not dataset.shared_reflection_field
    if own_field:
        reflection = new_reflection(dataset)
        reflection.create_random(normal, offset, camera_centers, scene.cameras_extent, opt.refl_init_points, opt.refl_init_radius_scale, opt.refl_init_color)
        reflection.training_setup(opt)
        reflection_motion = ExODEMotion(reflection, dataset.ode_chunk)
        print(f'ExODE-GS motion in G_R: {"on" if reflection_motion.dynamic else "off (static reflection)"}')
    else:
        # Shared-field ablation: the reflection is the frozen interface itself, seen from the virtual camera.
        reflection, reflection_motion = interface, interface_motion
        if blend_optimizer is None:
            print('Shared reflection field with pure Fresnel mixing: nothing to train, evaluating only.')

    num_iterations = opt.iterations if (own_field or blend_optimizer is not None) else 0
    testing_iterations = sorted(set(i for i in testing_iterations if i <= num_iterations) | {num_iterations})
    saving_iterations = sorted(set(i for i in saving_iterations if 0 < i <= num_iterations) | {num_iterations})

    def evaluate(iteration):
        torch.cuda.empty_cache()
        results = []
        train_subset = [viewpoint_stack[idx % num_frames] for idx in range(5, 30, 5)]
        for name, cameras in (('test', scene.getTestCameras()), ('train', train_subset)):
            if not cameras:
                continue
            metrics = evaluate_composite(cameras, interface, interface_motion, reflection, reflection_motion, pipe, background, normal, offset, blend, opt.gt_glass_mask, use_virtual_camera)
            results.append({'iteration': iteration, 'split': name, **metrics})
            print(
                f'\n[ITER {iteration}] Evaluating {name}: PSNR {metrics["psnr"]:.3f} (direct only {metrics["psnr_direct"]:.3f}), '
                f'glass PSNR {metrics["glass_psnr"]:.3f} (direct only {metrics["glass_psnr_direct"]:.3f})'
            )
            if tb_writer:
                for key, value in metrics.items():
                    tb_writer.add_scalar(f'{name}/{key}', value, iteration)
        torch.cuda.empty_cache()
        return results

    def save(iteration):
        print('\n[ITER {}] Saving reflection'.format(iteration))
        save_reflection(dataset.model_path, iteration, blend, reflection if own_field else None)

    history = []
    ema_loss_dict = {}
    first = 1
    checkpoint = load_checkpoint(reflection_path, num_iterations) if num_iterations > 0 else None
    if checkpoint is not None:
        if own_field:
            reflection.restore(to_cuda(checkpoint['field']), opt)
        blend.load_state_dict(checkpoint['blend'])
        if blend_optimizer is not None:
            blend_optimizer.load_state_dict(checkpoint['blend_optimizer'])
        history, ema_loss_dict = checkpoint['history'], checkpoint['ema_loss_dict']
        set_rng_state(checkpoint['rng'])
        first = checkpoint['iteration'] + 1
        del checkpoint
    progress_bar = tqdm(range(0, num_iterations), desc='Reflection training', initial=first - 1)

    for iteration in range(first, num_iterations + 1):
        if own_field:
            reflection.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                reflection.oneupSHdegree()

        viewpoint_cam = viewpoint_stack[np.random.randint(num_frames)]
        direct, glass_map = direct_cache(viewpoint_cam)

        # The reflection field trains statically first; a shared (frozen) field always uses its own motion.
        moving = (iteration > opt.warm_up if own_field else True) and reflection_motion.dynamic
        noise = time_noise_std(opt, iteration, num_frames) if own_field else 0.0
        geometry = reflection_motion.geometry(viewpoint_cam.time, enabled=moving, time_noise=noise)

        out = render_composite(viewpoint_cam, direct, glass_map, reflection, pipe, normal, offset, reflection_geometry=geometry, blend=blend, use_virtual_camera=use_virtual_camera)
        motion_losses = reflection_motion.regularizers(geometry[2], opt, scene.cameras_extent) if (own_field and moving) else {}
        loss_dict = reflection_losses(out, viewpoint_cam, opt, motion_losses)
        loss = sum(loss_dict.values())
        loss.backward()

        with torch.no_grad():
            for key, value in loss_dict.items():
                ema_loss_dict[key] = 0.4 * value.item() + 0.6 * ema_loss_dict.get(key, value.item())
            if iteration % 10 == 0:
                progress_bar.set_postfix({**{k: f'{v:.4f}' for k, v in ema_loss_dict.items()}, 'Points': reflection.num_points})
                progress_bar.update(10)

            if tb_writer:
                tb_writer.add_scalar('train_loss/total', loss.item(), iteration)
                for key, value in loss_dict.items():
                    tb_writer.add_scalar(f'train_loss/{key}', value.item(), iteration)
                tb_writer.add_scalar('total_points', reflection.num_points, iteration)

            if iteration in testing_iterations:
                history.extend(evaluate(iteration))
            if iteration in saving_iterations:
                save(iteration)

            # Densification
            if own_field and iteration < opt.densify_until_iter:
                visibility_filter, radii = out['visibility_filter'], out['radii']
                reflection.max_radii2D[visibility_filter] = torch.max(reflection.max_radii2D[visibility_filter], radii[visibility_filter])
                reflection.add_densification_stats(out['viewspace_points'], visibility_filter, iteration)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    reflection.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0:
                    reflection.reset_opacity()

            if iteration < num_iterations:
                if own_field:
                    reflection.optimizer.step()
                    reflection.optimizer.zero_grad(set_to_none=True)
                if blend_optimizer is not None:
                    blend_optimizer.step()
                    blend_optimizer.zero_grad(set_to_none=True)
                if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                    save_checkpoint(reflection_path, iteration, num_iterations, {
                        'field': reflection.capture() if own_field else None, 'blend': blend.state_dict(),
                        'blend_optimizer': blend_optimizer.state_dict() if blend_optimizer is not None else None,
                        'history': history, 'ema_loss_dict': ema_loss_dict})
    progress_bar.close()
    remove_checkpoint(reflection_path)  # the final model is saved; the checkpoint is no longer needed

    if num_iterations == 0:
        history.extend(evaluate(0))
        save(0)

    if tb_writer:
        tb_writer.close()
    return history


if __name__ == '__main__':
    parser = ArgumentParser(description='Phase II: reflection reconstruction with ExODE-GS motion')
    lp = ModelParams(parser, sentinel=True)  # the scene settings come from Phase I's cfg_args
    op = ReflectionOptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--interface_iteration', default=-1, type=int, help='Phase I iteration to load (default: the latest)')
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs='+', type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[])
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='save the training state every N iterations to resume from (0: never)')
    parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser, cfg_subdir=INTERFACE_DIR)

    print('Optimizing ' + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, getattr(args, 'interface_iteration', -1), args.checkpoint_interval)

    print('\nReflection training complete.')
