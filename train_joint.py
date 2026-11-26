"""
Ablation "joint optimization": fine-tune the reconstruction and the refiner together, end to end.

The default pipeline trains in three separate stages (Phase I, Phase II, refiner), each with what came before
frozen. This ablation starts from a finished model (all three stages, in its own copy of the model folder) and then
optimizes everything at once through the refined output:

    trainable   G_I and Theta_I, G_R and Theta_R, the blend (F0, h_psi), the refiner f_theta
    loss        L_rgb(I_hat_t, I_t)                               refined output vs original image (training frames)
                + L_rgb(B_t, I_t^clean)                           the Phase I target, so B keeps its meaning (--no_keep_clean drops it)
                + L_ode + lambda_scale L_scale for both fields    the ExODE-GS regularizers
    frozen      the glass plane, and the optical flows (computed once from the starting model's base renders)

Clips of consecutive frames are rendered and refined in time order exactly as in train_refiner.py, but the base
renders are now differentiable. Densification is off (the Gaussian sets stay fixed); learning rates are the
phases' final rates (positions) or a tenth of their initial rates.

Writes interface/ and reflection/ at iteration (loaded + joint_iterations), refiner/refiner_joint.pth and the
refined test renders test/joint_<joint_iterations>/.

    python train_joint.py -m output/<scene>_joint          # a copy of a finished model folder
"""

import json
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from arguments import InterfaceOptimizationParams, ModelParams, PipelineParams, ReflectionOptimizationParams, get_combined_args
from gaussian_renderer.interface import render_interface
from gaussian_renderer.vigor import render_composite
from refiner.network import CausalRefiner
from refiner.sequence import refine_step, run_sequence, to_srgb
from train_refiner import REFINER_DIR, History, prepare, refiner_loss, write_split
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.vigor_utils import INTERFACE_DIR, REFLECTION_DIR, save_reflection


def phase_defaults(group_class):
    parser = ArgumentParser()
    group = group_class(parser)
    return group.extract(parser.parse_args([]))


def fine_tune_rates(opt, scale=0.1):
    """A phase's optimization settings with the rates of a late fine-tune and no densification."""
    opt.position_lr_init = opt.position_lr_final
    opt.position_lr_delay_mult = 1.0
    for name in ('feature_lr', 'opacity_lr', 'scaling_lr', 'rotation_lr', 'ode_A_lr', 'ode_b_lr', 'ode_omega_lr', 'ode_kappa_lr'):
        setattr(opt, name, getattr(opt, name) * scale)
    if hasattr(opt, 'occupancy_lr'):
        opt.occupancy_lr *= scale
    opt.ode_lr_final_ratio = 1.0
    return opt


def main():
    parser = ArgumentParser(description='Ablation: joint fine-tuning of the reconstruction and the refiner')
    model_params = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--joint_iterations', default=3000, type=int)
    parser.add_argument('--joint_clip', default=3, type=int)
    parser.add_argument('--joint_refiner_lr', default=5e-5, type=float)
    parser.add_argument('--no_keep_clean', action='store_true', help='drop the Phase I term on B')
    parser.add_argument('--joint_seed', default=0, type=int)
    parser.add_argument('--quiet', action='store_true')
    args = get_combined_args(parser, cfg_subdir=REFLECTION_DIR)
    safe_state(args.quiet)
    random.seed(args.joint_seed)
    np.random.seed(args.joint_seed)
    torch.manual_seed(args.joint_seed)
    dataset, pipe = model_params.extract(args), pipeline.extract(args)

    with open(os.path.join(dataset.model_path, REFINER_DIR, 'config.json')) as f:
        refiner_config = json.load(f)
    scene, model, label, frames, flows, masks = prepare(dataset, pipe, refiner_config['history'], refiner_config['flow_iterations'])
    net = CausalRefiner(refiner_config['history'], refiner_config['width']).cuda()
    net.load_state_dict(torch.load(os.path.join(dataset.model_path, REFINER_DIR, 'refiner.pth')))
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device='cuda')

    interface, reflection, blend = model['interface'], model['reflection'], model['blend']
    interface_motion, reflection_motion = model['interface_motion'], model['reflection_motion']
    interface_opt = fine_tune_rates(phase_defaults(InterfaceOptimizationParams))
    reflection_opt = fine_tune_rates(phase_defaults(ReflectionOptimizationParams))
    for field in (interface, reflection):
        field.spatial_lr_scale = scene.cameras_extent
    interface.training_setup(interface_opt)
    reflection.training_setup(reflection_opt)
    blend_params = list(blend.parameters())
    optimizers = [interface.optimizer, reflection.optimizer, torch.optim.Adam(net.parameters(), lr=args.joint_refiner_lr)]
    if blend_params:
        optimizers.append(torch.optim.Adam(blend_params, lr=reflection_opt.f0_lr * 0.1))

    def base(cam):
        """Differentiable I_t^base (sRGB) and B_t (training color space) of one camera."""
        direct_out = render_interface(cam, interface, pipe, background, geometry=interface_motion.geometry(cam.time))
        out = render_composite(cam, direct_out['direct'], direct_out['glass_map'].detach(), reflection, pipe, model['normal'], model['offset'],
                               reflection_geometry=reflection_motion.geometry(cam.time), blend=blend, use_virtual_camera=model['use_virtual_camera'])
        return to_srgb(out['composite'], dataset.linear_rgb), direct_out['direct']

    # The history bank starts from the trained refiner's own outputs.
    bank = run_sequence(net, frames, flows, masks)
    clip = min(args.joint_clip, len(frames))
    starts = [s for s in range(len(frames) - clip + 1) if any(frames[t].split == 'train' for t in range(s, s + clip))]
    net.train()
    start_time, ema, log = time.time(), None, []
    progress = tqdm(range(1, args.joint_iterations + 1), desc='Joint fine-tuning')
    for iteration in progress:
        s = random.choice(starts)
        history = History(bank, s)
        losses = []
        for t in range(s, s + clip):
            frame = frames[t]
            base_t, direct = base(frame.cam)
            refined, _ = refine_step(net, base_t, history, t, flows[t], masks[t])
            history.clip[t] = refined[0]
            if frame.split != 'train':
                continue
            losses.append(refiner_loss(refined, frame.gt.cuda().float()[None], 0.2))
            if not args.no_keep_clean:
                clean = frame.cam.clean_image.cuda()
                losses.append((1 - interface_opt.lambda_dssim) * l1_loss(direct, clean) + interface_opt.lambda_dssim * (1 - ssim(direct, clean)))
            for motion, opt in ((interface_motion, interface_opt), (reflection_motion, reflection_opt)):
                geometry = motion.geometry(frame.cam.time)
                losses.extend(motion.regularizers(geometry[2], opt, scene.cameras_extent).values())
        loss = torch.stack(losses).sum() / max(1, sum(frames[t].split == 'train' for t in range(s, s + clip)))
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            for t in history.clip:
                bank[t] = history.full(t)
        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
        if iteration % 10 == 0:
            progress.set_postfix({'loss': f'{ema:.4f}'})
        if iteration % 100 == 0:
            log.append({'iteration': iteration, 'loss': ema})

    # Save the fine-tuned reconstruction as new iterations, and the refiner next to the separately trained one.
    interface_iter = int(scene.loaded_iter) + args.joint_iterations
    reflection_iter = int(label.split('_')[-1]) + args.joint_iterations
    interface.save_ply(os.path.join(dataset.model_path, INTERFACE_DIR, 'point_cloud', f'iteration_{interface_iter}', 'point_cloud.ply'))
    save_reflection(dataset.model_path, reflection_iter, blend, reflection)
    torch.save(net.state_dict(), os.path.join(dataset.model_path, REFINER_DIR, 'refiner_joint.pth'))
    with open(os.path.join(dataset.model_path, REFINER_DIR, 'joint.json'), 'w') as f:
        json.dump({'iterations': args.joint_iterations, 'keep_clean': not args.no_keep_clean, 'minutes': (time.time() - start_time) / 60, 'log': log}, f, indent=1)

    # Final images: base renders of the fine-tuned reconstruction, the same (frozen) flows, the fine-tuned refiner.
    with torch.no_grad():
        for frame in frames:
            frame.base = base(frame.cam)[0].clamp(0, 1).half().cpu()
    refined = run_sequence(net, frames, flows, masks)
    print('Joint renders:', write_split(dataset.model_path, 'test', f'joint_{args.joint_iterations}', frames, refined, scene, dataset.linear_rgb))


if __name__ == '__main__':
    main()
