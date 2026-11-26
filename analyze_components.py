"""
Component analysis of a rendered test split (report, Section 5.1.3).

Reads what render_vigor.py writes for one model and test iteration:

    <test_dir>/renders/<i>.png             blended composite I_t^base = (1 - alpha) B + alpha R
    <test_dir>/gt/<i>.png                  original image I_t
    <test_dir>/transparent_masks/<i>.png   glass mask M_t
    <test_dir>/vis/direct_<i>.png          direct image B_t
    <test_dir>/vis/alpha_<i>.png           reflection weight alpha_t (8-bit, no colour transform)
    <test_dir>/vis/glass_map_<i>.png       predicted glass map g_t (8-bit)

and optionally the refined output I_hat_t of train_refiner.py (`--refined_dir`, e.g. <model>/test/refined_4000).

Reported (all PSNR in dB, on sRGB PNGs in [0, 1]):
    B_t / I_t^base / I_hat_t PSNR_M   mean over test frames of the PSNR over the glass pixels (mask > 0.5)
    ... PSNR outside                  the same over the pixels outside the mask
    Phase II gain                     mean of the per-frame I_t^base minus B_t PSNR_M, with a percentile bootstrap
                                      95% interval over frames (10,000 resamples, seed 0)
    alpha inside glass                median, 25th and 75th percentiles and share above 0.2, pooled over the glass
                                      pixels of all test frames
    alpha outside glass               per frame, the share of the frame's total alpha that lies outside the mask,
                                      averaged over frames
    IoU of g_t > 0.5 and M_t          pooled over all test frames (summed intersections over summed unions)
    mean g_t outside glass            pooled over the pixels outside the mask of all test frames
    DAI vs. original                  (needs --scene_dir) PSNR between the reflection-removed image and the original,
                                      inside and outside the glass mask, on the training frames at the training
                                      resolution (`--resolution`, resized like eval_masked_baselines: PIL default
                                      filter, mask > 0 after the resize), averaged over frames

These definitions reproduce the VIGOR-GS component tables of the report from the VIGOR-GS test renders.

Usage:
    python analyze_components.py --test_dir <model>/test/ours_30000 --refined_dir <model>/test/refined_4000 \
        --scene_dir <scene with colmap/> --resolution 2 --out components.json
"""

import json
import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image

BOOTSTRAP = 10000
SEED = 0


def load(path, gray=False):
    return np.asarray(Image.open(path).convert('L' if gray else 'RGB')).astype(np.float64) / 255.0


def region_psnr(image, reference, region):
    """PSNR over the pixels of `region` (H x W bool), all channels."""
    if not region.any():
        return float('nan')
    mse = ((image - reference) ** 2)[region].mean()
    return float(10.0 * np.log10(1.0 / max(mse, 1e-10)))


def bootstrap_ci(values, n=BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def analyze_test(test_dir, refined_dir=None):
    names = sorted(os.listdir(os.path.join(test_dir, 'renders')))
    per_frame, alpha_in = [], []
    inter = union = 0
    g_out_sum = g_out_count = 0.0
    for name in names:
        gt = load(os.path.join(test_dir, 'gt', name))
        mask = load(os.path.join(test_dir, 'transparent_masks', name), gray=True) > 0.5
        direct = load(os.path.join(test_dir, 'vis', 'direct_' + name))
        base = load(os.path.join(test_dir, 'renders', name))
        alpha = load(os.path.join(test_dir, 'vis', 'alpha_' + name), gray=True)
        glass = load(os.path.join(test_dir, 'vis', 'glass_map_' + name), gray=True)
        row = {
            'file': name,
            'mask fraction': float(mask.mean()),
            'B PSNR_M': region_psnr(direct, gt, mask),
            'base PSNR_M': region_psnr(base, gt, mask),
            'B PSNR outside': region_psnr(direct, gt, ~mask),
            'base PSNR outside': region_psnr(base, gt, ~mask),
            'alpha outside share': float(alpha[~mask].sum() / max(alpha.sum(), 1e-12)),
        }
        if refined_dir:
            refined = load(os.path.join(refined_dir, 'renders', name))
            row['refined PSNR_M'] = region_psnr(refined, gt, mask)
            row['refined PSNR outside'] = region_psnr(refined, gt, ~mask)
        row['Phase II gain'] = row['base PSNR_M'] - row['B PSNR_M']
        per_frame.append(row)
        alpha_in.append(alpha[mask])
        pred = glass > 0.5
        inter += int((pred & mask).sum())
        union += int((pred | mask).sum())
        g_out_sum += float(glass[~mask].sum())
        g_out_count += float((~mask).sum())

    def mean(key):
        return float(np.nanmean([r[key] for r in per_frame]))

    alpha_in = np.concatenate(alpha_in)
    gains = np.array([r['Phase II gain'] for r in per_frame])
    summary = {
        'frames': len(names),
        'B PSNR_M': mean('B PSNR_M'),
        'base PSNR_M': mean('base PSNR_M'),
        'Phase II gain': float(gains.mean()),
        'Phase II gain 95% CI': bootstrap_ci(gains),
        'B PSNR outside': mean('B PSNR outside'),
        'base PSNR outside': mean('base PSNR outside'),
        'alpha inside median': float(np.median(alpha_in)),
        'alpha inside p25': float(np.percentile(alpha_in, 25)),
        'alpha inside p75': float(np.percentile(alpha_in, 75)),
        'glass pixels with alpha > 0.2': float((alpha_in > 0.2).mean()),
        'alpha outside share': mean('alpha outside share'),
        'IoU g>0.5 vs M': inter / max(union, 1),
        'mean g outside': g_out_sum / max(g_out_count, 1.0),
    }
    if refined_dir:
        summary['refined PSNR_M'] = mean('refined PSNR_M')
        summary['refined PSNR outside'] = mean('refined PSNR outside')
    return summary, per_frame


def find_clean(clean_dir, frame, stem):
    folders = []
    if frame.get('source_image'):
        folders.append(os.path.join(clean_dir, os.path.dirname(frame['source_image'])))
    folders += [os.path.join(clean_dir, 'train', 'images'), clean_dir]
    for folder in folders:
        if os.path.isdir(folder):
            for entry in sorted(os.listdir(folder)):
                if os.path.splitext(entry)[0] == stem:
                    return os.path.join(folder, entry)
    raise FileNotFoundError(f'No clean image for {stem} under {clean_dir}')


def analyze_dai(scene_dir, resolution, clean_images='clean_image'):
    """Reflection-removed targets vs. the originals on the training frames, at the training resolution."""
    colmap = os.path.join(scene_dir, 'colmap')
    with open(os.path.join(colmap, 'split.json')) as f:
        train = set(json.load(f)['train'])
    with open(os.path.join(colmap, 'frames.json')) as f:
        frames = [fr for fr in json.load(f)['frames'] if fr['name'] in train]
    clean_dir = os.path.join(colmap, clean_images)
    if not os.path.isdir(clean_dir):
        clean_dir = os.path.join(scene_dir, clean_images)
    inside, outside = [], []
    for frame in frames:
        stem = os.path.splitext(frame['name'])[0]
        image = Image.open(os.path.join(colmap, 'images', frame['name'])).convert('RGB')
        size = (round(image.width / resolution), round(image.height / resolution))
        image = np.asarray(image.resize(size)).astype(np.float64) / 255.0
        clean = Image.open(find_clean(clean_dir, frame, stem)).convert('RGB')
        clean = np.asarray(clean.resize(size)).astype(np.float64) / 255.0
        mask = np.asarray(Image.open(os.path.join(colmap, 'masks', stem + '.png')).convert('L').resize(size)) > 0
        inside.append(region_psnr(clean, image, mask))
        outside.append(region_psnr(clean, image, ~mask))
    return {'training frames': len(frames), 'size': list(size), 'DAI PSNR_M': float(np.nanmean(inside)), 'DAI PSNR outside': float(np.nanmean(outside))}


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--test_dir', required=True, help='<model>/test/ours_<iteration> written by render_vigor.py')
    parser.add_argument('--refined_dir', default=None, help='<model>/test/refined_<iteration> written by train_refiner.py')
    parser.add_argument('--scene_dir', default=None, help='scene folder with colmap/ (enables the DAI column)')
    parser.add_argument('--resolution', '-r', type=int, default=1, help='training resolution divisor of the run')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    summary, per_frame = analyze_test(args.test_dir, args.refined_dir)
    if args.scene_dir:
        summary.update(analyze_dai(args.scene_dir, args.resolution))
    with open(args.out, 'w') as f:
        json.dump({'test_dir': args.test_dir, 'refined_dir': args.refined_dir, 'summary': summary, 'per_frame': per_frame}, f, indent=1)
    print(json.dumps(summary, indent=1))
