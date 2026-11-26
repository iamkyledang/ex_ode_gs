"""
Component figure of the report (Section 5.1.3): for one test frame per dataset, the original I_t, direct image B_t,
reflection image R_t, reflection contribution alpha_t R_t (scaled by four), weight alpha_t (Blues, 0 to 0.3) and glass
map g_t with the outline of the glass mask M_t. Reads the layout render_vigor.py writes (<test_dir>/gt, vis, ...).

The layout (1615 x 1723 px at 100 dpi, panel positions, fonts) follows the original VIGOR-GS figure of the report.

Usage:
    python make_component_figure.py --out fig5_3_decomposition.jpg \
        --panel VSGD <model>/test/ours_30000 19 \
        --panel WayveScenes101 <model>/test/ours_30000 26 \
        --panel 3D-FRONT-T <model>/test/ours_30000 12
"""

import os
from argparse import ArgumentParser

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

WIDTH, HEIGHT, DPI = 1615, 1723, 100
COLUMNS = (45, 593, 1140)  # left edge of each panel column (px)
PANEL_W = 462
# top edge of the two panel rows of each dataset (px); the panel height follows the image aspect
ROW_TOPS = ((38, 338), (686, 886), (1148, 1449))
TITLES = (
    (r'Reference $I_t$', r'Direct image $B_t$', r'Reflection image $R_t$'),
    (r'Reflection contribution $\alpha_t R_t$ ($\times$4)', r'Blend weight $\alpha_t$ (0 → 0.3)', r'Glass map $g_t$, mask $M_t$ outlined'),
)
TITLE_SIZE, LABEL_SIZE = 14.2, 16
OUTLINE = '#17becf'


def load(path, gray=False):
    return np.asarray(Image.open(path).convert('L' if gray else 'RGB')).astype(np.float64) / 255.0


def draw(out, panels):
    fig = plt.figure(figsize=((WIDTH + 0.5) / DPI, (HEIGHT + 0.5) / DPI), dpi=DPI)  # +0.5 px: float sizes round down
    for (label, test_dir, index), tops in zip(panels, ROW_TOPS):
        name = '{0:05d}.png'.format(int(index))
        vis = os.path.join(test_dir, 'vis')
        gt = load(os.path.join(test_dir, 'gt', name))
        mask = load(os.path.join(test_dir, 'transparent_masks', name), gray=True) > 0.5
        images = (
            (gt, {}),
            (load(os.path.join(vis, 'direct_' + name)), {}),
            (load(os.path.join(vis, 'reflection_' + name)), {}),
            (np.clip(4.0 * load(os.path.join(vis, 'reflection_contribution_' + name)), 0.0, 1.0), {}),
            (load(os.path.join(vis, 'alpha_' + name), gray=True), {'cmap': 'Blues', 'vmin': 0.0, 'vmax': 0.3}),
            # lifted grey scale (0 -> 0.1, 1 -> 0.95) so that the black background stays distinct from g_t = 0
            (0.1 + 0.85 * load(os.path.join(vis, 'glass_map_' + name), gray=True), {'cmap': 'gray', 'vmin': 0.0, 'vmax': 1.0}),
        )
        h, w = gt.shape[:2]
        panel_h = PANEL_W * h / w
        for k, (image, style) in enumerate(images):
            row, col = divmod(k, 3)
            left, top = COLUMNS[col], tops[row]
            ax = fig.add_axes((left / WIDTH, 1 - (top + panel_h) / HEIGHT, PANEL_W / WIDTH, panel_h / HEIGHT))
            ax.imshow(image, interpolation='antialiased', **style)
            if k == 5:
                ax.contour(mask.astype(float), levels=[0.5], colors=OUTLINE, linewidths=1.0)
            ax.set_title(TITLES[row][col], fontsize=TITLE_SIZE, pad=3)
            ax.set_axis_off()
        centre = tops[0] + panel_h / 2  # the dataset label sits beside the first row
        fig.text(22 / WIDTH, 1 - centre / HEIGHT, label, rotation=90, fontsize=LABEL_SIZE, ha='center', va='center')
    fig.savefig(out, dpi=DPI, pil_kwargs={'quality': 92})
    plt.close(fig)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--panel', nargs=3, action='append', required=True, metavar=('LABEL', 'TEST_DIR', 'INDEX'),
                        help='dataset label, <model>/test/ours_<iteration>, test frame index (three times)')
    args = parser.parse_args()
    if len(args.panel) != 3:
        parser.error('give --panel three times (one per dataset)')
    draw(args.out, args.panel)
