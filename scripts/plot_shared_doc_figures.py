"""
plot_shared_doc_figures.py
--------------------------
Figures 1-5 of the shared working doc, redrawn with the corrected-run numbers, collected in
RESULTS_DIR/figures_shared_doc/ (PNG + PDF):

  fig1_system_overview      (a) classification pathway + supervised AUCs, (b) scanning pathway,
                            per-subject Method A scores and re-identification (real values);
                            panels marked "illustration" show the method, not data
  fig2_pca_window_level     copied from layer_analysis/ (run scripts/analyze_layers.py first)
  fig3_pca_subject_level    copied from layer_analysis/
  fig4_cross_validation     fold design and subject-level evaluation
  fig5_reid_vs_bins         re-identification Top-1 vs bin count, single axis

Reads: patchtst/minirocket/inceptiontime/boss_svm_results.json, deepscan_results_layer_2.json,
privacy_bins_sweep.json.

Run:
    python scripts/plot_shared_doc_figures.py
"""

import sys, os, os.path as osp, json, shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
if 'RESULTS_DIR' in os.environ:
    RESULTS_DIR = os.environ['RESULTS_DIR']
else:
    sys.path.insert(0, SCRIPT_DIR)
    import config as C
    RESULTS_DIR = C.RESULTS_DIR
OUT_DIR = osp.join(RESULTS_DIR, 'figures_shared_doc')
os.makedirs(OUT_DIR, exist_ok=True)

# Same validated palette as plot_results_v2.py; HC / PD keep aqua / orange throughout
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
HC_C, PD_C = AQUA, ORANGE
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'legend.frameon': False, 'lines.linewidth': 2,
})


def load(name):
    p = osp.join(RESULTS_DIR, name)
    if not osp.exists(p):
        print(f'  [missing] {name}')
        return None
    with open(p) as f:
        return json.load(f)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(osp.join(OUT_DIR, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  saved {name}.png/.pdf')


def tag(ax, text='illustration', left=False):
    ax.text(0.02 if left else 0.98, 0.98, text, transform=ax.transAxes,
            ha='left' if left else 'right', va='top', fontsize=7,
            color=MUTED, style='italic', zorder=5,
            bbox=dict(fc=SURFACE, ec='none', alpha=0.85, pad=1))


def arrows(fig, axes, y):
    """Arrows in the gaps between consecutive panels, measured from their drawn extents."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    boxes = [inv.transform(a.get_tightbbox(fig.canvas.get_renderer())) for a in axes]
    for b0, b1 in zip(boxes, boxes[1:]):
        x0, x1 = b0[1][0] + 0.004, b1[0][0] - 0.004
        if x1 - x0 > 0.006:
            fig.add_artist(matplotlib.patches.FancyArrowPatch(
                (x0, y), (x1, y), transform=fig.transFigure, arrowstyle='-|>',
                mutation_scale=12, color=INK2, lw=1.2))


def box(ax, x, y, w, h, text, edge=INK2, fill='none', alpha=1.0, weight='normal', size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.015',
                                fc=fill, ec=edge, lw=1.2, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', color=INK,
            fontsize=size, fontweight=weight)


rng = np.random.default_rng(7)
MODELS = [('MiniROCKET', 'minirocket_results.json'), ('PatchTST', 'patchtst_results.json'),
          ('InceptionTime', 'inceptiontime_results.json'), ('BOSS', 'boss_svm_results.json')]
models = [(n, load(f)) for n, f in MODELS]
models = [(n, r['aggregate']) for n, r in models if r]
ds = load('deepscan_results_layer_2.json')
sweep = load('privacy_bins_sweep.json')

# ── Fig 1: system overview ────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7.4))
top = fig.add_gridspec(1, 6, left=0.03, right=0.975, top=0.88, bottom=0.58, wspace=0.75)
bot = fig.add_gridspec(1, 5, left=0.03, right=0.99, top=0.40, bottom=0.08, wspace=0.6)
top_axes, bot_axes = [], []
fig.text(0.01, 0.95, '(a) Classification pathway', fontsize=12, fontweight='bold', color=INK)
fig.text(0.01, 0.47, '(b) Histogram-based subset scanning and privacy', fontsize=12,
         fontweight='bold', color=INK)

t = np.linspace(0, 2, 200)
ax = fig.add_subplot(top[0])
for i, ch in enumerate(['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']):
    sig = np.sin(2 * np.pi * (1.2 + 0.4 * i) * t + i) * 0.35 + 0.08 * rng.standard_normal(200)
    ax.plot(t, sig + i, lw=0.8, color=BLUE if i < 3 else ORANGE)
ax.set_yticks(range(6), ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
ax.set_xlabel('Time (s)')
ax.set_title('Raw IMU signal\n6 channels, 100 Hz')
tag(ax)

ax = fig.add_subplot(top[1])
for k in range(5):
    sig = np.sin(2 * np.pi * 1.6 * t + k) * 0.3 + 0.08 * rng.standard_normal(200)
    ax.plot(t, sig + k, lw=0.8, color=BLUE)
ax.set_yticks(range(5), [f'w{k + 1}' for k in range(5)])
ax.set_xlabel('200 samples (2 s)')
ax.set_title('Windows\n5 per recording, 110 per subject')
tag(ax)

ax = fig.add_subplot(top[2])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
for k in range(4):
    box(ax, 0.02, 0.78 - k * 0.2, 0.3, 0.14, f'P{k + 1}', edge=BLUE, size=8)
ax.text(0.17, 0.02, '24 patches\nper channel', ha='center', fontsize=7.5, color=INK2)
for k in range(3):
    box(ax, 0.5, 0.7 - k * 0.27, 0.48, 0.2, f'Transformer\nlayer {k}', size=8)
ax.annotate('', (0.5, 0.5), (0.33, 0.5), arrowprops=dict(arrowstyle='-|>', color=INK2))
ax.set_title('PatchTST encoder\nd = 128, 8 heads, 3 layers')

ax = fig.add_subplot(top[3])
ax.imshow(rng.standard_normal((40, 128)), aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
ax.set_xticks([0, 64, 127], ['0', '64', '128']); ax.set_yticks([])
ax.set_xlabel('neuron'); ax.set_ylabel('windows')
ax.set_title('Activations\n128 per window')
tag(ax)

ax = fig.add_subplot(top[4])
vals = np.convolve(rng.standard_normal(140), np.ones(12) / 12, mode='valid')[:128]
ax.bar(range(128), vals, width=1.0, color=BLUE)
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xticks([0, 64, 127], ['0', '64', '128']); ax.set_yticks([])
ax.set_xlabel('neuron')
ax.set_title('Subject representation\nmean over 110 windows')
tag(ax)

ax = fig.add_subplot(top[5])
ys = np.arange(len(models))[::-1]
for y, (n, a) in zip(ys, models):
    ax.errorbar(a['auc_mean'], y, xerr=a['auc_std'], fmt='o', color=BLUE, ms=6,
                capsize=3, lw=1.5)
    ax.text(a['auc_mean'], y + 0.28, f"{a['auc_mean']:.3f}", ha='center', fontsize=7.5, color=INK)
ax.set_yticks(ys, [n for n, _ in models])
ax.set_xlim(0.7, 1.0)
ax.set_ylim(-0.6, len(models) - 0.3)
ax.grid(axis='x')
ax.set_xlabel('AUC, mean ± SD (5 folds)')
ax.set_title('Supervised classifiers\nsubject-level AUC')


ax = fig.add_subplot(bot[0])
ax.imshow(rng.standard_normal((40, 128)), aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
ax.set_xticks([0, 64, 127], ['0', '64', '128']); ax.set_yticks([])
ax.set_xlabel('neuron'); ax.set_ylabel('healthy windows')
ax.set_title('Healthy reference\nheld-out HC activations')
tag(ax)

ax = fig.add_subplot(bot[1])
counts = np.array([3, 9, 20, 34, 40, 33, 22, 11, 6, 2], float)
cols = [BLUE] * 7 + [ORANGE] * 3
ax.bar(range(10), counts, width=0.92, color=cols)
ax.axvline(6.6, color=INK, lw=1, ls='--')
ax.text(6.75, 38, 'test\nactivation', fontsize=7.5, color=INK)
p = (counts[7:].sum() + 3) / (counts.sum() + 10)
ax.text(9.4, 27, f'p = {p:.2f}', ha='right', fontsize=8, color=INK)
ax.set_xticks(range(10), ['−∞'] + [''] * 8 + ['+∞'])
ax.set_xlabel('bin (edges fixed centrally, 10 bins)')
ax.set_yticks([])
ax.set_title('One neuron\'s histogram\np = share of HC at least as extreme')
tag(ax, left=True)

ax = fig.add_subplot(bot[2])
pv = np.sort(np.r_[rng.uniform(0.01, 0.25, 30), rng.uniform(0.05, 1, 98)])
ax.bar(range(128), pv, width=1.0, color=[ORANGE if v <= 0.25 else BLUE for v in pv])
ax.axhline(0.25, color=INK, lw=1, ls='--')
ax.text(2, 0.28, 'α', ha='left', fontsize=9, color=INK)
ax.set_xlabel('neurons, sorted by p-value')
ax.set_ylabel('p-value')
ax.set_title('Scan one window\'s 128 p-values\nfor the most unusual subset')
tag(ax)

ax = fig.add_subplot(bot[3])
if ds:
    ss = ds['subject_scores']
    y = np.array(ss['label']); sc = np.array(ss['mean'], float)
    for row, lv, c in [(1, 0, HC_C), (0, 1, PD_C)]:
        v = sc[y == lv]
        ax.scatter(v, row + rng.uniform(-0.28, 0.28, len(v)), s=10, color=c, alpha=0.7,
                   linewidths=0)
    ax.set_yticks([0, 1], [f'PD ({(y == 1).sum()})', f'HC ({(y == 0).sum()})'])
    ax.set_ylim(-0.6, 1.6)
    pooled = ds['aggregate_mean']['pooled_bootstrap']
    ax.set_title(f"Method A score per subject\npooled AUC {pooled['auc']:.3f} "
                 f"[{pooled['ci_low']:.3f}, {pooled['ci_high']:.3f}]")
ax.set_xlabel('anomaly score (1 − mean p)')

ax = fig.add_subplot(bot[4])
if sweep:
    b10 = sweep['per_bin']['10']
    bars = [('Raw', sweep['raw_reid']['top1_mean'], INK2),
            ('Quantised', b10['quant_reid_top1_mean'], BLUE),
            ('p-values', b10['reid_top1_mean'], BLUE)]
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v * 100, width=0.6, color=c)
        ax.text(i, v * 100 + 0.2, f'{v * 100:.2f}%', ha='center', fontsize=8, color=INK)
    ax.axhline(sweep['chance_top1'] * 100, color=MUTED, ls=':', lw=1.2)
    ax.text(3.1, sweep['chance_top1'] * 100 + 0.2, f"chance\n{sweep['chance_top1'] * 100:.1f}%",
            ha='right', fontsize=7.5, color=INK2)
    ax.set_xticks(range(3), [b[0] for b in bars])
    ax.set_xlim(-0.5, 3.15)
    ax.tick_params(axis='x', labelsize=7.5)
    ax.set_ylim(0, 10)
ax.set_ylabel('Top-1 re-identification (%)')
ax.set_title('Re-identification risk, 10 bins\n50–51 HC subjects per fold')

top_axes = fig.axes[:6]
bot_axes = fig.axes[6:11]
arrows(fig, top_axes, 0.73)
arrows(fig, bot_axes, 0.25)
save(fig, 'fig1_system_overview')

# ── Fig 2-3: PCA figures from analyze_layers.py ───────────────────────────────
for src, dst in [('pca_window_level', 'fig2_pca_window_level'),
                 ('pca_subject_level', 'fig3_pca_subject_level')]:
    for ext in ('png', 'pdf'):
        p = osp.join(RESULTS_DIR, 'layer_analysis', f'{src}.{ext}')
        if osp.exists(p):
            shutil.copy(p, osp.join(OUT_DIR, f'{dst}.{ext}'))
            print(f'  copied {dst}.{ext}')
        else:
            print(f'  [missing] layer_analysis/{src}.{ext} (run scripts/analyze_layers.py)')

# ── Fig 4: cross-validation design ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.5, 0.97, '5-fold stratified cross-validation, split by subject', ha='center',
        fontsize=13, fontweight='bold', color=INK)
ax.text(0.5, 0.915, '355 subjects (79 HC, 276 PD), both wrists, all 11 tasks; every window of '
        'a subject stays in one split', ha='center', fontsize=9.5, color=INK2)
# One fold, drawn to scale (subjects)
W, x0, y0 = 0.955 / 355, 0.02, 0.70
for n, lab, c in [(227, 'Train\n227 subjects (50 HC, 177 PD)', BLUE),
                  (57, 'Validation\n57 (13 HC, 44 PD)', ORANGE),
                  (71, 'Test\n71 (16 HC, 55 PD)', AQUA)]:
    box(ax, x0, y0, n * W, 0.11, '', edge=c, fill=c, alpha=0.18)
    box(ax, x0, y0, n * W, 0.11, lab, edge=c, size=8.5)
    x0 += n * W
ax.text(0.02, 0.6, 'One fold, drawn to scale. Across the 5 folds every subject is in the test set '
        'exactly once.\nValidation subjects (20% of the rest) are used for early stopping. '
        'Fold 5 has 51 HC subjects in training.', fontsize=8, color=INK2)

ys, bw = 0.33, 0.172
steps = ['Train the model\non the training\nsubjects',
         'Score every test\nwindow: 7,810 windows\n(71 subjects × 110)',
         'Average per subject:\none score per\nsubject',
         'Fold AUC over\n71 subjects',
         'Repeat for 5 folds:\nmean ± SD of\nthe fold AUCs']
gap = (0.955 - 5 * bw) / 4
xs = [0.02 + i * (bw + gap) for i in range(5)]
for x, st in zip(xs, steps):
    box(ax, x, ys, bw, 0.2, st, size=8.5)
for i in range(4):
    ax.annotate('', (xs[i + 1], ys + 0.1), (xs[i] + bw, ys + 0.1),
                arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.2))
box(ax, 0.02, 0.07, 0.955, 0.16,
    'Supervised result, subject-level AUC (mean ± SD over 5 folds)\n' + '   ·   '.join(
        f"{n} {a['auc_mean']:.3f} ± {a['auc_std']:.3f}" for n, a in models[:2]),
    edge=BLUE, weight='bold', size=10)
ax.text(0.02, 0.015, 'Windows: 2 s at 100 Hz. Subject-level evaluation as in Varghese et al. (2024).',
        fontsize=7.5, color=INK2)
save(fig, 'fig4_cross_validation')

# ── Fig 5: re-identification vs bin count ─────────────────────────────────────
if sweep:
    Ns = sweep['bin_counts']
    pb = sweep['per_bin']
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(Ns))
    for key, lab, c, m in [('reid_top1', 'Histogram p-values', BLUE, 'o'),
                           ('quant_reid_top1', 'Quantised activations', ORANGE, 's')]:
        mu = np.array([pb[str(n)][f'{key}_mean'] for n in Ns]) * 100
        sd = np.array([pb[str(n)][f'{key}_std'] for n in Ns]) * 100
        ax.errorbar(x, mu, yerr=sd, color=c, marker=m, ms=7, capsize=3, label=lab)
        ax.text(x[-1] + 0.12, mu[-1], lab, va='center', fontsize=8, color=INK)
    raw = sweep['raw_reid']['top1_mean'] * 100
    ch = sweep['chance_top1'] * 100
    ax.axhline(raw, color=INK2, ls='--', lw=1.2)
    ax.text(-0.3, raw + 0.25, f'raw activations {raw:.2f}%', fontsize=8, color=INK2)
    ax.axhline(ch, color=MUTED, ls=':', lw=1.2)
    ax.text(-0.3, ch + 0.25, f'chance {ch:.1f}%', fontsize=8, color=INK2)
    b10 = pb['10']
    ax.annotate(f"{b10['reid_top1_mean'] * 100:.2f}%", (1, b10['reid_top1_mean'] * 100),
                (1.25, 3.4), fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=INK2))
    ax.set_xticks(x, [str(n) for n in Ns])
    ax.set_xlim(-0.4, len(Ns) + 0.6)
    ax.set_ylim(0, 10)
    ax.grid(axis='y')
    ax.set_xlabel('Histogram bins (N)')
    ax.set_ylabel('Top-1 re-identification (%)')
    aucs = [pb[str(n)]['deepscan_auc_mean'] for n in Ns]
    ax.set_title(f'Re-identification against bin count\nMethod A detection AUC stays '
                 f'{min(aucs):.3f}–{max(aucs):.3f} at every N', fontsize=10)
    save(fig, 'fig5_reid_vs_bins')

print(f'Figures in {OUT_DIR}')
