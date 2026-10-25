"""
plot_results_v2.py
------------------
Publication figures from the result JSONs written by the full pipeline.
Reads RESULTS_DIR (scripts/config.py, or the RESULTS_DIR environment variable) and
writes PNG + PDF to RESULTS_DIR/figures_v2/:

  fig_roc_layer2          pooled subject-level ROC, Methods A/B/C (main run)
  fig_methods_by_config   Method A/B/C AUC with pooled 95% CI per configuration
  fig_membership          membership-inference AUC vs site size, per epsilon
  fig_reid_bins           re-identification Top-1 vs bin count (p-values vs quantised)

Run:
    python scripts/plot_results_v2.py
"""

import sys, os, os.path as osp, json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
if 'RESULTS_DIR' in os.environ:
    RESULTS_DIR = os.environ['RESULTS_DIR']
else:
    sys.path.insert(0, SCRIPT_DIR)
    import config as C
    RESULTS_DIR = C.RESULTS_DIR
OUT_DIR = osp.join(RESULTS_DIR, 'figures_v2')
os.makedirs(OUT_DIR, exist_ok=True)

# Validated categorical order (blue, orange, aqua, yellow); identity is also carried
# by marker shape and direct labels, since aqua/yellow are below 3:1 on the surface.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D']
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
METHODS = [('mean', 'A: mean p-value'), ('individ', 'B: per-window FGSS'),
           ('group', 'C: group FGSS')]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 10, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
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


# ── Fig: pooled ROC, layer 2 main run ─────────────────────────────────────────
r = load('deepscan_results_layer_2.json')
if r:
    s = r['subject_scores']
    y = np.array(s['label'])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], ls=':', lw=1, color=INK2)
    for i, (m, label) in enumerate(METHODS):
        if m not in s:
            continue
        sc = np.array(s[m], dtype=float)
        keep = ~np.isnan(sc)
        fpr, tpr, _ = roc_curve(y[keep], sc[keep])
        ci = r[f'aggregate_{m}']['pooled_bootstrap']
        ax.plot(fpr, tpr, color=SERIES[i],
                label=f"{label.split(':')[0]}  {ci['auc']:.2f} [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
    ax.set_xlabel('False positive rate (1 − specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect('equal')
    # Short labels keep the legend in the empty lower-right corner, clear of the curves
    ax.legend(loc='lower right', title='Method: pooled AUC [95% CI]', title_fontsize=8)
    save(fig, 'fig_roc_layer2')

# ── Fig: Method AUC with CI per configuration ─────────────────────────────────
configs = [('', 'Held-out HC ref.\n(all tasks)'), ('_hctrain', 'HC-train ref.\n(all tasks)'),
           ('_relaxed_right', 'Held-out HC ref.\n(Relaxed, right wrist)')]
rows = [(lab, load(f'deepscan_results_layer_2{tag}.json')) for tag, lab in configs]
rows = [(lab, rr) for lab, rr in rows if rr]
if rows:
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(rows))
    offs = np.linspace(-0.22, 0.22, len(METHODS))
    for i, (m, label) in enumerate(METHODS):
        pts = [(rr[f'aggregate_{m}']['pooled_bootstrap'] if f'aggregate_{m}' in rr else None)
               for _, rr in rows]
        xs = [x[j] + offs[i] for j, p in enumerate(pts) if p]
        ps = [p for p in pts if p]
        if not ps:
            continue
        auc = np.array([p['auc'] for p in ps])
        err = np.array([[p['auc'] - p['ci_low'] for p in ps], [p['ci_high'] - p['auc'] for p in ps]])
        ax.errorbar(xs, auc, yerr=err, fmt=MARKERS[i], ms=6, color=SERIES[i], ecolor=SERIES[i],
                    elinewidth=1.5, capsize=0, label=label, mec=SURFACE, mew=1)
    ax.axhline(0.5, ls=':', lw=1, color=INK2)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.text(len(rows) - 0.52, 0.51, 'chance', va='bottom', ha='right', fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([lab for lab, _ in rows])
    ax.set_ylabel('Subject-level AUC (pooled, 95% CI)')
    ax.set_ylim(0.4, 1.0); ax.grid(axis='x', visible=False)
    ax.legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.0))
    save(fig, 'fig_methods_by_config')

# ── Fig: membership inference vs site size ────────────────────────────────────
r = load('privacy_membership_layer_2.json')
if r:
    sizes, eps_list = r['config']['site_sizes'], r['config']['epsilons']
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ends = []
    for i, eps in enumerate(eps_list):
        key = lambda sz: f'size{sz}_eps{eps}'
        per_fold = np.array([[f['membership'][key(sz)][0] for sz in sizes] for f in r['folds']])
        mu, sd = per_fold.mean(0), per_fold.std(0)
        lab = 'No noise' if eps is None else f'ε = {eps:g}'
        ax.errorbar(sizes, mu, yerr=sd, fmt='-' + MARKERS[i], ms=6, color=SERIES[i],
                    elinewidth=1.2, capsize=0, mec=SURFACE, mew=1, label=lab)
        ends.append([mu[-1], lab])
    # Direct end labels, nudged apart so close series never collide
    ends.sort(key=lambda e: e[0])
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 0.05)
    for yv, lab in ends:
        ax.text(sizes[-1] + 1.2, yv, lab, va='center', fontsize=7.5, color=INK)
    # chance line stops short of the end labels
    ax.plot([sizes[0] - 2, sizes[-1]], [0.5, 0.5], ls=':', lw=1, color=INK2)
    ax.set_xticks(sizes)
    ax.set_xlabel('Healthy-control subjects per site')
    ax.set_ylabel('Membership-inference AUC')
    ax.set_ylim(0.35, 1.02); ax.set_xlim(sizes[0] - 2, sizes[-1] + 6)   # direct labels, no legend
    save(fig, 'fig_membership')

# ── Fig: re-identification vs bins ────────────────────────────────────────────
r = load('privacy_bins_sweep.json')
if r:
    bins = sorted(int(b) for b in r['per_bin'])
    pb = [r['per_bin'][str(b)] for b in bins]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for i, (k, lab) in enumerate([('reid_top1', 'Histogram p-values'),
                                  ('quant_reid_top1', 'Quantised activations')]):
        if f'{k}_mean' not in pb[0]:
            continue
        mu = np.array([v[f'{k}_mean'] for v in pb]) * 100
        sd = np.array([v[f'{k}_std'] for v in pb]) * 100
        ax.errorbar(bins, mu, yerr=sd, fmt='-' + MARKERS[i], ms=6, color=SERIES[i],
                    elinewidth=1.2, capsize=0, mec=SURFACE, mew=1, label=lab)
    raw = r['raw_reid']['top1_mean'] * 100
    ax.axhline(raw, ls='--', lw=1.2, color=INK2)
    ax.text(bins[0], raw + 0.2, f'Raw activations ({raw:.1f}%)', fontsize=7, color=INK2)
    ch = r.get('chance_top1', 0.02) * 100
    ax.axhline(ch, ls=':', lw=1, color=INK2)
    ax.text(bins[0], ch + 0.2, f'Chance ({ch:.1f}%)', fontsize=7, color=INK2)
    ax.set_xscale('log'); ax.set_xticks(bins); ax.set_xticklabels(bins); ax.minorticks_off()
    ax.set_xlabel('Histogram bins (N)')
    ax.set_ylabel('Top-1 re-identification accuracy (%)')
    ax.set_ylim(0, max(10, raw + 2))
    ax.legend(loc='lower right')
    save(fig, 'fig_reid_bins')

print(f'Figures in {OUT_DIR}')
