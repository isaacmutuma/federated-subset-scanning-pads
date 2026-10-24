"""
plot_privacy_utility_v3.py
--------------------------
Publication-quality privacy-utility trade-off figure.
Style matched to ICML / NeurIPS / MICCAI standards using tueplots guidelines.

Run:
    python scripts/plot_privacy_utility_v3.py
"""

import json, os, os.path as osp, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Locate results ────────────────────────────────────────────────────────────
CANDIDATES = [
    'results/privacy_bins_sweep.json',
    'privacy_bins_sweep.json',
]
try:
    SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
    sys.path.insert(0, SCRIPT_DIR)
    import config as C
    CANDIDATES.insert(0, osp.join(C.RESULTS_DIR, 'privacy_bins_sweep.json'))
except Exception:
    pass

data_path = None
for p in CANDIDATES:
    if osp.exists(p):
        data_path = p
        break

if data_path is None:
    print("ERROR: privacy_bins_sweep.json not found.")
    sys.exit(1)

with open(data_path) as f:
    data = json.load(f)

print(f"Loaded: {data_path}")

# ── Extract data ──────────────────────────────────────────────────────────────
bin_counts = sorted([int(b) for b in data['per_bin'].keys()])

reid_pval = [data['per_bin'][str(b)]['reid_top1_mean'] * 100 for b in bin_counts]
reid_pval_std = [data['per_bin'][str(b)]['reid_top1_std'] * 100 for b in bin_counts]
reid_raw  = data['raw_reid']['top1_mean'] * 100
reid_raw_std = data['raw_reid']['top1_std'] * 100
chance    = 2.0

has_ds = all('deepscan_auc_mean' in data['per_bin'][str(b)] for b in bin_counts)
if has_ds:
    ds_auc = [data['per_bin'][str(b)]['deepscan_auc_mean'] for b in bin_counts]
    ds_std = [data['per_bin'][str(b)]['deepscan_auc_std'] for b in bin_counts]

best_idx = int(np.argmin(reid_pval))
best_bin = bin_counts[best_idx]
best_val = reid_pval[best_idx]

# ── Publication style (tueplots ICML/NeurIPS conventions) ─────────────────────
plt.rcParams.update({
    # Fonts — serif to match paper body (Times)
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif',
                   'Nimbus Roman', 'TeX Gyre Termes'],
    'mathtext.fontset': 'cm',

    # Sizes — larger for readability
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,

    # Bold
    'axes.labelweight': 'bold',
    'font.weight': 'bold',

    # Lines — slightly heavier to match bold text
    'axes.linewidth': 1.0,
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
    'patch.linewidth': 0.8,
    'grid.linewidth': 0.5,

    # Ticks
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',

    # Layout
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.02,
    'axes.axisbelow': True,

    'axes.titleweight': 'bold',
})

# ── Colors — muted, colorblind-accessible ─────────────────────────────────────
C_REID  = '#c44e52'   # muted red
C_AUC   = '#4c72b0'   # muted blue
C_GRAY  = '#888888'

# ── Figure — two-column width ────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(7.5, 4.0))

# Use categorical x-axis (equal spacing) since bins aren't linearly spaced
x_pos = np.arange(len(bin_counts))
x_labels = [str(b) for b in bin_counts]

# ── Left axis: Re-identification accuracy ────────────────────────────────────
ax1.set_xlabel('Number of histogram bins ($N$)')
ax1.set_ylabel('Re-identification accuracy (%)', color=C_REID)

# Error bars
ax1.errorbar(x_pos, reid_pval, yerr=reid_pval_std,
             fmt='o-', color=C_REID, markerfacecolor='white',
             markeredgecolor=C_REID, markeredgewidth=1.0,
             markersize=8, capsize=3.5, capthick=1.0,
             linewidth=1.8, label='ReID accuracy (histogram $p$-values)',
             zorder=5)

# Raw activation baseline
ax1.axhline(y=reid_raw, color=C_REID, linestyle='--', linewidth=1.4,
            alpha=0.55, label=f'ReID accuracy (raw activations): {reid_raw:.1f}%')

# Chance baseline
ax1.axhline(y=chance, color=C_GRAY, linestyle=':', linewidth=1.0,
            alpha=0.5, label='Random chance (2.0%)')

# Light shaded region for "privacy gain" — between raw and p-value line
ax1.fill_between(x_pos, reid_pval, reid_raw,
                 color=C_REID, alpha=0.06)

# Annotate best operating point
best_x = x_pos[best_idx]
ax1.annotate(f'$N$={best_bin}: {best_val:.1f}%',
             xy=(best_x, best_val), xytext=(18, -15),
             textcoords='offset points',
             fontsize=12, fontweight='bold', color=C_REID,
             arrowprops=dict(arrowstyle='->', color=C_REID,
                             lw=1.2, connectionstyle='arc3,rad=-0.2'))

ax1.set_ylim(0, 11)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels)
ax1.tick_params(axis='y', labelcolor=C_REID)

# ── Right axis: Detection AUC ───────────────────────────────────────────────
if has_ds:
    ax2 = ax1.twinx()
    ax2.set_ylabel('Detection AUC (Method A)', color=C_AUC)

    ax2.errorbar(x_pos, ds_auc, yerr=ds_std,
                 fmt='s-', color=C_AUC, markerfacecolor='white',
                 markeredgecolor=C_AUC, markeredgewidth=1.0,
                 markersize=7, capsize=3.5, capthick=1.0,
                 linewidth=1.8,
                 label='DeepScan AUC (Method A)', zorder=5)

    ax2.set_ylim(0.65, 0.85)
    ax2.tick_params(axis='y', labelcolor=C_AUC)

    # Style right spine to match axis color
    ax2.spines['right'].set_color(C_AUC)
    ax2.spines['right'].set_linewidth(1.0)
    ax2.spines['top'].set_visible(False)

# Style left spine
ax1.spines['left'].set_color(C_REID)
ax1.spines['left'].set_linewidth(1.0)
ax1.spines['top'].set_visible(False)
if not has_ds:
    ax1.spines['right'].set_visible(False)

# ── Combined legend — below plot, single row ─────────────────────────────────
lines1, labels1 = ax1.get_legend_handles_labels()
if has_ds:
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2
else:
    all_lines = lines1
    all_labels = labels1

fig.legend(all_lines, all_labels,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.02),
           ncol=2, fontsize=10,
           frameon=True, edgecolor='#cccccc',
           fancybox=False, framealpha=0.9,
           handlelength=2.0, columnspacing=1.2,
           borderpad=0.4)

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = osp.dirname(data_path) if osp.dirname(data_path) else '.'
png_path = osp.join(out_dir, 'privacy_utility.png')
pdf_path = osp.join(out_dir, 'privacy_utility.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
