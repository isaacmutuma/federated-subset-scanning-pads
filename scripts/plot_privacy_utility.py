"""
plot_privacy_utility.py
-----------------------
Generates the privacy-utility trade-off figure from privacy_bins_sweep.json.

Run:
    python scripts/plot_privacy_utility.py
"""

import json, os, os.path as osp, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Locate results ────────────────────────────────────────────────────────────
# Try multiple locations
CANDIDATES = [
    'results/privacy_bins_sweep.json',
    'privacy_bins_sweep.json',
    osp.join(osp.dirname(osp.abspath(__file__)), '..', 'results', 'privacy_bins_sweep.json'),
]

# Also try config-based path
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
    print("Run privacy_analysis_bins.py first.")
    sys.exit(1)

with open(data_path) as f:
    data = json.load(f)

print(f"Loaded: {data_path}")

# ── Extract data ──────────────────────────────────────────────────────────────
bin_counts = sorted([int(b) for b in data['per_bin'].keys()])

reid_pval_top1 = [data['per_bin'][str(b)]['reid_top1_mean'] * 100 for b in bin_counts]
reid_raw_top1  = data['raw_reid']['top1_mean'] * 100
random_chance  = 2.0  # 1/50 subjects

has_deepscan = all('deepscan_auc_mean' in data['per_bin'][str(b)] for b in bin_counts)
if has_deepscan:
    deepscan_auc = [data['per_bin'][str(b)]['deepscan_auc_mean'] for b in bin_counts]
else:
    print("WARNING: DeepScan AUC not found for all bins.")
    print("Plot will show re-identification only.")

# PatchTST supervised AUC (from base model comparison)
patchtst_supervised = 0.904

# ── Best bin annotation ───────────────────────────────────────────────────────
best_idx = int(np.argmin(reid_pval_top1))
best_bin = bin_counts[best_idx]
best_val = reid_pval_top1[best_idx]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5.5))

color_reid = '#C0392B'
color_reid_raw = '#E74C3C'
color_auc = '#2471A3'

# Left axis: Re-identification accuracy
ax1.set_xlabel('Number of histogram bins', fontsize=22, fontweight='bold')
ax1.set_ylabel('Re-identification accuracy (%)', fontsize=22, fontweight='bold',
               color=color_reid)

ax1.plot(bin_counts, reid_pval_top1, 'o-', color=color_reid, linewidth=3,
         markersize=12, label='ReID accuracy (p-values)', zorder=5)

ax1.axhline(y=reid_raw_top1, color=color_reid_raw, linestyle='--', linewidth=2.5,
            label='ReID accuracy (raw activations)', alpha=0.8)

ax1.axhline(y=random_chance, color='gray', linestyle=':', linewidth=2.5,
            label='Random chance (2%)', alpha=0.7)

# Annotate best point
ax1.annotate(f'Best: N={best_bin}\n{best_val:.2f}%',
             xy=(best_bin, best_val),
             xytext=(best_bin + (bin_counts[-1] - bin_counts[0]) * 0.12,
                     best_val - 1.2),
             fontsize=18, fontweight='bold', color=color_reid,
             arrowprops=dict(arrowstyle='->', color=color_reid, lw=2),
             ha='left')

ax1.set_ylim(0, max(reid_raw_top1 + 4, 13))
ax1.tick_params(axis='y', labelcolor=color_reid, labelsize=18, width=2)
ax1.tick_params(axis='x', labelsize=18, width=2)
ax1.set_xticks(bin_counts)
ax1.set_xticklabels([str(b) for b in bin_counts], fontsize=18, fontweight='bold')

for label in ax1.get_yticklabels():
    label.set_fontweight('bold')
for label in ax1.get_xticklabels():
    label.set_fontweight('bold')

# Right axis: DeepScan AUC (only if data available)
if has_deepscan:
    ax2 = ax1.twinx()
    ax2.set_ylabel('Detection AUC', fontsize=22, fontweight='bold', color=color_auc)

    ax2.plot(bin_counts, deepscan_auc, 's--', color=color_auc, linewidth=3,
             markersize=12, label='DeepScan AUC (Method A, unsupervised)', zorder=5)

    ax2.axhline(y=patchtst_supervised, color='#5B6B7A', linestyle='-.',
                linewidth=2.5, label=f'PatchTST AUC (supervised, {patchtst_supervised})',
                alpha=0.7)

    ax2.set_ylim(0.80, 0.95)
    ax2.tick_params(axis='y', labelcolor=color_auc, labelsize=18, width=2)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                        loc='upper center', fontsize=13, framealpha=0.95,
                        edgecolor='black', fancybox=False, ncol=2,
                        bbox_to_anchor=(0.5, 0.99))
    legend.get_frame().set_linewidth(1.5)

    for spine in ax2.spines.values():
        spine.set_linewidth(2)
else:
    legend = ax1.legend(loc='upper right', fontsize=14, framealpha=0.95,
                        edgecolor='black', fancybox=False)
    legend.get_frame().set_linewidth(1.5)

# Title
ax1.set_title('Privacy-utility trade-off — histogram bin count\n'
              'Lower ReID = more private  |  Higher AUC = more useful',
              fontsize=20, fontweight='bold', pad=15)

# Bold spines
for spine in ax1.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()

# Save
out_dir = osp.dirname(data_path) if osp.dirname(data_path) else '.'
png_path = osp.join(out_dir, 'privacy_utility.png')
pdf_path = osp.join(out_dir, 'privacy_utility.pdf')
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
print("Upload the PNG or PDF to Overleaf.")
