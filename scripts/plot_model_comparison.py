"""
plot_model_comparison.py
------------------------
Publication-ready model comparison figure — final results.
Both wrists + all 11 tasks, subject-level mean pooling, 5-fold CV.

Run: python scripts/plot_model_comparison.py
Output: model_comparison.pdf  model_comparison.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Final results ─────────────────────────────────────────────────────────────
models = [
    'BOSS+SGD',
    'InceptionTime',
    'PatchTST',
    'MiniROCKET',
]

aucs = [0.711, 0.792, 0.801, 0.847]
stds = [0.044, 0.035, 0.058, 0.019]
specs = [0.722, 0.809, 0.848, 0.924]

# 0=classical, 1=deep/transformer
model_type  = [0, 1, 1, 0]
COLORS = {0: '#1baf7a', 1: '#2a78d6'}
bar_colors = [COLORS[t] for t in model_type]

# ── Figure layout: two panels ─────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
fig.subplots_adjust(wspace=0.38)

y_pos = np.arange(len(models))

# ── Panel 1: Mean AUC ─────────────────────────────────────────────────────────
bars1 = ax1.barh(y_pos, aucs, xerr=stds,
                 color=bar_colors, alpha=0.88,
                 error_kw=dict(ecolor='#52514e', capsize=4, capthick=1.2, elinewidth=1.2),
                 height=0.52, zorder=3)

for i, (auc, std) in enumerate(zip(aucs, stds)):
    ax1.text(auc + std + 0.005, i, f'{auc:.3f}',
             va='center', ha='left', fontsize=9.5, color='#0b0b0b', fontweight='500')

ax1.axvline(x=0.5, color='#c3c2b7', linestyle='--', linewidth=1, zorder=2)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models, fontsize=10.5)
ax1.set_xlabel('Mean AUC (subject-level, 5-fold CV)', fontsize=9.5)
ax1.set_xlim(0.52, 0.95)
ax1.set_ylim(-0.6, len(models) - 0.4)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax1.grid(axis='x', color='#e1e0d9', linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
ax1.spines[['top', 'right']].set_visible(False)
ax1.spines[['left', 'bottom']].set_color('#c3c2b7')
ax1.set_title('Mean AUC ± std dev', fontsize=10, color='#52514e', pad=6)

# MiniROCKET annotation
ax1.annotate('best', xy=(0.847, 3), xytext=(0.76, 3.42),
             fontsize=8, color='#1baf7a',
             arrowprops=dict(arrowstyle='->', color='#1baf7a', lw=1))

# PatchTST annotation
ax1.annotate('Phase 4\nbase model', xy=(0.801, 2), xytext=(0.62, 2.42),
             fontsize=7.5, color='#2a78d6',
             arrowprops=dict(arrowstyle='->', color='#2a78d6', lw=1))

# ── Panel 2: Specificity ─────────────────────────────────────────────────────
bars2 = ax2.barh(y_pos, specs,
                 color=bar_colors, alpha=0.88,
                 height=0.52, zorder=3)

for i, s in enumerate(specs):
    ax2.text(s + 0.005, i, f'{s:.3f}',
             va='center', ha='left', fontsize=9.5, color='#0b0b0b', fontweight='500')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(models, fontsize=10.5)
ax2.set_xlabel('Mean specificity (HC identification)', fontsize=9.5)
ax2.set_xlim(0.52, 1.02)
ax2.set_ylim(-0.6, len(models) - 0.4)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax2.grid(axis='x', color='#e1e0d9', linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color('#c3c2b7')
ax2.set_title('Mean specificity', fontsize=10, color='#52514e', pad=6)

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=COLORS[0], alpha=0.88, label='Classical'),
    mpatches.Patch(color=COLORS[1], alpha=0.88, label='Deep learning / Transformer'),
    plt.Line2D([0], [0], color='#c3c2b7', linestyle='--', linewidth=1, label='Chance (AUC=0.5)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=8.5, frameon=True, framealpha=0.9,
           edgecolor='#e1e0d9', bbox_to_anchor=(0.5, -0.05))

fig.suptitle('Base model comparison — PD detection from wrist IMU',
             fontsize=11, fontweight='500', y=1.02)
fig.text(0.5, 0.97,
         'Both wrists · all 11 tasks · 355 subjects (79 HC / 276 PD) · subject-level mean pooling',
         ha='center', fontsize=8.5, color='#52514e')

plt.savefig('model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison.pdf and model_comparison.png")
