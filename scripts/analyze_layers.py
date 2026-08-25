"""
analyze_layers.py
-----------------
Layer analysis for PatchTST — adapted from Celia Cintas's representation_extraction.ipynb.
Evaluates at SUBJECT level (mean-pool all windows per subject).

Clustering metrics per layer (Table 1 from arxiv 2505.24539):
  - Silhouette (Si)        — higher is better
  - Calinski-Harabasz (CH) — higher is better
  - Euclidean distance (ED)— higher is better (centroid separation)
  - Davies-Bouldin (DB)    — lower is better

Steps:
  1. Extract hidden states at each transformer block (embedding, layers 0-2)
  2. Aggregate per subject — mean over all 110 windows
  3. PCA visualization — HC vs PD per layer at subject level
  4. Clustering metrics per layer (Si, CH, ED, DB)
  5. Save per-layer activations as .npy for DeepScan

Run:
    python scripts/analyze_layers.py
"""

import sys, os, os.path as osp, json, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score)

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C
sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.data.dataset import PADSDataset
from src.data.preprocessing import compute_normalization_stats, apply_normalization
from transformers import PatchTSTConfig, PatchTSTModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

OUT_DIR = osp.join(C.RESULTS_DIR, 'layer_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load fold 1 ───────────────────────────────────────────────────────────────
windows     = np.load(osp.join(C.OUTPUT_DIR, 'windows.npy'))
labels      = np.load(osp.join(C.OUTPUT_DIR, 'labels.npy'))
subject_ids = np.load(osp.join(C.OUTPUT_DIR, 'subject_ids.npy'))

with open(osp.join(C.OUTPUT_DIR, 'fold_splits.pkl'), 'rb') as f:
    folds = pickle.load(f)

fold       = folds[0]
train_mask = np.isin(subject_ids, fold['train_subjects'])
test_mask  = np.isin(subject_ids, fold['test_subjects'])

train_win          = windows[train_mask]
test_win, test_lab = windows[test_mask], labels[test_mask]
test_subj          = subject_ids[test_mask]

mean, std = compute_normalization_stats(train_win)
test_win  = apply_normalization(test_win, mean, std)
print(f"Test windows: {test_win.shape}  HC={(test_lab==0).sum()}  PD={(test_lab==1).sum()}")

# ── Model ─────────────────────────────────────────────────────────────────────
config = PatchTSTConfig(
    num_input_channels=6, context_length=C.WINDOW_SIZE,
    patch_length=C.PATCH_LEN, stride=C.STRIDE,
    d_model=C.D_MODEL, num_attention_heads=C.NUM_HEADS,
    num_hidden_layers=C.NUM_LAYERS, ffn_dim=C.FFN_DIM,
    dropout=C.DROPOUT, head_dropout=C.DROPOUT,
    pooling_type='mean', channel_attention=False,
    scaling='std', loss='mse', num_targets=1,
)

class PatchTSTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = PatchTSTModel(config)
        self.classifier = nn.Sequential(
            nn.LayerNorm(C.D_MODEL*6), nn.Dropout(C.DROPOUT),
            nn.Linear(C.D_MODEL*6, 64), nn.GELU(),
            nn.Dropout(C.DROPOUT), nn.Linear(64, 2))
    def forward(self, x):
        out    = self.encoder(past_values=x.permute(0, 2, 1))
        pooled = out.last_hidden_state.mean(dim=2)
        return self.classifier(pooled.reshape(pooled.size(0), -1))

model = PatchTSTClassifier().to(device)
model.load_state_dict(torch.load(
    osp.join(C.CKPT_DIR, 'patchtst_fold1_best.pt'), map_location=device))
model.eval()
print("Model loaded.")

# ── Hooks ─────────────────────────────────────────────────────────────────────
hook_outputs = {}
def make_hook(name):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        hook_outputs[name] = h.detach().cpu()
    return hook

LAYER_NAMES = ['embedding', 'layer_0', 'layer_1', 'layer_2']
LAYER_DISPLAY = {
    'embedding': 'Patch Embedding',
    'layer_0':   'Transformer Layer 0',
    'layer_1':   'Transformer Layer 1',
    'layer_2':   'Transformer Layer 2',
}

model.encoder.encoder.embedder.register_forward_hook(make_hook('embedding'))
model.encoder.encoder.layers[0].register_forward_hook(make_hook('layer_0'))
model.encoder.encoder.layers[1].register_forward_hook(make_hook('layer_1'))
model.encoder.encoder.layers[2].register_forward_hook(make_hook('layer_2'))

def extract_window_acts(win_array, lab_array, batch_size=32):
    loader = DataLoader(PADSDataset(win_array, lab_array),
                        batch_size=batch_size, shuffle=False)
    collected = {n: [] for n in LAYER_NAMES}
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            for name in LAYER_NAMES:
                collected[name].append(hook_outputs[name].numpy().mean(axis=(1, 2)))
    return {n: np.concatenate(v, axis=0) for n, v in collected.items()}

# ── Extract and aggregate to subject level ────────────────────────────────────
print("\nExtracting window activations...")
win_acts = extract_window_acts(test_win, test_lab)

unique_subj = np.unique(test_subj)
subj_labels = np.array([test_lab[test_subj==s][0] for s in unique_subj])
print(f"Subjects: {len(unique_subj)}  HC={(subj_labels==0).sum()}  PD={(subj_labels==1).sum()}")

subj_acts = {}
for layer in LAYER_NAMES:
    subj_acts[layer] = np.array([
        win_acts[layer][test_subj==s].mean(axis=0) for s in unique_subj
    ])
    print(f"  {layer}: {subj_acts[layer].shape}")

# ── PCA visualization ─────────────────────────────────────────────────────────
colors = {0: '#1baf7a', 1: '#e05c3a'}
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('PCA — HC vs PD across PatchTST layers (subject level, 71 subjects)',
             fontsize=11, fontweight='500')

for ax, layer in zip(axes, LAYER_NAMES):
    feats  = subj_acts[layer]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)
    pca    = PCA(n_components=2, random_state=42)
    emb    = pca.fit_transform(scaled)
    var    = pca.explained_variance_ratio_

    for lv in [0, 1]:
        mask = subj_labels == lv
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=colors[lv], label='HC' if lv==0 else 'PD',
                   alpha=0.85, s=60, linewidths=0.5, edgecolors='white')

    ax.set_title(f"{LAYER_DISPLAY[layer]}\nPC1={var[0]:.1%}  PC2={var[1]:.1%}",
                 fontsize=9)
    ax.set_xlabel('PC1', fontsize=8); ax.set_ylabel('PC2', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[['top','right']].set_visible(False)
    if layer == 'embedding':
        ax.legend(markerscale=2, fontsize=8)

plt.tight_layout()
fig.savefig(osp.join(OUT_DIR, 'pca_subject_level.png'), dpi=150, bbox_inches='tight')
fig.savefig(osp.join(OUT_DIR, 'pca_subject_level.pdf'), dpi=150, bbox_inches='tight')
print("\nPCA plots saved.")

# ── Clustering metrics per layer ──────────────────────────────────────────────
# Following Celia's suggestion: Si, CH, ED, DB (Table 1, arxiv 2505.24539)
print("\n" + "="*72)
print(f"{'Layer':<30} {'Si':>8} {'CH':>10} {'ED':>10} {'DB':>8}")
print(f"{'':30} {'(↑)':>8} {'(↑)':>10} {'(↑)':>10} {'(↓)':>8}")
print("="*72)

metrics_results = {}
for layer in LAYER_NAMES:
    feats  = subj_acts[layer]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)

    si = float(silhouette_score(scaled, subj_labels))
    ch = float(calinski_harabasz_score(scaled, subj_labels))
    db = float(davies_bouldin_score(scaled, subj_labels))

    hc_mean = scaled[subj_labels==0].mean(axis=0)
    pd_mean = scaled[subj_labels==1].mean(axis=0)
    ed      = float(np.linalg.norm(hc_mean - pd_mean))

    metrics_results[layer] = {'Si': si, 'CH': ch, 'ED': ed, 'DB': db}
    print(f"{LAYER_DISPLAY[layer]:<30} {si:>8.4f} {ch:>10.2f} {ed:>10.4f} {db:>8.4f}")

print("="*72)
print("Si / CH / ED: higher is better  |  DB: lower is better")

# Pick best layer by silhouette (primary unsupervised metric)
best_si_layer = max(metrics_results, key=lambda k: metrics_results[k]['Si'])
print(f"\nBest layer (silhouette): {best_si_layer}  "
      f"Si={metrics_results[best_si_layer]['Si']:.4f}")

# ── Save per-layer subject activations for DeepScan ──────────────────────────
print("\nSaving per-layer subject activations...")
for layer in LAYER_NAMES:
    hc_feats = subj_acts[layer][subj_labels == 0]
    pd_feats = subj_acts[layer][subj_labels == 1]
    np.save(osp.join(OUT_DIR, f'subj_hc_{layer}.npy'), hc_feats)
    np.save(osp.join(OUT_DIR, f'subj_pd_{layer}.npy'), pd_feats)
    print(f"  {layer}: HC={hc_feats.shape}  PD={pd_feats.shape}")

with open(osp.join(OUT_DIR, 'layer_analysis_results.json'), 'w') as f:
    json.dump({
        'clustering_metrics_subject_level': metrics_results,
        'best_layer_silhouette': best_si_layer,
        'token_strategy': 'mean over channels and patches per window, then mean over windows per subject (128-dim)',
        'n_subjects': int(len(unique_subj)),
        'reference': 'Metrics: Si, CH, ED, DB — Table 1, arxiv 2505.24539'
    }, f, indent=2)

print(f"\nAll outputs saved to: {OUT_DIR}")
print("DONE.")
