"""
analyze_layers.py
-----------------
Layer analysis for PatchTST — following Celia Cintas's guidance.

For each of 4 extraction points:
  - After patch embedding
  - After transformer layer 0
  - After transformer layer 1
  - After transformer layer 2 (final, before pooling)

For each layer, tries 3 summarization strategies:
  A) Mean over patches → shape (6, 128) → flatten → 768-dim
  B) Last patch       → shape (6, 128) → flatten → 768-dim
  C) Mean over channels AND patches → shape (128,) → 128-dim

Then for each (layer × strategy):
  - UMAP to 2D → plot HC vs PD
  - Silhouette score → quantitative separation measure

Run:
    python scripts/analyze_layers.py
"""

import sys, os, os.path as osp, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.data.dataset import PADSDataset
from src.data.preprocessing import (compute_normalization_stats, apply_normalization)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import umap
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from transformers import PatchTSTConfig, PatchTSTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load fold 1 data ──────────────────────────────────────────────────────────
ACT_DIR  = osp.join(C.RESULTS_DIR, 'activations')
CKPT_DIR = C.CKPT_DIR

# Load saved arrays from activation extraction
windows     = np.load(osp.join(C.OUTPUT_DIR, 'windows.npy'))
labels      = np.load(osp.join(C.OUTPUT_DIR, 'labels.npy'))
subject_ids = np.load(osp.join(C.OUTPUT_DIR, 'subject_ids.npy'))

import pickle
with open(osp.join(C.OUTPUT_DIR, 'fold_splits.pkl'), 'rb') as f:
    folds = pickle.load(f)

fold = folds[0]   # fold 1 only for analysis
train_mask = np.isin(subject_ids, fold['train_subjects'])
test_mask  = np.isin(subject_ids, fold['test_subjects'])

train_win           = windows[train_mask]
test_win, test_lab  = windows[test_mask], labels[test_mask]

mean, std = compute_normalization_stats(train_win)
test_win  = apply_normalization(test_win, mean, std)

print(f"Test windows: {test_win.shape}  HC={(test_lab==0).sum()}  PD={(test_lab==1).sum()}")

# ── Build PatchTST model with hooks ──────────────────────────────────────────
config = PatchTSTConfig(
    num_input_channels=6,
    context_length=C.WINDOW_SIZE,
    patch_length=C.PATCH_LEN,
    stride=C.STRIDE,
    d_model=C.D_MODEL,
    num_attention_heads=C.NUM_HEADS,
    num_hidden_layers=C.NUM_LAYERS,
    ffn_dim=C.FFN_DIM,
    dropout=C.DROPOUT,
    head_dropout=C.DROPOUT,
    pooling_type='mean',
    channel_attention=False,
    scaling='std',
    loss='mse',
    num_targets=1,
)

class PatchTSTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder    = PatchTSTModel(config)
        encoder_dim     = C.D_MODEL * 6
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(C.DROPOUT),
            nn.Linear(encoder_dim, 64),
            nn.GELU(),
            nn.Dropout(C.DROPOUT),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out    = self.encoder(past_values=x.permute(0, 2, 1))
        pooled = out.last_hidden_state.mean(dim=2)
        return self.classifier(pooled.reshape(pooled.size(0), -1))

# Load checkpoint
model = PatchTSTClassifier().to(device)
ckpt  = osp.join(CKPT_DIR, 'patchtst_fold1_best.pt')
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print(f"Loaded: {ckpt}")

# ── Hook storage ──────────────────────────────────────────────────────────────
# We capture: embedding output + after each of 3 transformer layers
# Each hook receives tensor shape: (batch, n_channels, n_patches, d_model)
# n_patches = (200 - 16) / 8 + 1 = 24
# So shape is (batch, 6, 24, 128)

hook_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        # output shape: (batch, n_channels, n_patches, d_model)
        hook_outputs[name] = output.detach().cpu()
    return hook

# Register hooks
model.encoder.encoder.embedder.register_forward_hook(make_hook('embedding'))
model.encoder.encoder.layers[0].register_forward_hook(make_hook('layer_0'))
model.encoder.encoder.layers[1].register_forward_hook(make_hook('layer_1'))
model.encoder.encoder.layers[2].register_forward_hook(make_hook('layer_2'))

# ── Extract activations for all test windows ──────────────────────────────────
loader = DataLoader(PADSDataset(test_win, test_lab), batch_size=64, shuffle=False)

all_layer_acts = {name: [] for name in ['embedding', 'layer_0', 'layer_1', 'layer_2']}

with torch.no_grad():
    for x, _ in loader:
        _ = model(x.to(device))
        for name in all_layer_acts:
            all_layer_acts[name].append(hook_outputs[name].numpy())

for name in all_layer_acts:
    all_layer_acts[name] = np.concatenate(all_layer_acts[name], axis=0)
    print(f"{name}: {all_layer_acts[name].shape}")
    # shape: (n_windows, 6, 24, 128)

# ── Summarization strategies ──────────────────────────────────────────────────
def summarize(acts, strategy):
    """
    acts: (n_windows, 6, 24, 128)
    strategy A: mean over patches → (n_windows, 6, 128) → flatten → (n_windows, 768)
    strategy B: last patch        → (n_windows, 6, 128) → flatten → (n_windows, 768)
    strategy C: mean over channels AND patches → (n_windows, 128)
    """
    if strategy == 'A_mean_patches':
        return acts.mean(axis=2).reshape(len(acts), -1)   # (n, 6*128=768)
    elif strategy == 'B_last_patch':
        return acts[:, :, -1, :].reshape(len(acts), -1)   # (n, 768)
    elif strategy == 'C_mean_all':
        return acts.mean(axis=(1, 2))                       # (n, 128)

strategies = {
    'A_mean_patches': 'Mean over patches (768-dim)',
    'B_last_patch':   'Last patch (768-dim)',
    'C_mean_all':     'Mean over channels+patches (128-dim)',
}

layers = ['embedding', 'layer_0', 'layer_1', 'layer_2']
layer_labels = {
    'embedding': 'Patch Embedding',
    'layer_0':   'Transformer Layer 0',
    'layer_1':   'Transformer Layer 1',
    'layer_2':   'Transformer Layer 2 (final)',
}

# ── UMAP + Silhouette scores ──────────────────────────────────────────────────
OUT_DIR = osp.join(C.RESULTS_DIR, 'layer_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

results = {}

fig, axes = plt.subplots(len(strategies), len(layers),
                          figsize=(16, 10))
fig.suptitle('PatchTST Layer Analysis — HC vs PD Separation\n(Fold 1 test set)',
             fontsize=13, fontweight='500')

colors = {0: '#1baf7a', 1: '#e05c3a'}
labels_str = {0: 'HC', 1: 'PD'}

for s_idx, (strat_key, strat_label) in enumerate(strategies.items()):
    for l_idx, layer in enumerate(layers):
        ax = axes[s_idx][l_idx]

        acts   = all_layer_acts[layer]
        feats  = summarize(acts, strat_key)

        # Standardize
        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(feats)

        # Silhouette score (use subsample for speed if large)
        n_sub = min(2000, len(feats_scaled))
        idx   = np.random.choice(len(feats_scaled), n_sub, replace=False)
        sil   = silhouette_score(feats_scaled[idx], test_lab[idx])

        # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        emb     = reducer.fit_transform(feats_scaled)

        for label_val in [0, 1]:
            mask = test_lab == label_val
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=colors[label_val], label=labels_str[label_val],
                       alpha=0.5, s=8, linewidths=0)

        ax.set_title(f"{layer_labels[layer]}\n{strat_label}\nSilhouette: {sil:.3f}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['top','right','left','bottom']].set_color('#ddd')

        results[f"{layer}_{strat_key}"] = float(sil)

        if l_idx == 0:
            ax.set_ylabel(strat_label, fontsize=8)
        if s_idx == 0 and l_idx == 0:
            ax.legend(markerscale=3, fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig(osp.join(OUT_DIR, 'layer_analysis_umap.png'), dpi=150, bbox_inches='tight')
plt.savefig(osp.join(OUT_DIR, 'layer_analysis_umap.pdf'), dpi=150, bbox_inches='tight')
print(f"\nSaved UMAP plots to {OUT_DIR}")

# ── Print ranking ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SILHOUETTE SCORES — Higher is better (max=1.0)")
print("="*55)
ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in ranked:
    layer, strat = name.rsplit('_', 2)[0], '_'.join(name.split('_')[1:])
    print(f"  {name:45s}  {score:.4f}")

best_name, best_score = ranked[0]
print(f"\nBest layer+strategy: {best_name}  (silhouette={best_score:.4f})")

with open(osp.join(OUT_DIR, 'layer_analysis_results.json'), 'w') as f:
    json.dump({'silhouette_scores': results,
               'ranked': ranked,
               'best': best_name}, f, indent=2)

print(f"Results: {OUT_DIR}/layer_analysis_results.json")
print("DONE.")
