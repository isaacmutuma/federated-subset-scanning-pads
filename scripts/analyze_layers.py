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
  A) Mean over patches → (6, 128) → flatten → 768-dim
  B) Last patch        → (6, 128) → flatten → 768-dim
  C) Mean over channels AND patches → 128-dim

Then for each combination:
  - UMAP to 2D → plot HC vs PD
  - Silhouette score → quantitative separation

Run:
    python scripts/analyze_layers.py
"""

import sys, os, os.path as osp, json, warnings, pickle
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

OUT_DIR = osp.join(C.RESULTS_DIR, 'layer_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load fold 1 data ──────────────────────────────────────────────────────────
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

mean, std = compute_normalization_stats(train_win)
test_win  = apply_normalization(test_win, mean, std)
print(f"Test windows: {test_win.shape}  HC={(test_lab==0).sum()}  PD={(test_lab==1).sum()}")

# ── Model ─────────────────────────────────────────────────────────────────────
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

model = PatchTSTClassifier().to(device)
ckpt  = osp.join(C.CKPT_DIR, 'patchtst_fold1_best.pt')
model.load_state_dict(torch.load(ckpt, map_location=device))
model.eval()
print(f"Loaded: {ckpt}")

# ── Hooks — fixed for tuple output ───────────────────────────────────────────
hook_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        # PatchTSTEncoderLayer returns tuple — first element is hidden state
        if isinstance(output, tuple):
            hook_outputs[name] = output[0].detach().cpu()
        else:
            hook_outputs[name] = output.detach().cpu()
    return hook

model.encoder.encoder.embedder.register_forward_hook(make_hook('embedding'))
model.encoder.encoder.layers[0].register_forward_hook(make_hook('layer_0'))
model.encoder.encoder.layers[1].register_forward_hook(make_hook('layer_1'))
model.encoder.encoder.layers[2].register_forward_hook(make_hook('layer_2'))

# ── Extract activations ───────────────────────────────────────────────────────
loader = DataLoader(PADSDataset(test_win, test_lab), batch_size=64, shuffle=False)
all_layer_acts = {n: [] for n in ['embedding', 'layer_0', 'layer_1', 'layer_2']}

print("Extracting activations from all layers...")
with torch.no_grad():
    for x, _ in loader:
        _ = model(x.to(device))
        for name in all_layer_acts:
            all_layer_acts[name].append(hook_outputs[name].numpy())

for name in all_layer_acts:
    all_layer_acts[name] = np.concatenate(all_layer_acts[name], axis=0)
    print(f"  {name}: {all_layer_acts[name].shape}")
    # expected shape: (n_windows, 6, 24, 128)

# ── Summarization strategies ──────────────────────────────────────────────────
def summarize(acts, strategy):
    # acts: (n_windows, 6, 24, 128)
    if strategy == 'A_mean_patches':
        return acts.mean(axis=2).reshape(len(acts), -1)   # (n, 768)
    elif strategy == 'B_last_patch':
        return acts[:, :, -1, :].reshape(len(acts), -1)   # (n, 768)
    elif strategy == 'C_mean_all':
        return acts.mean(axis=(1, 2))                       # (n, 128)

strategies = {
    'A_mean_patches': 'Mean over patches (768-dim)',
    'B_last_patch':   'Last patch (768-dim)',
    'C_mean_all':     'Mean channels+patches (128-dim)',
}
layers = ['embedding', 'layer_0', 'layer_1', 'layer_2']
layer_labels = {
    'embedding': 'Patch Embedding',
    'layer_0':   'Transformer Layer 0',
    'layer_1':   'Transformer Layer 1',
    'layer_2':   'Transformer Layer 2',
}
colors = {0: '#1baf7a', 1: '#e05c3a'}

# ── UMAP + Silhouette ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(strategies), len(layers), figsize=(16, 10))
fig.suptitle('PatchTST Layer Analysis — HC vs PD Separation (Fold 1 test set)',
             fontsize=13, fontweight='500')

results = {}

for s_idx, (strat_key, strat_label) in enumerate(strategies.items()):
    for l_idx, layer in enumerate(layers):
        ax    = axes[s_idx][l_idx]
        acts  = all_layer_acts[layer]
        feats = summarize(acts, strat_key)

        scaler       = StandardScaler()
        feats_scaled = scaler.fit_transform(feats)

        # Silhouette on subsample for speed
        n_sub = min(2000, len(feats_scaled))
        idx   = np.random.choice(len(feats_scaled), n_sub, replace=False)
        sil   = silhouette_score(feats_scaled[idx], test_lab[idx])

        # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=30, min_dist=0.1)
        emb     = reducer.fit_transform(feats_scaled)

        for lv in [0, 1]:
            mask = test_lab == lv
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=colors[lv], label='HC' if lv==0 else 'PD',
                       alpha=0.4, s=6, linewidths=0)

        ax.set_title(f"{layer_labels[layer]}\nSilhouette: {sil:.3f}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[['top','right','left','bottom']].set_color('#ddd')

        if l_idx == 0:
            ax.set_ylabel(strat_label, fontsize=8)
        if s_idx == 0 and l_idx == 0:
            ax.legend(markerscale=4, fontsize=7)

        results[f"{layer}__{strat_key}"] = float(sil)

plt.tight_layout()
fig.savefig(osp.join(OUT_DIR, 'layer_analysis_umap.png'), dpi=150, bbox_inches='tight')
fig.savefig(osp.join(OUT_DIR, 'layer_analysis_umap.pdf'), dpi=150, bbox_inches='tight')
print(f"\nPlots saved to {OUT_DIR}")

# ── Ranked table ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SILHOUETTE SCORES — Higher = better HC/PD separation")
print("="*60)
ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
for name, score in ranked:
    print(f"  {name:50s}  {score:.4f}")

best_name, best_score = ranked[0]
print(f"\nBest: {best_name}  (silhouette={best_score:.4f})")

with open(osp.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump({'silhouette_scores': results, 'ranked': ranked,
               'best': best_name}, f, indent=2)
print("DONE.")
