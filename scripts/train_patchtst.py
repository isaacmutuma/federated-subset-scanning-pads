"""
train_patchtst.py
-----------------
Self-contained PatchTST training script for Kaggle.
Run with: !python scripts/train_patchtst.py

Outputs saved to /kaggle/working/results/patchtst_results.json
Model checkpoints saved to /kaggle/working/checkpoints/
"""

import sys, os, os.path as osp, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils import resample

# ── Repo setup ────────────────────────────────────────────────────────────────
REPO = '/kaggle/working/federated-subset-scanning-pads'
if not osp.exists(REPO):
    os.system(f'git clone https://github.com/isaacmutuma/federated-subset-scanning-pads.git {REPO}')
else:
    os.system(f'git -C {REPO} pull origin main')

sys.path.insert(0, REPO)
os.chdir(REPO)

from src.data.dataset import PADSDataset
from src.data.preprocessing import (bandpass_filter, segment_windows,
                                     compute_normalization_stats, apply_normalization)
from src.data.folds import generate_fold_splits, load_fold_splits
from src.training.metrics import (compute_metrics, aggregate_fold_metrics,
                                   print_fold_results, print_summary)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Data regeneration ─────────────────────────────────────────────────────────
import pandas as pd

OUTPUT_DIR = '/kaggle/working/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/results', exist_ok=True)

def find_pads_root(base):
    for root, dirs, files in os.walk(base):
        if 'movement' in dirs and 'patients' in dirs:
            return root
    return None

BASE_PATH = find_pads_root('/kaggle/input')
assert BASE_PATH, "PADS dataset not found"

OBSERVATION_DIR = osp.join(BASE_PATH, 'movement')
PATIENTS_DIR    = osp.join(BASE_PATH, 'patients')

def get_patient_label(patient_id):
    path = osp.join(PATIENTS_DIR, f'patient_{int(patient_id):03d}.json')
    with open(path) as f:
        return json.load(f).get('condition')

def build_manifest(n_patients=469):
    rows = []
    for pat_num in range(1, n_patients + 1):
        patient_id = f'{pat_num:03d}'
        label = get_patient_label(pat_num)
        obs_path = osp.join(OBSERVATION_DIR, f'observation_{patient_id}.json')
        with open(obs_path) as f:
            obs = json.load(f)
        for session in obs['session']:
            for record in session['records']:
                rows.append({'patient_id': patient_id, 'label': label,
                             'task': session.get('record_name'),
                             'wrist': record.get('device_location'),
                             'filepath': record.get('file_name')})
    return pd.DataFrame(rows)

print("Building manifest...")
manifest = build_manifest()
all_labels_found = manifest['label'].unique().tolist()
HC_STR = next(l for l in all_labels_found if 'healthy' in l.lower() or l == 'HC')
PD_STR = next(l for l in all_labels_found if 'parkinson' in l.lower() or l == 'PD')
LABEL_MAP = {HC_STR: 0, PD_STR: 1}
print(f'HC="{HC_STR}"  PD="{PD_STR}"')

filtered = manifest[manifest['label'].isin([HC_STR, PD_STR])].copy()
print(f'Filtered: {len(filtered)} rows across {filtered["patient_id"].nunique()} subjects')

def load_timeseries(filepath):
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 1:7].values.T.astype(np.float32)

FS, WINDOW_SIZE, N_WINDOWS = 100.0, 200, 10
all_windows, all_labels_out, all_subject_ids = [], [], []
all_tasks, all_wrists = [], []
skipped = 0

print("Processing recordings...")
for _, row in filtered.iterrows():
    filepath = osp.join(BASE_PATH, 'movement', row['filepath'])
    if not osp.exists(filepath): skipped += 1; continue
    try: signal = load_timeseries(filepath)
    except: skipped += 1; continue
    if signal.shape[1] < WINDOW_SIZE * N_WINDOWS: skipped += 1; continue
    sig_filt = bandpass_filter(signal, lowcut=1.0, highcut=20.0, fs=FS)
    wins = segment_windows(sig_filt, window_size=WINDOW_SIZE, step=WINDOW_SIZE)[:N_WINDOWS]
    if len(wins) < N_WINDOWS: skipped += 1; continue
    n = len(wins)
    all_windows.append(wins)
    all_labels_out.extend([LABEL_MAP[row['label']]] * n)
    all_subject_ids.extend([int(row['patient_id'])] * n)
    all_tasks.extend([row['task']] * n)
    all_wrists.extend([row['wrist']] * n)

windows     = np.concatenate(all_windows, axis=0)
labels      = np.array(all_labels_out,  dtype=np.int64)
subject_ids = np.array(all_subject_ids, dtype=np.int64)
tasks       = np.array(all_tasks)
wrists      = np.array(all_wrists)

relaxed_rw_mask = (tasks == 'Relaxed') & (wrists == 'RightWrist')
print(f'Windows: {windows.shape}  HC={(labels==0).sum()}  PD={(labels==1).sum()}')
print(f'Relaxed+RW: {relaxed_rw_mask.sum()} (HC={(( labels==0)&relaxed_rw_mask).sum()} PD={(( labels==1)&relaxed_rw_mask).sum()})')
print(f'Skipped: {skipped}')

unique_subjects = np.unique(subject_ids)
subject_labels  = np.array([labels[subject_ids == s][0] for s in unique_subjects])
folds = generate_fold_splits(unique_subjects, subject_labels,
                              n_splits=5, random_state=42, val_fraction=0.2,
                              save_path=osp.join(OUTPUT_DIR, 'fold_splits.pkl'))

np.save(osp.join(OUTPUT_DIR, 'windows.npy'),         windows)
np.save(osp.join(OUTPUT_DIR, 'labels.npy'),          labels)
np.save(osp.join(OUTPUT_DIR, 'subject_ids.npy'),     subject_ids)
np.save(osp.join(OUTPUT_DIR, 'relaxed_rw_mask.npy'), relaxed_rw_mask.astype(np.bool_))
print("Data saved.")

# ── Model definition ──────────────────────────────────────────────────────────
from transformers import PatchTSTConfig, PatchTSTModel

class PatchTSTClassifier(nn.Module):
    def __init__(self, n_channels=6, seq_len=200, patch_len=16,
                 stride=8, n_classes=2, dropout=0.3):
        super().__init__()
        config = PatchTSTConfig(
            num_input_channels=n_channels,
            context_length=seq_len,
            patch_length=patch_len,
            stride=stride,
            d_model=128,
            num_attention_heads=8,
            num_hidden_layers=3,
            ffn_dim=256,
            dropout=dropout,
            head_dropout=dropout,
            pooling_type='mean',
            channel_attention=False,
            scaling='std',
            loss='mse',
            num_targets=1,
        )
        self.encoder = PatchTSTModel(config)
        encoder_dim  = config.d_model * n_channels
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x_t    = x.permute(0, 2, 1)
        out    = self.encoder(past_values=x_t)
        pooled = out.last_hidden_state.mean(dim=2)
        flat   = pooled.reshape(pooled.size(0), -1)
        return self.classifier(flat)

    def get_activations(self, x):
        x_t    = x.permute(0, 2, 1)
        out    = self.encoder(past_values=x_t)
        pooled = out.last_hidden_state.mean(dim=2)
        return pooled.reshape(pooled.size(0), -1).detach()


# ── Training ──────────────────────────────────────────────────────────────────
fold_metrics = []

for fold_idx, fold in enumerate(folds):
    print(f"\n{'='*50}\nFOLD {fold_idx+1}/5\n{'='*50}")

    train_mask        = np.isin(subject_ids, fold['train_subjects'])
    val_mask          = np.isin(subject_ids, fold['val_subjects'])
    test_mask_relaxed = np.isin(subject_ids, fold['test_subjects']) & relaxed_rw_mask

    train_win, train_lab = windows[train_mask], labels[train_mask]
    val_win,   val_lab   = windows[val_mask],   labels[val_mask]
    test_win,  test_lab  = windows[test_mask_relaxed], labels[test_mask_relaxed]

    mean, std = compute_normalization_stats(train_win)
    train_win = apply_normalization(train_win, mean, std)
    val_win   = apply_normalization(val_win,   mean, std)
    test_win  = apply_normalization(test_win,  mean, std)

    # Oversample HC to balance
    hc_idx = np.where(train_lab == 0)[0]
    pd_idx = np.where(train_lab == 1)[0]
    hc_os  = resample(hc_idx, replace=True, n_samples=len(pd_idx), random_state=42)
    idx    = np.concatenate([hc_os, pd_idx])
    np.random.shuffle(idx)
    train_win, train_lab = train_win[idx], train_lab[idx]

    print(f"Train: {train_win.shape[0]} HC={(train_lab==0).sum()} PD={(train_lab==1).sum()}")
    print(f"Val:   {val_win.shape[0]}   Test: {test_win.shape[0]} HC={(test_lab==0).sum()} PD={(test_lab==1).sum()}")

    train_loader = DataLoader(PADSDataset(train_win, train_lab),
                              batch_size=64, shuffle=True, drop_last=True)
    val_loader   = DataLoader(PADSDataset(val_win,   val_lab),
                              batch_size=64, shuffle=False)
    test_loader  = DataLoader(PADSDataset(test_win,  test_lab),
                              batch_size=64, shuffle=False)

    model     = PatchTSTClassifier(n_channels=6, seq_len=200, patch_len=16,
                                    stride=8, n_classes=2, dropout=0.4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4,
        steps_per_epoch=len(train_loader), epochs=80,
        pct_start=0.1, anneal_strategy='cos'
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss    = float('inf')
    patience_counter = 0
    PATIENCE         = 20

    for epoch in range(1, 81):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(x)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/80  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(),
                       f'/kaggle/working/checkpoints/patchtst_fold{fold_idx+1}_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(
        f'/kaggle/working/checkpoints/patchtst_fold{fold_idx+1}_best.pt',
        map_location=device))
    model.eval()

    all_probs, all_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_true.append(y.numpy())

    metrics = compute_metrics(np.concatenate(all_true), np.concatenate(all_probs))
    fold_metrics.append(metrics)
    print_fold_results(fold_idx, metrics)

# ── Save results ──────────────────────────────────────────────────────────────
agg = aggregate_fold_metrics(fold_metrics)
print_summary(agg)

results = {'fold_metrics': fold_metrics, 'aggregate': agg}
with open('/kaggle/working/results/patchtst_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to /kaggle/working/results/patchtst_results.json")
print("Checkpoints saved to /kaggle/working/checkpoints/")
print("DONE.")
