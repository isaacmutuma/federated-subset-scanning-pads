"""
train_inceptiontime.py
----------------------
InceptionTime classifier for PADS IMU windows.
Subject-level evaluation: mean pool per-window probabilities per subject.
GPU recommended.

Run:
    python scripts/train_inceptiontime.py
"""

import sys, os, os.path as osp, json, warnings, zipfile
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.utils import resample
import pandas as pd

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

if not osp.exists(C.REPO_DIR):
    os.system(f'git clone https://github.com/isaacmutuma/federated-subset-scanning-pads.git {C.REPO_DIR}')
else:
    os.system(f'git -C {C.REPO_DIR} pull origin main')

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.data.dataset import PADSDataset
from src.data.preprocessing import (bandpass_filter, segment_windows,
                                     compute_normalization_stats, apply_normalization)
from src.data.folds import generate_fold_splits
from src.training.metrics import (compute_metrics, aggregate_fold_metrics,
                                   print_fold_results, print_summary)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

os.makedirs(C.OUTPUT_DIR,  exist_ok=True)
os.makedirs(C.CKPT_DIR,    exist_ok=True)
os.makedirs(C.RESULTS_DIR, exist_ok=True)

if C.PADS_ZIP and not osp.exists(C.PADS_ROOT):
    with zipfile.ZipFile(C.PADS_ZIP, 'r') as z:
        z.extractall(C.PADS_ROOT)

def find_pads_root(base):
    for root, dirs, files in os.walk(base):
        if 'movement' in dirs and 'patients' in dirs:
            return root
    return None

BASE_PATH = find_pads_root(C.PADS_ROOT)
assert BASE_PATH, f"PADS dataset not found under {C.PADS_ROOT}"
OBSERVATION_DIR = osp.join(BASE_PATH, 'movement')
PATIENTS_DIR    = osp.join(BASE_PATH, 'patients')
print(f"PADS root: {BASE_PATH}")

def get_patient_label(patient_id):
    path = osp.join(PATIENTS_DIR, f'patient_{int(patient_id):03d}.json')
    with open(path) as f:
        return json.load(f).get('condition')

def build_manifest(n_patients=469):
    rows = []
    for pat_num in range(1, n_patients + 1):
        patient_id = f'{pat_num:03d}'
        label      = get_patient_label(pat_num)
        obs_path   = osp.join(OBSERVATION_DIR, f'observation_{patient_id}.json')
        with open(obs_path) as f:
            obs = json.load(f)
        for session in obs['session']:
            for record in session['records']:
                rows.append({'patient_id': patient_id, 'label': label,
                             'task':  session.get('record_name'),
                             'wrist': record.get('device_location'),
                             'filepath': record.get('file_name')})
    return pd.DataFrame(rows)

def load_timeseries(filepath):
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 1:7].values.T.astype(np.float32)

print("Building manifest...")
manifest         = build_manifest()
all_labels_found = manifest['label'].unique().tolist()
HC_STR = next(l for l in all_labels_found if 'healthy'   in l.lower() or l == 'HC')
PD_STR = next(l for l in all_labels_found if 'parkinson' in l.lower() or l == 'PD')
LABEL_MAP = {HC_STR: 0, PD_STR: 1}
print(f'HC="{HC_STR}"  PD="{PD_STR}"')

filtered = manifest[
    manifest['label'].isin([HC_STR, PD_STR]) &
    (manifest['wrist'] == C.TRAIN_WRIST)
].copy()
print(f'Subjects: {filtered["patient_id"].nunique()}  Rows: {len(filtered)}')

all_windows, all_labels_out, all_subject_ids = [], [], []
skipped = 0
print("Windowing recordings...")
for _, row in filtered.iterrows():
    fp = osp.join(BASE_PATH, 'movement', row['filepath'])
    if not osp.exists(fp): skipped += 1; continue
    try:    signal = load_timeseries(fp)
    except: skipped += 1; continue
    if signal.shape[1] < C.WINDOW_SIZE * C.N_WINDOWS: skipped += 1; continue
    sig_filt = bandpass_filter(signal, lowcut=C.LOWCUT, highcut=C.HIGHCUT, fs=C.FS)
    wins     = segment_windows(sig_filt, window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)[:C.N_WINDOWS]
    if len(wins) < C.N_WINDOWS: skipped += 1; continue
    all_windows.append(wins)
    all_labels_out.extend([LABEL_MAP[row['label']]] * len(wins))
    all_subject_ids.extend([int(row['patient_id'])] * len(wins))

windows     = np.concatenate(all_windows, axis=0)
labels      = np.array(all_labels_out,  dtype=np.int64)
subject_ids = np.array(all_subject_ids, dtype=np.int64)
print(f"Windows: {windows.shape}  HC={(labels==0).sum()}  PD={(labels==1).sum()}")

unique_subjects = np.unique(subject_ids)
subject_labels  = np.array([labels[subject_ids == s][0] for s in unique_subjects])
folds = generate_fold_splits(unique_subjects, subject_labels,
                              n_splits=C.N_SPLITS, random_state=C.RANDOM_STATE,
                              val_fraction=0.2,
                              save_path=osp.join(C.OUTPUT_DIR, 'fold_splits.pkl'))

# ── InceptionTime Model ───────────────────────────────────────────────────────
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, nb_filters=32, kernel_sizes=(10, 20, 40)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(nb_filters, nb_filters, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )
        self.bn  = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        branches   = [conv(bottleneck) for conv in self.convs]
        branches  += [self.maxpool_conv(x)]
        min_len    = min(b.shape[-1] for b in branches)
        branches   = [b[..., :min_len] for b in branches]
        return self.act(self.bn(torch.cat(branches, dim=1)))


class InceptionTime(nn.Module):
    def __init__(self, in_channels=6, n_classes=2, nb_filters=32,
                 depth=3, kernel_sizes=(10, 20, 40), dropout=0.3):
        super().__init__()
        out_ch = nb_filters * (len(kernel_sizes) + 1)
        blocks, ch = [], in_channels
        for _ in range(depth):
            blocks.append(InceptionBlock(ch, nb_filters, kernel_sizes))
            ch = out_ch
        self.inception = nn.Sequential(*blocks)
        self.shortcut  = nn.Sequential(
            nn.Conv1d(in_channels, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act        = nn.ReLU()
        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_ch, n_classes)

    def forward(self, x):
        residual = self.shortcut(x)
        out      = self.inception(x)
        min_len  = min(out.shape[-1], residual.shape[-1])
        out      = self.act(out[..., :min_len] + residual[..., :min_len])
        out      = self.gap(out).squeeze(-1)
        return self.classifier(self.dropout(out))

# ── Training ──────────────────────────────────────────────────────────────────
RESULTS_FILE  = osp.join(C.RESULTS_DIR, 'inceptiontime_results.json')
IT_PATIENCE   = 20
IT_MAX_EPOCHS = 100
IT_LR         = 1e-3
IT_BATCH      = 64

fold_metrics = []

for fold_idx, fold in enumerate(folds):
    ckpt_path = osp.join(C.CKPT_DIR, f'inceptiontime_fold{fold_idx+1}_best.pt')

    print(f"\n{'='*55}\nFOLD {fold_idx+1}/{C.N_SPLITS}\n{'='*55}")

    train_mask = np.isin(subject_ids, fold['train_subjects'])
    val_mask   = np.isin(subject_ids, fold['val_subjects'])
    test_mask  = np.isin(subject_ids, fold['test_subjects'])

    train_win, train_lab = windows[train_mask], labels[train_mask]
    val_win,   val_lab   = windows[val_mask],   labels[val_mask]
    test_win,  test_lab  = windows[test_mask],  labels[test_mask]
    test_subj            = subject_ids[test_mask]

    mean, std = compute_normalization_stats(train_win)
    train_win = apply_normalization(train_win, mean, std)
    val_win   = apply_normalization(val_win,   mean, std)
    test_win  = apply_normalization(test_win,  mean, std)

    hc_idx = np.where(train_lab == 0)[0]
    pd_idx = np.where(train_lab == 1)[0]
    hc_os  = resample(hc_idx, replace=True, n_samples=len(pd_idx), random_state=C.RANDOM_STATE)
    idx    = np.concatenate([hc_os, pd_idx])
    np.random.shuffle(idx)
    train_win, train_lab = train_win[idx], train_lab[idx]

    train_loader = DataLoader(PADSDataset(train_win, train_lab), batch_size=IT_BATCH, shuffle=True, drop_last=True)
    val_loader   = DataLoader(PADSDataset(val_win,   val_lab),   batch_size=IT_BATCH, shuffle=False)
    test_loader  = DataLoader(PADSDataset(test_win,  test_lab),  batch_size=IT_BATCH, shuffle=False)

    if osp.exists(ckpt_path):
        print(f"Checkpoint found — skipping training for fold {fold_idx+1}")
        model = InceptionTime().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    else:
        print(f"Train: {train_win.shape[0]:,}  HC={(train_lab==0).sum():,}  PD={(train_lab==1).sum():,}")
        model     = InceptionTime().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=IT_LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        criterion = nn.CrossEntropyLoss()

        best_val_loss    = float('inf')
        patience_counter = 0

        for epoch in range(1, IT_MAX_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                train_loss += loss.item() * len(x)
                optimizer.step()
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    val_loss += criterion(model(x), y).item() * len(x)
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            print(f"Epoch {epoch:3d}/{IT_MAX_EPOCHS}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= IT_PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

    # ── Subject-level evaluation ──────────────────────────────────────────────
    all_probs, all_true, all_subj_ids = [], [], []
    with torch.no_grad():
        offset = 0
        for x, y in test_loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_true.append(y.numpy())
            all_subj_ids.append(test_subj[offset:offset + len(x)])
            offset += len(x)

    all_probs    = np.concatenate(all_probs)
    all_true     = np.concatenate(all_true)
    all_subj_ids = np.concatenate(all_subj_ids)

    unique_test_subj = np.unique(all_subj_ids)
    subj_probs = np.array([all_probs[all_subj_ids == s].mean() for s in unique_test_subj])
    subj_true  = np.array([all_true [all_subj_ids == s][0]     for s in unique_test_subj])

    print(f"Subject-level: {len(unique_test_subj)} subjects  "
          f"HC={(subj_true==0).sum()}  PD={(subj_true==1).sum()}")

    metrics = compute_metrics(subj_true, subj_probs)
    fold_metrics.append(metrics)
    print_fold_results(fold_idx, metrics)

    with open(RESULTS_FILE, 'w') as f:
        json.dump({'fold_metrics': fold_metrics, 'folds_complete': fold_idx + 1}, f, indent=2)

agg = aggregate_fold_metrics(fold_metrics)
print_summary(agg)
with open(RESULTS_FILE, 'w') as f:
    json.dump({'fold_metrics': fold_metrics, 'aggregate': agg,
               'evaluation': 'subject-level mean pooling'}, f, indent=2)
print(f"\nResults: {RESULTS_FILE}")
print("DONE.")
