"""
train_tspulse.py
----------------
TSPulse (IBM Granite) full fine-tuning for PADS IMU windows.
Full backbone unfrozen — trains end-to-end with lower LR.
Subject-level evaluation: mean pool per-window probabilities per subject.

Run:
    python scripts/train_tspulse.py
"""

import sys, os, os.path as osp, json, warnings, zipfile
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.utils import resample

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

if not osp.exists(C.REPO_DIR):
    os.system(f'git clone https://github.com/isaacmutuma/federated-subset-scanning-pads.git {C.REPO_DIR}')
else:
    os.system(f'git -C {C.REPO_DIR} pull origin main')

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.data.preprocessing import (bandpass_filter, segment_windows,
                                     compute_normalization_stats, apply_normalization)
from src.data.folds import generate_fold_splits
from src.training.metrics import (compute_metrics, aggregate_fold_metrics,
                                   print_fold_results, print_summary)

try:
    from tsfm_public.models.tspulse import TSPulseForClassification
except ImportError:
    print("Installing granite-tsfm...")
    os.system('pip install -q "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.3.1"')
    from tsfm_public.models.tspulse import TSPulseForClassification

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
        pid  = f'{pat_num:03d}'
        lbl  = get_patient_label(pat_num)
        obs_path = osp.join(OBSERVATION_DIR, f'observation_{pid}.json')
        with open(obs_path) as f:
            obs = json.load(f)
        for session in obs['session']:
            for record in session['records']:
                rows.append({'patient_id': pid, 'label': lbl,
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

# RightWrist only — consistent with all other models
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

# ── Dataset wrapper ───────────────────────────────────────────────────────────
class TSPulseDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels  = torch.tensor(labels,  dtype=torch.long)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].permute(1, 0)   # (200, 6)
        x = F.pad(x, (0, 0, 0, 312))          # (512, 6)
        return {'past_values': x, 'target_values': self.labels[idx]}


# ── Model config ──────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'head_reduce_d_model': 1,
    'decoder_mode': 'mix_channel',
    'head_gated_attention_activation': 'softmax',
    'mask_ratio': 0.3,
    'channel_virtual_expand_scale': 1,
    'loss': 'cross_entropy',
    'disable_mask_in_classification_eval': True,
    'ignore_mismatched_sizes': True,
    'num_input_channels': 6,
    'num_targets': 2,
}

TS_PATIENCE   = 20
TS_MAX_EPOCHS = 80
TS_LR         = 5e-5
TS_BATCH      = 32

# ── Training ──────────────────────────────────────────────────────────────────
RESULTS_FILE = osp.join(C.RESULTS_DIR, 'tspulse_results.json')
fold_metrics = []

for fold_idx, fold in enumerate(folds):
    ckpt_path = osp.join(C.CKPT_DIR, f'tspulse_fold{fold_idx+1}_best.pt')

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

    print(f"Train: {train_win.shape[0]:,}  HC={(train_lab==0).sum():,}  PD={(train_lab==1).sum():,}")
    print(f"Val:   {val_win.shape[0]:,}    Test subjects: {len(np.unique(test_subj))}")

    train_loader = DataLoader(TSPulseDataset(train_win, train_lab),
                              batch_size=TS_BATCH, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TSPulseDataset(val_win,   val_lab),
                              batch_size=TS_BATCH, shuffle=False)
    test_loader  = DataLoader(TSPulseDataset(test_win,  test_lab),
                              batch_size=TS_BATCH, shuffle=False)

    # ── Skip if checkpoint exists ─────────────────────────────────────────────
    if osp.exists(ckpt_path):
        print(f"Checkpoint found — skipping training for fold {fold_idx+1}")
        model = TSPulseForClassification.from_pretrained(
            'ibm-granite/granite-timeseries-tspulse-r1',
            revision='tspulse-block-dualhead-512-p16-r1',
            **MODEL_CONFIG
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()
    else:
        # ── Full fine-tuning ──────────────────────────────────────────────────
        model = TSPulseForClassification.from_pretrained(
            'ibm-granite/granite-timeseries-tspulse-r1',
            revision='tspulse-block-dualhead-512-p16-r1',
            **MODEL_CONFIG
        ).to(device)

        # Unfreeze all — pretrained weights provide better init than random
        for param in model.parameters():
            param.requires_grad = True

        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {total:,} (full fine-tuning)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=TS_LR, weight_decay=1e-2)

        def lr_lambda(epoch):
            warmup = 5
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(1, TS_MAX_EPOCHS - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_loss    = float('inf')
        patience_counter = 0

        for epoch in range(1, TS_MAX_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                past_values   = batch['past_values'].to(device)
                target_values = batch['target_values'].to(device)
                optimizer.zero_grad()
                outputs = model(past_values=past_values, target_values=target_values)
                loss    = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(past_values)
            train_loss /= len(train_loader.dataset)
            scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    past_values   = batch['past_values'].to(device)
                    target_values = batch['target_values'].to(device)
                    outputs = model(past_values=past_values, target_values=target_values)
                    val_loss += outputs.loss.item() * len(past_values)
            val_loss /= len(val_loader.dataset)

            print(f"Epoch {epoch:3d}/{TS_MAX_EPOCHS}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= TS_PATIENCE:
                    print(f"Early stopping at epoch {epoch}  (best val: {best_val_loss:.4f})")
                    break

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

    # ── Subject-level evaluation — mean pool window probabilities ─────────────
    all_probs, all_true, all_subj_ids = [], [], []
    with torch.no_grad():
        offset = 0
        for batch in test_loader:
            past_values = batch['past_values'].to(device)
            outputs     = model(past_values=past_values)
            probs       = torch.softmax(outputs.prediction_outputs, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_true.append(batch['target_values'].numpy())
            all_subj_ids.append(test_subj[offset:offset + len(past_values)])
            offset += len(past_values)

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
        json.dump({'fold_metrics': fold_metrics,
                   'folds_complete': fold_idx + 1}, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
agg = aggregate_fold_metrics(fold_metrics)
print_summary(agg)

with open(RESULTS_FILE, 'w') as f:
    json.dump({'fold_metrics': fold_metrics, 'aggregate': agg,
               'evaluation': 'subject-level mean pooling'}, f, indent=2)

print(f"\nResults:     {RESULTS_FILE}")
print(f"Checkpoints: {C.CKPT_DIR}")
print("DONE.")
