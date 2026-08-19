"""
extract_activations.py
----------------------
Phase 4: Extract PatchTST encoder activations for DeepScan.

For each fold:
  - Load best PatchTST checkpoint
  - Extract activations from HC train windows → background distribution
  - Extract activations from all test windows (HC + PD) → test set
  - Save to Drive as numpy arrays

Output structure (per fold):
  activations_fold{k}_hc_train.npy   — shape (n_hc_train_windows, d_model*6)
  activations_fold{k}_test.npy       — shape (n_test_windows, d_model*6)
  activations_fold{k}_test_labels.npy — shape (n_test_windows,) — 0=HC, 1=PD
  activations_fold{k}_test_subj.npy  — shape (n_test_windows,) — subject IDs

Run:
    python scripts/extract_activations.py
"""

import sys, os, os.path as osp, json, warnings, zipfile
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

os.makedirs(C.OUTPUT_DIR,  exist_ok=True)
os.makedirs(C.CKPT_DIR,    exist_ok=True)

# Output dir for activations
ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
os.makedirs(ACT_DIR, exist_ok=True)
print(f"Activations will be saved to: {ACT_DIR}")

# ── Extract zip if needed ─────────────────────────────────────────────────────
if C.PADS_ZIP and not osp.exists(C.PADS_ROOT):
    with zipfile.ZipFile(C.PADS_ZIP, 'r') as z:
        z.extractall(C.PADS_ROOT)

def find_pads_root(base):
    for root, dirs, files in os.walk(base):
        if 'movement' in dirs and 'patients' in dirs:
            return root
    return None

BASE_PATH = find_pads_root(C.PADS_ROOT)
assert BASE_PATH, f"PADS not found under {C.PADS_ROOT}"
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

filtered = manifest[manifest['label'].isin([HC_STR, PD_STR])].copy()
if C.TRAIN_WRIST is not None:
    filtered = filtered[filtered['wrist'] == C.TRAIN_WRIST]
print(f'Subjects: {filtered["patient_id"].nunique()}  Rows: {len(filtered)}')

all_windows, all_labels_out, all_subject_ids = [], [], []
skipped = 0
print("Windowing...")
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

# ── PatchTST model ────────────────────────────────────────────────────────────
from transformers import PatchTSTConfig, PatchTSTModel

class PatchTSTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
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

    def get_activations(self, x):
        """Returns pooled encoder representations — shape (batch, D_MODEL*6)"""
        out    = self.encoder(past_values=x.permute(0, 2, 1))
        pooled = out.last_hidden_state.mean(dim=2)
        return pooled.reshape(pooled.size(0), -1).detach()


# ── Extract activations per fold ──────────────────────────────────────────────
for fold_idx, fold in enumerate(folds):
    ckpt_path = osp.join(C.CKPT_DIR, f'patchtst_fold{fold_idx+1}_best.pt')
    assert osp.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    print(f"\n{'='*55}\nFOLD {fold_idx+1}/{C.N_SPLITS}\n{'='*55}")

    # Skip if already extracted
    hc_path = osp.join(ACT_DIR, f'activations_fold{fold_idx+1}_hc_train.npy')
    if osp.exists(hc_path):
        print(f"Already extracted — skipping fold {fold_idx+1}")
        continue

    train_mask = np.isin(subject_ids, fold['train_subjects'])
    test_mask  = np.isin(subject_ids, fold['test_subjects'])

    train_win, train_lab = windows[train_mask], labels[train_mask]
    test_win,  test_lab  = windows[test_mask],  labels[test_mask]
    test_subj            = subject_ids[test_mask]

    mean, std = compute_normalization_stats(train_win)
    train_win = apply_normalization(train_win, mean, std)
    test_win  = apply_normalization(test_win,  mean, std)

    # HC train windows only (background distribution for DeepScan)
    hc_train_mask = train_lab == 0
    hc_train_win  = train_win[hc_train_mask]
    print(f"HC train windows: {hc_train_win.shape[0]:,}")
    print(f"Test windows: {test_win.shape[0]:,}  HC={(test_lab==0).sum()}  PD={(test_lab==1).sum()}")

    # Load model
    model = PatchTSTClassifier().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # Extract HC train activations (background)
    hc_loader = DataLoader(PADSDataset(hc_train_win,
                                       np.zeros(len(hc_train_win), dtype=np.int64)),
                           batch_size=128, shuffle=False)
    hc_acts = []
    with torch.no_grad():
        for x, _ in hc_loader:
            hc_acts.append(model.get_activations(x.to(device)).cpu().numpy())
    hc_acts = np.concatenate(hc_acts, axis=0)
    print(f"HC activations shape: {hc_acts.shape}")

    # Extract test activations (all test windows — HC and PD)
    test_loader = DataLoader(PADSDataset(test_win, test_lab),
                             batch_size=128, shuffle=False)
    test_acts = []
    with torch.no_grad():
        for x, _ in test_loader:
            test_acts.append(model.get_activations(x.to(device)).cpu().numpy())
    test_acts = np.concatenate(test_acts, axis=0)
    print(f"Test activations shape: {test_acts.shape}")

    # Save
    np.save(osp.join(ACT_DIR, f'activations_fold{fold_idx+1}_hc_train.npy'),   hc_acts)
    np.save(osp.join(ACT_DIR, f'activations_fold{fold_idx+1}_test.npy'),        test_acts)
    np.save(osp.join(ACT_DIR, f'activations_fold{fold_idx+1}_test_labels.npy'), test_lab)
    np.save(osp.join(ACT_DIR, f'activations_fold{fold_idx+1}_test_subj.npy'),   test_subj)
    np.save(osp.join(ACT_DIR, f'activations_fold{fold_idx+1}_norm_mean.npy'),   mean)
    np.save(osp.join(ACT_DIR, f'activations_fold{fold_idx+1}_norm_std.npy'),    std)

    print(f"Saved fold {fold_idx+1} activations to {ACT_DIR}")

print("\nAll folds done.")
print(f"Activations directory: {ACT_DIR}")
print("Files saved:")
for f in sorted(os.listdir(ACT_DIR)):
    path = osp.join(ACT_DIR, f)
    print(f"  {f}  ({os.path.getsize(path)/1e6:.1f} MB)")
