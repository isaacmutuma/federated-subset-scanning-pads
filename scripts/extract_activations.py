"""
extract_activations.py
----------------------
Phase 4: Extract PatchTST encoder activations for DeepScan.

Extracts from TWO layers for comparison:
  - Layer 1 (best by clustering metrics: Si, CH, ED)
  - Layer 2 (best by classifier AUC — used in previous experiments)

Token strategy: mean over all channels AND patches → 128-dim per window
  (matches Strategy C from layer analysis, best silhouette)

For each fold saves (per layer):
  activations_fold{k}_{layer}_hc_train.npy  — HC train background (n_hc_train, 128)
  activations_fold{k}_{layer}_val_hc.npy    — HC val activations (n_hc_val, 128)
  activations_fold{k}_{layer}_val_pd.npy    — PD val for direction mask (n_pd_val, 128)
  activations_fold{k}_{layer}_test.npy      — all test windows (n_test, 128)
  activations_fold{k}_{layer}_test_labels.npy
  activations_fold{k}_{layer}_test_subj.npy
  activations_fold{k}_{split}_{subj,task,wrist}.npy — per-window metadata,
        split in {hc_train, val_hc, val_pd, test} (test_subj saved as before)
  activations_fold{k}_norm_mean.npy
  activations_fold{k}_norm_std.npy

Run:
    python scripts/extract_activations.py
"""

import sys, os, os.path as osp, json, warnings, zipfile, pickle
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

os.makedirs(C.OUTPUT_DIR, exist_ok=True)
os.makedirs(C.CKPT_DIR,   exist_ok=True)

ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
os.makedirs(ACT_DIR, exist_ok=True)
print(f"Activations will be saved to: {ACT_DIR}")

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

def window_recordings(df, label_of):
    """Window every recording in df. Returns windows, labels, subject_ids, tasks, wrists."""
    all_windows, all_labels_out, all_subject_ids = [], [], []
    all_tasks, all_wrists = [], []
    skipped = 0
    for _, row in df.iterrows():
        fp = osp.join(BASE_PATH, 'movement', row['filepath'])
        if not osp.exists(fp): skipped += 1; continue
        try:    signal = load_timeseries(fp)
        except: skipped += 1; continue
        if signal.shape[1] < C.WINDOW_SIZE * C.N_WINDOWS: skipped += 1; continue
        sig_filt = bandpass_filter(signal, lowcut=C.LOWCUT, highcut=C.HIGHCUT, fs=C.FS)
        wins     = segment_windows(sig_filt, window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)[:C.N_WINDOWS]
        if len(wins) < C.N_WINDOWS: skipped += 1; continue
        all_windows.append(wins)
        all_labels_out.extend([label_of(row['label'])] * len(wins))
        all_subject_ids.extend([int(row['patient_id'])] * len(wins))
        all_tasks.extend([str(row['task'])] * len(wins))
        all_wrists.extend([str(row['wrist'])] * len(wins))
    windows = np.concatenate(all_windows, axis=0)
    print(f"  Windows: {windows.shape}  Skipped recordings: {skipped}  "
          f"Windows per subject: {len(windows) / len(np.unique(all_subject_ids)):.1f}")
    return (windows, np.array(all_labels_out, dtype=np.int64),
            np.array(all_subject_ids, dtype=np.int64),
            np.array(all_tasks), np.array(all_wrists))

filtered = manifest[manifest['label'].isin([HC_STR, PD_STR])].copy()
if C.TRAIN_WRIST is not None:
    filtered = filtered[filtered['wrist'] == C.TRAIN_WRIST]
print(f'Subjects: {filtered["patient_id"].nunique()}  Rows: {len(filtered)}')

print("Windowing HC/PD...")
windows, labels, subject_ids, tasks, wrists = window_recordings(filtered, LABEL_MAP.get)
print(f"  HC={(labels==0).sum()}  PD={(labels==1).sum()}")

unique_subjects = np.unique(subject_ids)
subject_labels  = np.array([labels[subject_ids == s][0] for s in unique_subjects])
folds = generate_fold_splits(unique_subjects, subject_labels,
                              n_splits=C.N_SPLITS, random_state=C.RANDOM_STATE,
                              val_fraction=0.2,
                              save_path=osp.join(C.OUTPUT_DIR, 'fold_splits.pkl'))

LAYERS_TO_EXTRACT = ['layer_0', 'layer_1', 'layer_2']

# ── PatchTST model with hooks for Layer 1 and Layer 2 ────────────────────────
from transformers import PatchTSTConfig, PatchTSTModel

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


def extract_acts(model, win_array, lab_array, batch_size=128):
    """
    Run windows through encoder.
    Returns dict: {layer_name: (n, 128)} for every layer in LAYERS_TO_EXTRACT
    Token strategy: mean over channels AND patches → 128-dim
    """
    hook_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hook_outputs[name] = h.detach().cpu()
        return hook

    handles = [model.encoder.encoder.layers[int(name.split('_')[1])]
               .register_forward_hook(make_hook(name)) for name in LAYERS_TO_EXTRACT]

    loader = DataLoader(PADSDataset(win_array, lab_array),
                        batch_size=batch_size, shuffle=False)
    collected = {name: [] for name in LAYERS_TO_EXTRACT}
    with torch.no_grad():
        for x, _ in loader:
            _ = model(x.to(device))
            for name in collected:
                # mean over channels (6) and patches (24) → (batch, 128)
                collected[name].append(hook_outputs[name].numpy().mean(axis=(1, 2)))

    for h in handles: h.remove()
    return {n: np.concatenate(v, axis=0) for n, v in collected.items()}


# ── Extract per fold ──────────────────────────────────────────────────────────
# Set True after retraining checkpoints, otherwise stale activations are reused
FORCE_REEXTRACT = os.environ.get('FORCE_REEXTRACT', '0') == '1'

for fold_idx, fold in enumerate(folds):
    k = fold_idx + 1
    ckpt_path = osp.join(C.CKPT_DIR, f'patchtst_fold{k}_best.pt')
    assert osp.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")

    # Skip if already extracted for both layers (incl. metadata added for federated sim)
    files_needed = [f'activations_fold{k}_{layer}_hc_train.npy' for layer in LAYERS_TO_EXTRACT]
    files_needed.append(f'activations_fold{k}_hc_train_subj.npy')
    if not FORCE_REEXTRACT and all(osp.exists(osp.join(ACT_DIR, f)) for f in files_needed):
        print(f"Already extracted — skipping fold {k}")
        continue

    train_mask = np.isin(subject_ids, fold['train_subjects'])
    val_mask   = np.isin(subject_ids, fold['val_subjects'])
    test_mask  = np.isin(subject_ids, fold['test_subjects'])

    train_win, train_lab = windows[train_mask], labels[train_mask]
    val_win,   val_lab   = windows[val_mask],   labels[val_mask]
    test_win,  test_lab  = windows[test_mask],  labels[test_mask]
    test_subj            = subject_ids[test_mask]
    train_subj, val_subj = subject_ids[train_mask], subject_ids[val_mask]
    train_task, val_task, test_task = tasks[train_mask], tasks[val_mask], tasks[test_mask]
    train_wrist, val_wrist, test_wrist = wrists[train_mask], wrists[val_mask], wrists[test_mask]

    mean, std = compute_normalization_stats(train_win)
    train_win = apply_normalization(train_win, mean, std)
    val_win   = apply_normalization(val_win,   mean, std)
    test_win  = apply_normalization(test_win,  mean, std)

    hc_train_win = train_win[train_lab == 0]
    val_hc_win   = val_win[val_lab == 0]
    val_pd_win   = val_win[val_lab == 1]

    print(f"HC train: {hc_train_win.shape[0]:,}  Val HC: {val_hc_win.shape[0]:,}  "
          f"Val PD: {val_pd_win.shape[0]:,}  Test: {test_win.shape[0]:,}")

    model = PatchTSTClassifier().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded: {ckpt_path}")

    # Extract from both layers simultaneously
    print("Extracting HC train activations...")
    hc_train_acts = extract_acts(model, hc_train_win,
                                  np.zeros(len(hc_train_win), dtype=np.int64))

    print("Extracting val HC activations...")
    val_hc_acts = extract_acts(model, val_hc_win,
                                np.zeros(len(val_hc_win), dtype=np.int64))

    print("Extracting val PD activations...")
    val_pd_acts = extract_acts(model, val_pd_win,
                                np.ones(len(val_pd_win), dtype=np.int64))

    print("Extracting test activations...")
    test_acts = extract_acts(model, test_win, test_lab)

    # Save for each layer
    for layer in LAYERS_TO_EXTRACT:
        print(f"  Saving {layer}: {hc_train_acts[layer].shape}")
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{layer}_hc_train.npy'),
                hc_train_acts[layer])
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{layer}_val_hc.npy'),
                val_hc_acts[layer])
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{layer}_val_pd.npy'),
                val_pd_acts[layer])
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{layer}_test.npy'),
                test_acts[layer])

    # Shared files (same for both layers)
    np.save(osp.join(ACT_DIR, f'activations_fold{k}_test_labels.npy'), test_lab)
    np.save(osp.join(ACT_DIR, f'activations_fold{k}_test_subj.npy'),   test_subj)
    # Per-window metadata, row-aligned with the activation files above
    hc_tr, v_hc, v_pd = train_lab == 0, val_lab == 0, val_lab == 1
    meta = {'hc_train': (train_subj[hc_tr], train_task[hc_tr], train_wrist[hc_tr]),
            'val_hc':   (val_subj[v_hc],    val_task[v_hc],    val_wrist[v_hc]),
            'val_pd':   (val_subj[v_pd],    val_task[v_pd],    val_wrist[v_pd]),
            'test':     (test_subj,         test_task,         test_wrist)}
    for split, (s, t, w) in meta.items():
        if split != 'test':
            np.save(osp.join(ACT_DIR, f'activations_fold{k}_{split}_subj.npy'), s)
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{split}_task.npy'),  t)
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{split}_wrist.npy'), w)
    np.save(osp.join(ACT_DIR, f'activations_fold{k}_norm_mean.npy'),   mean)
    np.save(osp.join(ACT_DIR, f'activations_fold{k}_norm_std.npy'),    std)
    print(f"Saved fold {k} — {LAYERS_TO_EXTRACT} activations (128-dim each)")

print("\nAll folds done.")
print(f"Files in {ACT_DIR}:")
for f in sorted(os.listdir(ACT_DIR)):
    path = osp.join(ACT_DIR, f)
    print(f"  {f}  ({os.path.getsize(path)/1e6:.1f} MB)")
