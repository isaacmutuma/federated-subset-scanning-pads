"""
extract_minirocket_features.py
------------------------------
MiniROCKET side pipeline, step 1: per-window MiniROCKET features for DeepScan.

Same subjects, windows, folds and normalisation as scripts/extract_activations.py (PatchTST),
so the shared analysis scripts compare the two encoders on identical data. Per fold:
  - normalise with the fold's training-window statistics (as train_minirocket.py does)
  - fit MiniRocket on the fold's training windows (HC + PD, no labels: MiniROCKET's
    kernels are fixed and only the bias quantiles are fitted from the data)
  - transform every split and save it in the PatchTST activation file layout, with the
    layer name 'minirocket', under MR_ACT_DIR (default /content/minirocket_scan/activations)

Saved per fold (same names as extract_activations.py, layer = 'minirocket'):
  activations_fold{k}_minirocket_{hc_train,val_hc,val_pd,test}.npy  (5,544 features per window)
  activations_fold{k}_test_labels.npy, activations_fold{k}_test_subj.npy
  activations_fold{k}_{split}_{subj,task,wrist}.npy  (split in hc_train, val_hc, val_pd, test)

Run:
    python minirocket_scan/extract_minirocket_features.py
"""

import sys, os, os.path as osp, json, warnings, zipfile

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO_ROOT, 'scripts'))
sys.path.insert(0, REPO_ROOT)
import config as C

from src.data.preprocessing import (bandpass_filter, segment_windows,
                                     compute_normalization_stats, apply_normalization)
from src.data.folds import generate_fold_splits

try:
    from sktime.transformations.panel.rocket import MiniRocket
except ImportError:
    os.system('pip install -q sktime')
    from sktime.transformations.panel.rocket import MiniRocket

LAYER   = 'minirocket'
# 5,544 features per window (924 per channel x 6) make ~2.2 GB of features per run, so they
# go to the runtime's local disk by default; only the result JSONs are written to Drive.
ACT_DIR = os.environ.get('MR_ACT_DIR', '/content/minirocket_scan/activations')
os.makedirs(ACT_DIR, exist_ok=True)
os.makedirs(C.OUTPUT_DIR, exist_ok=True)
print(f"MiniROCKET features will be saved to: {ACT_DIR}")

if C.PADS_ZIP and not osp.exists(C.PADS_ROOT):
    with zipfile.ZipFile(C.PADS_ZIP, 'r') as z:
        z.extractall(C.PADS_ROOT)

# ── Data: identical to scripts/extract_activations.py ─────────────────────────
def find_pads_root(base):
    for root, dirs, files in os.walk(base):
        if 'movement' in dirs and 'patients' in dirs:
            return root
    return None

BASE_PATH = find_pads_root(C.PADS_ROOT)
assert BASE_PATH, f"PADS not found under {C.PADS_ROOT}"
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


def transform(rocket, win, batch=2000):
    """MiniROCKET features as float32, in batches to bound memory."""
    return np.concatenate([np.asarray(rocket.transform(win[i:i + batch]), dtype=np.float32)
                           for i in range(0, len(win), batch)], axis=0)


# ── Extract per fold ──────────────────────────────────────────────────────────
FORCE_REEXTRACT = os.environ.get('FORCE_REEXTRACT', '0') == '1'

for fold_idx, fold in enumerate(folds):
    k = fold_idx + 1
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")

    marker = osp.join(ACT_DIR, f'activations_fold{k}_hc_train_subj.npy')
    if not FORCE_REEXTRACT and osp.exists(osp.join(ACT_DIR, f'activations_fold{k}_{LAYER}_test.npy')) \
            and osp.exists(marker):
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

    hc_tr, v_hc, v_pd = train_lab == 0, val_lab == 0, val_lab == 1
    print(f"HC train: {hc_tr.sum():,}  Val HC: {v_hc.sum():,}  Val PD: {v_pd.sum():,}  "
          f"Test: {len(test_win):,}")

    # Unsupervised fit: no labels are passed; seeded so every run gives the same features
    rocket = MiniRocket(num_kernels=C.ROCKET_KERNELS, random_state=C.RANDOM_STATE)
    rocket.fit(train_win)

    feats = {'hc_train': transform(rocket, train_win[hc_tr]),
             'val_hc':   transform(rocket, val_win[v_hc]),
             'val_pd':   transform(rocket, val_win[v_pd]),
             'test':     transform(rocket, test_win)}
    print(f"  Features per window: {feats['test'].shape[1]}")

    for split, X in feats.items():
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{LAYER}_{split}.npy'), X)
    np.save(osp.join(ACT_DIR, f'activations_fold{k}_test_labels.npy'), test_lab)
    np.save(osp.join(ACT_DIR, f'activations_fold{k}_test_subj.npy'),   test_subj)
    meta = {'hc_train': (train_subj[hc_tr], train_task[hc_tr], train_wrist[hc_tr]),
            'val_hc':   (val_subj[v_hc],    val_task[v_hc],    val_wrist[v_hc]),
            'val_pd':   (val_subj[v_pd],    val_task[v_pd],    val_wrist[v_pd]),
            'test':     (test_subj,         test_task,         test_wrist)}
    for split, (s, t, w) in meta.items():
        if split != 'test':
            np.save(osp.join(ACT_DIR, f'activations_fold{k}_{split}_subj.npy'), s)
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{split}_task.npy'),  t)
        np.save(osp.join(ACT_DIR, f'activations_fold{k}_{split}_wrist.npy'), w)
    print(f"Saved fold {k}")

print("\nAll folds done.")
