"""
train_minirocket.py
-------------------
MiniROCKET classifier for PADS IMU windows.
CPU only — runs in minutes. No GPU needed.

All paths and hyperparameters from config.py.
Run:
    python scripts/train_minirocket.py
"""

import sys, os, os.path as osp, json, warnings, zipfile
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# ── Load config ───────────────────────────────────────────────────────────────
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

# ── Repo setup ────────────────────────────────────────────────────────────────
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

# ── Install sktime if needed ──────────────────────────────────────────────────
try:
    from sktime.transformations.panel.rocket import MiniRocket
except ImportError:
    print("Installing sktime...")
    os.system('pip install -q sktime')
    from sktime.transformations.panel.rocket import MiniRocket

print("MiniROCKET ready.")

os.makedirs(C.OUTPUT_DIR,  exist_ok=True)
os.makedirs(C.CKPT_DIR,    exist_ok=True)
os.makedirs(C.RESULTS_DIR, exist_ok=True)

# ── Extract zip if needed ─────────────────────────────────────────────────────
if C.PADS_ZIP and not osp.exists(C.PADS_ROOT):
    print(f"Extracting {C.PADS_ZIP} ...")
    with zipfile.ZipFile(C.PADS_ZIP, 'r') as z:
        z.extractall(C.PADS_ROOT)
    print("Done.")

# ── Find dataset root ─────────────────────────────────────────────────────────
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

# ── Build manifest ────────────────────────────────────────────────────────────
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
print(f'Subjects: {filtered["patient_id"].nunique()}  Rows: {len(filtered)}')

# ── Window recordings ─────────────────────────────────────────────────────────
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
print(f"Skipped: {skipped}")

unique_subjects = np.unique(subject_ids)
subject_labels  = np.array([labels[subject_ids == s][0] for s in unique_subjects])
folds = generate_fold_splits(unique_subjects, subject_labels,
                              n_splits=C.N_SPLITS, random_state=C.RANDOM_STATE,
                              val_fraction=0.2,
                              save_path=osp.join(C.OUTPUT_DIR, 'fold_splits.pkl'))

# ── MiniROCKET needs sktime panel format: (n, c, t) already — perfect ────────
# sktime MiniRocket expects numpy (n_samples, n_channels, n_timepoints)
# Our windows are already (N, 6, 200) — no reshaping needed

RESULTS_FILE = osp.join(C.RESULTS_DIR, 'minirocket_results.json')
fold_metrics = []

for fold_idx, fold in enumerate(folds):
    print(f"\n{'='*55}\nFOLD {fold_idx+1}/{C.N_SPLITS}\n{'='*55}")

    train_mask = np.isin(subject_ids, fold['train_subjects'])
    test_mask  = np.isin(subject_ids, fold['test_subjects'])

    train_win, train_lab = windows[train_mask], labels[train_mask]
    test_win,  test_lab  = windows[test_mask],  labels[test_mask]

    # Normalize using train stats
    mean, std = compute_normalization_stats(train_win)
    train_win = apply_normalization(train_win, mean, std)
    test_win  = apply_normalization(test_win,  mean, std)

    # Oversample HC to balance training set
    hc_idx = np.where(train_lab == 0)[0]
    pd_idx = np.where(train_lab == 1)[0]
    hc_os  = resample(hc_idx, replace=True, n_samples=len(pd_idx), random_state=C.RANDOM_STATE)
    idx    = np.concatenate([hc_os, pd_idx])
    np.random.shuffle(idx)
    train_win, train_lab = train_win[idx], train_lab[idx]

    print(f"Train: {train_win.shape[0]:,}  HC={(train_lab==0).sum():,}  PD={(train_lab==1).sum():,}")
    print(f"Test:  {test_win.shape[0]:,}   HC={(test_lab==0).sum():,}   PD={(test_lab==1).sum():,}")

    # ── Fit MiniROCKET ────────────────────────────────────────────────────────
    print("Fitting MiniROCKET...")
    rocket = MiniRocket(num_kernels=C.ROCKET_KERNELS, random_state=C.RANDOM_STATE)
    rocket.fit(train_win)

    print("Transforming train...")
    X_train = rocket.transform(train_win)   # (n_train, n_features)

    print("Transforming test...")
    X_test  = rocket.transform(test_win)    # (n_test, n_features)

    # Scale features
    scaler  = StandardScaler(with_mean=True)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Ridge classifier ──────────────────────────────────────────────────────
    print("Training Ridge classifier...")
    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(X_train, train_lab)

    # Decision function scores for AUC
    scores = clf.decision_function(X_test)

    metrics = compute_metrics(test_lab, scores)
    fold_metrics.append(metrics)
    print_fold_results(fold_idx, metrics)

    with open(RESULTS_FILE, 'w') as f:
        json.dump({'fold_metrics': fold_metrics,
                   'folds_complete': fold_idx + 1}, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
agg = aggregate_fold_metrics(fold_metrics)
print_summary(agg)

with open(RESULTS_FILE, 'w') as f:
    json.dump({'fold_metrics': fold_metrics, 'aggregate': agg}, f, indent=2)

print(f"\nResults: {RESULTS_FILE}")
print("DONE.")
