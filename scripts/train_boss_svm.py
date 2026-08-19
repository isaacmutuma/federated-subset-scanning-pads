"""
train_boss_svm.py
-----------------
BOSS + SGD classifier for PADS IMU windows. Pure numpy — no pyts.
Subject-level evaluation: mean pool per-window decision scores per subject.
Run: python scripts/train_boss_svm.py
"""
import sys, os, os.path as osp, json, warnings, zipfile
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
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
        pid = f'{pat_num:03d}'
        lbl = get_patient_label(pat_num)
        obs_path = osp.join(OBSERVATION_DIR, f'observation_{pid}.json')
        with open(obs_path) as f:
            obs = json.load(f)
        for session in obs['session']:
            for record in session['records']:
                rows.append({'patient_id': pid, 'label': lbl,
                             'task': session.get('record_name'),
                             'wrist': record.get('device_location'),
                             'filepath': record.get('file_name')})
    return pd.DataFrame(rows)

def load_timeseries(filepath):
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 1:7].values.T.astype(np.float32)

print("Building manifest...")
manifest = build_manifest()
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
    try: signal = load_timeseries(fp)
    except: skipped += 1; continue
    if signal.shape[1] < C.WINDOW_SIZE * C.N_WINDOWS: skipped += 1; continue
    sig_filt = bandpass_filter(signal, lowcut=C.LOWCUT, highcut=C.HIGHCUT, fs=C.FS)
    wins = segment_windows(sig_filt, window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)[:C.N_WINDOWS]
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

# ── BOSS (pure numpy) ─────────────────────────────────────────────────────────
BOSS_WINDOW_SIZES = [40]
BOSS_WORD_SIZE    = 4
BOSS_N_BINS       = 4


def fit_boss_channel(X_ch, window_size, word_size, n_bins):
    n_samples, n_timesteps = X_ch.shape
    step = max(1, window_size // 2)
    all_coeffs = []
    for i in range(n_samples):
        for start in range(0, n_timesteps - window_size + 1, step):
            sub = X_ch[i, start:start + window_size]
            dft = np.fft.rfft(sub)[:word_size]
            all_coeffs.append(np.real(dft))
    all_coeffs  = np.array(all_coeffs)
    breakpoints = np.percentile(all_coeffs,
                                np.linspace(0, 100, n_bins + 1)[1:-1], axis=0)
    return breakpoints


def transform_boss_channel(X_ch, window_size, word_size, breakpoints):
    n_samples, n_timesteps = X_ch.shape
    step = max(1, window_size // 2)
    bags = []
    for i in range(n_samples):
        bag = Counter()
        prev_word = None
        for start in range(0, n_timesteps - window_size + 1, step):
            sub     = X_ch[i, start:start + window_size]
            dft     = np.fft.rfft(sub)[:word_size]
            coeffs  = np.real(dft)
            letters = tuple(
                int(np.searchsorted(breakpoints[:, j], coeffs[j], side='right'))
                for j in range(word_size)
            )
            if letters != prev_word:
                bag[letters] += 1
                prev_word = letters
        bags.append(bag)
    return bags


def bags_to_matrix(bags, vocab=None):
    if vocab is None:
        vocab = sorted(set(k for b in bags for k in b.keys()))
    vocab_idx = {k: i for i, k in enumerate(vocab)}
    X = np.zeros((len(bags), max(1, len(vocab))), dtype=np.float32)
    for i, bag in enumerate(bags):
        for k, v in bag.items():
            if k in vocab_idx:
                X[i, vocab_idx[k]] = v
    return X, vocab


def extract_boss_features(X, window_sizes, word_size, n_bins, fitted_params=None):
    n_samples, n_channels, _ = X.shape
    all_features = []
    params_out   = [] if fitted_params is None else None
    param_idx    = 0
    for ws in window_sizes:
        for ch in range(n_channels):
            X_ch = X[:, ch, :]
            if fitted_params is None:
                bp           = fit_boss_channel(X_ch, ws, word_size, n_bins)
                bags         = transform_boss_channel(X_ch, ws, word_size, bp)
                feats, vocab = bags_to_matrix(bags)
                params_out.append((bp, vocab))
            else:
                bp, vocab = fitted_params[param_idx]
                bags      = transform_boss_channel(X_ch, ws, word_size, bp)
                feats, _  = bags_to_matrix(bags, vocab=vocab)
                param_idx += 1
            all_features.append(feats)
    return np.hstack(all_features), params_out


# ── Training ──────────────────────────────────────────────────────────────────
RESULTS_FILE = osp.join(C.RESULTS_DIR, 'boss_svm_results.json')
fold_metrics = []

for fold_idx, fold in enumerate(folds):
    print(f"\n{'='*55}\nFOLD {fold_idx+1}/{C.N_SPLITS}\n{'='*55}")

    train_mask = np.isin(subject_ids, fold['train_subjects'])
    test_mask  = np.isin(subject_ids, fold['test_subjects'])

    train_win, train_lab = windows[train_mask], labels[train_mask]
    test_win,  test_lab  = windows[test_mask],  labels[test_mask]
    test_subj            = subject_ids[test_mask]

    mean, std = compute_normalization_stats(train_win)
    train_win = apply_normalization(train_win, mean, std)
    test_win  = apply_normalization(test_win,  mean, std)

    hc_idx = np.where(train_lab == 0)[0]
    pd_idx = np.where(train_lab == 1)[0]
    hc_os  = resample(hc_idx, replace=True, n_samples=len(pd_idx), random_state=C.RANDOM_STATE)
    idx    = np.concatenate([hc_os, pd_idx])
    np.random.shuffle(idx)
    train_win, train_lab = train_win[idx], train_lab[idx]

    print(f"Train: {train_win.shape[0]:,}  HC={(train_lab==0).sum():,}  PD={(train_lab==1).sum():,}")

    print("Extracting BOSS features (train)...")
    X_train, fitted_params = extract_boss_features(
        train_win, BOSS_WINDOW_SIZES, BOSS_WORD_SIZE, BOSS_N_BINS)

    print("Extracting BOSS features (test)...")
    X_test, _ = extract_boss_features(
        test_win, BOSS_WINDOW_SIZES, BOSS_WORD_SIZE, BOSS_N_BINS,
        fitted_params=fitted_params)

    print(f"Features: train={X_train.shape}  test={X_test.shape}")

    scaler  = StandardScaler(with_mean=True)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Training SGD classifier...")
    clf = SGDClassifier(loss='modified_huber', class_weight='balanced',
                        max_iter=1000, random_state=C.RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, train_lab)

    # Per-window decision scores
    window_scores = clf.predict_proba(X_test)[:, 1]

    # ── Subject-level aggregation ─────────────────────────────────────────────
    unique_test_subj = np.unique(test_subj)
    subj_scores = np.array([window_scores[test_subj == s].mean() for s in unique_test_subj])
    subj_true   = np.array([test_lab     [test_subj == s][0]     for s in unique_test_subj])

    print(f"Subject-level: {len(unique_test_subj)} subjects  "
          f"HC={(subj_true==0).sum()}  PD={(subj_true==1).sum()}")

    metrics = compute_metrics(subj_true, subj_scores)
    fold_metrics.append(metrics)
    print_fold_results(fold_idx, metrics)

    with open(RESULTS_FILE, 'w') as f:
        json.dump({'fold_metrics': fold_metrics,
                   'folds_complete': fold_idx + 1}, f, indent=2)

agg = aggregate_fold_metrics(fold_metrics)
print_summary(agg)
with open(RESULTS_FILE, 'w') as f:
    json.dump({'fold_metrics': fold_metrics, 'aggregate': agg,
               'evaluation': 'subject-level mean pooling'}, f, indent=2)
print(f"\nResults: {RESULTS_FILE}")
print("DONE.")
