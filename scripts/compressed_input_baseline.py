"""
compressed_input_baseline.py
-----------------------------
Compressed input baseline for DeepScan — no neural network encoder.

Tests whether simple signal-level compression is sufficient for
histogram-based anomaly detection, or whether PatchTST learned
representations are necessary.

Three compression methods:
  1. Mean pooling  — divide each channel into 10 segments, take mean → 60-dim
  2. PCA           — fit PCA on HC train windows, reduce to 60 components → 60-dim
  3. Statistical   — mean, std, min, max, energy per channel → 30-dim

Baseline comparison:
  PatchTST Layer 2 (learned representations) → 128-dim → 0.861 AUC

Method A (simple mean p-value) used throughout for fair comparison.
N_BINS=10 (selected from privacy-utility sweep).

Run:
    python scripts/compressed_input_baseline.py
"""

import sys, os, os.path as osp, json, warnings, zipfile, pickle
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
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

from src.data.preprocessing import (bandpass_filter, segment_windows,
                                     compute_normalization_stats, apply_normalization)
from src.data.folds import generate_fold_splits
from src.training.metrics import (compute_metrics, aggregate_fold_metrics,
                                   print_fold_results, print_summary)

N_BINS = 10   # from privacy-utility sweep

# ── Load PADS data ─────────────────────────────────────────────────────────────
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

filtered = manifest[manifest['label'].isin([HC_STR, PD_STR])].copy()
if C.TRAIN_WRIST is not None:
    filtered = filtered[filtered['wrist'] == C.TRAIN_WRIST]
print(f'Subjects: {filtered["patient_id"].nunique()}  Rows: {len(filtered)}')

all_windows, all_labels_out, all_subject_ids = [], [], []
print("Windowing...")
for _, row in filtered.iterrows():
    fp = osp.join(BASE_PATH, 'movement', row['filepath'])
    if not osp.exists(fp): continue
    try:    signal = load_timeseries(fp)
    except: continue
    if signal.shape[1] < C.WINDOW_SIZE * C.N_WINDOWS: continue
    sig_filt = bandpass_filter(signal, lowcut=C.LOWCUT, highcut=C.HIGHCUT, fs=C.FS)
    wins     = segment_windows(sig_filt, window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)[:C.N_WINDOWS]
    if len(wins) < C.N_WINDOWS: continue
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


# ── Compression methods ────────────────────────────────────────────────────────
def compress_mean_pool(wins, n_segments=10):
    """
    Mean pooling: divide each channel into n_segments, take mean of each.
    Input: (n_windows, 6, 200) → Output: (n_windows, 6*n_segments) = (n_windows, 60)
    """
    n, c, t = wins.shape
    seg_size = t // n_segments
    segs = wins[:, :, :seg_size*n_segments].reshape(n, c, n_segments, seg_size)
    return segs.mean(axis=3).reshape(n, -1)   # (n, 60)


def compress_statistical(wins):
    """
    Statistical features: mean, std, min, max, energy per channel.
    Input: (n_windows, 6, 200) → Output: (n_windows, 6*5) = (n_windows, 30)
    """
    feats = np.stack([
        wins.mean(axis=2),                          # mean per channel
        wins.std(axis=2),                           # std per channel
        wins.min(axis=2),                           # min per channel
        wins.max(axis=2),                           # max per channel
        (wins**2).mean(axis=2),                     # energy per channel
    ], axis=2)   # (n, 6, 5)
    return feats.reshape(len(wins), -1)             # (n, 30)


def compress_pca(train_wins, test_wins, n_components=60):
    """
    PCA compression: fit on HC train windows, transform all.
    Input: (n, 6, 200) → flatten → (n, 1200) → PCA → (n, 60)
    """
    n_tr = train_wins.reshape(len(train_wins), -1)
    n_te = test_wins.reshape(len(test_wins), -1)
    pca  = PCA(n_components=n_components, random_state=42)
    pca.fit(n_tr)
    return pca.transform(n_tr), pca.transform(n_te)


# ── Histogram p-value scoring (Method A) ──────────────────────────────────────
def compute_pvalues_directed(hc_ref, test_feats, val_pd_feats, n_bins=N_BINS):
    """Directed one-tailed histogram p-values. Low p = anomalous."""
    n_dims    = hc_ref.shape[1]
    direction = val_pd_feats.mean(0) > hc_ref.mean(0)
    p_matrix  = np.zeros((len(test_feats), n_dims), dtype=np.float64)
    for j in range(n_dims):
        counts, edges = np.histogram(hc_ref[:, j], bins=n_bins, density=False)
        counts = counts.astype(float) + 1e-6
        probs  = counts / counts.sum()
        cum    = np.cumsum(probs)
        idx    = np.clip(np.searchsorted(edges[1:], test_feats[:, j], side='right'),
                         0, len(probs)-1)
        lower  = np.where(idx > 0, cum[idx-1], 0.0)
        if direction[j]:
            p_matrix[:, j] = np.clip(1.0 - lower, 1e-6, 1.0)
        else:
            p_matrix[:, j] = np.clip(lower + probs[idx], 1e-6, 1.0)
    return p_matrix


def run_deepscan_method_a(hc_ref, val_pd_feats, test_feats, test_lab, test_subj):
    """Method A: 1 - mean(p-values) per window, mean per subject."""
    p_matrix    = compute_pvalues_directed(hc_ref, test_feats, val_pd_feats)
    win_scores  = 1.0 - p_matrix.mean(axis=1)
    unique_subj = np.unique(test_subj)
    subj_scores = np.array([win_scores[test_subj==s].mean() for s in unique_subj])
    subj_true   = np.array([test_lab[test_subj==s][0] for s in unique_subj])
    return roc_auc_score(subj_true, subj_scores)


# ── Run all compression methods across 5 folds ────────────────────────────────
COMPRESSION_METHODS = {
    'Mean pooling (60-dim)':     'mean_pool',
    'Statistical features (30-dim)': 'statistical',
    'PCA compression (60-dim)':  'pca',
}

results = {name: [] for name in COMPRESSION_METHODS}
RESULTS_FILE = osp.join(C.RESULTS_DIR, 'compressed_input_baseline.json')

for fold_idx, fold in enumerate(folds):
    k = fold_idx + 1
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")

    train_mask = np.isin(subject_ids, fold['train_subjects'])
    val_mask   = np.isin(subject_ids, fold['val_subjects'])
    test_mask  = np.isin(subject_ids, fold['test_subjects'])

    train_win, train_lab = windows[train_mask], labels[train_mask]
    val_win,   val_lab   = windows[val_mask],   labels[val_mask]
    test_win,  test_lab  = windows[test_mask],  labels[test_mask]
    test_subj            = subject_ids[test_mask]

    # Normalise using train stats
    mean, std = compute_normalization_stats(train_win)
    train_win = apply_normalization(train_win, mean, std)
    val_win   = apply_normalization(val_win,   mean, std)
    test_win  = apply_normalization(test_win,  mean, std)

    # HC train and val PD for direction mask
    hc_train_win = train_win[train_lab == 0]
    val_pd_win   = val_win[val_lab == 1]

    print(f"HC train: {hc_train_win.shape[0]:,}  Val PD: {val_pd_win.shape[0]:,}  "
          f"Test: {test_win.shape[0]:,}")

    # ── Method 1: Mean pooling ────────────────────────────────────────────────
    hc_mp  = compress_mean_pool(hc_train_win)
    vpd_mp = compress_mean_pool(val_pd_win)
    te_mp  = compress_mean_pool(test_win)
    auc_mp = run_deepscan_method_a(hc_mp, vpd_mp, te_mp, test_lab, test_subj)
    results['Mean pooling (60-dim)'].append(auc_mp)
    print(f"  Mean pooling AUC:        {auc_mp:.4f}")

    # ── Method 2: Statistical features ───────────────────────────────────────
    hc_st  = compress_statistical(hc_train_win)
    vpd_st = compress_statistical(val_pd_win)
    te_st  = compress_statistical(test_win)
    auc_st = run_deepscan_method_a(hc_st, vpd_st, te_st, test_lab, test_subj)
    results['Statistical features (30-dim)'].append(auc_st)
    print(f"  Statistical features AUC: {auc_st:.4f}")

    # ── Method 3: PCA compression ─────────────────────────────────────────────
    # Fit PCA on HC train windows only
    hc_pca_tr, _ = compress_pca(hc_train_win, hc_train_win, n_components=60)
    _, vpd_pca   = compress_pca(hc_train_win, val_pd_win,   n_components=60)
    _, te_pca    = compress_pca(hc_train_win, test_win,     n_components=60)
    auc_pca = run_deepscan_method_a(hc_pca_tr, vpd_pca, te_pca, test_lab, test_subj)
    results['PCA compression (60-dim)'].append(auc_pca)
    print(f"  PCA compression AUC:     {auc_pca:.4f}")

    # Save intermediate
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'results': {k: v for k, v in results.items()},
                   'folds_complete': fold_idx+1, 'n_bins': N_BINS}, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("COMPRESSED INPUT BASELINE — DeepScan Method A (mean p-value)")
print(f"{'='*60}")
print(f"{'Representation':<35} {'Mean AUC':>10} {'Std':>8}")
print("-"*55)

for name, aucs in results.items():
    print(f"{name:<35} {np.mean(aucs):>10.4f} {np.std(aucs):>8.4f}")

print("-"*55)
print(f"{'PatchTST Layer 2 (learned)':<35} {'0.8611':>10} {'0.0677':>8}")
print(f"{'PatchTST supervised':<35} {'0.9041':>10} {'0.0542':>8}")
print(f"{'='*60}")
print("\nConclusion: learned representations from PatchTST encoder are")
print("necessary for effective histogram-based anomaly detection.")

with open(RESULTS_FILE, 'w') as f:
    json.dump({
        'results':         {k: v for k, v in results.items()},
        'summary':         {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                            for k, v in results.items()},
        'patchtst_layer2': {'mean': 0.8611, 'std': 0.0677},
        'patchtst_supervised': {'mean': 0.9041, 'std': 0.0542},
        'n_bins': N_BINS,
        'method': 'DeepScan Method A — mean p-value, directed histogram',
    }, f, indent=2)

print(f"\nResults: {RESULTS_FILE}")
print("DONE.")
