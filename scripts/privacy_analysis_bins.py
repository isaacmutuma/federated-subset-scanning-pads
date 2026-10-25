"""
privacy_analysis_bins.py
------------------------
Extended privacy analysis: runs re-identification attack across multiple
histogram bin counts to generate the privacy-utility trade-off curve.

Uses exactly the representation the deployed pipeline computes (src/scanning/histograms.py,
same settings as run_deepscan.py): shared edges from HC train, reference counts from
BACKGROUND, directed one-tailed p-values with Laplace smoothing.

Attack (closed-set window -> subject linkage):
  - HC train subjects of each fold (~50); 80% of each subject's windows train a
    multinomial logistic regression, 20% are attacked. Chance Top-1 = 1/n_subjects.

For each bin count, computes:
  - Re-identification accuracy (Top-1, Top-5) on histogram p-values
  - Re-identification on quantised raw activations (bin index, same edges) — baseline
    that separates the effect of p-value scoring from plain quantisation
  - DeepScan Method A AUC (mean p-value, subject-level) — matches run_deepscan.py

Raw activation re-identification is bin-independent (computed once).

Outputs: results/privacy_bins_sweep.json

Run:
    python scripts/privacy_analysis_bins.py
"""

import sys, os, os.path as osp, json, subprocess, warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.scanning.histograms import make_global_edges, site_counts, pvalues_from_counts, bin_index

ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"
os.makedirs(C.RESULTS_DIR, exist_ok=True)

BIN_COUNTS   = [5, 10, 20, 50, 100]
LAYER        = 'layer_2'
BACKGROUND   = os.environ.get('BACKGROUND', 'val_hc')   # keep identical to run_deepscan.py
ALPHA        = 1.0
TEST_FRAC    = 0.2
RANDOM_STATE = 42

try:
    GIT_COMMIT = subprocess.check_output(['git', '-C', C.REPO_DIR, 'rev-parse', 'HEAD'],
                                         text=True).strip()
except Exception:
    GIT_COMMIT = 'unknown'

print(f"Privacy-utility sweep across bin counts: {BIN_COUNTS}")
print(f"Layer: {LAYER}  |  Background: {BACKGROUND}  |  Test fraction: {TEST_FRAC}")


def act_path(k, name):
    return osp.join(ACT_DIR, f'activations_fold{k}_{name}.npy')


def top_k_accuracy(y_true, probs, k=5):
    """Fraction of samples where true label is in top-k predictions."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in top_k_preds[i] for i in range(len(y_true))]))


def reid_attack(X, y, train_idx, test_idx):
    """Multinomial LR re-identification. Returns (top1, top5)."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train_idx])
    Xte = scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0,
                             random_state=RANDOM_STATE, solver='lbfgs')
    clf.fit(Xtr, y[train_idx])
    probs = clf.predict_proba(Xte)
    return (accuracy_score(y[test_idx], clf.classes_[probs.argmax(axis=1)]),
            top_k_accuracy(np.searchsorted(clf.classes_, y[test_idx]), probs, k=5))


def method_a_auc(p_matrix, subj, lab):
    win = 1.0 - p_matrix.mean(axis=1)
    us = np.unique(subj)
    return roc_auc_score([lab[subj == s][0] for s in us], [win[subj == s].mean() for s in us])


def summarise(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values)),
            'per_fold': [float(x) for x in values]}


# ── Per-fold data + fixed attack split (identical for every representation) ───
fold_data = []
for k in range(1, C.N_SPLITS + 1):
    hc_train = np.load(act_path(k, f'{LAYER}_hc_train'))
    hc_subj  = np.load(act_path(k, 'hc_train_subj'))
    rng = np.random.RandomState(RANDOM_STATE)
    tr, te = [], []
    for s in np.unique(hc_subj):
        idx = np.where(hc_subj == s)[0]
        sel = rng.choice(idx, max(1, int(len(idx) * TEST_FRAC)), replace=False)
        tr.extend(np.setdiff1d(idx, sel)); te.extend(sel)
    fold_data.append({
        'k': k, 'hc_train': hc_train, 'hc_subj': hc_subj,
        'bg': np.load(act_path(k, f'{LAYER}_{BACKGROUND}')),
        'val_pd': np.load(act_path(k, f'{LAYER}_val_pd')),
        'test': np.load(act_path(k, f'{LAYER}_test')),
        'test_lab': np.load(act_path(k, 'test_labels')),
        'test_subj': np.load(act_path(k, 'test_subj')),
        'tr': np.array(tr), 'te': np.array(te),
        'n_subjects': int(len(np.unique(hc_subj))),
    })

n_subj = [f['n_subjects'] for f in fold_data]
all_results = {
    'bin_counts': BIN_COUNTS, 'layer': LAYER, 'background': BACKGROUND, 'alpha': ALPHA,
    'pvalues': 'directed one-tailed (same as run_deepscan.py)', 'git_commit': GIT_COMMIT,
    'attack': 'closed-set window->subject linkage, multinomial LR, 80/20 windows per subject',
    'n_attack_subjects_per_fold': n_subj,
    'chance_top1': float(np.mean([1 / n for n in n_subj])),
    'chance_top5': float(np.mean([5 / n for n in n_subj])),
    'per_bin': {}, 'raw_reid': {},
}

# First pass: raw activation re-identification (doesn't depend on bins)
raw1, raw5 = zip(*[reid_attack(f['hc_train'], f['hc_subj'], f['tr'], f['te']) for f in fold_data])
r1, r5 = summarise(raw1), summarise(raw5)
all_results['raw_reid'] = {'top1_mean': r1['mean'], 'top1_std': r1['std'],
                           'top5_mean': r5['mean'], 'top5_std': r5['std'],
                           'per_fold_top1': r1['per_fold'], 'per_fold_top5': r5['per_fold']}
print(f"\nRaw activations: Top-1 {r1['mean']:.4f} ± {r1['std']:.4f}  "
      f"Top-5 {r5['mean']:.4f} ± {r5['std']:.4f}  (chance Top-1 {all_results['chance_top1']:.4f})")

# Second pass: per-bin analysis
for n_bins in BIN_COUNTS:
    print(f"\n{'='*55}\nN_BINS = {n_bins}\n{'='*55}")
    p1, p5, q1, q5, aucs = [], [], [], [], []
    for f in fold_data:
        edges = make_global_edges(f['hc_train'], n_bins)
        direction = f['val_pd'].mean(0) > f['hc_train'].mean(0)
        counts = site_counts(f['bg'], edges)

        X_p = pvalues_from_counts(counts, edges, f['hc_train'], direction, ALPHA)
        a, b = reid_attack(X_p, f['hc_subj'], f['tr'], f['te']); p1.append(a); p5.append(b)

        X_q = bin_index(f['hc_train'], edges).astype(float)
        a, b = reid_attack(X_q, f['hc_subj'], f['tr'], f['te']); q1.append(a); q5.append(b)

        p_test = pvalues_from_counts(counts, edges, f['test'], direction, ALPHA)
        aucs.append(method_a_auc(p_test, f['test_subj'], f['test_lab']))

    s_p1, s_p5, s_q1, s_q5, s_auc = map(summarise, (p1, p5, q1, q5, aucs))
    all_results['per_bin'][str(n_bins)] = {
        'n_bins': n_bins,
        'reid_top1_mean': s_p1['mean'], 'reid_top1_std': s_p1['std'],
        'reid_top5_mean': s_p5['mean'], 'reid_top5_std': s_p5['std'],
        'per_fold_reid_top1': s_p1['per_fold'], 'per_fold_reid_top5': s_p5['per_fold'],
        'quant_reid_top1_mean': s_q1['mean'], 'quant_reid_top1_std': s_q1['std'],
        'quant_reid_top5_mean': s_q5['mean'], 'quant_reid_top5_std': s_q5['std'],
        'deepscan_auc_mean': s_auc['mean'], 'deepscan_auc_std': s_auc['std'],
        'per_fold_deepscan_auc': s_auc['per_fold'],
    }
    print(f"  ReID Top-1 p-values: {s_p1['mean']:.4f} ± {s_p1['std']:.4f}  "
          f"quantised raw: {s_q1['mean']:.4f} ± {s_q1['std']:.4f}")
    print(f"  DeepScan AUC (Method A): {s_auc['mean']:.4f} ± {s_auc['std']:.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────
out_path = osp.join(C.RESULTS_DIR, 'privacy_bins_sweep.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*55}")
print(f"Results saved: {out_path}")
print("Run plot_privacy_utility_v3.py to generate the figure.")
print("DONE.")
