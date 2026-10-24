"""
privacy_analysis_bins.py
------------------------
Extended privacy analysis: runs re-identification attack across multiple
histogram bin counts to generate the privacy-utility trade-off curve.

For each bin count, computes:
  - Re-identification accuracy (Top-1, Top-5) on histogram p-values
  - DeepScan Method A AUC (mean p-value, subject-level)

Raw activation re-identification is bin-independent (computed once).

Outputs: results/privacy_bins_sweep.json

Run:
    python scripts/privacy_analysis_bins.py
"""

import sys, os, os.path as osp, json, warnings, pickle
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

ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"
os.makedirs(C.RESULTS_DIR, exist_ok=True)

BIN_COUNTS   = [5, 10, 20, 50, 100]
LAYER        = 'layer_2'
TEST_FRAC    = 0.2
RANDOM_STATE = 42

print(f"Privacy-utility sweep across bin counts: {BIN_COUNTS}")
print(f"Layer: {LAYER}  |  Test fraction: {TEST_FRAC}")


def compute_pvalues(hc_acts_ref, query_acts, n_bins):
    """
    Build per-dimension histograms from hc_acts_ref.
    Compute two-tailed p-values for each query window.
    Returns p_matrix: (n_query, 128)
    """
    n_query, n_dims = query_acts.shape
    p_matrix = np.zeros((n_query, n_dims), dtype=np.float64)

    for j in range(n_dims):
        counts, edges = np.histogram(hc_acts_ref[:, j], bins=n_bins, density=False)
        counts  = counts.astype(float) + 1e-6
        probs   = counts / counts.sum()
        cum     = np.cumsum(probs)

        bin_idx = np.clip(np.searchsorted(edges[1:], query_acts[:, j], side='right'),
                          0, len(probs) - 1)
        lower   = np.where(bin_idx > 0, cum[bin_idx - 1], 0.0)
        p_upper = np.clip(1.0 - lower, 1e-6, 1.0)
        p_lower = np.clip(lower + probs[bin_idx], 1e-6, 1.0)
        p_matrix[:, j] = np.minimum(p_upper, p_lower)

    return p_matrix


def top_k_accuracy(y_true, probs, k=5):
    """Fraction of samples where true label is in top-k predictions."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)


def deepscan_method_a(hc_pvals, pd_pvals, hc_subjects, pd_subjects):
    """
    Method A: mean p-value per subject.
    Lower mean p-value = more anomalous.
    Returns subject-level AUC (HC=0 vs PD=1).
    """
    # HC subject scores
    hc_scores = []
    for s in np.unique(hc_subjects):
        mask = hc_subjects == s
        mean_p = hc_pvals[mask].mean()
        hc_scores.append(mean_p)

    # PD subject scores
    pd_scores = []
    for s in np.unique(pd_subjects):
        mask = pd_subjects == s
        mean_p = pd_pvals[mask].mean()
        pd_scores.append(mean_p)

    # AUC: lower mean p-value = anomalous (PD), so negate for AUC
    scores = np.array(hc_scores + pd_scores)
    labels = np.array([0]*len(hc_scores) + [1]*len(pd_scores))
    # Negate because lower p = more anomalous = PD
    auc = roc_auc_score(labels, -scores)
    return auc


# ── Load shared data ──────────────────────────────────────────────────────────
with open(osp.join(C.OUTPUT_DIR, 'fold_splits.pkl'), 'rb') as f:
    folds = pickle.load(f)

windows     = np.load(osp.join(C.OUTPUT_DIR, 'windows.npy'))
labels      = np.load(osp.join(C.OUTPUT_DIR, 'labels.npy'))
subject_ids = np.load(osp.join(C.OUTPUT_DIR, 'subject_ids.npy'))


# ── Main sweep ────────────────────────────────────────────────────────────────
all_results = {
    'bin_counts': BIN_COUNTS,
    'layer': LAYER,
    'per_bin': {},
    'raw_reid': {},  # bin-independent
}

# First pass: raw activation re-identification (doesn't depend on bins)
raw_top1_folds = []
raw_top5_folds = []

for fold_idx in range(C.N_SPLITS):
    k = fold_idx + 1
    hc_acts = np.load(osp.join(ACT_DIR, f'activations_fold{k}_{LAYER}_hc_train.npy'))

    fold       = folds[fold_idx]
    train_mask = np.isin(subject_ids, fold['train_subjects'])
    train_lab  = labels[train_mask]
    train_subj = subject_ids[train_mask]

    hc_mask    = train_lab == 0
    hc_subj    = train_subj[hc_mask]
    unique_hc  = np.unique(hc_subj)
    subj_to_idx = {s: i for i, s in enumerate(unique_hc)}
    subj_labels = np.array([subj_to_idx[s] for s in hc_subj])

    np.random.seed(RANDOM_STATE)
    train_idx, test_idx = [], []
    for s in unique_hc:
        idx = np.where(hc_subj == s)[0]
        n_test = max(1, int(len(idx) * TEST_FRAC))
        test_sel = np.random.choice(idx, n_test, replace=False)
        train_idx.extend(np.setdiff1d(idx, test_sel))
        test_idx.extend(test_sel)

    train_idx, test_idx = np.array(train_idx), np.array(test_idx)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(hc_acts[train_idx])
    Xte = scaler.transform(hc_acts[test_idx])
    y_tr, y_te = subj_labels[train_idx], subj_labels[test_idx]

    clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                             C=1.0, random_state=RANDOM_STATE,
                             multi_class='multinomial', solver='lbfgs')
    clf.fit(Xtr, y_tr)
    probs = clf.predict_proba(Xte)
    raw_top1_folds.append(accuracy_score(y_te, probs.argmax(axis=1)))
    raw_top5_folds.append(top_k_accuracy(y_te, probs, k=5))

all_results['raw_reid'] = {
    'top1_mean': float(np.mean(raw_top1_folds)),
    'top1_std':  float(np.std(raw_top1_folds)),
    'top5_mean': float(np.mean(raw_top5_folds)),
    'top5_std':  float(np.std(raw_top5_folds)),
    'per_fold_top1': [float(x) for x in raw_top1_folds],
    'per_fold_top5': [float(x) for x in raw_top5_folds],
}

print(f"\nRaw activations (bin-independent):")
print(f"  Top-1: {np.mean(raw_top1_folds):.4f} ± {np.std(raw_top1_folds):.4f}")
print(f"  Top-5: {np.mean(raw_top5_folds):.4f} ± {np.std(raw_top5_folds):.4f}")


# Second pass: per-bin analysis
for n_bins in BIN_COUNTS:
    print(f"\n{'='*55}")
    print(f"N_BINS = {n_bins}")
    print(f"{'='*55}")

    pval_top1_folds = []
    pval_top5_folds = []
    deepscan_auc_folds = []

    for fold_idx in range(C.N_SPLITS):
        k = fold_idx + 1

        # Load activations
        hc_train_acts = np.load(osp.join(ACT_DIR,
                                         f'activations_fold{k}_{LAYER}_hc_train.npy'))

        fold       = folds[fold_idx]
        train_mask = np.isin(subject_ids, fold['train_subjects'])
        test_mask  = np.isin(subject_ids, fold['test_subjects'])
        train_lab  = labels[train_mask]
        train_subj = subject_ids[train_mask]
        test_lab   = labels[test_mask]
        test_subj  = subject_ids[test_mask]

        # HC train windows for re-identification
        hc_mask     = train_lab == 0
        hc_subj     = train_subj[hc_mask]
        unique_hc   = np.unique(hc_subj)
        subj_to_idx = {s: i for i, s in enumerate(unique_hc)}
        subj_labels = np.array([subj_to_idx[s] for s in hc_subj])

        # Re-identification: train/test split within HC train windows
        np.random.seed(RANDOM_STATE)
        reid_train_idx, reid_test_idx = [], []
        for s in unique_hc:
            idx = np.where(hc_subj == s)[0]
            n_test = max(1, int(len(idx) * TEST_FRAC))
            test_sel = np.random.choice(idx, n_test, replace=False)
            reid_train_idx.extend(np.setdiff1d(idx, test_sel))
            reid_test_idx.extend(test_sel)

        reid_train_idx = np.array(reid_train_idx)
        reid_test_idx  = np.array(reid_test_idx)

        # Compute p-values with current n_bins
        X_pval_train = compute_pvalues(hc_train_acts[reid_train_idx],
                                        hc_train_acts[reid_train_idx], n_bins)
        X_pval_test  = compute_pvalues(hc_train_acts[reid_train_idx],
                                        hc_train_acts[reid_test_idx],  n_bins)

        # Re-identification on p-values
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_pval_train)
        Xte = scaler.transform(X_pval_test)
        y_tr = subj_labels[reid_train_idx]
        y_te = subj_labels[reid_test_idx]

        clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                                 C=1.0, random_state=RANDOM_STATE,
                                 multi_class='multinomial', solver='lbfgs')
        clf.fit(Xtr, y_tr)
        probs = clf.predict_proba(Xte)
        pval_top1_folds.append(accuracy_score(y_te, probs.argmax(axis=1)))
        pval_top5_folds.append(top_k_accuracy(y_te, probs, k=5))

        # DeepScan Method A: need HC test + PD test activations
        # HC test activations
        hc_test_path = osp.join(ACT_DIR, f'activations_fold{k}_{LAYER}_val_hc.npy')
        pd_test_path = osp.join(ACT_DIR, f'activations_fold{k}_{LAYER}_val_pd.npy')
        
        if osp.exists(hc_test_path) and osp.exists(pd_test_path):
            hc_test_acts = np.load(hc_test_path)
            pd_test_acts = np.load(pd_test_path)

            # p-values against HC train reference
            hc_test_pvals = compute_pvalues(hc_train_acts, hc_test_acts, n_bins)
            pd_test_pvals = compute_pvalues(hc_train_acts, pd_test_acts, n_bins)

            # Method A: mean p-value per subject
            # val_hc and val_pd are already split by class
            # Use test_subj to get subject IDs, split by test_labels
            test_subj_path   = osp.join(ACT_DIR, f'activations_fold{k}_test_subj.npy')
            test_labels_path = osp.join(ACT_DIR, f'activations_fold{k}_test_labels.npy')
            t_subj   = np.load(test_subj_path)
            t_labels = np.load(test_labels_path)
            hc_subj_ids = t_subj[t_labels == 0]
            pd_subj_ids = t_subj[t_labels == 1]

            # Compute per-subject mean p-value scores
            hc_scores = []
            for s in np.unique(hc_subj_ids):
                mask = hc_subj_ids == s
                hc_scores.append(hc_test_pvals[mask].mean())

            pd_scores = []
            for s in np.unique(pd_subj_ids):
                mask = pd_subj_ids == s
                pd_scores.append(pd_test_pvals[mask].mean())

            scores = np.array(hc_scores + pd_scores)
            auc_labels = np.array([0]*len(hc_scores) + [1]*len(pd_scores))
            auc = roc_auc_score(auc_labels, -scores)
            deepscan_auc_folds.append(auc)
        else:
            print(f"  [Fold {k}] HC/PD test activations not found, skipping DeepScan")

    bin_result = {
        'n_bins': n_bins,
        'reid_top1_mean': float(np.mean(pval_top1_folds)),
        'reid_top1_std':  float(np.std(pval_top1_folds)),
        'reid_top5_mean': float(np.mean(pval_top5_folds)),
        'reid_top5_std':  float(np.std(pval_top5_folds)),
        'per_fold_reid_top1': [float(x) for x in pval_top1_folds],
        'per_fold_reid_top5': [float(x) for x in pval_top5_folds],
    }

    if deepscan_auc_folds:
        bin_result['deepscan_auc_mean'] = float(np.mean(deepscan_auc_folds))
        bin_result['deepscan_auc_std']  = float(np.std(deepscan_auc_folds))
        bin_result['per_fold_deepscan_auc'] = [float(x) for x in deepscan_auc_folds]

    all_results['per_bin'][str(n_bins)] = bin_result

    print(f"  ReID Top-1: {np.mean(pval_top1_folds):.4f} ± {np.std(pval_top1_folds):.4f}")
    if deepscan_auc_folds:
        print(f"  DeepScan AUC: {np.mean(deepscan_auc_folds):.4f} ± {np.std(deepscan_auc_folds):.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────
out_path = osp.join(C.RESULTS_DIR, 'privacy_bins_sweep.json')
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*55}")
print(f"Results saved: {out_path}")
print("Run plot_privacy_utility.py to generate the figure.")
print("DONE.")
