"""
privacy_analysis.py
-------------------
Privacy analysis: re-identification attack on raw activations vs histogram p-values.

Measures how well an adversary can identify which specific HC subject a window
came from, given either:
  (A) Raw 128-dim activation vectors (Layer 2, mean over channels+patches)
  (B) 128-dim p-value vectors (histogram transformation of the same activations)

If raw activations allow higher re-identification accuracy than p-values,
the histogram transformation provides a meaningful privacy benefit.

Setup:
  - Only HC training subjects used (we know ground-truth subject IDs)
  - Per fold: 50 HC train subjects × 110 windows each
  - Train classifier on 80% of windows per subject
  - Test on held-out 20% of windows per subject
  - Classifier: logistic regression (50-class, one class per HC subject)
  - Metric: top-1 accuracy and top-5 accuracy
  - Baseline: random chance = 1/50 = 2%

Run:
    python scripts/privacy_analysis.py
"""

import sys, os, os.path as osp, json, warnings, pickle
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"
os.makedirs(C.RESULTS_DIR, exist_ok=True)

N_BINS    = 10
LAYER     = 'layer_2'   # confirmed best layer
TEST_FRAC = 0.2         # hold out 20% of windows per subject for testing
RANDOM_STATE = 42

print(f"Privacy analysis — re-identification attack")
print(f"Layer: {LAYER}  |  Test fraction: {TEST_FRAC}")
print(f"Baseline (random chance, 50 subjects): {1/50:.1%}")


def compute_pvalues(hc_acts_ref, query_acts, n_bins=N_BINS):
    """
    Build per-dimension histograms from hc_acts_ref.
    Compute two-tailed p-values for each query window.
    (Two-tailed for privacy — we flag any extreme value, not directional)
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
        # Two-tailed: minimum of both tails
        p_matrix[:, j] = np.minimum(p_upper, p_lower)

    return p_matrix


def top_k_accuracy(y_true, probs, k=5):
    """Fraction of samples where true label is in top-k predictions."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)


# ── Run across all 5 folds ────────────────────────────────────────────────────
fold_results = []

for fold_idx in range(C.N_SPLITS):
    k = fold_idx + 1
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")

    # Load HC train activations — shape (n_hc_train_windows, 128)
    hc_acts = np.load(osp.join(ACT_DIR,
                                f'activations_fold{k}_{LAYER}_hc_train.npy'))

    # Load fold splits to get subject IDs for HC train windows
    with open(osp.join(C.OUTPUT_DIR, 'fold_splits.pkl'), 'rb') as f:
        folds = pickle.load(f)

    windows     = np.load(osp.join(C.OUTPUT_DIR, 'windows.npy'))
    labels      = np.load(osp.join(C.OUTPUT_DIR, 'labels.npy'))
    subject_ids = np.load(osp.join(C.OUTPUT_DIR, 'subject_ids.npy'))

    fold      = folds[fold_idx]
    train_mask = np.isin(subject_ids, fold['train_subjects'])
    train_lab  = labels[train_mask]
    train_subj = subject_ids[train_mask]

    # HC train windows only
    hc_mask    = train_lab == 0
    hc_subj    = train_subj[hc_mask]   # subject ID per HC window

    assert len(hc_acts) == hc_mask.sum(), \
        f"Mismatch: {len(hc_acts)} activations vs {hc_mask.sum()} HC windows"

    # Unique HC subjects
    unique_hc_subj = np.unique(hc_subj)
    n_subjects     = len(unique_hc_subj)
    subj_to_idx    = {s: i for i, s in enumerate(unique_hc_subj)}
    subj_labels    = np.array([subj_to_idx[s] for s in hc_subj])

    print(f"HC subjects: {n_subjects}  Windows per subject: "
          f"{[(hc_subj==s).sum() for s in unique_hc_subj[:3]]}")

    # Train/test split — 80/20 per subject
    np.random.seed(RANDOM_STATE)
    train_idx, test_idx = [], []
    for s in unique_hc_subj:
        idx      = np.where(hc_subj == s)[0]
        n_test   = max(1, int(len(idx) * TEST_FRAC))
        test_sel = np.random.choice(idx, n_test, replace=False)
        train_sel = np.setdiff1d(idx, test_sel)
        train_idx.extend(train_sel)
        test_idx.extend(test_sel)

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    X_raw_train = hc_acts[train_idx]
    X_raw_test  = hc_acts[test_idx]
    y_train     = subj_labels[train_idx]
    y_test      = subj_labels[test_idx]

    print(f"Train windows: {len(train_idx):,}  Test windows: {len(test_idx):,}")

    # Build HC histograms from training windows only (no leakage)
    X_pval_train = compute_pvalues(X_raw_train, X_raw_train, N_BINS)
    X_pval_test  = compute_pvalues(X_raw_train, X_raw_test,  N_BINS)

    # ── Run A: raw activations ─────────────────────────────────────────────────
    scaler_raw  = StandardScaler()
    Xtr_raw     = scaler_raw.fit_transform(X_raw_train)
    Xte_raw     = scaler_raw.transform(X_raw_test)

    clf_raw = LogisticRegression(max_iter=1000, class_weight='balanced',
                                  C=1.0, random_state=RANDOM_STATE,
                                  multi_class='multinomial', solver='lbfgs')
    clf_raw.fit(Xtr_raw, y_train)
    probs_raw = clf_raw.predict_proba(Xte_raw)
    acc_raw_1  = accuracy_score(y_test, probs_raw.argmax(axis=1))
    acc_raw_5  = top_k_accuracy(y_test, probs_raw, k=5)

    # ── Run B: histogram p-values ──────────────────────────────────────────────
    scaler_pval = StandardScaler()
    Xtr_pval    = scaler_pval.fit_transform(X_pval_train)
    Xte_pval    = scaler_pval.transform(X_pval_test)

    clf_pval = LogisticRegression(max_iter=1000, class_weight='balanced',
                                   C=1.0, random_state=RANDOM_STATE,
                                   multi_class='multinomial', solver='lbfgs')
    clf_pval.fit(Xtr_pval, y_train)
    probs_pval = clf_pval.predict_proba(Xte_pval)
    acc_pval_1  = accuracy_score(y_test, probs_pval.argmax(axis=1))
    acc_pval_5  = top_k_accuracy(y_test, probs_pval, k=5)

    print(f"\nRe-identification accuracy (baseline chance={1/n_subjects:.1%}):")
    print(f"  Raw activations  — Top-1: {acc_raw_1:.4f}  Top-5: {acc_raw_5:.4f}")
    print(f"  Histogram pvals  — Top-1: {acc_pval_1:.4f}  Top-5: {acc_pval_5:.4f}")
    print(f"  Privacy gain     — Top-1: {acc_raw_1 - acc_pval_1:+.4f}  "
          f"Top-5: {acc_raw_5 - acc_pval_5:+.4f}")

    fold_results.append({
        'fold': k,
        'n_subjects': n_subjects,
        'chance_level': 1 / n_subjects,
        'raw_top1':  float(acc_raw_1),
        'raw_top5':  float(acc_raw_5),
        'pval_top1': float(acc_pval_1),
        'pval_top5': float(acc_pval_5),
        'privacy_gain_top1': float(acc_raw_1 - acc_pval_1),
        'privacy_gain_top5': float(acc_raw_5 - acc_pval_5),
    })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("PRIVACY ANALYSIS SUMMARY (5-fold mean)")
print('='*55)

chance      = np.mean([r['chance_level'] for r in fold_results])
raw_1_mean  = np.mean([r['raw_top1']  for r in fold_results])
raw_1_std   = np.std( [r['raw_top1']  for r in fold_results])
raw_5_mean  = np.mean([r['raw_top5']  for r in fold_results])
raw_5_std   = np.std( [r['raw_top5']  for r in fold_results])
pval_1_mean = np.mean([r['pval_top1'] for r in fold_results])
pval_1_std  = np.std( [r['pval_top1'] for r in fold_results])
pval_5_mean = np.mean([r['pval_top5'] for r in fold_results])
pval_5_std  = np.std( [r['pval_top5'] for r in fold_results])

print(f"Chance level (1/{int(1/chance)} subjects): {chance:.1%}")
print(f"\nRaw activations:  Top-1: {raw_1_mean:.4f} ± {raw_1_std:.4f}  "
      f"Top-5: {raw_5_mean:.4f} ± {raw_5_std:.4f}")
print(f"Histogram pvals:  Top-1: {pval_1_mean:.4f} ± {pval_1_std:.4f}  "
      f"Top-5: {pval_5_mean:.4f} ± {pval_5_std:.4f}")
print(f"Privacy gain:     Top-1: {raw_1_mean - pval_1_mean:+.4f}  "
      f"Top-5: {raw_5_mean - pval_5_mean:+.4f}")

results = {
    'fold_results': fold_results,
    'summary': {
        'chance_level': float(chance),
        'raw_top1_mean':  float(raw_1_mean),  'raw_top1_std':  float(raw_1_std),
        'raw_top5_mean':  float(raw_5_mean),  'raw_top5_std':  float(raw_5_std),
        'pval_top1_mean': float(pval_1_mean), 'pval_top1_std': float(pval_1_std),
        'pval_top5_mean': float(pval_5_mean), 'pval_top5_std': float(pval_5_std),
        'privacy_gain_top1': float(raw_1_mean - pval_1_mean),
        'privacy_gain_top5': float(raw_5_mean - pval_5_mean),
    },
    'layer': LAYER,
    'method': 'logistic regression re-identification attack',
    'n_bins': N_BINS,
    'test_fraction': TEST_FRAC,
}

out_path = osp.join(C.RESULTS_DIR, 'privacy_analysis.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved: {out_path}")
print("DONE.")
