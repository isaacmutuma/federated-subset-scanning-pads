"""
run_deepscan.py
---------------
Phase 4: Histogram-based anomaly scoring on PatchTST activations.

Method:
  1. Load HC train activations → build per-neuron histograms (100 bins)
  2. Estimate direction mask from VAL PD activations (no test leakage)
  3. Compute directed one-tailed p-values for each test window
  4. Anomaly score = 1 - mean(p-values across 768 neurons) per window
  5. Subject-level mean pooling → AUC

Run:
    python scripts/run_deepscan.py
"""

import sys, os, os.path as osp, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

if not osp.exists(C.REPO_DIR):
    os.system(f'git clone https://github.com/isaacmutuma/federated-subset-scanning-pads.git {C.REPO_DIR}')
else:
    os.system(f'git -C {C.REPO_DIR} pull origin main')

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.training.metrics import (compute_metrics, aggregate_fold_metrics,
                                   print_fold_results, print_summary)

try:
    from art.defences.detector.evasion.subsetscanning.scanner import Scanner
    from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions
    ART_AVAILABLE = True
    print("ART subsetscanning loaded.")
except ImportError:
    print("Installing adversarial-robustness-toolbox...")
    os.system('pip install -q adversarial-robustness-toolbox')
    try:
        from art.defences.detector.evasion.subsetscanning.scanner import Scanner
        from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions
        ART_AVAILABLE = True
    except Exception:
        ART_AVAILABLE = False
        print("ART not available — skipping FGSS scanner.")

ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"
os.makedirs(C.RESULTS_DIR, exist_ok=True)

N_BINS = 100
A_MAX  = 0.5


def compute_pvalues_directed(hc_acts, test_acts, direction_mask, n_bins=N_BINS):
    """
    Per neuron:
      - Build histogram from HC train activations
      - Compute one-tailed p-value for each test window
        Upper tail if direction_mask[j]=True (PD > HC for this neuron)
        Lower tail if direction_mask[j]=False (PD < HC for this neuron)

    Returns p_matrix: (n_test, n_neurons) — low p = anomalous
    """
    n_test, n_neurons = test_acts.shape
    p_matrix = np.zeros((n_test, n_neurons), dtype=np.float64)

    for j in range(n_neurons):
        hc_col   = hc_acts[:, j]
        test_col = test_acts[:, j]

        counts, edges = np.histogram(hc_col, bins=n_bins, density=False)
        counts  = counts.astype(float) + 1e-6   # Laplace smoothing
        probs   = counts / counts.sum()
        cum     = np.cumsum(probs)

        bin_idx   = np.searchsorted(edges[1:], test_col, side='right')
        bin_idx   = np.clip(bin_idx, 0, len(probs) - 1)
        lower_cum = np.where(bin_idx > 0, cum[bin_idx - 1], 0.0)

        if direction_mask[j]:
            p_matrix[:, j] = np.clip(1.0 - lower_cum, 1e-6, 1.0)
        else:
            p_matrix[:, j] = np.clip(lower_cum + probs[bin_idx], 1e-6, 1.0)

    return p_matrix


def fgss_score_window(pvals, a_max=A_MAX, score_fn=None):
    try:
        score, _, _, _ = Scanner.fgss_individ_for_nets(
            pvalues=pvals.reshape(1, -1).astype(np.float64),
            a_max=a_max, score_function=score_fn)
        return float(score)
    except (ValueError, Exception):
        return 0.0


RESULTS_FILE = osp.join(C.RESULTS_DIR, 'deepscan_results.json')
fold_metrics_mean = []
fold_metrics_fgss = []

for fold_idx in range(C.N_SPLITS):
    k = fold_idx + 1
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")

    hc_acts   = np.load(osp.join(ACT_DIR, f'activations_fold{k}_hc_train.npy'))
    val_pd    = np.load(osp.join(ACT_DIR, f'activations_fold{k}_val_pd.npy'))
    test_acts = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test.npy'))
    test_lab  = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test_labels.npy'))
    test_subj = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test_subj.npy'))

    print(f"HC background: {hc_acts.shape}")
    print(f"Val PD (direction mask): {val_pd.shape}")
    print(f"Test: {test_acts.shape}  HC={(test_lab==0).sum()}  PD={(test_lab==1).sum()}")

    # Direction mask — estimated from VAL PD, not test (no leakage)
    hc_train_mean  = hc_acts.mean(0)
    val_pd_mean    = val_pd.mean(0)
    direction_mask = val_pd_mean > hc_train_mean
    print(f"Direction: {direction_mask.sum()} upper-tail  "
          f"{(~direction_mask).sum()} lower-tail neurons")

    # P-value matrix
    print("Computing directed p-values...")
    p_matrix = compute_pvalues_directed(hc_acts, test_acts, direction_mask, N_BINS)
    print(f"  HC p-value mean: {p_matrix[test_lab==0].mean():.4f}")
    print(f"  PD p-value mean: {p_matrix[test_lab==1].mean():.4f}")

    # ── Primary: simple mean p-value ─────────────────────────────────────────
    window_scores = 1.0 - p_matrix.mean(axis=1)

    unique_subj = np.unique(test_subj)
    subj_scores = np.array([window_scores[test_subj==s].mean() for s in unique_subj])
    subj_true   = np.array([test_lab[test_subj==s][0] for s in unique_subj])

    print(f"Subject-level: {len(unique_subj)} subjects  "
          f"HC={(subj_true==0).sum()}  PD={(subj_true==1).sum()}")

    metrics_mean = compute_metrics(subj_true, subj_scores)
    fold_metrics_mean.append(metrics_mean)
    print("Mean p-value — ", end="")
    print_fold_results(fold_idx, metrics_mean)

    # ── Secondary: ART FGSS scanner ──────────────────────────────────────────
    if ART_AVAILABLE:
        print("Running ART fgss_individ_for_nets...")
        score_fn = ScoringFunctions.get_score_bj_fast
        window_scores_fgss = np.array([
            fgss_score_window(p_matrix[i], a_max=A_MAX, score_fn=score_fn)
            for i in range(len(p_matrix))
        ])
        subj_fgss    = np.array([window_scores_fgss[test_subj==s].mean() for s in unique_subj])
        metrics_fgss = compute_metrics(subj_true, subj_fgss)
        fold_metrics_fgss.append(metrics_fgss)
        print("FGSS scanner — ", end="")
        print_fold_results(fold_idx, metrics_fgss)

    with open(RESULTS_FILE, 'w') as f:
        json.dump({'fold_metrics_mean_pvalue': fold_metrics_mean,
                   'fold_metrics_fgss':        fold_metrics_fgss,
                   'folds_complete':            k,
                   'direction_source':          'val_pd (no test leakage)',
                   'n_bins': N_BINS, 'a_max': A_MAX}, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("PRIMARY: Simple mean p-value")
agg_mean = aggregate_fold_metrics(fold_metrics_mean)
print_summary(agg_mean)

if ART_AVAILABLE and fold_metrics_fgss:
    print(f"\n{'='*55}")
    print("SECONDARY: ART FGSS Berk-Jones")
    agg_fgss = aggregate_fold_metrics(fold_metrics_fgss)
    print_summary(agg_fgss)
else:
    agg_fgss = {}

with open(RESULTS_FILE, 'w') as f:
    json.dump({'fold_metrics_mean_pvalue': fold_metrics_mean,
               'aggregate_mean_pvalue':    agg_mean,
               'fold_metrics_fgss':        fold_metrics_fgss,
               'aggregate_fgss':           agg_fgss,
               'evaluation':               'subject-level mean pooling',
               'direction_source':         'val_pd (no test leakage)',
               'n_bins': N_BINS, 'a_max': A_MAX}, f, indent=2)

print(f"\nResults: {RESULTS_FILE}")
print("DONE.")
