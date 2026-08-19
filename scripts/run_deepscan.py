"""
run_deepscan.py
---------------
Phase 4: Histogram-based subset scanning using ART library.

Follows Celia Cintas / Akumu et al. pipeline:
  [0] ART subsetscanning: github.com/Trusted-AI/adversarial-robustness-toolbox
  [1] IBM personas repo:  github.com/IBM/personas-llms-analysis

For each fold:
  1. Load HC train activations → build per-neuron histograms
  2. Compute p-value matrix for test windows against HC histograms
  3. Score each test window using ART fgss_individ_for_nets (O(N log N), exact)
  4. Subject-level mean pooling → AUC

Run:
    python scripts/run_deepscan.py
"""

import sys, os, os.path as osp, json, warnings
warnings.filterwarnings('ignore')

import numpy as np

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

# ── Install ART if needed ─────────────────────────────────────────────────────
try:
    from art.defences.detector.evasion.subsetscanning.scanner import Scanner
    from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions
    print("ART subsetscanning loaded.")
except ImportError:
    print("Installing adversarial-robustness-toolbox...")
    os.system('pip install -q adversarial-robustness-toolbox')
    from art.defences.detector.evasion.subsetscanning.scanner import Scanner
    from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions
    print("ART subsetscanning loaded.")

ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"
os.makedirs(C.RESULTS_DIR, exist_ok=True)

N_BINS  = 20
A_MAX   = 0.5     # alpha threshold for FGSS (standard default)

# ── Histogram p-value computation (Akumu et al. method) ──────────────────────
def compute_pvalues_histogram(hc_acts, test_acts, n_bins=N_BINS):
    """
    For each neuron j:
      1. Build histogram over HC background activations
      2. Compute p-value = P(X >= x | HC histogram) for each test window

    Returns p_matrix: shape (n_test, n_neurons)
    Low p-value = anomalous (activation unlikely under HC distribution)
    """
    n_test, n_neurons = test_acts.shape
    p_matrix = np.zeros((n_test, n_neurons), dtype=np.float32)

    for j in range(n_neurons):
        hc_col   = hc_acts[:, j]
        test_col = test_acts[:, j]

        counts, edges = np.histogram(hc_col, bins=n_bins, density=False)
        counts = counts.astype(float) + 1e-6   # Laplace smoothing
        probs  = counts / counts.sum()
        cum    = np.cumsum(probs)               # empirical CDF

        bin_idx        = np.searchsorted(edges[1:], test_col, side='right')
        bin_idx        = np.clip(bin_idx, 0, len(probs) - 1)
        lower_cum      = np.where(bin_idx > 0, cum[bin_idx - 1], 0.0)
        p_matrix[:, j] = np.clip(1.0 - lower_cum, 1e-6, 1.0)

    return p_matrix


# ── Score individual windows using ART fgss_individ_for_nets ─────────────────
def score_windows(p_matrix, a_max=A_MAX, batch_size=200):
    """
    Score each test window using ART's exact individual scanner.
    fgss_individ_for_nets expects pvalues shape (1, n_neurons) per window.

    Returns scores array: shape (n_test,)
    Higher score = more anomalous = more likely PD.
    """
    n_windows  = p_matrix.shape[0]
    scores     = np.zeros(n_windows, dtype=np.float32)
    score_fn   = ScoringFunctions.get_score_bj_fast  # standard Berk-Jones

    for i in range(0, n_windows, batch_size):
        batch = p_matrix[i:i + batch_size]
        for j, pvals in enumerate(batch):
            # fgss_individ_for_nets: pvalues shape (1, n_neurons)
            best_score, _, _, _ = Scanner.fgss_individ_for_nets(
                pvalues=pvals.reshape(1, -1).astype(np.float64),
                a_max=a_max,
                score_function=score_fn,
            )
            scores[i + j] = best_score
        if (i // batch_size) % 5 == 0:
            print(f"  Scored {min(i + batch_size, n_windows)}/{n_windows} windows...")

    return scores


# ── Main loop ─────────────────────────────────────────────────────────────────
RESULTS_FILE = osp.join(C.RESULTS_DIR, 'deepscan_results.json')
fold_metrics = []

for fold_idx in range(C.N_SPLITS):
    k = fold_idx + 1
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")

    hc_acts   = np.load(osp.join(ACT_DIR, f'activations_fold{k}_hc_train.npy'))
    test_acts = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test.npy'))
    test_lab  = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test_labels.npy'))
    test_subj = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test_subj.npy'))

    print(f"HC background: {hc_acts.shape}")
    print(f"Test: {test_acts.shape}  HC={(test_lab==0).sum()}  PD={(test_lab==1).sum()}")

    # Step 1 — Compute p-value matrix from HC histograms
    print("Computing p-values from HC histograms...")
    p_matrix = compute_pvalues_histogram(hc_acts, test_acts, n_bins=N_BINS)
    print(f"P-matrix shape: {p_matrix.shape}")
    print(f"  HC windows  p-value mean: {p_matrix[test_lab==0].mean():.4f}")
    print(f"  PD windows  p-value mean: {p_matrix[test_lab==1].mean():.4f}")

    # Step 2 — ART FGSS individual scoring per window
    print("Running ART fgss_individ_for_nets per window...")
    window_scores = score_windows(p_matrix, a_max=A_MAX)
    print(f"Scan scores:")
    print(f"  HC mean: {window_scores[test_lab==0].mean():.4f}")
    print(f"  PD mean: {window_scores[test_lab==1].mean():.4f}")

    # Step 3 — Subject-level aggregation (mean pool)
    unique_test_subj = np.unique(test_subj)
    subj_scores = np.array([window_scores[test_subj == s].mean()
                             for s in unique_test_subj])
    subj_true   = np.array([test_lab[test_subj == s][0]
                             for s in unique_test_subj])

    print(f"Subject-level: {len(unique_test_subj)} subjects  "
          f"HC={(subj_true==0).sum()}  PD={(subj_true==1).sum()}")

    metrics = compute_metrics(subj_true, subj_scores)
    fold_metrics.append(metrics)
    print_fold_results(fold_idx, metrics)

    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'fold_metrics':   fold_metrics,
            'folds_complete': k,
            'method':         'ART fgss_individ_for_nets (Berk-Jones)',
            'n_bins':         N_BINS,
            'a_max':          A_MAX,
            'n_neurons':      hc_acts.shape[1],
        }, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
agg = aggregate_fold_metrics(fold_metrics)
print_summary(agg)

with open(RESULTS_FILE, 'w') as f:
    json.dump({
        'fold_metrics': fold_metrics,
        'aggregate':    agg,
        'method':       'ART fgss_individ_for_nets (Berk-Jones)',
        'evaluation':   'subject-level mean pooling',
        'n_bins':       N_BINS,
        'a_max':        A_MAX,
    }, f, indent=2)

print(f"\nResults: {RESULTS_FILE}")
print("DONE.")
