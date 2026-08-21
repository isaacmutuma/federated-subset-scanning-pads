"""
run_deepscan.py
---------------
Phase 4: Histogram-based subset scanning on PatchTST activations.

Three methods:
  A) Simple mean p-value — 1 - mean(p) per window, then mean per subject
  B) Per-window FGSS    — fgss_individ_for_nets per window, then mean per subject
  C) Group FGSS         — fgss_for_nets on ALL windows from one subject (SubsetGAN)

Method C follows Cintas et al. SubsetGAN (2021):
  pvalues must be shape (n_images, n_nodes, 2) — [lower, upper] p-value range
  For a single p-value p, use [p, p] as the range.

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
        print("ART not available.")

ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"
os.makedirs(C.RESULTS_DIR, exist_ok=True)

N_BINS   = 100
A_MAX    = 0.5
RESTARTS = 5


# ── Histogram p-value computation ─────────────────────────────────────────────
def compute_pvalues_directed(hc_acts, test_acts, direction_mask, n_bins=N_BINS):
    """
    Per neuron: build HC histogram, compute directed one-tailed p-values.
    Returns p_matrix: (n_test, n_neurons) — low p = anomalous.
    """
    n_test, n_neurons = test_acts.shape
    p_matrix = np.zeros((n_test, n_neurons), dtype=np.float64)
    for j in range(n_neurons):
        counts, edges = np.histogram(hc_acts[:, j], bins=n_bins, density=False)
        counts  = counts.astype(float) + 1e-6
        probs   = counts / counts.sum()
        cum     = np.cumsum(probs)
        bin_idx = np.clip(np.searchsorted(edges[1:], test_acts[:, j], side='right'),
                          0, len(probs) - 1)
        lower   = np.where(bin_idx > 0, cum[bin_idx - 1], 0.0)
        if direction_mask[j]:
            p_matrix[:, j] = np.clip(1.0 - lower, 1e-6, 1.0)
        else:
            p_matrix[:, j] = np.clip(lower + probs[bin_idx], 1e-6, 1.0)
    return p_matrix


# ── Safe scorers ──────────────────────────────────────────────────────────────
def safe_individ_score(pvals_1d, a_max=A_MAX, score_fn=None):
    """Score one window with fgss_individ_for_nets. Returns 0 on failure."""
    try:
        score, _, _, _ = Scanner.fgss_individ_for_nets(
            pvalues=pvals_1d.reshape(1, -1).astype(np.float64),
            a_max=a_max, score_function=score_fn)
        return float(score)
    except Exception:
        return 0.0


def safe_group_score(p_subj_2d, a_max=A_MAX, restarts=RESTARTS, score_fn=None):
    """
    Group scan for one subject.
    p_subj_2d: shape (n_windows, n_neurons)
    ART fgss_for_nets requires shape (n_windows, n_neurons, 2).
    We use [p, p] as the p-value range (lower=upper=p).
    Returns (best_score, n_anomalous_windows, n_anomalous_nodes).
    """
    try:
        # Expand to 3D: (n_windows, n_neurons, 2)
        p_3d = np.stack([p_subj_2d, p_subj_2d], axis=2).astype(np.float64)
        best_score, image_sub, node_sub, optimal_alpha = Scanner.fgss_for_nets(
            pvalues=p_3d,
            a_max=a_max,
            restarts=restarts,
            score_function=score_fn,
        )
        return float(best_score), len(image_sub), len(node_sub)
    except Exception:
        return 0.0, 0, 0


# ── Main loop ─────────────────────────────────────────────────────────────────
RESULTS_FILE = osp.join(C.RESULTS_DIR, 'deepscan_results.json')
fold_metrics_mean    = []
fold_metrics_individ = []
fold_metrics_group   = []

for fold_idx in range(C.N_SPLITS):
    k = fold_idx + 1
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")

    hc_acts   = np.load(osp.join(ACT_DIR, f'activations_fold{k}_hc_train.npy'))
    val_pd    = np.load(osp.join(ACT_DIR, f'activations_fold{k}_val_pd.npy'))
    test_acts = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test.npy'))
    test_lab  = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test_labels.npy'))
    test_subj = np.load(osp.join(ACT_DIR, f'activations_fold{k}_test_subj.npy'))

    print(f"HC background: {hc_acts.shape}  Val PD: {val_pd.shape}")
    print(f"Test: {test_acts.shape}  HC={(test_lab==0).sum()}  PD={(test_lab==1).sum()}")

    # Direction mask from val PD — no test leakage
    direction_mask = val_pd.mean(0) > hc_acts.mean(0)
    print(f"Direction: {direction_mask.sum()} upper  {(~direction_mask).sum()} lower")

    # P-value matrix: (n_test_windows, n_neurons)
    print("Computing directed p-values...")
    p_matrix = compute_pvalues_directed(hc_acts, test_acts, direction_mask, N_BINS)
    print(f"  HC p mean: {p_matrix[test_lab==0].mean():.4f}  "
          f"PD p mean: {p_matrix[test_lab==1].mean():.4f}")

    unique_subj = np.unique(test_subj)
    subj_true   = np.array([test_lab[test_subj==s][0] for s in unique_subj])

    # ── Method A: simple mean p-value ─────────────────────────────────────────
    win_scores_mean = 1.0 - p_matrix.mean(axis=1)
    subj_mean = np.array([win_scores_mean[test_subj==s].mean() for s in unique_subj])
    metrics_mean = compute_metrics(subj_true, subj_mean)
    fold_metrics_mean.append(metrics_mean)
    print("A) Mean p-value — ", end="")
    print_fold_results(fold_idx, metrics_mean)

    if ART_AVAILABLE:
        score_fn = ScoringFunctions.get_score_bj_fast

        # ── Method B: per-window FGSS, then average per subject ───────────────
        print("B) Per-window fgss_individ_for_nets...")
        win_scores_individ = np.array([
            safe_individ_score(p_matrix[i], a_max=A_MAX, score_fn=score_fn)
            for i in range(len(p_matrix))
        ])
        subj_individ = np.array([win_scores_individ[test_subj==s].mean()
                                  for s in unique_subj])
        metrics_individ = compute_metrics(subj_true, subj_individ)
        fold_metrics_individ.append(metrics_individ)
        print("B) Per-window FGSS — ", end="")
        print_fold_results(fold_idx, metrics_individ)

        # ── Method C: group FGSS per subject (SubsetGAN) ─────────────────────
        # pvalues shape must be (n_windows, n_neurons, 2) — [p, p] range
        print("C) Group fgss_for_nets per subject (SubsetGAN approach)...")
        subj_group_scores = []
        for s_idx, s in enumerate(unique_subj):
            p_subj = p_matrix[test_subj == s]   # (n_windows_s, 768)
            score, n_img, n_node = safe_group_score(
                p_subj, a_max=A_MAX, restarts=RESTARTS, score_fn=score_fn)
            subj_group_scores.append(score)
            if s_idx % 15 == 0:
                label = "PD" if subj_true[s_idx] == 1 else "HC"
                print(f"  Subject {s} ({label}): score={score:.1f}  "
                      f"anom_windows={n_img}/{p_subj.shape[0]}  nodes={n_node}")

        subj_group = np.array(subj_group_scores)
        print(f"  HC mean: {subj_group[subj_true==0].mean():.1f}  "
              f"PD mean: {subj_group[subj_true==1].mean():.1f}")
        metrics_group = compute_metrics(subj_true, subj_group)
        fold_metrics_group.append(metrics_group)
        print("C) Group FGSS — ", end="")
        print_fold_results(fold_idx, metrics_group)

    # Save after each fold
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'fold_metrics_mean':    fold_metrics_mean,
                   'fold_metrics_individ': fold_metrics_individ,
                   'fold_metrics_group':   fold_metrics_group,
                   'folds_complete': k,
                   'direction_source': 'val_pd (no test leakage)',
                   'n_bins': N_BINS, 'a_max': A_MAX,
                   'restarts': RESTARTS}, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("A) Simple mean p-value")
agg_mean = aggregate_fold_metrics(fold_metrics_mean)
print_summary(agg_mean)

if ART_AVAILABLE:
    print(f"\n{'='*55}")
    print("B) Per-window FGSS, averaged per subject")
    agg_individ = aggregate_fold_metrics(fold_metrics_individ)
    print_summary(agg_individ)

    print(f"\n{'='*55}")
    print("C) Group FGSS per subject (SubsetGAN — pvalues shape n_windows×n_neurons×2)")
    agg_group = aggregate_fold_metrics(fold_metrics_group)
    print_summary(agg_group)
else:
    agg_individ = {}
    agg_group   = {}

with open(RESULTS_FILE, 'w') as f:
    json.dump({'fold_metrics_mean':    fold_metrics_mean,
               'aggregate_mean':       agg_mean,
               'fold_metrics_individ': fold_metrics_individ,
               'aggregate_individ':    agg_individ,
               'fold_metrics_group':   fold_metrics_group,
               'aggregate_group':      agg_group,
               'evaluation':           'subject-level',
               'direction_source':     'val_pd (no test leakage)',
               'n_bins': N_BINS, 'a_max': A_MAX,
               'restarts': RESTARTS}, f, indent=2)

print(f"\nResults: {RESULTS_FILE}")
print("DONE.")
