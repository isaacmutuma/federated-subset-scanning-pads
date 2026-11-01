"""
control_reference_counts.py
---------------------------
Control: how much of Method A's detection comes from the HC reference COUNTS, and how much from
the quantities the server fixes centrally (bin edges from PADS HC train, tail direction per neuron
from PADS val PD vs HC train)?

Motivation: in privacy_membership_v2.py the eps = 1 Laplace noise (scale 128 x 10 / eps = 1,280)
is about 100 times the reference counts (~13 per bin from 130 capped windows), yet Method A AUC
stayed at 0.863. That suggests the edges and direction, not the counts, carry the signal.

Every condition uses the main pipeline's edges, test set and Method A score (1 - mean p per window,
mean over a subject's windows); only the counts or the direction change:

  real                    : val-HC counts, central direction (= run_deepscan main result)
  flat                    : every count equal (no reference information at all)
  random                  : counts drawn uniformly at random (N_DRAWS draws)
  dp_eps10 / dp_eps1      : 10-window-capped val-HC counts + Laplace noise, as privacy_membership_v2
  real_two_tailed         : real counts, no direction (two-tailed p-values, needs no PD data)
  real_flipped_direction  : real counts, every direction reversed
  real_random_direction   : real counts, random direction per neuron (N_DRAWS draws)
  flat_random_direction   : flat counts, random direction (N_DRAWS draws)

Pooled AUC = subject scores of all 5 folds pooled (355 subjects), bootstrap 95% CI. Paired
bootstrap: real - flat. Reading fixed before the run: the counts carry detection signal beyond
edges + direction only if the real - flat CI lower bound is > 0.

Run:
    python scripts/control_reference_counts.py
"""

import sys, os, os.path as osp, json, subprocess, warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.scanning.histograms import make_global_edges, site_counts, pvalues_from_counts
from src.training.metrics import bootstrap_auc_ci, paired_bootstrap_auc_diff, spec_at_sens

ACT_DIR = os.environ.get('ACT_DIR', osp.join(C.RESULTS_DIR, 'activations'))
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"

LAYER        = os.environ.get('LAYER', 'layer_2')
N_BINS       = 10
ALPHA        = 1.0
BACKGROUND   = 'val_hc'
W_CAP        = 10                       # as privacy_membership_v2.py
N_DRAWS      = int(os.environ.get('N_DRAWS', 20))
RANDOM_STATE = 42

try:
    GIT_COMMIT = subprocess.check_output(['git', '-C', C.REPO_DIR, 'rev-parse', 'HEAD'],
                                         text=True).strip()
except Exception:
    GIT_COMMIT = 'unknown'
RUN_CONFIG = {'layer': LAYER, 'n_bins': N_BINS, 'alpha': ALPHA, 'background': BACKGROUND,
              'w_cap': W_CAP, 'n_draws': N_DRAWS, 'edges_from': 'hc_train',
              'direction_source': 'val_pd vs hc_train', 'git_commit': GIT_COMMIT}
print(f"Run config: {RUN_CONFIG}")

DETERMINISTIC = ['real', 'flat', 'real_two_tailed', 'real_flipped_direction']
RANDOMISED = ['random', 'dp_eps10', 'dp_eps1', 'real_random_direction', 'flat_random_direction']


def act_path(k, name):
    return osp.join(ACT_DIR, f'activations_fold{k}_{name}.npy')


def method_a(counts, edges, test, direction, subj, us):
    win = 1.0 - pvalues_from_counts(counts, edges, test, direction, ALPHA).mean(axis=1)
    return np.array([win[subj == s].mean() for s in us])


def capped(acts, subj, rng):
    keep = np.concatenate([rng.permutation(np.where(subj == s)[0])[:W_CAP] for s in np.unique(subj)])
    return acts[keep]


def laplace(counts, eps, rng):
    scale = counts.shape[0] * W_CAP / eps
    return np.clip(counts + rng.laplace(0.0, scale, counts.shape), 0.0, None)


# scores[condition][draw] -> list over folds of subject-score arrays
scores = {c: [[]] for c in DETERMINISTIC}
scores.update({c: [[] for _ in range(N_DRAWS)] for c in RANDOMISED})
labels, fold_aucs = [], {c: [] for c in DETERMINISTIC + RANDOMISED}

for k in range(1, C.N_SPLITS + 1):
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")
    rng = np.random.RandomState(RANDOM_STATE + k)
    hc_train = np.load(act_path(k, f'{LAYER}_hc_train'))
    val_pd   = np.load(act_path(k, f'{LAYER}_val_pd'))
    bg       = np.load(act_path(k, f'{LAYER}_{BACKGROUND}'))
    bg_subj  = np.load(act_path(k, f'{BACKGROUND}_subj'))
    test, lab, subj = (np.load(act_path(k, f'{LAYER}_test')), np.load(act_path(k, 'test_labels')),
                       np.load(act_path(k, 'test_subj')))
    us = np.unique(subj)
    y = np.array([lab[subj == s][0] for s in us])
    labels.append(y)

    edges = make_global_edges(hc_train, N_BINS)
    direction = val_pd.mean(0) > hc_train.mean(0)
    real = site_counts(bg, edges).astype(float)
    flat = np.full_like(real, real.sum(axis=1, keepdims=True).mean() / N_BINS)
    n_neurons = real.shape[0]

    def run(counts, d):
        return method_a(counts, edges, test, d, subj, us)

    det = {'real': run(real, direction),
           'flat': run(flat, direction),
           'real_two_tailed': run(real, None),
           'real_flipped_direction': run(real, ~direction)}
    for c, s in det.items():
        scores[c][0].append(s)
        fold_aucs[c].append(roc_auc_score(y, s))

    for d in range(N_DRAWS):
        rand_dir = rng.rand(n_neurons) < 0.5
        cap = site_counts(capped(bg, bg_subj, rng), edges).astype(float)
        rnd = {'random': run(rng.randint(0, 2 * int(real.sum(1).mean() / N_BINS) + 1,
                                                    real.shape).astype(float), direction),
               'dp_eps10': run(laplace(cap, 10.0, rng), direction),
               'dp_eps1': run(laplace(cap, 1.0, rng), direction),
               'real_random_direction': run(real, rand_dir),
               'flat_random_direction': run(flat, rand_dir)}
        for c, s in rnd.items():
            scores[c][d].append(s)
            fold_aucs[c].append(roc_auc_score(y, s))

    print("  " + "  ".join(f"{c} {np.mean(fold_aucs[c][-1 if c in DETERMINISTIC else -N_DRAWS:]):.4f}"
                           for c in DETERMINISTIC + RANDOMISED))

# Pooled over folds (355 subjects)
y_all = np.concatenate(labels)
results = {'config': RUN_CONFIG, 'n_subjects': int(len(y_all)), 'conditions': {}}
for c in DETERMINISTIC + RANDOMISED:
    pooled = [np.concatenate(draw) for draw in scores[c]]
    aucs = [roc_auc_score(y_all, s) for s in pooled]
    rec = {'fold_auc_mean': round(float(np.mean(fold_aucs[c])), 4),
           'fold_auc_std': round(float(np.std(fold_aucs[c])), 4),
           'pooled_auc_mean': round(float(np.mean(aucs)), 4),
           'pooled_auc_std_over_draws': round(float(np.std(aucs)), 4),
           'spec_at_90sens_mean': round(float(np.mean([spec_at_sens(y_all, s, 0.9) for s in pooled])), 4)}
    if c in DETERMINISTIC:
        ci = bootstrap_auc_ci(y_all, pooled[0])
        rec['pooled_ci'] = [ci['ci_low'], ci['ci_high']]
    results['conditions'][c] = rec

real_p, flat_p = np.concatenate(scores['real'][0]), np.concatenate(scores['flat'][0])
results['paired_real_minus_flat'] = paired_bootstrap_auc_diff(y_all, real_p, flat_p)
results['counts_carry_signal'] = bool(results['paired_real_minus_flat']['ci_low'] > 0)

print(f"\n{'condition':<24} {'fold AUC':>16} {'pooled AUC':>12}  95% CI / sd over draws   spec@90")
for c, r in results['conditions'].items():
    spread = (f"[{r['pooled_ci'][0]:.3f}, {r['pooled_ci'][1]:.3f}]" if 'pooled_ci' in r
              else f"sd {r['pooled_auc_std_over_draws']:.3f}")
    print(f"{c:<24} {r['fold_auc_mean']:.3f} ± {r['fold_auc_std']:.3f}   {r['pooled_auc_mean']:>10.3f}  "
          f"{spread:<22}  {r['spec_at_90sens_mean']:.3f}")
p = results['paired_real_minus_flat']
print(f"\nreal - flat: {p['diff']:+.4f}  95% CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]  p={p['p']:.3f}"
      f"  -> counts carry signal beyond edges + direction: {results['counts_carry_signal']}")

out = osp.join(C.RESULTS_DIR, f'control_reference_counts_{LAYER}.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}\nDONE.")
