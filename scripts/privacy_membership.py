"""
privacy_membership.py
---------------------
Membership inference on the SHARED artefact: a site's HC histogram count matrix.

Threat model: the server (or anyone who sees a site's transmitted counts) holds a target
person's windows and wants to know whether that person is in the site's HC reference.
Attacker also has a population reference built from disjoint HC subjects.

Attack score for target t (likelihood ratio, averaged over t's windows, summed over neurons):
    LR(t) = mean_w sum_j [ log p_site(bin_j(w)) - log p_pop(bin_j(w)) ]
Members inflate the site counts in their own bins, so they score higher. Membership AUC 0.5 = no leakage.

Defence: per-subject contribution cap (W_CAP windows) + Laplace noise on every count.
One subject changes at most W_CAP counts per neuron, so the L1 sensitivity of the full
count matrix is n_neurons * W_CAP and Laplace(scale = n_neurons * W_CAP / eps) gives
eps-DP per site release (subject-level). Negative noisy counts are clipped to 0 (post-processing).

Utility: Method A AUC on the fold's test set using a (noisy) reference from BACKGROUND.

Run:
    python scripts/privacy_membership.py
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

from src.scanning.histograms import make_global_edges, site_counts, pvalues_from_counts, bin_index

ACT_DIR = osp.join(C.RESULTS_DIR, 'activations')
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"

LAYER      = 'layer_2'
N_BINS     = 10
ALPHA      = 1.0
BACKGROUND = 'val_hc'
SITE_SIZES = [5, 10, 20]           # HC subjects per site
EPSILONS   = [None, 10.0, 1.0]     # None = no noise
W_CAP      = 10                    # max windows per subject counted into a histogram
N_REPEATS  = 20
RANDOM_STATE = 42

try:
    GIT_COMMIT = subprocess.check_output(['git', '-C', C.REPO_DIR, 'rev-parse', 'HEAD'],
                                         text=True).strip()
except Exception:
    GIT_COMMIT = 'unknown'
RUN_CONFIG = {'layer': LAYER, 'n_bins': N_BINS, 'alpha': ALPHA, 'background': BACKGROUND,
              'site_sizes': SITE_SIZES, 'epsilons': EPSILONS, 'w_cap': W_CAP,
              'n_repeats': N_REPEATS, 'git_commit': GIT_COMMIT}
print(f"Run config: {RUN_CONFIG}")


def act_path(k, name):
    return osp.join(ACT_DIR, f'activations_fold{k}_{name}.npy')


def capped(acts, subj, rng):
    """Keep at most W_CAP random windows per subject (bounds per-subject sensitivity)."""
    keep = np.concatenate([rng.permutation(np.where(subj == s)[0])[:W_CAP] for s in np.unique(subj)])
    return acts[keep], subj[keep]


def release(acts, edges, eps, rng):
    """Counts a site would transmit, optionally with eps-DP Laplace noise."""
    counts = site_counts(acts, edges).astype(float)
    if eps is not None:
        scale = counts.shape[0] * W_CAP / eps
        counts = np.clip(counts + rng.laplace(0.0, scale, counts.shape), 0.0, None)
    return counts


def log_probs(counts):
    return np.log((counts + ALPHA) / (counts.sum(axis=1, keepdims=True) + ALPHA * counts.shape[1]))


def lr_score(bins, site_lp, pop_lp):
    rows = np.arange(site_lp.shape[0])
    return float((site_lp[rows, bins] - pop_lp[rows, bins]).sum(axis=1).mean())


results = {'config': RUN_CONFIG, 'folds': []}

for k in range(1, C.N_SPLITS + 1):
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")
    rng = np.random.RandomState(RANDOM_STATE + k)
    hc_train = np.load(act_path(k, f'{LAYER}_hc_train'))
    hc_subj  = np.load(act_path(k, 'hc_train_subj'))
    val_pd   = np.load(act_path(k, f'{LAYER}_val_pd'))
    edges = make_global_edges(hc_train, N_BINS)
    direction = val_pd.mean(0) > hc_train.mean(0)

    pool_acts = np.concatenate([hc_train, np.load(act_path(k, f'{LAYER}_val_hc'))])
    pool_subj = np.concatenate([hc_subj, np.load(act_path(k, 'val_hc_subj'))])
    pool_acts, pool_subj = capped(pool_acts, pool_subj, rng)
    pool_bins = bin_index(pool_acts, edges)
    subjects = np.unique(pool_subj)

    fold_rec = {'fold': k, 'n_pool_subjects': int(len(subjects)), 'membership': {}, 'utility': {}}

    # Membership inference
    for size in SITE_SIZES:
        for eps in EPSILONS:
            aucs = []
            for _ in range(N_REPEATS):
                perm = rng.permutation(subjects)
                pop, rest = perm[:len(perm) // 3], perm[len(perm) // 3:]
                site, nonmem = rest[:size], rest[size:]
                if len(nonmem) == 0:
                    continue
                site_lp = log_probs(release(pool_acts[np.isin(pool_subj, site)], edges, eps, rng))
                pop_lp  = log_probs(site_counts(pool_acts[np.isin(pool_subj, pop)], edges))
                scores = [lr_score(pool_bins[pool_subj == s], site_lp, pop_lp)
                          for s in np.r_[site, nonmem]]
                y = np.r_[np.ones(len(site)), np.zeros(len(nonmem))]
                aucs.append(roc_auc_score(y, scores))
            key = f'size{size}_eps{eps}'
            fold_rec['membership'][key] = [round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)]
            print(f"Membership AUC  site={size:>3}  eps={str(eps):>5}: "
                  f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Utility under the same noise
    bg = np.load(act_path(k, f'{LAYER}_{BACKGROUND}'))
    bg, _ = capped(bg, np.load(act_path(k, f'{BACKGROUND}_subj')), rng)
    test, lab, subj = (np.load(act_path(k, f'{LAYER}_test')), np.load(act_path(k, 'test_labels')),
                       np.load(act_path(k, 'test_subj')))
    us = np.unique(subj)
    y = np.array([lab[subj == s][0] for s in us])
    for eps in EPSILONS:
        aucs = []
        for _ in range(N_REPEATS if eps is not None else 1):
            p = pvalues_from_counts(release(bg, edges, eps, rng), edges, test, direction, ALPHA)
            win = 1.0 - p.mean(axis=1)
            aucs.append(roc_auc_score(y, [win[subj == s].mean() for s in us]))
        fold_rec['utility'][f'eps{eps}'] = [round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)]
        print(f"Method A AUC  eps={str(eps):>5}: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    results['folds'].append(fold_rec)

results['summary'] = {
    sec: {key: round(float(np.mean([f[sec][key][0] for f in results['folds']])), 4)
          for key in results['folds'][0][sec]}
    for sec in ['membership', 'utility']}
print(f"\nSUMMARY (mean over folds): {json.dumps(results['summary'], indent=2)}")

out = osp.join(C.RESULTS_DIR, f'privacy_membership_{LAYER}.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}\nDONE.")
