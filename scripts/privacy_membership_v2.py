"""
privacy_membership_v2.py
------------------------
Privacy of the healthy volunteers' shared histograms, and detection with Methods A, B and C
when the reference is noised. Extends privacy_membership.py (whose results file is untouched).

Deployment this models: healthy volunteers' HC histogram counts go to the central server, which
merges them into the reference installed on patients' watches. Patients' data never leaves the
watch, so only the volunteers' counts need protecting. Bin edges and tail directions are fixed
centrally from PADS (training HC; validation PD vs training HC) before any volunteer data is used.
PADS HC subjects stand in for the volunteers.

Part 1 — membership inference on one released histogram:
  volunteers per histogram : 1, 5, 10, 20
  noise                    : none, eps = 10, eps = 1 (subject-level eps-DP, as privacy_membership.py)
  attacker                 : 'exact' holds the W_CAP windows the volunteer contributed;
                             'fresh' holds W_CAP other windows of the same volunteer (e.g. a new
                             recording). Both attack the same released histogram.
  Score: likelihood ratio of the target's windows under the released vs a population histogram.
  Membership AUC 0.5 = no leakage.

Part 2 — detection under the same noise: Methods A, B and C on the fold's test set, with the
reference built from capped BACKGROUND windows and noised as in Part 1. B and C are scored exactly
as in run_deepscan.py (Berk-Jones, a_max 0.5, C seeded per subject). Reports subject-level AUC and
specificity at 90% sensitivity, averaged over noise draws.

Writes RESULTS_DIR/privacy_membership_v2_{LAYER}.json.

Run:
    python scripts/privacy_membership_v2.py
"""

import sys, os, os.path as osp, json, subprocess, warnings, zlib, time
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.scanning.histograms import make_global_edges, site_counts, pvalues_from_counts, bin_index
from src.training.metrics import spec_at_sens

try:
    from art.defences.detector.evasion.subsetscanning.scanner import Scanner
    from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions
except ImportError:
    os.system('pip install -q adversarial-robustness-toolbox')
    from art.defences.detector.evasion.subsetscanning.scanner import Scanner
    from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions

ACT_DIR = os.environ.get('ACT_DIR', osp.join(C.RESULTS_DIR, 'activations'))
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"

LAYER         = os.environ.get('LAYER', 'layer_2')
N_BINS        = 10
ALPHA         = 1.0
BACKGROUND    = 'val_hc'
SITE_SIZES    = [1, 5, 10, 20]       # volunteers per released histogram
EPSILONS      = [None, 10.0, 1.0]    # None = no noise
ATTACKERS     = ['exact', 'fresh']
W_CAP         = 10                   # max windows per volunteer counted into a histogram
N_REPEATS     = int(os.environ.get('N_REPEATS', 20))
N_REPEATS_ONE = int(os.environ.get('N_REPEATS_ONE', 100))   # one member per repeat: noisier AUC
N_NOISE_DRAWS = int(os.environ.get('N_NOISE_DRAWS', 5))     # detection under noise (B, C are slow)
A_MAX         = 0.5
RESTARTS      = 5
GROUP_SEED    = 0
RANDOM_STATE  = 42

try:
    GIT_COMMIT = subprocess.check_output(['git', '-C', C.REPO_DIR, 'rev-parse', 'HEAD'],
                                         text=True).strip()
except Exception:
    GIT_COMMIT = 'unknown'
RUN_CONFIG = {'layer': LAYER, 'n_bins': N_BINS, 'alpha': ALPHA, 'background': BACKGROUND,
              'site_sizes': SITE_SIZES, 'epsilons': EPSILONS, 'attackers': ATTACKERS,
              'w_cap': W_CAP, 'n_repeats': N_REPEATS, 'n_repeats_one': N_REPEATS_ONE,
              'n_noise_draws': N_NOISE_DRAWS, 'a_max': A_MAX, 'restarts': RESTARTS,
              'group_seed': GROUP_SEED, 'edges_from': 'hc_train (PADS)',
              'direction_source': 'val_pd vs hc_train (PADS)', 'git_commit': GIT_COMMIT}
print(f"Run config: {RUN_CONFIG}")


def act_path(k, name):
    return osp.join(ACT_DIR, f'activations_fold{k}_{name}.npy')


def split_windows(subj, rng):
    """Per subject: W_CAP contributed windows and W_CAP other ('fresh') windows, disjoint."""
    contrib, fresh = [], []
    for s in np.unique(subj):
        idx = rng.permutation(np.where(subj == s)[0])
        contrib.append(idx[:W_CAP])
        fresh.append(idx[W_CAP:2 * W_CAP])
    return np.concatenate(contrib), np.concatenate(fresh)


def release(acts, edges, eps, rng):
    """Counts a volunteer group would transmit, optionally with eps-DP Laplace noise."""
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


# Method B / C scoring: same as run_deepscan.py
FAILURES = {'individ': 0, 'group': 0}

def individ_score(p):
    if p.min() > A_MAX:          # empty subset: the scan score is 0
        return 0.0
    try:
        score, _, _, _ = Scanner.fgss_individ_for_nets(
            pvalues=np.stack([p, p], axis=1).astype(np.float64), a_max=A_MAX,
            score_function=ScoringFunctions.get_score_bj_fast)
        return float(score)
    except Exception:
        FAILURES['individ'] += 1
        return np.nan


def group_score(p_subj, s):
    np.random.seed(zlib.crc32(f'{GROUP_SEED}-{s}'.encode()))
    try:
        score, _, _, _ = Scanner.fgss_for_nets(
            pvalues=np.stack([p_subj, p_subj], axis=2).astype(np.float64), a_max=A_MAX,
            restarts=RESTARTS, score_function=ScoringFunctions.get_score_bj_fast)
        return float(score)
    except Exception:
        FAILURES['group'] += 1
        return np.nan


def detection(p, subj, us, y):
    """Subject-level AUC and spec@90%sens for Methods A, B, C from window p-values."""
    win_a = 1.0 - p.mean(axis=1)
    win_b = np.array([individ_score(w) for w in p])
    scores = {'A': np.array([win_a[subj == s].mean() for s in us]),
              'B': np.array([np.nanmean(win_b[subj == s]) for s in us]),
              'C': np.array([group_score(p[subj == s], s) for s in us])}
    out = {}
    for m, sc in scores.items():
        keep = ~np.isnan(sc)
        out[m] = (roc_auc_score(y[keep], sc[keep]), spec_at_sens(y[keep], sc[keep], 0.9))
    return out


results = {'config': RUN_CONFIG, 'folds': []}
t0 = time.time()

for k in range(1, C.N_SPLITS + 1):
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")
    rng = np.random.RandomState(RANDOM_STATE + k)
    hc_train = np.load(act_path(k, f'{LAYER}_hc_train'))
    val_pd   = np.load(act_path(k, f'{LAYER}_val_pd'))
    edges = make_global_edges(hc_train, N_BINS)
    direction = val_pd.mean(0) > hc_train.mean(0)

    # Volunteer pool (PADS HC stand-ins): contributed and fresh windows per person
    pool_acts = np.concatenate([hc_train, np.load(act_path(k, f'{LAYER}_val_hc'))])
    pool_subj = np.concatenate([np.load(act_path(k, 'hc_train_subj')),
                                np.load(act_path(k, 'val_hc_subj'))])
    i_con, i_fre = split_windows(pool_subj, rng)
    con_acts, con_subj = pool_acts[i_con], pool_subj[i_con]
    bins = {'exact': (bin_index(pool_acts[i_con], edges), con_subj),
            'fresh': (bin_index(pool_acts[i_fre], edges), pool_subj[i_fre])}
    subjects = np.unique(pool_subj)

    fold_rec = {'fold': k, 'n_pool_subjects': int(len(subjects)), 'membership': {}, 'detection': {}}

    # Part 1: membership inference
    for size in SITE_SIZES:
        n_rep = N_REPEATS_ONE if size == 1 else N_REPEATS
        for eps in EPSILONS:
            aucs = {a: [] for a in ATTACKERS}
            for _ in range(n_rep):
                perm = rng.permutation(subjects)
                pop, rest = perm[:len(perm) // 3], perm[len(perm) // 3:]
                site, nonmem = rest[:size], rest[size:]
                if len(nonmem) == 0:
                    continue
                site_lp = log_probs(release(con_acts[np.isin(con_subj, site)], edges, eps, rng))
                pop_lp  = log_probs(site_counts(con_acts[np.isin(con_subj, pop)], edges))
                y = np.r_[np.ones(len(site)), np.zeros(len(nonmem))]
                for a in ATTACKERS:
                    b, bs = bins[a]
                    scores = [lr_score(b[bs == s], site_lp, pop_lp) for s in np.r_[site, nonmem]]
                    aucs[a].append(roc_auc_score(y, scores))
            for a in ATTACKERS:
                fold_rec['membership'][f'{a}_size{size}_eps{eps}'] = [
                    round(float(np.mean(aucs[a])), 4), round(float(np.std(aucs[a])), 4)]
            print(f"Membership AUC  volunteers={size:>2}  eps={str(eps):>5}:  "
                  + "  ".join(f"{a} {np.mean(aucs[a]):.4f} ± {np.std(aucs[a]):.4f}" for a in ATTACKERS))

    # Part 2: detection with A, B, C under the same noise
    bg = np.load(act_path(k, f'{LAYER}_{BACKGROUND}'))
    bg_subj = np.load(act_path(k, f'{BACKGROUND}_subj'))
    bg = bg[split_windows(bg_subj, rng)[0]]
    test, lab, subj = (np.load(act_path(k, f'{LAYER}_test')), np.load(act_path(k, 'test_labels')),
                       np.load(act_path(k, 'test_subj')))
    us = np.unique(subj)
    y = np.array([lab[subj == s][0] for s in us])
    for eps in EPSILONS:
        draws = [detection(pvalues_from_counts(release(bg, edges, eps, rng), edges, test,
                                               direction, ALPHA), subj, us, y)
                 for _ in range(1 if eps is None else N_NOISE_DRAWS)]
        for m in ['A', 'B', 'C']:
            auc = [d[m][0] for d in draws]
            spec = [d[m][1] for d in draws]
            fold_rec['detection'][f'{m}_eps{eps}'] = {
                'auc': [round(float(np.mean(auc)), 4), round(float(np.std(auc)), 4)],
                'spec_at_90sens': [round(float(np.mean(spec)), 4), round(float(np.std(spec)), 4)]}
        print(f"Detection  eps={str(eps):>5}:  " + "  ".join(
            f"{m} AUC {fold_rec['detection'][f'{m}_eps{eps}']['auc'][0]:.4f}" for m in ['A', 'B', 'C'])
            + f"   ({time.time() - t0:.0f} s)")

    results['folds'].append(fold_rec)

results['scan_failures'] = FAILURES
results['summary'] = {
    'membership': {key: round(float(np.mean([f['membership'][key][0] for f in results['folds']])), 4)
                   for key in results['folds'][0]['membership']},
    'detection': {key: {stat: round(float(np.mean([f['detection'][key][stat][0]
                                                    for f in results['folds']])), 4)
                        for stat in ['auc', 'spec_at_90sens']}
                  for key in results['folds'][0]['detection']}}
print(f"\nSUMMARY (mean over folds): {json.dumps(results['summary'], indent=2)}")
print(f"Scan failures: {FAILURES}")

out = osp.join(C.RESULTS_DIR, f'privacy_membership_v2_{LAYER}.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}\nDONE.")
