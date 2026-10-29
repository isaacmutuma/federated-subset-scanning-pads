"""
run_federated_sim.py
--------------------
Simulated multi-site deployment of the histogram reference (Method A).

Per fold, HC reference subjects and test subjects are randomly assigned to N_SITES pseudo-sites.
The server fixes bin edges from its HC train activations; each site bins only its own HC windows
with those shared edges and sends the count matrix. Three references are compared:

  central   : one histogram over all reference HC windows pooled (upper bound, needs data sharing)
  federated : sum of site count matrices (what the protocol transmits)
  local     : each site uses only its own counts (no collaboration)

Checks that federated == central exactly, and reports the AUC gain of federating over local-only.
Each site's test subjects are scored locally; subject scores are pooled across sites per fold.

Site-count sweep (H1b: does merging help when each site has only a few HC volunteers?):
SITE_COUNTS="3,5,10,15,20" with ASSIGN=balanced splits the 63 HC per fold into equal-size sites
(3-21 HC each). Per repeat, test subjects from all folds are pooled and the paired subject-level
bootstrap gives federated - local AUC with a 95% CI. Decision rule (fixed before the run):
merging helps at a site size if the median-over-repeats CI lower bound is > 0.

Requires activations from extract_activations.py (with *_subj.npy metadata).

Run:
    python scripts/run_federated_sim.py                       # 3 sites, random assignment (paper)
    SITE_COUNTS=3,5,10,15,20 ASSIGN=balanced RUN_TAG=_sites \
        python scripts/run_federated_sim.py                   # site-count sweep
"""

import sys, os, os.path as osp, json, subprocess, warnings
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

from src.training.metrics import compute_metrics, paired_bootstrap_auc_diff
from src.scanning.histograms import (make_global_edges, site_counts, merge_counts,
                                     pvalues_from_counts)

ACT_DIR = os.environ.get('ACT_DIR', osp.join(C.RESULTS_DIR, 'activations'))
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"

LAYER      = os.environ.get('LAYER', 'layer_2')
N_BINS     = 10
ALPHA      = 1.0
SITE_COUNTS = [int(x) for x in os.environ.get('SITE_COUNTS', '3').split(',')]
# 'random': each subject draws a site independently (original run; sites can be uneven)
# 'balanced': shuffled subjects split into equal-size sites (needed at 10-20 sites, else empty sites)
ASSIGN     = os.environ.get('ASSIGN', 'random')
RUN_TAG    = os.environ.get('RUN_TAG', '')
N_REPEATS  = 10                       # site assignments per fold
# HC splits that sites hold as their local healthy reference. 'val_hc' alone is ~13 subjects
# per fold; adding 'hc_train' gives realistic site sizes (applies equally to all three settings).
SITE_HC_SPLITS = ['val_hc', 'hc_train']

try:
    GIT_COMMIT = subprocess.check_output(['git', '-C', C.REPO_DIR, 'rev-parse', 'HEAD'],
                                         text=True).strip()
except Exception:
    GIT_COMMIT = 'unknown'
RUN_CONFIG = {'layer': LAYER, 'n_bins': N_BINS, 'alpha': ALPHA, 'site_counts': SITE_COUNTS,
              'n_sites': SITE_COUNTS[0], 'assign': ASSIGN, 'n_repeats': N_REPEATS,
              'site_hc_splits': SITE_HC_SPLITS, 'git_commit': GIT_COMMIT}
print(f"Run config: {RUN_CONFIG}")


def act_path(k, name):
    return osp.join(ACT_DIR, f'activations_fold{k}_{name}.npy')


def subject_auc(p_matrix, subj, labels):
    win = 1.0 - p_matrix.mean(axis=1)
    us = np.unique(subj)
    scores = np.array([win[subj == s].mean() for s in us])
    y = np.array([labels[subj == s][0] for s in us])
    return y, scores


def assign_sites(subjects, n_sites, rng):
    """subject id -> site index."""
    if ASSIGN == 'random':
        return {s: rng.randint(n_sites) for s in subjects}
    perm = rng.permutation(subjects)
    return {s: i for i, part in enumerate(np.array_split(perm, n_sites)) for s in part}


# Load each fold once; the server fixes edges + tail direction from its HC train activations
folds = []
for k in range(1, C.N_SPLITS + 1):
    hc_train = np.load(act_path(k, f'{LAYER}_hc_train'))
    val_pd   = np.load(act_path(k, f'{LAYER}_val_pd'))
    ref_acts = np.concatenate([np.load(act_path(k, f'{LAYER}_{s}')) for s in SITE_HC_SPLITS])
    edges = make_global_edges(hc_train, N_BINS)
    folds.append({'k': k, 'edges': edges, 'direction': val_pd.mean(0) > hc_train.mean(0),
                  'ref_acts': ref_acts,
                  'ref_subj': np.concatenate([np.load(act_path(k, f'{s}_subj'))
                                              for s in SITE_HC_SPLITS]),
                  'test_acts': np.load(act_path(k, f'{LAYER}_test')),
                  'test_lab': np.load(act_path(k, 'test_labels')),
                  'test_subj': np.load(act_path(k, 'test_subj')),
                  'central_counts': site_counts(ref_acts, edges)})


def run_sites(n_sites):
    out = {'n_sites': n_sites, 'folds': []}
    pooled = [{'y': [], 'fed': [], 'loc': []} for _ in range(N_REPEATS)]
    for F in folds:
        k, edges, direction = F['k'], F['edges'], F['direction']
        ref_acts, ref_subj = F['ref_acts'], F['ref_subj']
        test_acts, test_lab, test_subj = F['test_acts'], F['test_lab'], F['test_subj']
        central_counts = F['central_counts']
        print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}  ({n_sites} sites, {ASSIGN})\n{'='*55}")

        y_c, s_c = subject_auc(pvalues_from_counts(central_counts, edges, test_acts, direction, ALPHA),
                               test_subj, test_lab)
        central_auc = compute_metrics(y_c, s_c)['auc']

        fold_rec = {'fold': k, 'central_auc': central_auc, 'repeats': []}
        # Same seed per fold for every site count (3 sites + random = original run, bit for bit)
        rng = np.random.RandomState(C.RANDOM_STATE + k)
        for r in range(N_REPEATS):
            ref_sites = assign_sites(np.unique(ref_subj), n_sites, rng)
            test_sites = assign_sites(np.unique(test_subj), n_sites, rng)
            ref_site_of = np.array([ref_sites[s] for s in ref_subj])
            test_site_of = np.array([test_sites[s] for s in test_subj])

            counts = [site_counts(ref_acts[ref_site_of == i], edges) for i in range(n_sites)]
            fed_counts = merge_counts(counts)
            assert np.array_equal(fed_counts, central_counts), "federated merge != central"

            y_all, s_fed, s_loc = [], [], []
            for i in range(n_sites):
                m = test_site_of == i
                if not m.any():
                    continue
                y, sf = subject_auc(pvalues_from_counts(fed_counts, edges, test_acts[m], direction,
                                                        ALPHA), test_subj[m], test_lab[m])
                _, sl = subject_auc(pvalues_from_counts(counts[i], edges, test_acts[m], direction,
                                                        ALPHA), test_subj[m], test_lab[m])
                y_all.append(y); s_fed.append(sf); s_loc.append(sl)
            y_all, s_fed, s_loc = map(np.concatenate, (y_all, s_fed, s_loc))
            pooled[r]['y'].append(y_all); pooled[r]['fed'].append(s_fed); pooled[r]['loc'].append(s_loc)
            site_hc = [int(sum(v == i for v in ref_sites.values())) for i in range(n_sites)]
            rec = {'federated_auc': compute_metrics(y_all, s_fed)['auc'],
                   'local_auc': compute_metrics(y_all, s_loc)['auc'],
                   'local_spec_at_90sens': compute_metrics(y_all, s_loc)['spec_at_90sens'],
                   'site_hc_subjects': site_hc,
                   'site_hc_windows': [int(c[0].sum()) for c in counts]}
            fold_rec['repeats'].append(rec)

        fed = [r['federated_auc'] for r in fold_rec['repeats']]
        loc = [r['local_auc'] for r in fold_rec['repeats']]
        fold_rec['federated_auc_mean'] = round(float(np.mean(fed)), 4)
        fold_rec['local_auc_mean'] = round(float(np.mean(loc)), 4)
        fold_rec['local_spec_at_90sens_mean'] = round(float(np.mean(
            [r['local_spec_at_90sens'] for r in fold_rec['repeats']])), 4)
        print(f"Central AUC: {central_auc:.4f}  Federated: {np.mean(fed):.4f}  "
              f"Local-only: {np.mean(loc):.4f} ± {np.std(loc):.4f}")
        out['folds'].append(fold_rec)

    out['summary'] = {key: [round(float(np.mean([f[key] for f in out['folds']])), 4),
                            round(float(np.std([f[key] for f in out['folds']])), 4)]
                      for key in ['central_auc', 'federated_auc_mean', 'local_auc_mean',
                                  'local_spec_at_90sens_mean']}
    sizes = [n for f in out['folds'] for r in f['repeats'] for n in r['site_hc_subjects']]
    out['summary']['site_hc_subjects_min_max'] = [int(min(sizes)), int(max(sizes))]

    # Federated - local on all test subjects (folds pooled), one paired bootstrap per repeat
    gains = [paired_bootstrap_auc_diff(np.concatenate(p['y']), np.concatenate(p['fed']),
                                       np.concatenate(p['loc'])) for p in pooled]
    out['pooled_gain_per_repeat'] = gains
    med = {key: round(float(np.median([g[key] for g in gains])), 4)
           for key in ['diff', 'ci_low', 'ci_high', 'p']}
    out['pooled_gain_median'] = {**med, 'merging_helps': bool(med['ci_low'] > 0)}
    print(f"\n[{n_sites} sites] SUMMARY (mean, std over folds): {out['summary']}")
    print(f"[{n_sites} sites] federated - local (pooled, median over repeats): "
          f"{med['diff']:+.4f}  95% CI [{med['ci_low']:+.4f}, {med['ci_high']:+.4f}]  p={med['p']:.3f}")
    return out


results = {'config': RUN_CONFIG,
           'by_n_sites': {str(n): run_sites(n) for n in SITE_COUNTS}}
# Top-level keys for single-count runs (read by the notebook summary and minirocket_scan)
first = results['by_n_sites'][str(SITE_COUNTS[0])]
results['folds'], results['summary'] = first['folds'], first['summary']

# Communication cost per site: one int count matrix vs raw activations
n_neurons = folds[0]['central_counts'].shape[0]
results['bytes_per_site_counts'] = int(n_neurons * N_BINS * 4)   # int32
print(f"\nEach site transmits {results['bytes_per_site_counts']} bytes of counts "
      f"({n_neurons} neurons x {N_BINS} bins)")
if len(SITE_COUNTS) > 1:
    print(f"\n{'sites':>5} {'HC/site':>8} {'central':>8} {'local':>8} {'fed-local':>10}  95% CI")
    for n in SITE_COUNTS:
        o = results['by_n_sites'][str(n)]
        g, sm = o['pooled_gain_median'], o['summary']
        print(f"{n:>5} {'%d-%d' % tuple(sm['site_hc_subjects_min_max']):>8} {sm['central_auc'][0]:>8.4f} "
              f"{sm['local_auc_mean'][0]:>8.4f} {g['diff']:>+10.4f}  [{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]"
              f"{'  merging helps' if g['merging_helps'] else ''}")

out = osp.join(C.RESULTS_DIR, f'federated_sim_{LAYER}{RUN_TAG}.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}\nDONE.")
