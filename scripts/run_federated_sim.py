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

Requires activations from extract_activations.py (with *_subj.npy metadata).

Run:
    python scripts/run_federated_sim.py
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

from src.training.metrics import compute_metrics
from src.scanning.histograms import (make_global_edges, site_counts, merge_counts,
                                     pvalues_from_counts)

ACT_DIR = os.environ.get('ACT_DIR', osp.join(C.RESULTS_DIR, 'activations'))
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"

LAYER      = os.environ.get('LAYER', 'layer_2')
N_BINS     = 10
ALPHA      = 1.0
N_SITES    = 3
N_REPEATS  = 10                       # random site assignments per fold
# HC splits that sites hold as their local healthy reference. 'val_hc' alone is ~13 subjects
# per fold; adding 'hc_train' gives realistic site sizes (applies equally to all three settings).
SITE_HC_SPLITS = ['val_hc', 'hc_train']

try:
    GIT_COMMIT = subprocess.check_output(['git', '-C', C.REPO_DIR, 'rev-parse', 'HEAD'],
                                         text=True).strip()
except Exception:
    GIT_COMMIT = 'unknown'
RUN_CONFIG = {'layer': LAYER, 'n_bins': N_BINS, 'alpha': ALPHA, 'n_sites': N_SITES,
              'n_repeats': N_REPEATS, 'site_hc_splits': SITE_HC_SPLITS, 'git_commit': GIT_COMMIT}
print(f"Run config: {RUN_CONFIG}")


def act_path(k, name):
    return osp.join(ACT_DIR, f'activations_fold{k}_{name}.npy')


def subject_auc(p_matrix, subj, labels):
    win = 1.0 - p_matrix.mean(axis=1)
    us = np.unique(subj)
    scores = np.array([win[subj == s].mean() for s in us])
    y = np.array([labels[subj == s][0] for s in us])
    return y, scores


results = {'config': RUN_CONFIG, 'folds': []}

for k in range(1, C.N_SPLITS + 1):
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")
    hc_train  = np.load(act_path(k, f'{LAYER}_hc_train'))
    val_pd    = np.load(act_path(k, f'{LAYER}_val_pd'))
    test_acts = np.load(act_path(k, f'{LAYER}_test'))
    test_lab  = np.load(act_path(k, 'test_labels'))
    test_subj = np.load(act_path(k, 'test_subj'))

    ref_acts = np.concatenate([np.load(act_path(k, f'{LAYER}_{s}')) for s in SITE_HC_SPLITS])
    ref_subj = np.concatenate([np.load(act_path(k, f'{s}_subj')) for s in SITE_HC_SPLITS])

    # Server: shared edges + tail direction, broadcast once
    edges = make_global_edges(hc_train, N_BINS)
    direction = val_pd.mean(0) > hc_train.mean(0)

    central_counts = site_counts(ref_acts, edges)
    y_c, s_c = subject_auc(pvalues_from_counts(central_counts, edges, test_acts, direction, ALPHA),
                           test_subj, test_lab)
    central_auc = compute_metrics(y_c, s_c)['auc']

    fold_rec = {'fold': k, 'central_auc': central_auc, 'repeats': []}
    rng = np.random.RandomState(C.RANDOM_STATE + k)
    for r in range(N_REPEATS):
        ref_sites = {s: rng.randint(N_SITES) for s in np.unique(ref_subj)}
        test_sites = {s: rng.randint(N_SITES) for s in np.unique(test_subj)}
        ref_site_of = np.array([ref_sites[s] for s in ref_subj])
        test_site_of = np.array([test_sites[s] for s in test_subj])

        counts = [site_counts(ref_acts[ref_site_of == i], edges) for i in range(N_SITES)]
        fed_counts = merge_counts(counts)
        assert np.array_equal(fed_counts, central_counts), "federated merge != central"

        y_all, s_fed, s_loc = [], [], []
        for i in range(N_SITES):
            m = test_site_of == i
            if not m.any():
                continue
            y, sf = subject_auc(pvalues_from_counts(fed_counts, edges, test_acts[m], direction, ALPHA),
                                test_subj[m], test_lab[m])
            _, sl = subject_auc(pvalues_from_counts(counts[i], edges, test_acts[m], direction, ALPHA),
                                test_subj[m], test_lab[m])
            y_all.append(y); s_fed.append(sf); s_loc.append(sl)
        y_all, s_fed, s_loc = map(np.concatenate, (y_all, s_fed, s_loc))
        rec = {'federated_auc': compute_metrics(y_all, s_fed)['auc'],
               'local_auc': compute_metrics(y_all, s_loc)['auc'],
               'site_hc_subjects': [int(sum(v == i for v in ref_sites.values()))
                                    for i in range(N_SITES)]}
        fold_rec['repeats'].append(rec)

    fed = [r['federated_auc'] for r in fold_rec['repeats']]
    loc = [r['local_auc'] for r in fold_rec['repeats']]
    fold_rec['federated_auc_mean'] = round(float(np.mean(fed)), 4)
    fold_rec['local_auc_mean'] = round(float(np.mean(loc)), 4)
    print(f"Central AUC: {central_auc:.4f}  Federated: {np.mean(fed):.4f}  "
          f"Local-only: {np.mean(loc):.4f} ± {np.std(loc):.4f}")
    results['folds'].append(fold_rec)

# Communication cost per site: one int count matrix vs raw activations
n_neurons = central_counts.shape[0]
results['bytes_per_site_counts'] = int(n_neurons * N_BINS * 4)   # int32
results['summary'] = {key: [round(float(np.mean([f[key] for f in results['folds']])), 4),
                            round(float(np.std([f[key] for f in results['folds']])), 4)]
                      for key in ['central_auc', 'federated_auc_mean', 'local_auc_mean']}
print(f"\nSUMMARY (mean, std over folds): {results['summary']}")
print(f"Each site transmits {results['bytes_per_site_counts']} bytes of counts "
      f"({n_neurons} neurons x {N_BINS} bins)")

out = osp.join(C.RESULTS_DIR, f'federated_sim_{LAYER}.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}\nDONE.")
