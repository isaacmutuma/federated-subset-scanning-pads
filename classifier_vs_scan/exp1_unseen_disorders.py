"""
exp1_unseen_disorders.py — Experiment 1: disorders the classifier was never trained on.

The PatchTST classifier was trained on PD vs HC only. The 114 PADS subjects with other movement
disorders (DD) are in no fold. Every method scores them against local healthy volunteers (val_hc)
with no labels from the screened population.

Comparisons (pooled over folds; each DD subject scored by one fold's model, assigned round-robin):
  pd  : HC vs PD          (sanity: reproduces the main results)
  dd  : HC vs DD          <- the test of "detect, don't classify"
  any : HC vs PD + DD     (the screening population)
  dd:<diagnosis> : HC vs each DD diagnosis with >= 5 subjects
Metrics: pooled AUC with bootstrap CI, per-fold AUC, sensitivity when 10% of local healthy
volunteers are referred (threshold from leave-one-out volunteer scores), realised specificity.

Rule fixed before the run: scanning flags unseen disorders better than the classifier if
scan_A - clf_K1 on 'dd' has a paired-bootstrap AUC CI above 0 (also reported vs clf_K2, maha,
knn, and for scan_A_2t / scan_C).

Writes OUT_DIR/exp1_unseen_disorders.json.

Run:
    python classifier_vs_scan/exp1_unseen_disorders.py          # RUN_C=0 skips Method C
"""

import os, os.path as osp, json
import numpy as np

from common import (C, FEAT_DIR, git_commit, save_json, load_fold, make_global_edges, subject_scores,
                    local_scores, evaluate, print_eval)

RUN_C = os.environ.get('RUN_C', '1') == '1'
SCAN_METHODS = ['scan_A', 'scan_A_2t'] + (['scan_C'] if RUN_C else [])
FIT_METHODS = SCAN_METHODS + ['maha', 'knn']
dd_groups = {int(k): v for k, v in json.load(open(osp.join(FEAT_DIR, 'dd_diagnosis.json'))).items()}


def fold_scores_for(shift, k, ref_shift=None, methods=None):
    """Scores of every method for one fold. ref_shift: build references/thresholds from the
    volunteers of another shift (exp 2's 'European reference' arms). methods: subset to compute."""
    methods = methods or FIT_METHODS + ['clf_K1', 'clf_K2']
    d = load_fold(shift, k)
    r = d if ref_shift is None else load_fold(ref_shift, k)
    edges = make_global_edges(d['hc_train_act'], 10)
    direction = d['val_pd_act'].mean(0) > d['hc_train_act'].mean(0)
    ts, tl = d['test_subj'], d['test_label']
    us_t = np.unique(ts)
    y = np.array([tl[ts == s][0] for s in us_t])
    out = {}
    eval_sets = {'test': (d['test_act'], ts), 'dd': (d['dd_act'], d['dd_subj'])}
    for m in [m for m in FIT_METHODS if m in methods]:
        sc = local_scores(r['val_hc_act'], r['val_hc_subj'], edges, direction, m, eval_sets)
        out[m] = {'test': (us_t, sc['test'][1], y), 'dd': sc['dd'], 'val_hc_loso': sc['val_hc_loso']}
    for name, key, loso_key in [('clf_K1', 'p1', 'p1'), ('clf_K2', 'p2', 'p2_loso')]:
        if name not in methods or (ref_shift is not None and name == 'clf_K2'):
            continue                                    # K2 is by construction local
        out[name] = {'test': (us_t, subject_scores(d[f'test_{key}'], ts)[1], y),
                     'dd': subject_scores(d[f'dd_{key}'], d['dd_subj']),
                     'val_hc_loso': subject_scores(r[f'val_hc_{loso_key}'], r['val_hc_subj'])}
    return out


if __name__ == '__main__':
    print(f"Methods: {FIT_METHODS + ['clf_K1', 'clf_K2']}")
    folds = []
    for k in range(1, C.N_SPLITS + 1):
        folds.append(fold_scores_for('none', k))
        print(f"fold {k} done")
    res = evaluate(folds, dd_groups, primary=SCAN_METHODS, competitors=['clf_K1', 'clf_K2', 'maha', 'knn'])
    res['config'] = {'shift': 'none', 'methods': list(folds[0].keys()), 'run_c': RUN_C,
                     'refer_rate': 0.10, 'reference': 'val_hc', 'git_commit': git_commit()}
    key = 'scan_A - clf_K1 | dd'
    res['rule'] = {'statement': 'scan_A beats clf_K1 on HC vs DD (paired AUC CI above 0)',
                   'met': bool(res['paired'].get(key, {}).get('wins_auc', False))}
    print_eval(res)
    print(f"\nRULE: {res['rule']['statement']} -> {res['rule']['met']}")
    save_json(res, 'exp1_unseen_disorders.json')
