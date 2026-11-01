"""
exp2_population_shift.py — Experiment 2: a different population (stand-in for rural Kenya).

The screened population (local volunteers, test subjects, DD subjects) is shifted at the raw-signal
level (see common.SHIFTS: amplitude, sensor noise, 25 Hz watch, rotated watch, slower movement,
and a combination); the server side (classifier training, bin edges, tail directions) stays PADS.

Arms (no labels from the shifted population in any of them):
  local re-referencing : scan_A, scan_A_2t, maha, knn fitted on the shifted volunteers;
                         clf_K1 threshold from the shifted volunteers; clf_K2 feature re-normalised
  no re-referencing    : scan_A_euro (reference + threshold from unshifted volunteers),
                         clf_K1_euro (threshold from unshifted volunteers)
Expected: amp0.7 changes nothing (PatchTST standardises each window internally) — a sanity check.

Rule fixed before the run (per shift, reported for all): scanning adapts better than the classifier
if scan_A - clf_K1 AND scan_A - clf_K2 sensitivity at 10% referral (HC vs PD+DD) both have
paired-bootstrap CIs above 0.

Writes OUT_DIR/exp2_population_shift.json.

Run:
    SHIFTS=amp0.7,noise,rate25,rotate30,speed0.8,combined python classifier_vs_scan/exp2_population_shift.py
"""

import os
os.environ.setdefault('RUN_C', '0')            # Method C is slow; set RUN_C=1 to include it

from common import C, git_commit, save_json, evaluate, print_eval
from exp1_unseen_disorders import fold_scores_for, dd_groups, SCAN_METHODS

SHIFT_LIST = os.environ.get('SHIFTS', 'amp0.7,noise,rate25,rotate30,speed0.8,combined').split(',')

results = {'config': {'shifts': SHIFT_LIST, 'refer_rate': 0.10, 'git_commit': git_commit()}, 'by_shift': {}}
for shift in SHIFT_LIST:
    print(f"\n{'='*60}\nSHIFT {shift}\n{'='*60}")
    folds = []
    for k in range(1, C.N_SPLITS + 1):
        local = fold_scores_for(shift, k)
        euro = fold_scores_for(shift, k, ref_shift='none', methods=['scan_A', 'clf_K1'])
        local.update({f'{m}_euro': v for m, v in euro.items()})
        folds.append(local)
    res = evaluate(folds, dd_groups, primary=SCAN_METHODS,
                   competitors=['clf_K1', 'clf_K2', 'maha', 'knn', 'clf_K1_euro', 'scan_A_euro'])
    k1 = res['paired'].get('scan_A - clf_K1 | any', {}).get('wins_sensitivity', False)
    k2 = res['paired'].get('scan_A - clf_K2 | any', {}).get('wins_sensitivity', False)
    res['rule'] = {'statement': 'scan_A beats clf_K1 and clf_K2 on sensitivity at 10% referral (HC vs PD+DD)',
                   'met': bool(k1 and k2), 'beats_K1': bool(k1), 'beats_K2': bool(k2)}
    print_eval(res, comps=('pd', 'dd', 'any'))
    print(f"RULE [{shift}]: {res['rule']}")
    results['by_shift'][shift] = res

print(f"\n{'shift':<10} {'scan_A':>14} {'clf_K1':>14} {'clf_K2':>14} {'K1 euro':>14}  (AUC any / sens@10% any / realised spec)")
for shift, r in results['by_shift'].items():
    cells = []
    for m in ['scan_A', 'clf_K1', 'clf_K2', 'clf_K1_euro']:
        v = r['methods'].get(m)
        cells.append(f"{v['comparisons']['any']['pooled_auc']:.3f}/{v['comparisons']['any']['sensitivity_at_10pct_referral']:.2f}/"
                     f"{v['realised_specificity']:.2f}" if v else '-')
    print(f"{shift:<10} " + " ".join(f"{c:>14}" for c in cells) + f"  rule met: {r['rule']['met']}")
save_json(results, 'exp2_population_shift.json')
