"""
exp6_habets_monitoring.py — Experiment 6: does the personal-baseline scan follow REAL symptom changes?

Data: Habets et al. home monitoring (habets_fetch.py): 20 people with PD, ~14 days at home, both wrists,
~7 symptom questionnaires (EMA beeps) a day. For each beep, the PRE_MIN minutes of wrist movement before it.

Monitoring as deployed on the watch: the first BASE_DAYS days of each person = personal baseline
(histogram built on the watch with the server's bin edges); every later beep is a follow-up scored
against it (monitor_common.py; scores per wrist, averaged over the two wrists and over the 5 fold models).

Outcomes per beep (self-reported, Likert 1-7):
  motor = mean(tremor, slowness, stiffness, 8 - walk_well, 8 - mobility_well)   (higher = worse)
  OFF   = sanpar_onoff == 1 (OFF) vs 3 (ON); "changing" answers are left out
Metrics (follow-up beeps only):
  within-person Spearman rho(score, motor), people with >= 10 beeps and a non-constant motor score;
  mean over people with bootstrap CI over people; paired difference pscan_2t - each other method.
  within-person AUC OFF vs ON (people with >= 3 OFF and >= 3 ON beeps).
  alarm rate (monitor_common.py rule) overall, in OFF vs ON beeps, and above vs below the person's median motor.

Rule fixed before the run: the personal scan tracks real symptom changes better than the classifier
and better than plain movement intensity if, for motor, mean rho of pscan_2t has CI > 0 AND the paired
differences pscan_2t - clf and pscan_2t - intensity have CIs > 0.
A tie with intensity means the scan mostly measures how much the person moved, not how they moved.

Limitations: 20 people, all with PD (no healthy controls; see exp7 for false alarms in healthy people);
self-reported symptoms; MOX sensors (not the PADS watch); activity before the beep is uncontrolled.

    python classifier_vs_scan/exp6_habets_monitoring.py
"""

import os, os.path as osp, json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import C, OUT_DIR, git_commit, save_json
from monitor_common import FOLDS, METHODS, MIN_WIN, Server, segment_scores, boot_mean_ci

HABETS_DIR = os.environ.get('HABETS_DIR', osp.join(OUT_DIR, 'habets'))
BASE_DAYS  = int(os.environ.get('BASE_DAYS', '3'))
MIN_BEEPS  = 10
PRIMARY    = 'pscan_2t'
SCORE_DIR  = osp.join(OUT_DIR, 'exp6_scores')
os.makedirs(SCORE_DIR, exist_ok=True)


def beeps_table():
    b = pd.read_csv(osp.join(HABETS_DIR, 'beeps.csv'), dtype={'ID': str}, parse_dates=['beep_time_start'])
    b['motor'] = b[['tremor', 'slowness', 'stiffness']].mean(axis=1) * 3 / 5 \
        + ((8 - b['walk_well']) + (8 - b['mobility_well'])) / 5
    b['off'] = np.where(b['sanpar_onoff'] == 1, 1.0, np.where(b['sanpar_onoff'] == 3, 0.0, np.nan))
    first = b.groupby('ID')['beep_time_start'].transform('min').dt.normalize()
    b['day'] = (b['beep_time_start'].dt.normalize() - first).dt.days
    b['is_base'] = b['day'] < BASE_DAYS
    return b


def run_fold(k, beeps):
    path = osp.join(SCORE_DIR, f'fold{k}.json')
    cached = json.load(open(path)) if osp.exists(path) else {}
    srv = None
    for pid in sorted(beeps['ID'].unique()):
        if pid in cached or not osp.exists(osp.join(HABETS_DIR, f'{pid}.npz')):
            continue
        srv = srv or Server(k)
        z = np.load(osp.join(HABETS_DIR, f'{pid}.npz'))
        act, p1 = srv.features(z['windows'])
        pb = beeps[beeps['ID'] == pid]
        base_beeps = set(pb.loc[pb['is_base'], 'beep'].astype(int))
        res = {}
        for side in ('left', 'right'):
            m = z['side'] == side
            sc, thr = segment_scores(srv, act[m], p1[m], z['intensity'][m], z['beep'][m], base_beeps)
            if sc is not None:
                res[side] = {'scores': {str(int(b)): v for b, v in sc.items()}, 'thr': thr}
        cached[pid] = res
        json.dump(cached, open(path, 'w'))
        print(f'  fold {k} {pid}: {len(z["beep"])} windows, sides {list(res)}')
    return cached


def combine(folds_out, beeps):
    """Per beep: score per method averaged over sides then folds; alarm if any side's fold-mean score
    exceeds that side's fold-mean threshold."""
    rows = []
    for pid in sorted(beeps['ID'].unique()):
        per_side = {}
        for side in ('left', 'right'):
            fo = [f[pid][side] for f in folds_out if pid in f and side in f[pid]]
            if not fo:
                continue
            bs = set.intersection(*[set(f['scores']) for f in fo])
            thr = {m: np.mean([f['thr'][m] for f in fo]) for m in METHODS}
            per_side[side] = {b: ({m: np.mean([f['scores'][b][m] for f in fo]) for m in METHODS},
                                  fo[0]['scores'][b].get('baseline', False), thr) for b in bs}
        for b in sorted(set().union(*[set(v) for v in per_side.values()]) if per_side else [], key=int):
            sides = [per_side[s][b] for s in per_side if b in per_side[s]]
            r = {'ID': pid, 'beep': int(b), 'baseline_seg': any(x[1] for x in sides), 'n_sides': len(sides)}
            for m in METHODS:
                r[m] = float(np.mean([x[0][m] for x in sides]))
                r[f'alarm_{m}'] = bool(any(x[0][m] > x[2][m] for x in sides))
            rows.append(r)
    s = pd.DataFrame(rows)
    return s.merge(beeps[['ID', 'beep', 'motor', 'off', 'day', 'is_base', 'beep_time_start']], on=['ID', 'beep'])


def evaluate(df):
    fu = df[~df['is_base']]
    res = {'n_people': int(df['ID'].nunique()), 'n_followup_beeps': int(len(fu)),
           'n_baseline_beeps': int(df['is_base'].sum()), 'motor': {}, 'off_on': {}, 'alarms': {}, 'paired': {}}
    rho, auc = {m: {} for m in METHODS}, {m: {} for m in METHODS}
    for pid, g in fu.groupby('ID'):
        g = g.dropna(subset=['motor'])
        if len(g) >= MIN_BEEPS and g['motor'].std() > 0:
            for m in METHODS:
                rho[m][pid] = spearmanr(g[m], g['motor']).correlation
        h = fu[(fu['ID'] == pid)].dropna(subset=['off'])
        if (h['off'] == 1).sum() >= 3 and (h['off'] == 0).sum() >= 3:
            for m in METHODS:
                auc[m][pid] = roc_auc_score(h['off'], h[m])
    for m in METHODS:
        res['motor'][m] = boot_mean_ci(list(rho[m].values()))
        res['off_on'][m] = boot_mean_ci(np.array(list(auc[m].values())) - 0.5)
        if res['off_on'][m]['mean'] is not None:
            res['off_on'][m]['mean_auc'] = round(res['off_on'][m]['mean'] + 0.5, 4)
        a = fu[f'alarm_{m}']
        med = fu.groupby('ID')['motor'].transform('median')
        res['alarms'][m] = {'followup': round(float(a.mean()), 4),
                            'off': round(float(a[fu['off'] == 1].mean()), 4) if (fu['off'] == 1).any() else None,
                            'on': round(float(a[fu['off'] == 0].mean()), 4) if (fu['off'] == 0).any() else None,
                            'motor_above_median': round(float(a[fu['motor'] > med].mean()), 4),
                            'motor_at_or_below_median': round(float(a[fu['motor'] <= med].mean()), 4)}
    for b in METHODS:
        if b == PRIMARY:
            continue
        for name, src in (('motor', rho), ('off_on', auc)):
            common_ = sorted(set(src[PRIMARY]) & set(src[b]))
            d = [src[PRIMARY][p] - src[b][p] for p in common_]
            r = boot_mean_ci(d)
            r['wins'] = bool(r['ci'][0] is not None and r['ci'][0] > 0)
            r['loses'] = bool(r['ci'][1] is not None and r['ci'][1] < 0)
            res['paired'][f'{PRIMARY} - {b} | {name}'] = r
    # also: does the scan track symptoms beyond intensity? partial (rank) correlation per person
    part = []
    for pid, g in fu.dropna(subset=['motor']).groupby('ID'):
        if len(g) >= MIN_BEEPS and g['motor'].std() > 0:
            r = g[[PRIMARY, 'motor', 'intensity']].rank()
            c = np.corrcoef(r.T.values)
            denom = np.sqrt((1 - c[0, 2] ** 2) * (1 - c[1, 2] ** 2))
            part.append((c[0, 1] - c[0, 2] * c[1, 2]) / denom if denom > 0 else np.nan)
    res['motor_partial_rho_given_intensity'] = boot_mean_ci(part)
    res['per_person_rho'] = {m: {p: round(float(v), 3) for p, v in rho[m].items()} for m in METHODS}
    m, pr = res['motor'][PRIMARY], res['paired']
    rule = {'statement': f'motor: mean rho({PRIMARY}) CI > 0 and {PRIMARY} - clf, {PRIMARY} - intensity paired CIs > 0',
            'rho_ci_above_0': bool(m['ci'][0] is not None and m['ci'][0] > 0),
            'beats_clf': pr[f'{PRIMARY} - clf | motor']['wins'],
            'beats_intensity': pr[f'{PRIMARY} - intensity | motor']['wins']}
    rule['met'] = bool(rule['rho_ci_above_0'] and rule['beats_clf'] and rule['beats_intensity'])
    res['rule'] = rule
    return res


def print_results(res):
    print(f"\n{res['n_people']} people, {res['n_followup_beeps']} follow-up beeps, {res['n_baseline_beeps']} baseline beeps")
    print(f"\n{'method':<10} | motor rho mean [CI] (n) | OFF-vs-ON AUC (n) | alarms fu / OFF / ON / >med / <=med")
    for m in METHODS:
        r, o, a = res['motor'][m], res['off_on'][m], res['alarms'][m]
        print(f"{m:<10} | {r['mean']} [{r['ci'][0]},{r['ci'][1]}] ({r['n']}) | {o.get('mean_auc')} ({o['n']}) | "
              f"{a['followup']} / {a['off']} / {a['on']} / {a['motor_above_median']} / {a['motor_at_or_below_median']}")
    print('\nPaired (per person):')
    for k, d in res['paired'].items():
        print(f"  {k:<34} {d['mean']} [{d['ci'][0]},{d['ci'][1]}] n={d['n']}{' WIN' if d['wins'] else ' LOSE' if d['loses'] else ''}")
    print(f"\npartial rho(pscan_2t, motor | intensity): {res['motor_partial_rho_given_intensity']}")
    print(f"\nRULE: {res['rule']}")


if __name__ == '__main__':
    t0 = time.time()
    beeps = beeps_table()
    folds_out = [run_fold(k, beeps) for k in FOLDS]
    df = combine(folds_out, beeps)
    df.to_csv(osp.join(OUT_DIR, 'exp6_beep_scores.csv'), index=False)
    res = evaluate(df)
    print_results(res)
    save_json({**res, 'config': {'base_days': BASE_DAYS, 'folds': FOLDS, 'min_windows': MIN_WIN,
                                 'min_beeps': MIN_BEEPS, 'methods': METHODS, 'git_commit': git_commit()}},
              'exp6_habets_monitoring.json')
    print(f'DONE {time.time() - t0:.0f} s')
