"""
exp7_imu50_false_alarms.py — Experiment 7: how often does personal monitoring raise a FALSE alarm in
healthy people living normally over several days?

Data: IMU50 (imu50_fetch.py): healthy adults 40-75, 5 days free living, non-dominant wrist. Segments =
first SEG_MIN minutes of each daytime hour. Nobody's movement changes for a medical reason, so every
follow-up alarm is a false alarm.

Same monitoring as exp6 (monitor_common.py): personal baseline = segments of day 0 and day 1 (the first
BASE_DAYS calendar days with data); later segments are follow-ups; alarm if a segment's score exceeds the
95th percentile of the person's leave-one-out baseline scores. Scores averaged over the 5 fold models.

Metrics per method: false-alarm rate over follow-up segments (nominal 5% if nothing drifts), per person
(median, IQR), and alarm DAYS: a day is flagged if >= DAY_K of its segments alarm (DAY_K = 3: under the
nominal 5% per segment and ~13 segments/day this happens on ~3% of days). Also how the false-alarm rate
grows with days since the baseline (drift).

Rule fixed before the run (M2): personal monitoring is usable in healthy people if the median per-person
false-alarm rate of pscan_2t is <= 10% (twice nominal) AND flagged days <= 10% of follow-up days.
Compared with exp6: the alarm rate in Habets OFF beeps should be clearly above this healthy rate.

    python classifier_vs_scan/exp7_imu50_false_alarms.py
"""

import os, os.path as osp, json, time
import numpy as np
import pandas as pd

from common import OUT_DIR, git_commit, save_json
from monitor_common import FOLDS, METHODS, MIN_WIN, Server, segment_scores, boot_mean_ci

IMU50_DIR = os.environ.get('IMU50_DIR', osp.join(OUT_DIR, 'imu50'))
BASE_DAYS = int(os.environ.get('BASE_DAYS', '2'))
DAY_K     = int(os.environ.get('DAY_K', '3'))
PRIMARY   = 'pscan_2t'
SCORE_DIR = osp.join(OUT_DIR, 'exp7_scores')
os.makedirs(SCORE_DIR, exist_ok=True)


def run_fold(k, segs):
    path = osp.join(SCORE_DIR, f'fold{k}.json')
    cached = json.load(open(path)) if osp.exists(path) else {}
    srv = None
    for sid in sorted(segs['subject'].unique()):
        if sid in cached or not osp.exists(osp.join(IMU50_DIR, f'{sid}.npz')):
            continue
        srv = srv or Server(k)
        z = np.load(osp.join(IMU50_DIR, f'{sid}.npz'))
        act, p1 = srv.features(z['windows'])
        ss = segs[segs['subject'] == sid]
        base = set(ss.loc[ss['day'] < BASE_DAYS, 'segment'].astype(int))
        sc, thr = segment_scores(srv, act, p1, z['intensity'], z['segment'], base)
        cached[sid] = None if sc is None else {'scores': {str(int(s)): v for s, v in sc.items()}, 'thr': thr}
        json.dump(cached, open(path, 'w'))
        print(f'  fold {k} {sid}: {len(z["segment"])} windows, {0 if sc is None else len(sc)} segments')
    return cached


def combine(folds_out, segs):
    rows = []
    for sid in sorted(segs['subject'].unique()):
        fo = [f[sid] for f in folds_out if f.get(sid)]
        if not fo:
            continue
        common_ = set.intersection(*[set(f['scores']) for f in fo])
        thr = {m: np.mean([f['thr'][m] for f in fo]) for m in METHODS}
        for s in common_:
            r = {'subject': sid, 'segment': int(s), 'baseline_seg': bool(fo[0]['scores'][s].get('baseline', False))}
            for m in METHODS:
                r[m] = float(np.mean([f['scores'][s][m] for f in fo]))
                r[f'alarm_{m}'] = bool(r[m] > thr[m])
            rows.append(r)
    df = pd.DataFrame(rows)
    return df.merge(segs[['subject', 'segment', 'day', 'hour', 'n_windows']], on=['subject', 'segment'])


def evaluate(df):
    fu = df[df['day'] >= BASE_DAYS]
    res = {'n_people': int(fu['subject'].nunique()), 'n_followup_segments': int(len(fu)),
           'n_followup_days': int(fu.groupby(['subject', 'day']).ngroups), 'methods': {}}
    for m in METHODS:
        a = f'alarm_{m}'
        per = fu.groupby('subject')[a].mean()
        days = fu.groupby(['subject', 'day'])[a].sum() >= DAY_K
        by_day = fu.groupby('day')[a].mean()
        res['methods'][m] = {
            'false_alarm_rate': round(float(fu[a].mean()), 4),
            'per_person_median': round(float(per.median()), 4),
            'per_person_iqr': [round(float(per.quantile(.25)), 4), round(float(per.quantile(.75)), 4)],
            'per_person_mean_ci': boot_mean_ci(per.values),
            'flagged_day_rate': round(float(days.mean()), 4),
            'rate_by_day_since_start': {int(d): round(float(v), 4) for d, v in by_day.items()}}
    p = res['methods'][PRIMARY]
    res['rule'] = {'statement': f'median per-person false-alarm rate of {PRIMARY} <= 0.10 and flagged days <= 0.10',
                   'median_rate': p['per_person_median'], 'flagged_day_rate': p['flagged_day_rate'],
                   'met': bool(p['per_person_median'] <= 0.10 and p['flagged_day_rate'] <= 0.10)}
    return res


def print_results(res):
    print(f"\n{res['n_people']} healthy people, {res['n_followup_segments']} follow-up segments, "
          f"{res['n_followup_days']} follow-up days (nominal false-alarm rate 0.05)")
    print(f"{'method':<10} | rate | per-person median [IQR] | flagged days | by day since start")
    for m, r in res['methods'].items():
        print(f"{m:<10} | {r['false_alarm_rate']:.3f} | {r['per_person_median']:.3f} [{r['per_person_iqr'][0]:.3f},"
              f"{r['per_person_iqr'][1]:.3f}] | {r['flagged_day_rate']:.3f} | {r['rate_by_day_since_start']}")
    print(f"\nRULE: {res['rule']}")


if __name__ == '__main__':
    t0 = time.time()
    segs = pd.read_csv(osp.join(IMU50_DIR, 'segments.csv'), dtype={'subject': str})
    segs = segs[segs['n_windows'] >= MIN_WIN]
    folds_out = [run_fold(k, segs) for k in FOLDS]
    df = combine(folds_out, segs)
    df.to_csv(osp.join(OUT_DIR, 'exp7_segment_scores.csv'), index=False)
    res = evaluate(df)
    print_results(res)
    save_json({**res, 'config': {'base_days': BASE_DAYS, 'day_k': DAY_K, 'folds': FOLDS, 'min_windows': MIN_WIN,
                                 'methods': METHODS, 'git_commit': git_commit()}}, 'exp7_imu50_false_alarms.json')
    print(f'DONE {time.time() - t0:.0f} s')
