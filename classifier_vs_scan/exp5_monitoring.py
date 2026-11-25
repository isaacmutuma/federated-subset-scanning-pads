"""
exp5_monitoring.py — Experiment 5: monitoring a person over time (simulated on PADS).

The goal of the system is monitoring, not diagnosis: flag when a person's movement CHANGES relative to
their own past. PADS has one session per person, so follow-up "days" are simulated within the session:

  personal baseline : windows 1-3 of every recording (11 tasks x 2 wrists x 3 = 66 windows)
  follow-up         : windows 4-5 of every recording (44 windows), either unchanged (control) or with a
                      change injected into the raw signal before filtering, at strength s:
      tremor_pd  : 4-6 Hz oscillation, amplitude s x channel SD          (what the classifier was trained on)
      tremor_et  : 8-12 Hz oscillation, amplitude s x channel SD         (essential-tremor-like)
      slowing    : movement time-stretched by 1 + 0.3 s                   (bradykinesia-like)
      irregular  : added noise, SD 0.5 s x channel SD                     (irregular / jerky movement)
      amplitude  : signal x (1 - 0.4 s)  -- sanity: PatchTST scales each window, so all methods should
                   be near chance; strong detection here would point to a pipeline artefact
Monitored people: each fold's test subjects (HC + PD) and the DD subjects assigned to that fold (every
person once). Patients have no healthy "normal", so a personal baseline is the only reference for them.

Methods (score = how much the follow-up differs from the person's baseline; higher = more change):
  pscan_2t      : personal histogram scan, two-tailed (server bin edges; no PD information)   <- primary
  pscan_A       : personal histogram scan, directed by the server's PD tail directions
  hcscan_delta  : healthy-reference scan A: score(follow-up) - score(baseline)
  clf_trend     : classifier P(PD): mean(follow-up) - mean(baseline)
  clf_trend_abs : |clf_trend| (any change in P(PD), either direction)
  pmaha         : Mahalanobis distance of follow-up windows from the personal baseline (128-d)

Metrics per change x strength, for all monitored people and separately for HC / patients (PD + DD):
AUC changed vs unchanged follow-up (bootstrap CI), detection rate at a 5% false-alarm rate (threshold =
95th percentile of the unchanged follow-ups), paired AUC differences pscan_2t - each competitor.

Rule fixed before the run: personal scanning monitors change better than the classifier if, at strength
0.5 and over all monitored people, pscan_2t - clf_trend AND pscan_2t - clf_trend_abs have paired AUC CIs
above 0 for at least 2 of the 3 non-PD changes (tremor_et, slowing, irregular). tremor_pd is reported
(a tie or loss there is expected: it is what the classifier was trained on).

Limitation: changes are simulated within one recording session; no real progression is observed.

Writes OUT_DIR/exp5_monitoring.json; per-fold subject scores cached in OUT_DIR/exp5_scores/ (restart resumes).

Run (GPU recommended; needs features/none from extract_features.py for bin edges and directions):
    python classifier_vs_scan/exp5_monitoring.py
    CHANGES=tremor_et STRENGTHS=1.0 FOLDS=1 python classifier_vs_scan/exp5_monitoring.py   # quick test
"""

import os, os.path as osp, json, pickle, time
import numpy as np
import torch

from common import (C, OUT_DIR, git_commit, save_json, pads_paths, load_manifest, load_raw, make_folds,
                    build_classifier, encode, head_probs, load_fold, make_global_edges, site_counts, scan_a,
                    Mahalanobis, dd_fold_assignment, bootstrap_auc_ci, paired_bootstrap_auc_diff, N_BOOT)
from src.data.preprocessing import (bandpass_filter, segment_windows, compute_normalization_stats,
                                    apply_normalization)

CHANGES   = os.environ.get('CHANGES', 'tremor_pd,tremor_et,slowing,irregular,amplitude').split(',')
STRENGTHS = [float(x) for x in os.environ.get('STRENGTHS', '0.25,0.5,1.0').split(',')]
FOLDS     = [int(x) for x in os.environ.get('FOLDS', ','.join(str(k) for k in range(1, C.N_SPLITS + 1))).split(',')]
RAW_CACHE = os.environ.get('RAW_CACHE', '/content/cvs_raw.pkl')
BASE_POS, FU_POS = [0, 1, 2], [3, 4]
FA_RATE   = 0.05
RULE_STRENGTH = 0.5
NON_PD    = ['tremor_et', 'slowing', 'irregular']
METHODS   = ['pscan_2t', 'pscan_A', 'hcscan_delta', 'clf_trend', 'clf_trend_abs', 'pmaha']
PRIMARY, COMPETITORS = 'pscan_2t', ['clf_trend', 'clf_trend_abs', 'pscan_A', 'hcscan_delta', 'pmaha']
SCORE_DIR = osp.join(OUT_DIR, 'exp5_scores')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(SCORE_DIR, exist_ok=True)


# ── simulated changes (raw signal, 6 x T, before filtering) ──────────────────
def _tremor(sig, lo, hi, s, rng):
    t = np.arange(sig.shape[1]) / C.FS
    f = rng.uniform(lo, hi)
    osc = np.sin(2 * np.pi * f * t[None, :] + rng.uniform(0, 2 * np.pi, (sig.shape[0], 1)))
    return (sig + s * sig.std(axis=1, keepdims=True) * osc).astype(np.float32)


def _slow(sig, s):
    T = sig.shape[1]
    n = int(round(T * (1 + 0.3 * s)))
    x = np.linspace(0, T - 1, n)
    return np.stack([np.interp(x, np.arange(T), ch) for ch in sig])[:, :T].astype(np.float32)


CHANGE_FN = {
    'tremor_pd': lambda sig, s, rng: _tremor(sig, 4, 6, s, rng),
    'tremor_et': lambda sig, s, rng: _tremor(sig, 8, 12, s, rng),
    'slowing':   lambda sig, s, rng: _slow(sig, s),
    'irregular': lambda sig, s, rng: (sig + rng.normal(0, 1, sig.shape) * 0.5 * s
                                      * sig.std(axis=1, keepdims=True)).astype(np.float32),
    'amplitude': lambda sig, s, rng: (sig * (1 - 0.4 * s)).astype(np.float32),
}


def windows_of(sig):
    if sig.shape[1] < C.WINDOW_SIZE * C.N_WINDOWS:
        return None
    w = segment_windows(bandpass_filter(sig, lowcut=C.LOWCUT, highcut=C.HIGHCUT, fs=C.FS),
                        window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)[:C.N_WINDOWS]
    return w if len(w) == C.N_WINDOWS else None


def build(records, positions, change=None, s=0.0, seed=0):
    """Windows at the given positions of every recording (optionally changed). Returns W, subj."""
    W, S = [], []
    for i, r in enumerate(records):
        sig = r['signal']
        if change is not None:
            sig = CHANGE_FN[change](sig, s, np.random.RandomState(seed * 100003 + i))
        w = windows_of(sig)
        if w is None or windows_of(r['signal']) is None:      # same recordings in every version
            continue
        W.append(w[positions]); S += [r['subj']] * len(positions)
    return np.concatenate(W).astype(np.float32), np.array(S, dtype=np.int64)


def person_scores(base, fu, ctx):
    """Per-subject change scores for one follow-up version. base/fu: dict subj -> (act, p1)."""
    edges, direction, hc_counts = ctx['edges'], ctx['direction'], ctx['hc_counts']
    out = {m: {} for m in METHODS}
    for s, (ba, bp) in base.items():
        fa, fp = fu[s]
        pc = site_counts(ba, edges)
        out['pscan_2t'][s] = float(scan_a(pc, edges, fa, None).mean())
        out['pscan_A'][s] = float(scan_a(pc, edges, fa, direction).mean())
        out['hcscan_delta'][s] = float(scan_a(hc_counts, edges, fa, direction).mean()
                                       - scan_a(hc_counts, edges, ba, direction).mean())
        out['clf_trend'][s] = float(fp.mean() - bp.mean())
        out['clf_trend_abs'][s] = abs(out['clf_trend'][s])
        out['pmaha'][s] = float(Mahalanobis(ba).score(fa).mean())
    return out


def run_fold(k, fold, records, W_tr, dd_assign, labels_num):
    path = osp.join(SCORE_DIR, f'fold{k}.json')
    cached = json.load(open(path)) if osp.exists(path) else {}
    needed = ['none'] + [f'{c}@{s}' for c in CHANGES for s in STRENGTHS]
    if all(v in cached for v in needed):
        print(f"fold {k}: cached"); return cached

    sup = load_fold('none', k)
    edges = make_global_edges(sup['hc_train_act'], 10)
    ctx = {'edges': edges, 'direction': sup['val_pd_act'].mean(0) > sup['hc_train_act'].mean(0),
           'hc_counts': site_counts(sup['val_hc_act'], edges)}

    model, _ = build_classifier()
    model.load_state_dict(torch.load(osp.join(C.CKPT_DIR, f'patchtst_fold{k}_best.pt'), map_location=device))
    model.to(device).eval()
    mean, std = compute_normalization_stats(W_tr)

    monitored = set(int(s) for s in fold['test_subjects']) | {s for s, f in dd_assign.items() if f == k}
    recs = [r for r in records if r['subj'] in monitored]

    def feats(W, S):
        act, pool = encode(model.encoder, apply_normalization(W, mean, std).astype(np.float32), device)
        p1 = head_probs(model.classifier, pool, device)
        return {int(s): (act[S == s], p1[S == s]) for s in np.unique(S)}

    t0 = time.time()
    base = feats(*build(recs, BASE_POS))
    cached['labels'] = {str(s): int(labels_num[s]) for s in base}
    for v in needed:
        if v in cached:
            continue
        if v == 'none':
            fu = feats(*build(recs, FU_POS))
        else:
            c, s = v.split('@')
            fu = feats(*build(recs, FU_POS, c, float(s), seed=k))
        cached[v] = {m: {str(p): x for p, x in d.items()} for m, d in person_scores(base, fu, ctx).items()}
        json.dump(cached, open(path, 'w'))
        print(f"  fold {k}: {v:<16} done ({time.time() - t0:.0f} s)")
    return cached


def evaluate(folds_out):
    from sklearn.metrics import roc_auc_score  # noqa: F401  (bootstrap_auc_ci uses it)
    labels = {int(p): l for fo in folds_out for p, l in fo['labels'].items()}
    groups = {'all': lambda l: True, 'hc': lambda l: l == 0, 'patients': lambda l: l > 0}

    def collect(version, m, g):
        return np.array([x for fo in folds_out for p, x in fo[version][m].items() if groups[g](labels[int(p)])])

    res = {}
    for c in CHANGES:
        for s in STRENGTHS:
            v = f'{c}@{s}'
            res[v] = {}
            for g in groups:
                neg = {m: collect('none', m, g) for m in METHODS}
                pos = {m: collect(v, m, g) for m in METHODS}
                y = np.r_[np.zeros(len(neg[PRIMARY])), np.ones(len(pos[PRIMARY]))]
                rec = {'n_people': int(len(pos[PRIMARY])), 'methods': {}, 'paired': {}}
                for m in METHODS:
                    sc = np.r_[neg[m], pos[m]]
                    ci = bootstrap_auc_ci(y, sc, n_boot=N_BOOT)
                    thr = np.quantile(neg[m], 1 - FA_RATE)
                    rec['methods'][m] = {'auc': ci['auc'], 'ci': [ci['ci_low'], ci['ci_high']],
                                         'detection_at_5pct_false_alarm': round(float((pos[m] > thr).mean()), 4)}
                for b in COMPETITORS:
                    d = paired_bootstrap_auc_diff(y, np.r_[neg[PRIMARY], pos[PRIMARY]], np.r_[neg[b], pos[b]],
                                                  n_boot=N_BOOT)
                    rec['paired'][f'{PRIMARY} - {b}'] = {**d, 'wins': bool(d['ci_low'] > 0),
                                                        'loses': bool(d['ci_high'] < 0)}
                res[v][g] = rec

    # smallest strength reaching 50% detection at 5% false alarms
    min_strength = {c: {m: next((s for s in sorted(STRENGTHS)
                                 if res[f'{c}@{s}']['all']['methods'][m]['detection_at_5pct_false_alarm'] >= 0.5), None)
                        for m in METHODS} for c in CHANGES}

    rule = None
    rs = RULE_STRENGTH if RULE_STRENGTH in STRENGTHS else None
    tested = [c for c in NON_PD if c in CHANGES]
    if rs is not None and tested:
        per = {c: all(res[f'{c}@{rs}']['all']['paired'][f'{PRIMARY} - {b}']['wins']
                      for b in ['clf_trend', 'clf_trend_abs']) for c in tested}
        rule = {'statement': f'at strength {rs}, {PRIMARY} beats clf_trend and clf_trend_abs (paired AUC CI > 0) '
                             f'for >= 2 of {NON_PD}', 'per_change': per,
                'met': bool(sum(per.values()) >= 2) if len(tested) == len(NON_PD) else None}
    return res, min_strength, rule


def print_results(res, min_strength, rule):
    print(f"\n{'change@strength':<18} group     n | " + "  ".join(f"{m:>13}" for m in METHODS))
    for v, byg in res.items():
        for g, r in byg.items():
            print(f"{v:<18} {g:<8} {r['n_people']:>3} | " + "  ".join(
                f"{r['methods'][m]['auc']:.3f}/{r['methods'][m]['detection_at_5pct_false_alarm']:.2f}" for m in METHODS))
    print("\n(AUC changed vs unchanged / detection at 5% false alarms)")
    print("\nPaired, all monitored people:")
    for v, byg in res.items():
        for key, d in byg['all']['paired'].items():
            print(f"  {v:<18} {key:<26} {d['diff']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]"
                  f"{' WIN' if d['wins'] else ' LOSE' if d['loses'] else ''}")
    print("\nSmallest strength with >= 50% detection at 5% false alarms:")
    for c, d in min_strength.items():
        print(f"  {c:<10} " + "  ".join(f"{m} {d[m]}" for m in METHODS))
    print(f"\nRULE: {rule}")


if __name__ == '__main__':
    base_dir = pads_paths()
    manifest, HC, PD, DD = load_manifest(base_dir)
    records = pickle.load(open(RAW_CACHE, 'rb')) if osp.exists(RAW_CACHE) else load_raw(base_dir, manifest)
    labels_num = {r['subj']: (0 if r['label'] == HC else 1 if r['label'] == PD else 2) for r in records}

    # folds and training windows exactly as extract_features.py
    from common import window_records
    W0, L0, S0, _, _ = window_records(records, 'none', seed=C.RANDOM_STATE)
    N0 = np.where(L0 == HC, 0, np.where(L0 == PD, 1, 2))
    folds = make_folds(S0[N0 < 2], N0[N0 < 2])
    dd_assign = dd_fold_assignment(sorted(s for s, l in labels_num.items() if l == 2))

    out = []
    for k in FOLDS:
        fold = folds[k - 1]
        out.append(run_fold(k, fold, records, W0[np.isin(S0, fold['train_subjects'])], dd_assign, labels_num))
    res, min_strength, rule = evaluate(out)
    print_results(res, min_strength, rule)
    save_json({'results': res, 'min_strength_50pct_detection': min_strength, 'rule': rule,
               'config': {'changes': CHANGES, 'strengths': STRENGTHS, 'folds': FOLDS, 'baseline_positions': BASE_POS,
                          'followup_positions': FU_POS, 'false_alarm_rate': FA_RATE, 'methods': METHODS,
                          'git_commit': git_commit()}},
              'exp5_monitoring.json')
