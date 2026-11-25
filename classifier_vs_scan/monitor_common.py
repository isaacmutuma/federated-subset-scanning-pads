"""
monitor_common.py — shared code for the real-data monitoring experiments (exp6 Habets, exp7 IMU50).

Everything the watch would hold comes from the PADS server side of fold k: the frozen PatchTST
(patchtst_fold{k}_best.pt), the per-channel normalisation of its training windows, the bin edges
(hc_train activations) and the PD tail directions (val_pd vs hc_train). The personal reference is built
on the watch from the person's own baseline windows; nothing about the monitored person is ever sent.

Segment = one block of windows (Habets: the PRE_MIN minutes before a symptom report on one wrist;
IMU50: the first minutes of a daytime hour). Change scores per segment (higher = more different):
  pscan_2t  : 1 - mean two-tailed p-value vs the personal baseline histogram (server edges)   <- primary
  pscan_A   : same, directed by the server's PD tail directions
  hcscan    : healthy-reference scan A (PADS healthy volunteers), level, not change
  clf       : mean classifier P(PD), level
  pmaha     : mean Mahalanobis distance from the personal baseline windows (128-d)
  intensity : mean acc-magnitude SD (g) — trivial activity baseline; if it tracks the outcome as well
              as the scan, the scan adds nothing beyond "how much the person moved"
Alarm rule (fixed): a segment alarms if its score exceeds the 95th percentile of the person's baseline
segments, each baseline segment scored against the baseline without itself (leave-one-segment-out).
"""

import os, os.path as osp, json, pickle
import numpy as np
import torch

from common import (C, OUT_DIR, pads_paths, load_manifest, load_raw, make_folds, window_records,
                    build_classifier, encode, head_probs, load_fold, make_global_edges, site_counts,
                    scan_a, Mahalanobis)
from src.data.preprocessing import compute_normalization_stats, apply_normalization

FOLDS    = [int(x) for x in os.environ.get('FOLDS', '1,2,3,4,5').split(',')]
METHODS  = ['pscan_2t', 'pscan_A', 'hcscan', 'clf', 'pmaha', 'intensity']
ALARM_Q  = 0.95
MIN_WIN  = int(os.environ.get('MIN_WIN', '10'))      # segments with fewer windows are dropped
RAW_CACHE = os.environ.get('RAW_CACHE', '/content/cvs_raw.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def norm_stats(k):
    """Per-channel mean/std of fold k's PADS training windows (cached: PADS is loaded only once)."""
    path = osp.join(OUT_DIR, f'norm_fold{k}.json')
    if osp.exists(path):
        d = json.load(open(path)); return np.array(d['mean'], np.float32), np.array(d['std'], np.float32)
    base = pads_paths()
    manifest, HC, PD, _ = load_manifest(base)
    records = pickle.load(open(RAW_CACHE, 'rb')) if osp.exists(RAW_CACHE) else load_raw(base, manifest)
    W, L, S, _, _ = window_records(records, 'none', seed=C.RANDOM_STATE)
    N = np.where(L == HC, 0, np.where(L == PD, 1, 2))
    folds = make_folds(S[N < 2], N[N < 2])
    for j, f in enumerate(folds, start=1):
        m, s = compute_normalization_stats(W[np.isin(S, f['train_subjects'])])
        json.dump({'mean': m.tolist(), 'std': s.tolist()}, open(osp.join(OUT_DIR, f'norm_fold{j}.json'), 'w'))
    d = json.load(open(path)); return np.array(d['mean'], np.float32), np.array(d['std'], np.float32)


class Server:
    """What fold k's server ships to the watch."""
    def __init__(self, k):
        sup = load_fold('none', k)
        self.edges = make_global_edges(sup['hc_train_act'], 10)
        self.direction = sup['val_pd_act'].mean(0) > sup['hc_train_act'].mean(0)
        self.hc_counts = site_counts(sup['val_hc_act'], self.edges)
        self.mean, self.std = norm_stats(k)
        self.model, _ = build_classifier()
        self.model.load_state_dict(torch.load(osp.join(C.CKPT_DIR, f'patchtst_fold{k}_best.pt'), map_location=device))
        self.model.to(device).eval()

    def features(self, windows):
        x = apply_normalization(windows.astype(np.float32), self.mean, self.std).astype(np.float32)
        act, pooled = encode(self.model.encoder, x, device)
        return act, head_probs(self.model.classifier, pooled, device)


def segment_scores(srv, act, p1, inten, seg, base_segs):
    """
    act/p1/inten: per window; seg: segment id per window; base_segs: ids forming the personal baseline.
    Returns {segment: {method: score}} for every segment with >= MIN_WIN windows, and the alarm
    threshold per method (LOSO over baseline segments).
    """
    segs = [s for s in np.unique(seg) if (seg == s).sum() >= MIN_WIN]
    base = [s for s in segs if s in set(base_segs)]
    if len(base) < 3:
        return None, None
    bmask = np.isin(seg, base)
    pc = site_counts(act[bmask], srv.edges)
    maha = Mahalanobis(act[bmask])

    def score(m, counts, mh):
        a, p, i = act[m], p1[m], inten[m]
        return {'pscan_2t': float(scan_a(counts, srv.edges, a, None).mean()),
                'pscan_A': float(scan_a(counts, srv.edges, a, srv.direction).mean()),
                'hcscan': float(scan_a(srv.hc_counts, srv.edges, a, srv.direction).mean()),
                'clf': float(p.mean()), 'pmaha': float(mh.score(a).mean()), 'intensity': float(i.mean())}

    out = {s: score(seg == s, pc, maha) for s in segs if s not in base}
    loso = []
    for s in base:
        m = seg == s
        rest = bmask & ~m
        loso.append(score(m, pc - site_counts(act[m], srv.edges), Mahalanobis(act[rest])))
    thr = {k: float(np.quantile([d[k] for d in loso], ALARM_Q)) for k in METHODS}
    for s, d in zip(base, loso):
        out[s] = {**d, 'baseline': True}
    return out, thr


def boot_mean_ci(x, n_boot=2000, seed=42):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return {'mean': None, 'ci': [None, None], 'n': 0}
    rng = np.random.RandomState(seed)
    b = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n_boot)]
    p = 2 * min((np.array(b) <= 0).mean(), (np.array(b) >= 0).mean())
    return {'mean': round(float(x.mean()), 4), 'ci': [round(float(np.percentile(b, 2.5)), 4),
            round(float(np.percentile(b, 97.5)), 4)], 'p': round(float(min(p, 1.0)), 4), 'n': int(len(x))}
