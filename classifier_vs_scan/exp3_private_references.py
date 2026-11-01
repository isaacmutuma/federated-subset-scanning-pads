"""
exp3_private_references.py — Experiment 3: histograms vs simpler trackers at equal privacy.

Each method shares a different summary of the healthy volunteers. All are released with
person-level eps-differential privacy (Laplace mechanism; each volunteer contributes at most W_CAP
windows; activations standardised with the server's PADS statistics and clipped to [-B, B]):

  hist       : 128 x 10 counts (the scan reference)          L1 sensitivity 128 * W_CAP
  diag_gauss : per-neuron sum and sum of squares (256 numbers) 128 * B * W_CAP, 128 * B^2 * W_CAP
  full_gauss : sum and sum of outer products (Mahalanobis)     128 * B * W_CAP, (128 B)^2 * W_CAP
  knn        : needs the raw volunteer embeddings - no private release exists (no-noise row only)
(eps split equally between the two statistics of a Gaussian.) All of these merge exactly across
sites when summed, so mergeability does not separate them; accuracy and leakage at equal eps do.

Detection: reference = the fold's val-HC volunteers capped at W_CAP windows (as the privacy runs),
scored on the test set: fold AUC, pooled AUC, specificity at 90% sensitivity; N_DRAWS noise draws.
Membership: likelihood-ratio attack on releases of 1, 5, 10 volunteers (exact / fresh attacker),
as privacy_membership_v2.py, with each reference type's own likelihood.

Rule fixed before the run: at eps = 1 only the histogram keeps detection if hist - diag_gauss and
hist - full_gauss pooled AUC differences (paired bootstrap, scores averaged over draws) have CIs
above 0.

Writes OUT_DIR/exp3_private_references.json.

Run:
    python classifier_vs_scan/exp3_private_references.py
"""

import os
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from common import (C, git_commit, save_json, load_fold, make_global_edges, scan_a, bin_index,
                    subject_scores, KNN, bootstrap_auc_ci, paired_bootstrap_auc_diff, N_BOOT)
from src.training.metrics import spec_at_sens

EPSILONS    = [None, 10.0, 1.0, 0.3]
SIZES       = [1, 5, 10]
W_CAP       = 10
B           = 3.0
N_DRAWS     = int(os.environ.get('N_DRAWS', 5))
N_REP       = int(os.environ.get('N_REPEATS', 20))
N_REP_ONE   = int(os.environ.get('N_REPEATS_ONE', 100))
VAR_FLOOR   = 0.05
TYPES       = ['hist', 'diag_gauss', 'full_gauss']
RS          = 42


def lap(x, sens, eps, rng):
    return x if eps is None else x + rng.laplace(0.0, sens / eps, np.shape(x))


def release(kind, z, bins, eps, rng):
    """z: standardised + clipped activations of the contributed windows; bins: their bin indices."""
    d, n = z.shape[1], len(z)
    if kind == 'hist':
        counts = np.stack([(bins == b).sum(0) for b in range(10)], axis=1).astype(float)
        return np.clip(lap(counts, d * W_CAP, eps, rng), 0, None)
    e = None if eps is None else eps / 2
    s1 = lap(z.sum(0), d * B * W_CAP, e, rng)
    mu = s1 / n
    if kind == 'diag_gauss':
        s2 = lap((z ** 2).sum(0), d * B ** 2 * W_CAP, e, rng)
        return mu, np.maximum(s2 / n - mu ** 2, VAR_FLOOR)
    noise = np.zeros((d, d)) if e is None else np.triu(rng.laplace(0, (d * B) ** 2 * W_CAP / e, (d, d)))
    so = z.T @ z + noise + np.triu(noise, 1).T
    cov = so / n - np.outer(mu, mu)
    w, v = np.linalg.eigh((cov + cov.T) / 2)
    w = np.maximum(w, VAR_FLOOR)
    return mu, (v / w) @ v.T, float(np.log(w).sum())          # mean, inverse covariance, log det


def window_score(kind, ref, z, bins, edges, acts, direction):
    """Higher = more anomalous."""
    if kind == 'hist':
        return scan_a(ref, edges, acts, direction)
    if kind == 'diag_gauss':
        mu, var = ref
        t = (z - mu) / np.sqrt(var)
        return norm.cdf(np.where(direction, t, -t)).mean(axis=1)
    mu, icov, _ = ref
    diff = z - mu
    return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff, icov, diff), 0))


def log_lik(kind, ref, z, bins):
    """Per-window log-likelihood of windows under a released reference (membership attack)."""
    if kind == 'hist':
        lp = np.log((ref + 1.0) / (ref.sum(1, keepdims=True) + 10.0))
        return lp[np.arange(lp.shape[0]), bins].sum(axis=1)
    if kind == 'diag_gauss':
        mu, var = ref
        return (-0.5 * ((z - mu) ** 2 / var + np.log(var))).sum(axis=1)
    mu, icov, logdet = ref
    diff = z - mu
    return -0.5 * (np.einsum('ij,jk,ik->i', diff, icov, diff) + logdet)


def capped(subj, rng):
    con, fre = [], []
    for s in np.unique(subj):
        idx = rng.permutation(np.where(subj == s)[0])
        con.append(idx[:W_CAP]); fre.append(idx[W_CAP:2 * W_CAP])
    return np.concatenate(con), np.concatenate(fre)


det = {t: {str(e): [] for e in EPSILONS} for t in TYPES + ['knn']}       # per fold: (y, subj scores per draw)
mem = {t: {str(e): {f'{a}_size{s}': [] for a in ['exact', 'fresh'] for s in SIZES} for e in EPSILONS} for t in TYPES}

for k in range(1, C.N_SPLITS + 1):
    print(f"\n=== FOLD {k} ===")
    rng = np.random.RandomState(RS + k)
    d = load_fold('none', k)
    hc_tr = d['hc_train_act']
    edges = make_global_edges(hc_tr, 10)
    direction = d['val_pd_act'].mean(0) > hc_tr.mean(0)
    m_s, s_s = hc_tr.mean(0), hc_tr.std(0) + 1e-8
    std = lambda a: np.clip((a - m_s) / s_s, -B, B)

    test, ts, tl = d['test_act'], d['test_subj'], d['test_label']
    us = np.unique(ts); y = np.array([tl[ts == s][0] for s in us])
    zt, bt = std(test), bin_index(test, edges)
    vh, vs = d['val_hc_act'], d['val_hc_subj']
    con, _ = capped(vs, rng)
    zc, bc = std(vh[con]), bin_index(vh[con], edges)

    det['knn']['None'].append((y, [subject_scores(KNN(vh[con]).score(test), ts)[1]]))
    for t in TYPES:
        for e in EPSILONS:
            draws = [subject_scores(window_score(t, release(t, zc, bc, e, rng), zt, bt, edges, test, direction), ts)[1]
                     for _ in range(1 if e is None else N_DRAWS)]
            det[t][str(e)].append((y, draws))
        print(f"  detection {t}: " + "  ".join(
            f"eps={e}: {np.mean([roc_auc_score(y, s) for s in det[t][str(e)][-1][1]]):.3f}" for e in EPSILONS))

    # membership on the HC pool (train + val volunteers)
    pool = np.concatenate([hc_tr, vh]); psub = np.concatenate([d['hc_train_subj'], vs])
    pc, pf = capped(psub, rng)
    Z = {'exact': (std(pool[pc]), bin_index(pool[pc], edges), psub[pc]),
         'fresh': (std(pool[pf]), bin_index(pool[pf], edges), psub[pf])}
    zcon, bcon, scon = Z['exact']
    subjects = np.unique(psub)
    for t in TYPES:
        for e in EPSILONS:
            for size in SIZES:
                aucs = {a: [] for a in Z}
                for _ in range(N_REP_ONE if size == 1 else N_REP):
                    perm = rng.permutation(subjects)
                    pop, rest = perm[:len(perm) // 3], perm[len(perm) // 3:]
                    site, non = rest[:size], rest[size:]
                    ms, mp = np.isin(scon, site), np.isin(scon, pop)
                    rel = release(t, zcon[ms], bcon[ms], e, rng)
                    popr = release(t, zcon[mp], bcon[mp], None, rng)
                    lab = np.r_[np.ones(len(site)), np.zeros(len(non))]
                    for a, (za, ba, sa) in Z.items():
                        sc = [(log_lik(t, rel, za[sa == s], ba[sa == s]) - log_lik(t, popr, za[sa == s], ba[sa == s])).mean()
                              for s in np.r_[site, non]]
                        aucs[a].append(roc_auc_score(lab, sc))
                for a in Z:
                    mem[t][str(e)][f'{a}_size{size}'].append(float(np.mean(aucs[a])))
        print(f"  membership {t} size1 exact: " + "  ".join(
            f"eps={e}: {mem[t][str(e)]['exact_size1'][-1]:.3f}" for e in EPSILONS))

# ── summary ──────────────────────────────────────────────────────────────────
def pooled(t, e):
    ys = np.concatenate([f[0] for f in det[t][e]])
    sc = np.concatenate([np.mean(f[1], axis=0) for f in det[t][e]])
    return ys, sc

summary = {'detection': {}, 'membership': {}, 'paired_at_eps1': {}}
for t in TYPES + ['knn']:
    for e in det[t]:
        if not det[t][e]:
            continue
        fold_auc = [np.mean([roc_auc_score(f[0], s) for s in f[1]]) for f in det[t][e]]
        ys, sc = pooled(t, e)
        ci = bootstrap_auc_ci(ys, sc, n_boot=N_BOOT)
        summary['detection'][f'{t}|eps={e}'] = {
            'fold_auc': [round(float(np.mean(fold_auc)), 4), round(float(np.std(fold_auc)), 4)],
            'pooled_auc': ci['auc'], 'ci': [ci['ci_low'], ci['ci_high']],
            'spec_at_90sens': spec_at_sens(ys, sc, 0.9)}
for t in TYPES:
    for e in mem[t]:
        summary['membership'][f'{t}|eps={e}'] = {key: round(float(np.mean(v)), 4) for key, v in mem[t][e].items()}
yh, sh = pooled('hist', '1.0')
for t in ['diag_gauss', 'full_gauss']:
    _, so = pooled(t, '1.0')
    summary['paired_at_eps1'][f'hist - {t}'] = paired_bootstrap_auc_diff(yh, sh, so, n_boot=N_BOOT)
rule = all(v['ci_low'] > 0 for v in summary['paired_at_eps1'].values())
summary['rule'] = {'statement': 'at eps=1, hist beats diag_gauss and full_gauss on pooled AUC (CI above 0)', 'met': bool(rule)}

print(f"\n{'reference':<12} {'eps':>5} | {'fold AUC':>14} {'pooled AUC [CI]':>22} {'spec@90':>8} | "
      f"MIA exact/fresh size1  size5  size10")
for key, v in summary['detection'].items():
    t, e = key.split('|eps=')
    m = summary['membership'].get(key, {})
    mia = (f"{m['exact_size1']:.3f}/{m['fresh_size1']:.3f}  {m['exact_size5']:.3f}/{m['fresh_size5']:.3f}  "
           f"{m['exact_size10']:.3f}/{m['fresh_size10']:.3f}") if m else 'no private release'
    print(f"{t:<12} {e:>5} | {v['fold_auc'][0]:.3f} ± {v['fold_auc'][1]:.3f}  {v['pooled_auc']:.3f} "
          f"[{v['ci'][0]:.3f},{v['ci'][1]:.3f}] {v['spec_at_90sens']:>8.3f} | {mia}")
for key, v in summary['paired_at_eps1'].items():
    print(f"  eps=1  {key}: {v['diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]  p={v['p']:.3f}")
print(f"RULE: {summary['rule']}")

save_json({'config': {'epsilons': EPSILONS, 'sizes': SIZES, 'w_cap': W_CAP, 'clip_B': B, 'n_draws': N_DRAWS,
                      'n_repeats': N_REP, 'n_repeats_one': N_REP_ONE, 'git_commit': git_commit()},
           'summary': summary}, 'exp3_private_references.json')
