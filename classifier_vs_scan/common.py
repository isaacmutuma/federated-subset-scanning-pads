"""
common.py — shared code for the classifier-vs-scan experiments (classifier_vs_scan/).

Self-contained: reads the main pipeline's checkpoints, config and src/ helpers, never writes to
the main results. Everything goes to OUT_DIR = RESULTS_DIR/classifier_vs_scan (override CVS_DIR).

Competitors (none uses labels from the deployment population):
  clf_K1  : PatchTST classifier P(PD), subject mean; threshold from local healthy volunteers
  clf_K2  : same classifier after re-normalising its 768 encoder features so local healthy
            volunteers match the training healthy statistics (feature-level adaptation)
  maha    : Mahalanobis distance of the 128-d scan activations from local healthy volunteers
  knn     : mean distance to the 10 nearest local healthy windows (standardised 128-d)
  scan_A  : histogram scan, Method A (1 - mean p), directed by the server's tail directions
  scan_A_2t : Method A, two-tailed (no PD information beyond the encoder)
  scan_C  : group subset scan (Method C) on directed p-values (ART, seeded per subject)

Referral rule used everywhere: refer anyone scoring above the 90th percentile of the local
healthy volunteers' subject scores (10% of healthy people referred). Local volunteers are scored
leave-one-subject-out, so a method fitted on them is not scored on its own training people.
"""

import sys, os, os.path as osp, json, zlib, subprocess, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(HERE)
sys.path.insert(0, osp.join(REPO, 'scripts'))
import config as C                                                # noqa: E402
sys.path.insert(0, C.REPO_DIR if osp.exists(C.REPO_DIR) else REPO)

from src.scanning.histograms import (make_global_edges, site_counts,  # noqa: E402
                                     pvalues_from_counts, bin_index)
from src.training.metrics import paired_bootstrap_auc_diff, bootstrap_auc_ci  # noqa: E402

OUT_DIR  = os.environ.get('CVS_DIR', osp.join(C.RESULTS_DIR, 'classifier_vs_scan'))
FEAT_DIR = os.environ.get('CVS_FEAT_DIR', osp.join(OUT_DIR, 'features'))
os.makedirs(OUT_DIR, exist_ok=True)

LAYER_IDX    = 2               # layer_2, as the main pipeline
N_BINS       = 10
ALPHA        = 1.0
A_MAX        = 0.5
RESTARTS     = 5
GROUP_SEED   = 0
REFER_RATE   = 0.10            # fraction of local healthy volunteers referred
KNN_K        = 10
N_BOOT       = 2000
RANDOM_STATE = 42
SPLITS       = ['hc_train', 'val_hc', 'val_pd', 'test', 'dd']
LOCAL_SPLITS = ['val_hc', 'test', 'dd']        # the deployment population (shifted in exp 2)


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def save_json(obj, name):
    path = osp.join(OUT_DIR, name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}")


# ── PADS loading (same windowing as scripts/extract_activations.py) ──────────
def pads_paths():
    for root, dirs, _ in os.walk(C.PADS_ROOT):
        if 'movement' in dirs and 'patients' in dirs:
            return root
    raise FileNotFoundError(f"PADS not found under {C.PADS_ROOT}")


def load_manifest(base):
    rows = []
    for pat in range(1, 470):
        with open(osp.join(base, 'patients', f'patient_{pat:03d}.json')) as f:
            label = json.load(f).get('condition')
        with open(osp.join(base, 'movement', f'observation_{pat:03d}.json')) as f:
            obs = json.load(f)
        for session in obs['session']:
            for record in session['records']:
                rows.append({'patient_id': pat, 'label': label, 'task': session.get('record_name'),
                             'wrist': record.get('device_location'), 'filepath': record.get('file_name')})
    m = pd.DataFrame(rows)
    labels = m['label'].unique().tolist()
    hc = next(l for l in labels if 'healthy' in l.lower() or l == 'HC')
    pd_ = next(l for l in labels if 'parkinson' in l.lower() or l == 'PD')
    dd = [l for l in labels if l not in (hc, pd_)]
    return m, hc, pd_, dd


def load_raw(base, manifest):
    """All recordings in memory: list of dicts with the raw (6, T) signal."""
    out = []
    for _, r in manifest.iterrows():
        fp = osp.join(base, 'movement', r['filepath'])
        try:
            sig = pd.read_csv(fp, header=None).iloc[:, 1:7].values.T.astype(np.float32)
        except Exception:
            continue
        out.append({'subj': int(r['patient_id']), 'label': r['label'], 'task': str(r['task']),
                    'wrist': str(r['wrist']), 'signal': sig})
    return out


# ── Population shifts, applied to the raw signal before filtering ────────────
def _rotate(sig, deg):
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]], dtype=np.float32)
    out = sig.copy()
    out[0:3], out[3:6] = R @ sig[0:3], R @ sig[3:6]
    return out


def _resample(sig, up, down):
    from scipy.signal import resample_poly
    return resample_poly(sig, up, down, axis=1).astype(np.float32)


def _noise(sig, frac, rng):
    return sig + rng.normal(0, 1, sig.shape).astype(np.float32) * frac * sig.std(axis=1, keepdims=True)


SHIFTS = {
    'none':     lambda s, rng: s,
    'amp0.7':   lambda s, rng: s * 0.7,                                   # weaker movement / gain
    'noise':    lambda s, rng: _noise(s, 0.3, rng),                      # cheaper, noisier sensor
    'rate25':   lambda s, rng: _resample(_resample(s, 1, 4), 4, 1),      # 25 Hz watch, upsampled
    'rotate30': lambda s, rng: _rotate(s, 30),                           # watch worn rotated
    'speed0.8': lambda s, rng: _resample(s, 5, 4),                       # movements 20% slower
    'combined': lambda s, rng: _noise(_rotate(_resample(_resample(_resample(s, 10, 9), 1, 4), 4, 1),
                                              20), 0.1, rng),            # slower + 25 Hz + rotated + noise
}


def window_records(records, shift='none', seed=0):
    from src.data.preprocessing import bandpass_filter, segment_windows
    rng = np.random.RandomState(seed)
    f = SHIFTS[shift]
    W, L, S, T, R = [], [], [], [], []
    for r in records:
        sig = f(r['signal'], rng)
        if sig.shape[1] < C.WINDOW_SIZE * C.N_WINDOWS:
            continue
        wins = segment_windows(bandpass_filter(sig, lowcut=C.LOWCUT, highcut=C.HIGHCUT, fs=C.FS),
                               window_size=C.WINDOW_SIZE, step=C.WINDOW_SIZE)[:C.N_WINDOWS]
        if len(wins) < C.N_WINDOWS:
            continue
        W.append(wins); n = len(wins)
        L += [r['label']] * n; S += [r['subj']] * n; T += [r['task']] * n; R += [r['wrist']] * n
    return (np.concatenate(W).astype(np.float32), np.array(L), np.array(S, dtype=np.int64),
            np.array(T), np.array(R))


def make_folds(subj, label_num):
    """Identical folds to the main pipeline (HC/PD subjects only)."""
    from src.data.folds import generate_fold_splits
    us = np.unique(subj)
    ul = np.array([label_num[subj == s][0] for s in us])
    return generate_fold_splits(us, ul, n_splits=C.N_SPLITS, random_state=C.RANDOM_STATE,
                                val_fraction=0.2, save_path=None)


# ── Model (identical architecture to scripts/train_patchtst.py) ──────────────
def build_classifier():
    import torch.nn as nn
    from transformers import PatchTSTConfig, PatchTSTModel
    cfg = PatchTSTConfig(num_input_channels=6, context_length=C.WINDOW_SIZE, patch_length=C.PATCH_LEN,
                         stride=C.STRIDE, d_model=C.D_MODEL, num_attention_heads=C.NUM_HEADS,
                         num_hidden_layers=C.NUM_LAYERS, ffn_dim=C.FFN_DIM, dropout=C.DROPOUT,
                         head_dropout=C.DROPOUT, pooling_type='mean', channel_attention=False,
                         scaling='std', loss='mse', num_targets=1)

    class PatchTSTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = PatchTSTModel(cfg)
            d = C.D_MODEL * 6
            self.classifier = nn.Sequential(nn.LayerNorm(d), nn.Dropout(C.DROPOUT), nn.Linear(d, 64),
                                            nn.GELU(), nn.Dropout(C.DROPOUT), nn.Linear(64, 2))
    return PatchTSTClassifier(), cfg


def encode(encoder, windows, device, batch=256):
    """Run PatchTSTModel; returns (act128 at LAYER_IDX, pooled 768) per window."""
    import torch
    store = {}
    h = encoder.encoder.layers[LAYER_IDX].register_forward_hook(
        lambda m, i, o: store.__setitem__('a', (o[0] if isinstance(o, tuple) else o).detach()))
    acts, pooled = [], []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            x = torch.tensor(windows[i:i + batch], device=device).permute(0, 2, 1)
            out = encoder(past_values=x)
            pooled.append(out.last_hidden_state.mean(dim=2).reshape(len(x), -1).cpu().numpy())
            acts.append(store['a'].mean(dim=(1, 2)).cpu().numpy())
    h.remove()
    return np.concatenate(acts), np.concatenate(pooled)


def head_probs(head, pooled, device):
    import torch
    with torch.no_grad():
        return torch.softmax(head(torch.tensor(pooled, device=device)), dim=1)[:, 1].cpu().numpy()


# ── Scores ───────────────────────────────────────────────────────────────────
def subject_scores(win_scores, subj):
    us = np.unique(subj)
    return us, np.array([np.nanmean(win_scores[subj == s]) for s in us])


def scan_a(counts, edges, acts, direction):
    return 1.0 - pvalues_from_counts(counts, edges, acts, direction, ALPHA).mean(axis=1)


_ART = {}
def scan_c_subject(p_subj, s):
    if not _ART:
        try:
            from art.defences.detector.evasion.subsetscanning.scanner import Scanner
            from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions
        except ImportError:
            os.system('pip install -q adversarial-robustness-toolbox')
            from art.defences.detector.evasion.subsetscanning.scanner import Scanner
            from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions
        _ART.update(Scanner=Scanner, SF=ScoringFunctions, failures=0)
    np.random.seed(zlib.crc32(f'{GROUP_SEED}-{s}'.encode()))
    try:
        score, _, _, _ = _ART['Scanner'].fgss_for_nets(
            pvalues=np.stack([p_subj, p_subj], axis=2).astype(np.float64), a_max=A_MAX,
            restarts=RESTARTS, score_function=_ART['SF'].get_score_bj_fast)
        return float(score)
    except Exception:
        _ART['failures'] += 1
        return np.nan


def scan_c_scores(counts, edges, acts, direction, subj):
    p = pvalues_from_counts(counts, edges, acts, direction, ALPHA)
    us = np.unique(subj)
    return us, np.array([scan_c_subject(p[subj == s], s) for s in us])


class Mahalanobis:
    def __init__(self, ref):
        from sklearn.covariance import LedoitWolf
        self.m = LedoitWolf().fit(ref)

    def score(self, x):
        return np.sqrt(np.maximum(self.m.mahalanobis(x), 0))


class KNN:
    def __init__(self, ref):
        from sklearn.neighbors import NearestNeighbors
        self.mu, self.sd = ref.mean(0), ref.std(0) + 1e-8
        self.nn = NearestNeighbors(n_neighbors=KNN_K).fit((ref - self.mu) / self.sd)

    def score(self, x):
        d, _ = self.nn.kneighbors((x - self.mu) / self.sd)
        return d.mean(axis=1)


def local_scores(ref_acts, ref_subj, edges, direction, method, eval_sets, run_c=False):
    """
    Scores for one fitted-on-local-volunteers method.
    eval_sets: {name: (acts, subj)} scored against the reference built from all volunteers.
    Returns {name: (subjects, scores)} plus 'val_hc_loso': each volunteer scored against a
    reference built from the other volunteers.
    """
    def fit(ra):
        if method in ('scan_A', 'scan_A_2t', 'scan_C'):
            return site_counts(ra, edges)
        return Mahalanobis(ra) if method == 'maha' else KNN(ra)

    def apply(model, acts, subj):
        if method == 'scan_A':
            return subject_scores(scan_a(model, edges, acts, direction), subj)
        if method == 'scan_A_2t':
            return subject_scores(scan_a(model, edges, acts, None), subj)
        if method == 'scan_C':
            return scan_c_scores(model, edges, acts, direction, subj)
        return subject_scores(model.score(acts), subj)

    full = fit(ref_acts)
    out = {name: apply(full, a, s) for name, (a, s) in eval_sets.items()}
    us = np.unique(ref_subj)
    loso = []
    for s in us:
        m = ref_subj == s
        model = (full - site_counts(ref_acts[m], edges)) if method.startswith('scan') else fit(ref_acts[~m])
        loso.append(apply(model, ref_acts[m], ref_subj[m])[1][0])
    out['val_hc_loso'] = (us, np.array(loso))
    return out


# ── Evaluation shared by all experiments ─────────────────────────────────────
def dd_fold_assignment(dd_subjects):
    """Each DD subject is scored by exactly one fold's model in the pooled analysis."""
    us = np.unique(dd_subjects)
    return {s: (i % C.N_SPLITS) + 1 for i, s in enumerate(us)}


def _boot_rate_diff(fa, fb, rng):
    n = len(fa)
    d = [(fa[i] .mean() - fb[i].mean()) for i in (rng.randint(0, n, n) for _ in range(N_BOOT))]
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {'diff': round(float(fa.mean() - fb.mean()), 4), 'ci_low': round(float(np.percentile(d, 2.5)), 4),
            'ci_high': round(float(np.percentile(d, 97.5)), 4), 'p': round(float(min(p, 1.0)), 4)}


def evaluate(fold_scores, dd_groups, primary, competitors):
    """
    fold_scores[k][method] = {'test': (subj, score, label01), 'dd': (subj, score),
                              'val_hc_loso': (subj, score)}   (higher = more anomalous)
    dd_groups: {dd_subject: diagnosis}. Returns pooled + per-fold metrics and paired comparisons
    of each primary method against each competitor. A comparison 'wins' if its CI is above 0.
    """
    from sklearn.metrics import roc_auc_score
    methods = list(fold_scores[0].keys())
    assign = dd_fold_assignment([s for s in dd_groups])
    pooled = {m: {'hc': [], 'pd': [], 'dd': [], 'dd_subj': [], 'hc_flag': [], 'pd_flag': [], 'dd_flag': []}
              for m in methods}
    per_fold = {m: {} for m in methods}
    for k, fs in enumerate(fold_scores, start=1):
        for m, d in fs.items():
            thr = np.quantile(d['val_hc_loso'][1], 1 - REFER_RATE)
            ts, sc, lab = d['test']
            ds, dsc = d['dd'] if 'dd' in d else (np.array([]), np.array([]))
            for comp, pos in [('pd', sc[lab == 1]), ('dd', dsc)]:
                if len(pos):
                    y = np.r_[np.zeros((lab == 0).sum()), np.ones(len(pos))]
                    per_fold[m].setdefault(comp, []).append(roc_auc_score(y, np.r_[sc[lab == 0], pos]))
            pooled[m]['hc'] += list(sc[lab == 0]); pooled[m]['pd'] += list(sc[lab == 1])
            pooled[m]['hc_flag'] += list(sc[lab == 0] > thr); pooled[m]['pd_flag'] += list(sc[lab == 1] > thr)
            mine = np.array([assign.get(s) == k for s in ds], dtype=bool)
            pooled[m]['dd'] += list(dsc[mine]); pooled[m]['dd_subj'] += list(ds[mine])
            pooled[m]['dd_flag'] += list(dsc[mine] > thr)

    def comparisons(m):
        P = pooled[m]
        dd_subj = np.array(P['dd_subj'])
        out = {'pd': (np.array(P['pd']), np.array(P['pd_flag']))}
        if len(P['dd']):
            out['dd'] = (np.array(P['dd']), np.array(P['dd_flag']))
            out['any'] = (np.r_[P['pd'], P['dd']], np.r_[P['pd_flag'], P['dd_flag']])
            for g in sorted(set(dd_groups.values())):
                sel = np.array([dd_groups[s] == g for s in dd_subj])
                if sel.sum() >= 5:
                    out[f'dd:{g}'] = (np.array(P['dd'])[sel], np.array(P['dd_flag'])[sel])
        return out

    res = {'n_hc': len(pooled[methods[0]]['hc']), 'methods': {}, 'paired': {}}
    for m in methods:
        hc, hcf = np.array(pooled[m]['hc']), np.array(pooled[m]['hc_flag'])
        rec = {'realised_specificity': round(float(1 - hcf.mean()), 4), 'comparisons': {}}
        for comp, (pos, flag) in comparisons(m).items():
            y, s = np.r_[np.zeros(len(hc)), np.ones(len(pos))], np.r_[hc, pos]
            ci = bootstrap_auc_ci(y, s, n_boot=N_BOOT)
            rec['comparisons'][comp] = {'n_pos': int(len(pos)), 'pooled_auc': ci['auc'],
                                        'ci': [ci['ci_low'], ci['ci_high']],
                                        'sensitivity_at_10pct_referral': round(float(flag.mean()), 4)}
            if comp in per_fold[m]:
                rec['comparisons'][comp]['fold_auc'] = [round(float(np.mean(per_fold[m][comp])), 4),
                                                        round(float(np.std(per_fold[m][comp])), 4)]
        res['methods'][m] = rec

    rng = np.random.RandomState(RANDOM_STATE)
    for a in primary:
        if a not in pooled:
            continue
        for b in competitors:
            if b not in pooled or b == a:
                continue
            ca, cb = comparisons(a), comparisons(b)
            hca, hcb = np.array(pooled[a]['hc']), np.array(pooled[b]['hc'])
            for comp in ca:
                if comp not in cb:
                    continue
                y = np.r_[np.zeros(len(hca)), np.ones(len(ca[comp][0]))]
                auc = paired_bootstrap_auc_diff(y, np.r_[hca, ca[comp][0]], np.r_[hcb, cb[comp][0]],
                                                n_boot=N_BOOT)
                sens = _boot_rate_diff(ca[comp][1].astype(float), cb[comp][1].astype(float), rng)
                res['paired'][f'{a} - {b} | {comp}'] = {
                    'auc': auc, 'sensitivity': sens,
                    'wins_auc': bool(auc['ci_low'] > 0), 'loses_auc': bool(auc['ci_high'] < 0),
                    'wins_sensitivity': bool(sens['ci_low'] > 0)}
    return res


def print_eval(res, comps=('pd', 'dd', 'any')):
    print(f"\n{'method':<12} {'spec':>6} | " + " | ".join(f"{c:>6} AUC [95% CI]   sens@10%" for c in comps))
    for m, r in res['methods'].items():
        cells = []
        for c in comps:
            v = r['comparisons'].get(c)
            cells.append(f"{v['pooled_auc']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]  {v['sensitivity_at_10pct_referral']:.3f}"
                         if v else f"{'-':>27}")
        print(f"{m:<12} {r['realised_specificity']:>6.3f} | " + " | ".join(cells))
    groups = sorted({c for r in res['methods'].values() for c in r['comparisons'] if c.startswith('dd:')})
    if groups:
        print("\nPer diagnosis (pooled AUC, sens@10%):")
        for g in groups:
            print(f"  {g:<40} " + "  ".join(
                f"{m} {r['comparisons'][g]['pooled_auc']:.3f}/{r['comparisons'][g]['sensitivity_at_10pct_referral']:.2f}"
                for m, r in res['methods'].items() if g in r['comparisons']))
    print("\nPaired (primary - competitor): AUC diff [CI] | sensitivity diff [CI]")
    for key, v in res['paired'].items():
        if key.split('| ')[1] in comps:
            a, s = v['auc'], v['sensitivity']
            print(f"  {key:<34} {a['diff']:+.3f} [{a['ci_low']:+.3f},{a['ci_high']:+.3f}]"
                  f"{' WIN' if v['wins_auc'] else ' LOSE' if v['loses_auc'] else ''} | "
                  f"{s['diff']:+.3f} [{s['ci_low']:+.3f},{s['ci_high']:+.3f}]{' WIN' if v['wins_sensitivity'] else ''}")


def load_fold(shift, k):
    return dict(np.load(osp.join(FEAT_DIR, shift, f'fold{k}.npz'), allow_pickle=True))
