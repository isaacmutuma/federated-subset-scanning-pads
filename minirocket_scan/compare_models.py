"""
compare_models.py
-----------------
MiniROCKET side pipeline, last step: PatchTST (Layer 2) vs MiniROCKET features, side by side,
from the result JSONs both pipelines write. Same subjects, folds, reference and scripts.

Decision rule (fixed before the results are seen):
  1. Primary: paired bootstrap of the Method A pooled subject-level AUC difference
     (MiniROCKET - PatchTST) on the same 355 subjects. If the 95% CI excludes 0, the model
     with the higher AUC is better at detection and is chosen.
  2. If the CI includes 0, detection is tied and the choice goes to the deployment criteria,
     in order: lower membership-inference AUC on unnoised site histograms (privacy), higher
     Method A AUC under eps = 1 noise (utility under DP), smaller site histogram (bytes).

Writes RESULTS_DIR/minirocket_scan/model_comparison.json.

Run:
    python minirocket_scan/compare_models.py
"""

import sys, os, os.path as osp, json

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO_ROOT, 'scripts'))
sys.path.insert(0, REPO_ROOT)
import config as C
from src.training.metrics import paired_bootstrap_auc_diff

PT_DIR = os.environ.get('PT_DIR', C.RESULTS_DIR)
MR_DIR = os.environ.get('MR_RESULTS_DIR', osp.join(C.RESULTS_DIR, 'minirocket_scan'))
MODELS = {'PatchTST (layer_2)': (PT_DIR, 'layer_2'), 'MiniROCKET': (MR_DIR, 'minirocket')}
METHODS = [('mean', 'A'), ('individ', 'B'), ('group', 'C')]


def load(d, name):
    p = osp.join(d, name)
    if not osp.exists(p):
        print(f'  [missing] {p}')
        return None
    with open(p) as f:
        return json.load(f)


def fmt(m, s):
    return f'{m:.3f} ± {s:.3f}'


out = {'decision_rule': __doc__.split('Decision rule')[1].split('Writes')[0].strip(), 'models': {}}

# ── Supervised classifiers (same folds) ───────────────────────────────────────
print('=== Supervised classifier (subject-level, 5-fold) ===')
for name, f in [('PatchTST', 'patchtst_results.json'), ('MiniROCKET', 'minirocket_results.json')]:
    r = load(PT_DIR, f)
    if r:
        a = r['aggregate']
        print(f"  {name:<12} AUC {fmt(a['auc_mean'], a['auc_std'])}  "
              f"spec@90%sens {fmt(a.get('spec_at_90sens_mean', np.nan), a.get('spec_at_90sens_std', np.nan))}")
        out.setdefault('supervised', {})[name] = a

# ── Per-model scanning, federation and privacy results ────────────────────────
res = {}
for name, (d, layer) in MODELS.items():
    res[name] = {'deepscan': load(d, f'deepscan_results_{layer}.json'),
                 'federated': load(d, f'federated_sim_{layer}.json'),
                 'reid': load(d, 'privacy_bins_sweep.json'),
                 'membership': load(d, f'privacy_membership_{layer}.json')}

print('\n=== DeepScan detection, PD vs HC (held-out HC reference, N=10) ===')
for name, r in res.items():
    ds = r['deepscan']
    if not ds:
        continue
    rec = {'commit': ds['config']['git_commit'][:7]}
    print(f"  {name}  (commit {rec['commit']})")
    for m, lab in METHODS:
        a = ds.get(f'aggregate_{m}')
        if a:
            ci = a['pooled_bootstrap']
            print(f"     {lab}: AUC {fmt(a['auc_mean'], a['auc_std'])}  pooled {ci['auc']:.3f} "
                  f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  spec@90%sens {a['spec_at_90sens_mean']:.3f}")
            rec[lab] = {'auc_mean': a['auc_mean'], 'auc_std': a['auc_std'], 'pooled': ci,
                        'spec_at_90sens_mean': a['spec_at_90sens_mean']}
    out['models'][name] = rec

# Paired model comparison on the same subjects
pt, mr = res['PatchTST (layer_2)']['deepscan'], res['MiniROCKET']['deepscan']
if pt and mr:
    sp, sm = pt['subject_scores'], mr['subject_scores']
    assert sp['subj'] == sm['subj'] and sp['label'] == sm['label'], \
        'Subject order differs between pipelines: the comparison would not be paired'
    y = np.array(sp['label'])
    out['paired_mr_minus_pt'] = {}
    print(f'\n=== Paired difference MiniROCKET − PatchTST (same {len(y)} subjects) ===')
    for m, lab in METHODS:
        if m in sp and m in sm:
            a, b = np.array(sm[m], dtype=float), np.array(sp[m], dtype=float)
            keep = ~(np.isnan(a) | np.isnan(b))
            d = paired_bootstrap_auc_diff(y[keep], a[keep], b[keep])
            out['paired_mr_minus_pt'][lab] = d
            print(f"     {lab}: {d['diff']:+.3f}  95% CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  p={d['p']:.3f}")

print('\n=== Federated reference (3 pseudo-sites) ===')
for name, r in res.items():
    f = r['federated']
    if f:
        s = f['summary']
        print(f"  {name:<20} central {fmt(*s['central_auc'])} | federated {fmt(*s['federated_auc_mean'])} | "
              f"local-only {fmt(*s['local_auc_mean'])} | {f['bytes_per_site_counts']:,} bytes/site")
        out['models'].setdefault(name, {})['federated'] = {**s, 'bytes_per_site': f['bytes_per_site_counts']}

print('\n=== Re-identification (closed-set, Top-1; chance ≈ 2%) ===')
for name, r in res.items():
    p = r['reid']
    if p:
        b = p['per_bin']['10']
        print(f"  {name:<20} raw {p['raw_reid']['top1_mean']:.2%} | p-values N=10 {b['reid_top1_mean']:.2%} | "
              f"quantised N=10 {b['quant_reid_top1_mean']:.2%}")
        out['models'].setdefault(name, {})['reid_top1'] = {
            'raw': p['raw_reid']['top1_mean'], 'pvalues_n10': b['reid_top1_mean'],
            'quantised_n10': b['quant_reid_top1_mean']}

print('\n=== Membership inference on shared histograms (AUC; 0.5 = no leakage) + utility ===')
for name, r in res.items():
    mm = r['membership']
    if mm:
        s = mm['summary']
        print(f"  {name}")
        for size in mm['config']['site_sizes']:
            row = '  '.join(f"eps={str(e):>4}: {s['membership'][f'size{size}_eps{e}']:.3f}"
                            for e in mm['config']['epsilons'])
            print(f"     site {size:>2} HC   {row}")
        print('     Method A AUC   ' + '  '.join(f"{k}: {v:.3f}" for k, v in s['utility'].items()))
        out['models'].setdefault(name, {})['membership'] = s

# ── Apply the decision rule ───────────────────────────────────────────────────
print('\n=== Decision ===')
names = list(MODELS)
pa = out.get('paired_mr_minus_pt', {}).get('A')
verdict = None
if pa:
    if pa['ci_low'] > 0:
        verdict = ('MiniROCKET', 'Method A AUC higher; paired 95% CI excludes 0')
    elif pa['ci_high'] < 0:
        verdict = ('PatchTST (layer_2)', 'Method A AUC higher; paired 95% CI excludes 0')
    else:
        print(f"  Detection tied (Method A difference {pa['diff']:+.3f}, CI includes 0) -> deployment criteria")
        crit = []
        mem = {n: out['models'][n]['membership']['membership'] for n in names
               if 'membership' in out['models'].get(n, {})}
        if len(mem) == 2:
            leak = {n: np.nanmean([v for k, v in m.items() if k.endswith('epsNone')]) for n, m in mem.items()}
            util = {n: out['models'][n]['membership']['utility']['eps1.0'] for n in names}
            crit.append(('lower unnoised membership AUC (mean over site sizes)', leak, min))
            crit.append(('higher Method A AUC at eps = 1', util, max))
        fed = {n: out['models'][n]['federated']['bytes_per_site'] for n in names
               if 'federated' in out['models'].get(n, {})}
        if len(fed) == 2:
            crit.append(('smaller site histogram (bytes)', fed, min))
        for desc, vals, pick in crit:
            vs = '  '.join(f'{n}: {v:.4g}' for n, v in vals.items())
            print(f'  {desc}: {vs}')
            v = np.array(list(vals.values()), dtype=float)
            if verdict is None and not np.isnan(v).any() and len(set(np.round(v, 4))) > 1:
                verdict = (pick(vals, key=vals.get), f'detection tied; decided by {desc}')
if verdict:
    print(f'  -> Better model: {verdict[0]}  ({verdict[1]})')
    out['verdict'] = {'model': verdict[0], 'reason': verdict[1]}
else:
    print('  -> Not decidable from the available files')

with open(osp.join(MR_DIR, 'model_comparison.json'), 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {osp.join(MR_DIR, 'model_comparison.json')}")
