"""
extract_features.py — features for the classifier-vs-scan experiments.

For every population shift and fold, loads the main pipeline's PatchTST checkpoint (read only) and
saves, per split, the 128-d layer-2 scan activations and the classifier's P(PD) without (K1) and
with (K2) local feature re-normalisation. Splits:

  hc_train, val_pd : server side (PADS, never shifted) -> bin edges, tail directions, K2 target stats
  val_hc           : local healthy volunteers (shifted)
  test             : HC + PD test subjects of the fold (shifted)
  dd               : all 114 subjects with other movement disorders (shifted; in no fold, so every
                     fold's model scores all of them)

K2: pooled 768-d encoder features are re-normalised so the local volunteers' mean/std match the
training healthy subjects', then passed through the classifier head. val_hc also gets K2
probabilities computed leave-one-subject-out (val_hc_p2_loso) for the referral threshold.

Writes FEAT_DIR/<shift>/fold<k>.npz and FEAT_DIR/dd_diagnosis.json. Sanity check printed for
shift 'none': classifier AUC (~0.904) and Method A AUC (~0.860) per fold.

Run:
    SHIFTS=none python classifier_vs_scan/extract_features.py
    SHIFTS=none,amp0.7,noise,rate25,rotate30,speed0.8,combined python classifier_vs_scan/extract_features.py
"""

import os, os.path as osp, json, pickle, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import (C, FEAT_DIR, SHIFTS, LOCAL_SPLITS, pads_paths, load_manifest, load_raw,
                    window_records, make_folds, build_classifier, encode, head_probs,
                    make_global_edges, site_counts, scan_a, subject_scores)
from src.data.preprocessing import compute_normalization_stats, apply_normalization

SHIFT_LIST = os.environ.get('SHIFTS', 'none').split(',')
FORCE      = os.environ.get('FORCE', '0') == '1'
RAW_CACHE  = os.environ.get('RAW_CACHE', '/content/cvs_raw.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for s in SHIFT_LIST:
    assert s in SHIFTS, f"unknown shift {s}; options {list(SHIFTS)}"
print(f"Device {device} | shifts {SHIFT_LIST} | features -> {FEAT_DIR}")

base = pads_paths()
manifest, HC, PD, DD = load_manifest(base)
print(f"HC='{HC}' PD='{PD}' DD labels={DD}")
if osp.exists(RAW_CACHE):
    with open(RAW_CACHE, 'rb') as f:
        records = pickle.load(f)
else:
    t0 = time.time()
    records = load_raw(base, manifest)
    try:
        with open(RAW_CACHE, 'wb') as f:
            pickle.dump(records, f)
    except OSError:
        pass
    print(f"Loaded {len(records)} recordings in {time.time() - t0:.0f} s")

os.makedirs(FEAT_DIR, exist_ok=True)
dd_diag = {int(r['subj']): r['label'] for r in records if r['label'] in DD}
with open(osp.join(FEAT_DIR, 'dd_diagnosis.json'), 'w') as f:
    json.dump({str(k): v for k, v in sorted(dd_diag.items())}, f, indent=1)
print(f"DD subjects: {len(dd_diag)}  " + ", ".join(f"{g}: {sum(v == g for v in dd_diag.values())}"
                                               for g in sorted(set(dd_diag.values()))))


def windows_for(shift):
    W, L, S, T, R = window_records(records, shift, seed=C.RANDOM_STATE)
    num = np.where(L == HC, 0, np.where(L == PD, 1, 2))
    print(f"  [{shift}] windows {W.shape}  HC {(num == 0).sum()}  PD {(num == 1).sum()}  DD {(num == 2).sum()}")
    return W, num, S, T, R


clean = windows_for('none')
W0, N0, S0, T0, R0 = clean
hcpd = N0 < 2
folds = make_folds(S0[hcpd], N0[hcpd])

for shift in SHIFT_LIST:
    W1, N1, S1, T1, R1 = clean if shift == 'none' else windows_for(shift)
    os.makedirs(osp.join(FEAT_DIR, shift), exist_ok=True)
    for k, fold in enumerate(folds, start=1):
        out_path = osp.join(FEAT_DIR, shift, f'fold{k}.npz')
        if osp.exists(out_path) and not FORCE:
            print(f"  [{shift}] fold {k}: exists, skipped")
            continue
        model, _ = build_classifier()
        model.load_state_dict(torch.load(osp.join(C.CKPT_DIR, f'patchtst_fold{k}_best.pt'), map_location=device))
        model.to(device).eval()

        tr0 = np.isin(S0, fold['train_subjects'])
        mean, std = compute_normalization_stats(W0[tr0])          # as extract_activations.py
        masks = {'hc_train': (0, tr0 & (N0 == 0)),
                 'val_pd':   (0, np.isin(S0, fold['val_subjects']) & (N0 == 1)),
                 'val_hc':   (1, np.isin(S1, fold['val_subjects']) & (N1 == 0)),
                 'test':     (1, np.isin(S1, fold['test_subjects'])),
                 'dd':       (1, N1 == 2)}
        data, pooled = {}, {}
        for split, (src, m) in masks.items():
            W, Nn, S, T, R = (W0, N0, S0, T0, R0) if src == 0 else (W1, N1, S1, T1, R1)
            act, pool = encode(model.encoder, apply_normalization(W[m], mean, std).astype(np.float32), device)
            pooled[split] = pool
            data.update({f'{split}_act': act.astype(np.float32), f'{split}_p1': head_probs(model.classifier, pool, device),
                         f'{split}_subj': S[m], f'{split}_label': Nn[m], f'{split}_task': T[m], f'{split}_wrist': R[m]})

        # K2: align local volunteers' feature statistics to the training healthy subjects'
        mu_t, sd_t = pooled['hc_train'].mean(0), pooled['hc_train'].std(0) + 1e-6
        def k2(pool, ref):
            return head_probs(model.classifier,
                              ((pool - ref.mean(0)) / (ref.std(0) + 1e-6) * sd_t + mu_t).astype(np.float32), device)
        for split in ['hc_train', 'val_pd']:
            data[f'{split}_p2'] = data[f'{split}_p1']
        for split in LOCAL_SPLITS:
            data[f'{split}_p2'] = k2(pooled[split], pooled['val_hc'])
        vs = data['val_hc_subj']
        loso = np.empty(len(vs), dtype=np.float32)
        for s in np.unique(vs):
            loso[vs == s] = k2(pooled['val_hc'][vs == s], pooled['val_hc'][vs != s])
        data['val_hc_p2_loso'] = loso
        np.savez_compressed(out_path, **data)

        # sanity check
        ts, lab = data['test_subj'], data['test_label']
        us, p1 = subject_scores(data['test_p1'], ts)
        y = np.array([lab[ts == s][0] for s in us])
        edges = make_global_edges(data['hc_train_act'], 10)
        direction = data['val_pd_act'].mean(0) > data['hc_train_act'].mean(0)
        _, sa = subject_scores(scan_a(site_counts(data['val_hc_act'], edges), edges, data['test_act'], direction), ts)
        print(f"  [{shift}] fold {k}: classifier AUC {roc_auc_score(y, p1):.4f}  Method A AUC {roc_auc_score(y, sa):.4f}"
              f"  (DD windows {len(data['dd_act'])})")

print("DONE.")
