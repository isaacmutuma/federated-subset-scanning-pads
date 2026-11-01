"""
exp4_label_free_encoder.py — Experiment 4: a pipeline that never uses patient labels.

A classifier needs labelled patients by definition. Here the encoder is trained WITHOUT labels by
PatchTST's masked-patch reconstruction (an encoder-decoder pretext task; same architecture as the
supervised model), then its layer-2 activations are scanned against local healthy volunteers:

  ssl_hc  : pretrained on the fold's healthy training windows only -> no patient data anywhere
  ssl_all : pretrained on all training windows (HC + PD), labels discarded

Scored with scan_A_2t (two-tailed: no PD information at all), maha and knn; scan_A (directions
from labelled PD validation data) is reported for reference. Compared with the supervised
classifier (clf_K1) and the supervised encoder's scans (sup:scan_A, sup:scan_A_2t), on HC vs PD,
HC vs other disorders (DD) and HC vs PD + DD.

No pass/fail rule: the result is the price in AUC of a pipeline with no patient labels (and, for
ssl_hc, no patient data), stated next to what it removes.

Checkpoints: OUT_DIR/ssl_ckpt/<variant>_fold<k>.pt; features: OUT_DIR/features_ssl/<variant>/.
Writes OUT_DIR/exp4_label_free_encoder.json.

Run (GPU recommended):
    python classifier_vs_scan/exp4_label_free_encoder.py          # EPOCHS=40 by default
"""

import os, os.path as osp, json, pickle, time
import numpy as np
import torch
from transformers import PatchTSTConfig, PatchTSTForPretraining, PatchTSTModel

from common import (C, OUT_DIR, FEAT_DIR, git_commit, save_json, pads_paths, load_manifest, load_raw,
                    window_records, make_folds, encode, load_fold, make_global_edges, subject_scores,
                    local_scores, evaluate, print_eval)
from src.data.preprocessing import compute_normalization_stats, apply_normalization

VARIANTS  = os.environ.get('VARIANTS', 'ssl_hc,ssl_all').split(',')
EPOCHS    = int(os.environ.get('EPOCHS', 40))
PATIENCE  = 8
BATCH     = 256
LR        = 5e-4
MASK      = 0.4
RAW_CACHE = os.environ.get('RAW_CACHE', '/content/cvs_raw.pkl')
CKPT_DIR  = osp.join(OUT_DIR, 'ssl_ckpt')
SSL_DIR   = osp.join(OUT_DIR, 'features_ssl')
METHODS   = ['scan_A_2t', 'scan_A', 'maha', 'knn']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(CKPT_DIR, exist_ok=True)


def cfg(mask):
    return PatchTSTConfig(num_input_channels=6, context_length=C.WINDOW_SIZE, patch_length=C.PATCH_LEN,
                          stride=C.STRIDE, d_model=C.D_MODEL, num_attention_heads=C.NUM_HEADS,
                          num_hidden_layers=C.NUM_LAYERS, ffn_dim=C.FFN_DIM, dropout=C.DROPOUT,
                          head_dropout=C.DROPOUT, pooling_type='mean', channel_attention=False, scaling='std',
                          do_mask_input=mask, mask_type='random', random_mask_ratio=MASK,
                          channel_consistent_masking=False, mask_value=0)


def pretrain(train_w, val_w, seed):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    model = PatchTSTForPretraining(cfg(True)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    to_t = lambda w: torch.tensor(w, device=device).permute(0, 2, 1)
    best, best_state, wait = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        for i in np.array_split(rng.permutation(len(train_w)), max(1, len(train_w) // BATCH)):
            loss = model(past_values=to_t(train_w[i])).loss
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            torch.manual_seed(seed)                                    # same masks every epoch
            vl = float(np.mean([model(past_values=to_t(val_w[j:j + BATCH])).loss.item()
                                for j in range(0, len(val_w), BATCH)]))
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    print(f"    pretrain stopped at epoch {ep + 1}, best val reconstruction loss {best:.4f}")
    return best_state, best


if __name__ == '__main__':
    base = pads_paths()
    manifest, HC, PD, DD = load_manifest(base)
    if osp.exists(RAW_CACHE):
        records = pickle.load(open(RAW_CACHE, 'rb'))
    else:
        records = load_raw(base, manifest)
    W, L, S, T, R = window_records(records, 'none', seed=C.RANDOM_STATE)
    N = np.where(L == HC, 0, np.where(L == PD, 1, 2))
    folds = make_folds(S[N < 2], N[N < 2])
    dd_groups = {int(k): v for k, v in json.load(open(osp.join(FEAT_DIR, 'dd_diagnosis.json'))).items()}
    log = {}

    for variant in VARIANTS:
        os.makedirs(osp.join(SSL_DIR, variant), exist_ok=True)
        for k, fold in enumerate(folds, start=1):
            fpath = osp.join(SSL_DIR, variant, f'fold{k}.npz')
            if osp.exists(fpath):
                print(f"[{variant}] fold {k}: features exist, skipped"); continue
            tr, va = np.isin(S, fold['train_subjects']), np.isin(S, fold['val_subjects'])
            keep = (N == 0) if variant == 'ssl_hc' else (N < 2)
            mean, std = compute_normalization_stats(W[tr & keep])
            norm = lambda m: apply_normalization(W[m], mean, std).astype(np.float32)
            ck = osp.join(CKPT_DIR, f'{variant}_fold{k}.pt')
            t0 = time.time()
            if osp.exists(ck):
                state = torch.load(ck, map_location='cpu'); vloss = None
            else:
                print(f"[{variant}] fold {k}: pretraining on {int((tr & keep).sum())} windows")
                state, vloss = pretrain(norm(tr & keep), norm(va & keep), seed=C.RANDOM_STATE + k)
                torch.save(state, ck)
            enc = PatchTSTModel(cfg(False)).to(device)
            missing = enc.load_state_dict({kk[len('model.'):]: v for kk, v in state.items()
                                           if kk.startswith('model.')}, strict=False)
            assert not missing.missing_keys, f"encoder weights missing: {missing.missing_keys[:5]}"
            enc.eval()
            masks = {'hc_train': tr & (N == 0), 'val_hc': va & (N == 0), 'val_pd': va & (N == 1),
                     'test': np.isin(S, fold['test_subjects']), 'dd': N == 2}
            data = {}
            for split, m in masks.items():
                act, _ = encode(enc, norm(m), device)
                data.update({f'{split}_act': act, f'{split}_subj': S[m], f'{split}_label': N[m]})
            np.savez_compressed(fpath, **data)
            log[f'{variant}_fold{k}'] = {'val_loss': vloss, 'seconds': round(time.time() - t0)}
            print(f"[{variant}] fold {k}: saved ({time.time() - t0:.0f} s)")

    # ── evaluation ────────────────────────────────────────────────────────────
    fold_scores = []
    for k in range(1, C.N_SPLITS + 1):
        fs = {}
        sup = load_fold('none', k)
        for name, d in [('sup', sup)] + [(v, dict(np.load(osp.join(SSL_DIR, v, f'fold{k}.npz')))) for v in VARIANTS]:
            edges = make_global_edges(d['hc_train_act'], 10)
            direction = d['val_pd_act'].mean(0) > d['hc_train_act'].mean(0)
            ts, tl = d['test_subj'], d['test_label']
            us = np.unique(ts); y = np.array([tl[ts == s][0] for s in us])
            for m in (['scan_A', 'scan_A_2t'] if name == 'sup' else METHODS):
                sc = local_scores(d['val_hc_act'], d['val_hc_subj'], edges, direction, m,
                                  {'test': (d['test_act'], ts), 'dd': (d['dd_act'], d['dd_subj'])})
                fs[f'{name}:{m}'] = {'test': (us, sc['test'][1], y), 'dd': sc['dd'], 'val_hc_loso': sc['val_hc_loso']}
        ts = sup['test_subj']; us = np.unique(ts); y = np.array([sup['test_label'][ts == s][0] for s in us])
        fs['clf_K1'] = {'test': (us, subject_scores(sup['test_p1'], ts)[1], y),
                        'dd': subject_scores(sup['dd_p1'], sup['dd_subj']),
                        'val_hc_loso': subject_scores(sup['val_hc_p1'], sup['val_hc_subj'])}
        fold_scores.append(fs)
    res = evaluate(fold_scores, dd_groups, primary=[f'{v}:scan_A_2t' for v in VARIANTS] + [f'{v}:maha' for v in VARIANTS],
                   competitors=['clf_K1', 'sup:scan_A', 'sup:scan_A_2t'])
    res['config'] = {'variants': VARIANTS, 'epochs': EPOCHS, 'patience': PATIENCE, 'lr': LR, 'mask_ratio': MASK,
                     'pretrain_log': log, 'git_commit': git_commit()}
    print_eval(res)
    save_json(res, 'exp4_label_free_encoder.json')
