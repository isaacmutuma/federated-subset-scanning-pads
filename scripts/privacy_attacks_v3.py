"""
privacy_attacks_v3.py
---------------------
Stronger attacks on released volunteer histograms, over a sweep of noise levels.

An adversary steals a released histogram (one volunteer, or a group) and tries to (1) identify
which person it came from, (2) tell whether a given person contributed, and (3) reconstruct the
person's encoder activations with a decoder network or a conditional GAN. Attackers know the
encoder, the bin edges and the noise mechanism, and train on simulated releases with the same noise.

Data per fold: the 63 PADS HC (train + val) stand in for volunteers. Subjects are split at random
into ATTACKER subjects (the adversary's own population data, all windows) and CANDIDATES (the
people under attack). Each candidate contributes W_CAP windows to a release; the adversary also
holds the candidate's other windows ('gallery', e.g. a different recording of the same person).

Attacks
  Identification (single-volunteer release, closed set over the candidates; chance = 1/n):
    likelihood : candidate whose gallery windows are most likely under the release
    classifier : multinomial LR trained on simulated noisy releases from each candidate's gallery
    moment     : no training; per-neuron mean read off the histogram (bin probabilities x bin
                 centres); the reconstruction is matched to the candidates' gallery means
    decoder    : MLP trained on attacker subjects' simulated releases to output the mean of the
                 windows that produced each release; matched as above
    cgan       : conditional GAN (adversarial + reconstruction loss, as pix2pix) trained the same way
                 to generate windows from a release; the mean of 256 generated windows is matched
  Learned attackers see per-neuron mean and spread computed from the release (256 numbers).
  Reconstruction skill (moment, decoder, cgan), against the mean of the windows actually released:
    1 - |rec - released mean|^2 / |population mean - released mean|^2.
    0 = no better than guessing an average person; 1 = perfect. With no noise the attacks must
    succeed (positive control); otherwise their failure under noise would mean nothing.
  Membership (releases of 1, 5, 10 candidates; members vs the other candidates):
    lr_exact / lr_fresh   : likelihood-ratio test with the contributed / other windows (as v2)
    shadow_exact / _fresh : LR classifier on 128 per-neuron log-ratio features, trained on
                            shadow releases built from attacker subjects (Shokri et al., 2017)
  DP bound: under eps-DP (person level) NO attack, whatever model it uses, can exceed the
    membership AUC of the tradeoff curve TPR <= e^eps FPR, 1 - FPR <= e^eps (1 - TPR)
    (post-processing). Reported per eps as dp_max_membership_auc.
  Utility: Method A detection AUC with the 10-window-capped val-HC reference under the same noise
    (as privacy_membership_v2), and with flat counts (see control_reference_counts.py).

Raw signals are not reconstructed: the adversary's best raw reconstruction could be pushed through
the public encoder, so identification and reconstruction are tested on the activations directly.

Writes RESULTS_DIR/privacy_attacks_v3_{LAYER}.json.

Run:
    python scripts/privacy_attacks_v3.py
    FAST=1 python scripts/privacy_attacks_v3.py     # fewer repeats / GAN steps (smoke test)
"""

import sys, os, os.path as osp, json, subprocess, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import config as C

sys.path.insert(0, C.REPO_DIR)
os.chdir(C.REPO_DIR)

from src.scanning.histograms import make_global_edges, site_counts, pvalues_from_counts, bin_index

ACT_DIR = os.environ.get('ACT_DIR', osp.join(C.RESULTS_DIR, 'activations'))
assert osp.exists(ACT_DIR), f"Run extract_activations.py first. Not found: {ACT_DIR}"

FAST          = os.environ.get('FAST', '0') == '1'
LAYER         = os.environ.get('LAYER', 'layer_2')
N_BINS        = 10
ALPHA         = 1.0
BACKGROUND    = 'val_hc'
W_CAP         = 10                                   # windows per volunteer in a release
EPSILONS      = [None if e == 'none' else float(e) for e in
                 os.environ.get('EPSILONS', 'none,100,30,10,3,1,0.3,0.1').split(',')]
MEMBER_SIZES  = [1, 5, 10]
N_ID_REPEATS  = 1 if FAST else 3                     # noise draws of every candidate's release
N_MEM_REPEATS = 5 if FAST else 20
N_MEM_ONE     = 10 if FAST else 100                  # size-1 releases: one member per repeat
N_SIM         = 30 if FAST else 100                  # simulated releases per person (training)
N_SHADOW      = 50 if FAST else 200
GAN_STEPS     = int(os.environ.get('GAN_STEPS', 300 if FAST else 2000))
N_GEN         = 256
GAN_REC_WEIGHT = 200.0                 # chosen so the no-noise positive control succeeds
N_UTIL_DRAWS  = 2 if FAST else 5
RANDOM_STATE  = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    GIT_COMMIT = subprocess.check_output(['git', '-C', C.REPO_DIR, 'rev-parse', 'HEAD'],
                                         text=True).strip()
except Exception:
    GIT_COMMIT = 'unknown'
RUN_CONFIG = {'layer': LAYER, 'n_bins': N_BINS, 'alpha': ALPHA, 'w_cap': W_CAP,
              'epsilons': EPSILONS, 'member_sizes': MEMBER_SIZES, 'n_id_repeats': N_ID_REPEATS,
              'n_mem_repeats': N_MEM_REPEATS, 'n_mem_one': N_MEM_ONE, 'n_sim': N_SIM,
              'n_shadow': N_SHADOW, 'gan_steps': GAN_STEPS, 'gan_rec_weight': GAN_REC_WEIGHT, 'n_gen': N_GEN,
              'n_util_draws': N_UTIL_DRAWS, 'fast': FAST, 'device': DEVICE,
              'edges_from': 'hc_train', 'git_commit': GIT_COMMIT}
print(f"Run config: {RUN_CONFIG}")


def act_path(k, name):
    return osp.join(ACT_DIR, f'activations_fold{k}_{name}.npy')


def dp_max_membership_auc(eps):
    """Largest membership AUC any attack can reach against a pure eps-DP release."""
    if eps is None:
        return 1.0
    e = np.exp(eps)
    x = 1.0 / (1.0 + e)                                  # where the two bounds meet
    return float(e * x ** 2 / 2 + (1 - x) - (1 - x) ** 2 / (2 * e))


def counts_of(bins):
    """(n_windows, n_neurons) bin indices -> (n_neurons, N_BINS) counts."""
    return np.stack([(bins == b).sum(axis=0) for b in range(N_BINS)], axis=1).astype(float)


def noisy(counts, eps, rng):
    if eps is None:
        return counts
    scale = counts.shape[0] * W_CAP / eps               # person-level L1 sensitivity / eps
    return np.clip(counts + rng.laplace(0.0, scale, counts.shape), 0.0, None)


def log_probs(counts):
    return np.log((counts + ALPHA) / (counts.sum(axis=1, keepdims=True) + ALPHA * counts.shape[1]))


def feature(counts):
    """What the identification classifier sees: per-neuron log bin probabilities, flattened."""
    return log_probs(counts).ravel()


def moments(counts, centres):
    """Per-neuron mean and spread implied by a release (centres in standardised units)."""
    p = np.exp(log_probs(counts))
    m1 = (p * centres).sum(axis=1)
    return np.r_[m1, np.sqrt(np.maximum((p * centres ** 2).sum(axis=1) - m1 ** 2, 0.0))]


def topk(score_matrix, k):
    """score_matrix[t, c]: higher = c more likely the source of release t. Top-k accuracy."""
    order = np.argsort(-score_matrix, axis=1)[:, :k]
    return float(np.mean([t in order[t] for t in range(len(score_matrix))]))


def nearest_mean_scores(recon, gallery_means):
    return -((recon[:, None, :] - gallery_means[None, :, :]) ** 2).sum(axis=2)


def skill(recon, true_means, pop_mean):
    num = ((recon - true_means) ** 2).sum(axis=1)
    den = ((pop_mean[None, :] - true_means) ** 2).sum(axis=1)
    return float(np.mean(1.0 - num / den))


class Gen(nn.Module):
    def __init__(self, c_dim, x_dim, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(nn.Linear(c_dim + z_dim, 256), nn.LeakyReLU(0.2),
                                 nn.Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, x_dim))

    def forward(self, c, z):
        return self.net(torch.cat([c, z], dim=1))


class Disc(nn.Module):
    def __init__(self, c_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c_dim + x_dim, 256), nn.LeakyReLU(0.2),
                                 nn.Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, c, x):
        return self.net(torch.cat([c, x], dim=1))


def train_cgan(conds, cond_windows, seed):
    """conds: (n, c_dim) scaled release features; cond_windows: (n, W_CAP, x_dim) released windows."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    c_dim, x_dim = conds.shape[1], cond_windows.shape[2]
    G, D = Gen(c_dim, x_dim).to(DEVICE), Disc(c_dim, x_dim).to(DEVICE)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    for _ in range(GAN_STEPS):
        i = rng.randint(0, len(conds), 128)
        x = cond_windows[i, rng.randint(0, W_CAP, len(i))]
        c = torch.tensor(conds[i], dtype=torch.float32, device=DEVICE)
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        fake = G(c, torch.randn(len(c), G.z_dim, device=DEVICE))
        loss_d = (bce(D(c, x), torch.full((len(c), 1), 0.9, device=DEVICE))
                  + bce(D(c, fake.detach()), torch.zeros(len(c), 1, device=DEVICE)))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()
        # adversarial + reconstruction loss (pix2pix-style): ties samples to the condition
        loss_g = (bce(D(c, fake), torch.ones(len(c), 1, device=DEVICE))
                  + GAN_REC_WEIGHT * ((fake - x) ** 2).mean())
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()
    return G


def generate_means(G, conds):
    with torch.no_grad():
        out = []
        for c in conds:
            ct = torch.tensor(np.repeat(c[None], N_GEN, 0), dtype=torch.float32, device=DEVICE)
            out.append(G(ct, torch.randn(N_GEN, G.z_dim, device=DEVICE)).mean(0).cpu().numpy())
    return np.stack(out)


results = {'config': RUN_CONFIG,
           'dp_max_membership_auc': {str(e): round(dp_max_membership_auc(e), 4) for e in EPSILONS},
           'folds': []}
t0 = time.time()

for k in range(1, C.N_SPLITS + 1):
    print(f"\n{'='*55}\nFOLD {k}/{C.N_SPLITS}\n{'='*55}")
    rng = np.random.RandomState(RANDOM_STATE + k)
    hc_train = np.load(act_path(k, f'{LAYER}_hc_train'))
    val_pd   = np.load(act_path(k, f'{LAYER}_val_pd'))
    edges = make_global_edges(hc_train, N_BINS)
    direction = val_pd.mean(0) > hc_train.mean(0)

    pool = np.concatenate([hc_train, np.load(act_path(k, f'{LAYER}_val_hc'))])
    pool_subj = np.concatenate([np.load(act_path(k, 'hc_train_subj')), np.load(act_path(k, 'val_hc_subj'))])
    bins = bin_index(pool, edges)
    scaler_x = StandardScaler().fit(pool)
    pool_z = scaler_x.transform(pool)
    inner = edges[:, 1:-1]
    width = inner[:, 1] - inner[:, 0]
    centres = np.concatenate([inner[:, :1] - width[:, None] / 2, (inner[:, 1:] + inner[:, :-1]) / 2,
                              inner[:, -1:] + width[:, None] / 2], axis=1)
    centres = (centres - scaler_x.mean_[:, None]) / scaler_x.scale_[:, None]

    subjects = rng.permutation(np.unique(pool_subj))
    attackers, cands = subjects[:len(subjects) // 2], subjects[len(subjects) // 2:]
    att_pop, att_shadow = attackers[:len(attackers) // 2], attackers[len(attackers) // 2:]
    contrib, gallery = {}, {}
    for s in subjects:
        idx = rng.permutation(np.where(pool_subj == s)[0])
        contrib[s], gallery[s] = idx[:W_CAP], idx[W_CAP:]
    n_c = len(cands)
    rel_means = np.stack([pool_z[contrib[s]].mean(0) for s in cands])
    gal_means = np.stack([pool_z[gallery[s]].mean(0) for s in cands])
    att_idx = np.where(np.isin(pool_subj, attackers))[0]
    pop_mean = pool_z[att_idx].mean(0)
    pop_lp = log_probs(counts_of(bins[np.concatenate([contrib[s] for s in att_pop])]))
    print(f"Volunteers {len(subjects)}: attacker subjects {len(attackers)}, candidates {n_c} "
          f"(chance top-1 {1 / n_c:.3f})")

    # Utility under the same noise (val-HC reference capped at W_CAP windows per person)
    bg = np.load(act_path(k, f'{LAYER}_{BACKGROUND}'))
    bg_subj = np.load(act_path(k, f'{BACKGROUND}_subj'))
    bg = bg[np.concatenate([rng.permutation(np.where(bg_subj == s)[0])[:W_CAP]
                            for s in np.unique(bg_subj)])]
    bg_counts = site_counts(bg, edges).astype(float)
    test, lab, tsubj = (np.load(act_path(k, f'{LAYER}_test')), np.load(act_path(k, 'test_labels')),
                        np.load(act_path(k, 'test_subj')))
    us = np.unique(tsubj)
    y_test = np.array([lab[tsubj == s][0] for s in us])

    def method_a_auc(counts):
        win = 1.0 - pvalues_from_counts(counts, edges, test, direction, ALPHA).mean(axis=1)
        return roc_auc_score(y_test, [win[tsubj == s].mean() for s in us])

    fold_rec = {'fold': k, 'n_candidates': int(n_c), 'n_attacker_subjects': int(len(attackers)),
                'chance_top1': round(1 / n_c, 4), 'chance_top5': round(min(5, n_c) / n_c, 4),
                'utility_flat_counts': round(method_a_auc(np.ones_like(bg_counts)), 4), 'by_eps': {}}

    for eps in EPSILONS:
        rec = {}
        # ---- training banks: simulated releases (attacker knows the mechanism) ----
        def sim_release(s, src):
            return feature(noisy(counts_of(bins[rng.choice(src, W_CAP, replace=False)]), eps, rng))
        bank_X, bank_W = [], []                           # decoder / cGAN: (release, its windows)
        for s in attackers:
            own = np.where(pool_subj == s)[0]
            for _ in range(N_SIM):
                w = rng.choice(own, W_CAP, replace=False)
                bank_X.append(moments(noisy(counts_of(bins[w]), eps, rng), centres))
                bank_W.append(pool_z[w])
        bank_X, bank_W = np.array(bank_X), np.array(bank_W)
        scaler_c = StandardScaler().fit(bank_X)
        bank_Xs = scaler_c.transform(bank_X)

        clf_X, clf_y = [], []                             # classifier: candidates' galleries
        for ci, s in enumerate(cands):
            for _ in range(N_SIM):
                clf_X.append(sim_release(s, gallery[s])); clf_y.append(ci)
        clf_scaler = StandardScaler().fit(clf_X)
        clf = LogisticRegression(max_iter=300, C=0.1).fit(clf_scaler.transform(clf_X), clf_y)

        dec = MLPRegressor(hidden_layer_sizes=(256,), max_iter=200, early_stopping=True,
                           random_state=RANDOM_STATE).fit(bank_Xs, bank_W.mean(axis=1))
        G = train_cgan(bank_Xs, bank_W, seed=RANDOM_STATE + k)

        # ---- identification of single-volunteer releases ----
        ident = {m: {'top1': [], 'top5': []}
                 for m in ['likelihood', 'classifier', 'moment', 'decoder', 'cgan']}
        sk = {'moment': [], 'decoder': [], 'cgan': []}
        for _ in range(N_ID_REPEATS):
            rel = [noisy(counts_of(bins[contrib[s]]), eps, rng) for s in cands]
            rel_lp = [log_probs(r) for r in rel]
            rows = np.arange(bins.shape[1])
            S_lik = np.array([[rel_lp[t][rows, bins[gallery[c]]].sum(axis=1).mean() for c in cands]
                              for t in range(n_c)])
            S_clf = clf.predict_log_proba(clf_scaler.transform([feature(r) for r in rel]))
            mom = np.array([moments(r, centres) for r in rel])
            feats = scaler_c.transform(mom)
            recon = {'moment': mom[:, :mom.shape[1] // 2], 'decoder': dec.predict(feats),
                     'cgan': generate_means(G, feats)}
            for m, S in [('likelihood', S_lik), ('classifier', S_clf)] + \
                        [(m, nearest_mean_scores(r, gal_means)) for m, r in recon.items()]:
                ident[m]['top1'].append(topk(S, 1)); ident[m]['top5'].append(topk(S, 5))
            for m, r in recon.items():
                sk[m].append(skill(r, rel_means, pop_mean))
        rec['identification'] = {m: {q: round(float(np.mean(v)), 4) for q, v in d.items()}
                                 for m, d in ident.items()}
        rec['reconstruction_skill'] = {m: round(float(np.mean(v)), 4) for m, v in sk.items()}

        # ---- membership ----
        rows = np.arange(bins.shape[1])
        def lr_feats(rel_lp, src):
            b = bins[src]
            return (rel_lp[rows, b] - pop_lp[rows, b]).mean(axis=0)     # 128 per-neuron log-ratios
        rec['membership'] = {}
        for size in MEMBER_SIZES:
            # shadow model trained on attacker subjects only
            sh_X = {'exact': [], 'fresh': []}; sh_y = []
            for _ in range(N_SHADOW):
                perm = rng.permutation(att_shadow)
                mem, non = perm[:size], perm[size:size + max(size, 3)]
                rl = log_probs(noisy(counts_of(bins[np.concatenate([contrib[s] for s in mem])]), eps, rng))
                for s, lab_ in [(s, 1) for s in mem] + [(s, 0) for s in non]:
                    sh_X['exact'].append(lr_feats(rl, contrib[s]))
                    sh_X['fresh'].append(lr_feats(rl, gallery[s])); sh_y.append(lab_)
            shadow = {a: LogisticRegression(max_iter=500).fit(sh_X[a], sh_y) for a in sh_X}
            aucs = {key: [] for key in ['lr_exact', 'lr_fresh', 'shadow_exact', 'shadow_fresh']}
            for _ in range(N_MEM_ONE if size == 1 else N_MEM_REPEATS):
                perm = rng.permutation(cands)
                mem, non = perm[:size], perm[size:]
                rl = log_probs(noisy(counts_of(bins[np.concatenate([contrib[s] for s in mem])]), eps, rng))
                y = np.r_[np.ones(len(mem)), np.zeros(len(non))]
                for a, src in [('exact', contrib), ('fresh', gallery)]:
                    F = np.array([lr_feats(rl, src[s]) for s in np.r_[mem, non]])
                    aucs[f'lr_{a}'].append(roc_auc_score(y, F.sum(axis=1)))
                    aucs[f'shadow_{a}'].append(roc_auc_score(y, shadow[a].decision_function(F)))
            rec['membership'][f'size{size}'] = {key: round(float(np.mean(v)), 4) for key, v in aucs.items()}

        # ---- utility under the same noise ----
        rec['utility_method_a_auc'] = round(float(np.mean(
            [method_a_auc(noisy(bg_counts, eps, rng)) for _ in range(1 if eps is None else N_UTIL_DRAWS)])), 4)

        fold_rec['by_eps'][str(eps)] = rec
        idn, m1 = rec['identification'], rec['membership']['size1']
        print(f"eps={str(eps):>5}  ID top-1 lik {idn['likelihood']['top1']:.3f} clf {idn['classifier']['top1']:.3f} "
              f"mom {idn['moment']['top1']:.3f} dec {idn['decoder']['top1']:.3f} gan {idn['cgan']['top1']:.3f} | skill mom "
              f"{rec['reconstruction_skill']['moment']:+.2f} dec {rec['reconstruction_skill']['decoder']:+.2f} "
              f"gan {rec['reconstruction_skill']['cgan']:+.2f} | "
              f"MIA(1) lr {m1['lr_exact']:.3f}/{m1['lr_fresh']:.3f} shadow {m1['shadow_exact']:.3f}/"
              f"{m1['shadow_fresh']:.3f} | utility A {rec['utility_method_a_auc']:.3f}  ({time.time() - t0:.0f} s)")
    results['folds'].append(fold_rec)


def fold_mean(path):
    vals = []
    for f in results['folds']:
        v = f
        for p in path:
            v = v[p]
        vals.append(v)
    return round(float(np.mean(vals)), 4)


summary = {'chance_top1': fold_mean(['chance_top1']), 'chance_top5': fold_mean(['chance_top5']),
           'utility_flat_counts': fold_mean(['utility_flat_counts']), 'by_eps': {}}
for e in map(str, EPSILONS):
    ex = results['folds'][0]['by_eps'][e]
    summary['by_eps'][e] = {
        'dp_max_membership_auc': results['dp_max_membership_auc'][e],
        'identification': {m: {q: fold_mean(['by_eps', e, 'identification', m, q]) for q in d}
                           for m, d in ex['identification'].items()},
        'reconstruction_skill': {m: fold_mean(['by_eps', e, 'reconstruction_skill', m])
                                 for m in ex['reconstruction_skill']},
        'membership': {sz: {a: fold_mean(['by_eps', e, 'membership', sz, a]) for a in d}
                       for sz, d in ex['membership'].items()},
        'utility_method_a_auc': fold_mean(['by_eps', e, 'utility_method_a_auc'])}
results['summary'] = summary

print(f"\nSUMMARY (mean over folds)  chance top-1 {summary['chance_top1']:.3f}  top-5 {summary['chance_top5']:.3f}"
      f"  | utility with flat counts {summary['utility_flat_counts']:.3f}")
print(f"{'eps':>6} {'DPmax':>6} | {'ID top-1: lik':>13} {'clf':>6} {'mom':>6} {'dec':>6} {'gan':>6} | "
      f"{'skill mom':>9} {'dec':>6} {'gan':>6} | "
      + " | ".join(f"MIA({s}) lr-ex lr-fr sh-ex sh-fr" for s in MEMBER_SIZES) + " | util A")
for e, r in summary['by_eps'].items():
    idn = r['identification']
    print(f"{e:>6} {r['dp_max_membership_auc']:>6.3f} | {idn['likelihood']['top1']:>13.3f} {idn['classifier']['top1']:>6.3f} "
          f"{idn['moment']['top1']:>6.3f} {idn['decoder']['top1']:>6.3f} {idn['cgan']['top1']:>6.3f} | "
          f"{r['reconstruction_skill']['moment']:>+9.2f} {r['reconstruction_skill']['decoder']:>+6.2f} "
          f"{r['reconstruction_skill']['cgan']:>+6.2f} | "
          + " | ".join("        " + " ".join(f"{r['membership'][f'size{s}'][a]:.3f}"
                                             for a in ['lr_exact', 'lr_fresh', 'shadow_exact', 'shadow_fresh'])
                       for s in MEMBER_SIZES)
          + f" | {r['utility_method_a_auc']:.3f}")

out = osp.join(C.RESULTS_DIR, f'privacy_attacks_v3_{LAYER}.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {out}\nDONE.")
