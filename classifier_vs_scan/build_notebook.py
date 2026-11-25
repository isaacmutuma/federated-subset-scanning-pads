"""
Builds classifier_vs_scan/classifier_vs_scan_pipeline.ipynb.

    python classifier_vs_scan/build_notebook.py classifier_vs_scan/classifier_vs_scan_pipeline.ipynb
"""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n")})

md("""
# Classifier vs scan — can a PD classifier do the screening job?

Four experiments that test the scan against the strongest label-free uses of the PatchTST
classifier (threshold re-tuned on local healthy volunteers; features re-normalised on them) and
against simple distance trackers (Mahalanobis, kNN). Nothing in the main pipeline is changed:
checkpoints and config are only read, and all outputs go to `RESULTS_DIR/classifier_vs_scan/`.

| Step | What | Time (approx., GPU) |
|---|---|---|
| 0–1 | Setup, data | 5 min |
| 2 | Features: no shift (incl. the 114 other-disorder patients) | ~5 min |
| 3 | **Exp 1** — disorders the classifier never saw | ~10 min (Method C) |
| 4 | Features under 6 population shifts | ~20 min |
| 5 | **Exp 2** — population shift with local re-referencing | ~10 min |
| 6 | **Exp 3** — histograms vs simpler trackers at equal privacy | ~10 min |
| 7 | **Exp 4** — encoder trained without labels (masked reconstruction) | ~30–60 min |
| 8 | **Exp 5** — monitoring: change vs the person's own baseline (simulated) | ~20–30 min |
| 10 | Real home data: fetch Habets (20 PD, 14 days, symptom reports) + **Exp 6** | ~20 + ~20 min |
| 11 | Real healthy data: fetch IMU50 (50 healthy, 5 days) + **Exp 7** false alarms | ~1–2 h + ~30 min |
| 9 | Summary + backup (run last) | 1 min |

Sections 10–11 need no GPU for the fetch step and no data on Drive beforehand: they read the open
datasets straight from DataverseNL / Zenodo with byte-range requests and keep only the windows used.

Each experiment states its decision rule (fixed before the run) at the top of its script.
""")

md("## 0. Setup")
code("""
from google.colab import drive
drive.mount('/content/drive')

import os
REPO = '/content/federated-subset-scanning-pads'
if not os.path.exists(REPO):
    !git clone https://github.com/isaacmutuma/federated-subset-scanning-pads.git {REPO}
else:
    !git -C {REPO} fetch origin && git -C {REPO} reset --hard origin/main
%cd {REPO}
PIN_COMMIT = None
if PIN_COMMIT:
    !git -C {REPO} checkout {PIN_COMMIT}
!git -C {REPO} log -1 --oneline
!pip install -q transformers adversarial-robustness-toolbox
""")

md("## 1. Data")
code("""
import zipfile, sys
ZIP_PATH     = '/content/drive/MyDrive/pads-parkisons-dataset-folder/pads-parkinsons-disease-smartwatch-dataset-1.0.0.zip'
EXTRACT_PATH = '/content/pads_dataset'
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_PATH)
sys.path.insert(0, f'{REPO}/scripts')
import config as C
CVS_DIR = os.path.join(C.RESULTS_DIR, 'classifier_vs_scan')
print('checkpoints:', C.CKPT_DIR, '| outputs:', CVS_DIR)
for k in range(1, 6):
    p = os.path.join(C.CKPT_DIR, f'patchtst_fold{k}_best.pt')
    print(f"  {'found  ' if os.path.exists(p) else 'MISSING'} {p}")
""")

md("""
## 2. Features, no shift
Sanity check printed per fold: classifier AUC should be close to the main results (0.904 mean) and
Method A close to 0.860, since the windows, folds and checkpoints are the same.
""")
code("""
!SHIFTS=none python classifier_vs_scan/extract_features.py 2>&1 | grep -vE "Redirects|^W[0-9]" | tail -20
""")

md("""
## 3. Experiment 1 — disorders the classifier was never trained on
Rule: `scan_A − clf_K1` on HC vs other disorders, paired AUC CI above 0.
""")
code("""
!python classifier_vs_scan/exp1_unseen_disorders.py > /content/cvs_exp1.log 2>&1; tail -n 45 /content/cvs_exp1.log
""")

md("""
## 4. Features under population shifts
amp0.7, noise, rate25 (25 Hz watch), rotate30 (watch worn rotated), speed0.8 (slower movement),
combined. Only the screened population is shifted; the server side stays PADS.
""")
code("""
!SHIFTS=amp0.7,noise,rate25,rotate30,speed0.8,combined python classifier_vs_scan/extract_features.py 2>&1 | grep -E "\\] fold|windows|DONE|Error|Traceback"
""")

md("""
## 5. Experiment 2 — a different population
Rule per shift: `scan_A` beats both `clf_K1` and `clf_K2` on sensitivity at 10% referral (HC vs PD+DD).
""")
code("""
!python classifier_vs_scan/exp2_population_shift.py > /content/cvs_exp2.log 2>&1; tail -n 12 /content/cvs_exp2.log
""")

md("""
## 6. Experiment 3 — histograms vs simpler trackers at equal privacy
Rule: at ε = 1, `hist` beats `diag_gauss` and `full_gauss` on pooled AUC (paired CI above 0).
""")
code("""
!python classifier_vs_scan/exp3_private_references.py > /content/cvs_exp3.log 2>&1; tail -n 20 /content/cvs_exp3.log
""")

md("""
## 7. Experiment 4 — an encoder trained without labels
Masked-patch reconstruction pretraining per fold: `ssl_hc` (healthy windows only — no patient data)
and `ssl_all` (all training windows, labels discarded). Checkpoints are saved, so a restart resumes.
""")
code("""
!python classifier_vs_scan/exp4_label_free_encoder.py > /content/cvs_exp4.log 2>&1; grep -E "pretrain|saved" /content/cvs_exp4.log | tail -12; tail -n 30 /content/cvs_exp4.log
""")

md("""
## 8. Experiment 5 — monitoring a person over time (simulated)
Personal baseline = windows 1–3 of each recording; follow-up = windows 4–5, unchanged or with a simulated
change (tremor 4–6 Hz / 8–12 Hz, slowing, irregular movement, amplitude). Rule: at strength 0.5, the personal
two-tailed scan beats the classifier trend (signed and absolute) for ≥ 2 of the 3 non-PD changes.
Per-fold scores are cached, so a restart resumes.
""")
code("""
!python classifier_vs_scan/exp5_monitoring.py > /content/cvs_exp5.log 2>&1; grep -E "done|cached" /content/cvs_exp5.log | tail -5; tail -n 45 /content/cvs_exp5.log
""")

md("""
## 10. Real monitoring — Habets home data (Exp 6)
20 people with PD at home for ~14 days, both wrists, ~7 symptom questionnaires a day (CC0,
doi:10.34894/5HHK8H). The fetch keeps only the 10 minutes before each questionnaire (~0.3 GB, resumable).
Rule: the personal scan's within-person correlation with self-reported motor symptoms has CI > 0 and beats
both the classifier P(PD) and plain movement intensity (paired CIs > 0).
""")
code("""
!python classifier_vs_scan/habets_fetch.py 2>&1 | tail -n 25
!python classifier_vs_scan/exp6_habets_monitoring.py > /content/cvs_exp6.log 2>&1; tail -n 30 /content/cvs_exp6.log
""")

md("""
## 11. False alarms in healthy people — IMU50 (Exp 7)
50 healthy adults, 5 days free living, non-dominant wrist (CC BY 4.0, Zenodo 21468410). The 46.7 GB zip is
streamed, never stored; only the first 10 minutes of each daytime hour are kept. CPU-bound (csv parsing);
WORKERS=4 subjects in parallel. Resumable: re-run the cell if the runtime disconnects.
Rule: median per-person false-alarm rate of the personal scan <= 10% and flagged days <= 10%.
""")
code("""
!WORKERS=4 python classifier_vs_scan/imu50_fetch.py 2>&1 | tail -n 55
!python classifier_vs_scan/exp7_imu50_false_alarms.py > /content/cvs_exp7.log 2>&1; tail -n 14 /content/cvs_exp7.log
""")

md("## 9. Summary and backup")
code("""
import json, shutil, datetime
def load(n):
    p = os.path.join(CVS_DIR, n)
    return json.load(open(p)) if os.path.exists(p) else None
for name in ['exp1_unseen_disorders.json', 'exp2_population_shift.json', 'exp3_private_references.json', 'exp5_monitoring.json',
             'exp6_habets_monitoring.json', 'exp7_imu50_false_alarms.json']:
    r = load(name)
    if r is None:
        print('[missing]', name); continue
    rules = [r['rule']] if r.get('rule') is not None else [v['rule'] for v in r.get('by_shift', {}).values()] or [r['summary']['rule']]
    print(name, '->', rules)
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
bdir = f'/content/drive/MyDrive/results_backup_cvs_{stamp}'
os.makedirs(bdir, exist_ok=True)
for f in os.listdir(CVS_DIR):
    if f.endswith('.json') or f.endswith('_scores.csv'):
        shutil.copy(os.path.join(CVS_DIR, f), bdir)
print('backup ->', bdir)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("cells:", len(cells))
