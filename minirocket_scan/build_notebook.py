"""
Builds minirocket_scan/minirocket_scan_pipeline.ipynb.

    python minirocket_scan/build_notebook.py minirocket_scan/minirocket_scan_pipeline.ipynb
"""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n")})

md("""
# MiniROCKET side pipeline — scanning MiniROCKET features instead of PatchTST activations

Runs the **same** DeepScan, federated and privacy scripts as the main pipeline on MiniROCKET
features, with the same 355 PD/HC subjects, windows, folds, held-out HC reference and N = 10 bins.
Only the representation changes. The last cell compares the two encoders and applies a decision
rule fixed in `minirocket_scan/compare_models.py` before the results were seen.

**Prerequisite:** the main pipeline (`notebooks/000_full_pipeline.ipynb`) has been run, so the
PatchTST results and `minirocket_results.json` exist in `RESULTS_DIR`.

| Step | What | Time (approx.) |
|---|---|---|
| 0 | Setup: Drive, repo, packages | 2 min |
| 1 | Data: unzip PADS | 2 min |
| 2 | MiniROCKET features per fold | ~10 min |
| 3 | DeepScan Methods A/B/C | ~15 min (5,544 features vs PatchTST's 128) |
| 4 | Federated pseudo-site simulation | ~5 min |
| 5 | Privacy: re-identification sweep | ~15 min |
| 6 | Privacy: membership inference + DP | ~10 min |
| 7 | Comparison with PatchTST + decision | instant |
| 8 | Backup | 1 min |

MiniROCKET results go to `RESULTS_DIR/minirocket_scan/`; the PatchTST results are only read.
A GPU is not needed.
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

# Optional: pin an exact commit for a reproducible paper run, e.g. PIN_COMMIT = 'abc1234'
PIN_COMMIT = None
if PIN_COMMIT:
    !git -C {REPO} checkout {PIN_COMMIT}
!git -C {REPO} log -1 --oneline
""")
code("""
!pip install -q scipy scikit-learn adversarial-robustness-toolbox sktime
import sklearn, numpy as np, sktime
print('sklearn', sklearn.__version__, '| numpy', np.__version__, '| sktime', sktime.__version__)
""")

md("## 1. Data")
code("""
import zipfile
ZIP_PATH     = '/content/drive/MyDrive/pads-parkisons-dataset-folder/pads-parkinsons-disease-smartwatch-dataset-1.0.0.zip'
EXTRACT_PATH = '/content/pads_dataset'
if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_PATH)
print(os.listdir(EXTRACT_PATH))

import sys
sys.path.insert(0, f'{REPO}/scripts')
import config as C
RESULTS_DIR = C.RESULTS_DIR                                   # PatchTST results (read only)
MR_DIR      = os.path.join(RESULTS_DIR, 'minirocket_scan')    # MiniROCKET results (Drive)
MR_ACT      = '/content/minirocket_scan/activations'        # MiniROCKET features (local disk, ~2.2 GB)
os.makedirs(MR_DIR, exist_ok=True)
print('PatchTST results:  ', RESULTS_DIR)
print('MiniROCKET results:', MR_DIR)
for f in ['deepscan_results_layer_2.json', 'minirocket_results.json', 'privacy_membership_layer_2.json']:
    print(f"  {'found  ' if os.path.exists(os.path.join(RESULTS_DIR, f)) else 'MISSING'} {f}")
""")

md("""
## 2. MiniROCKET features
Fitted per fold on the training windows without labels (seeded), then applied to every split.
Features (5,544 per window) stay on the runtime's local disk; if the runtime restarts they are
rebuilt automatically. Folds that already exist are skipped; `FORCE_REEXTRACT = True` rebuilds them.
""")
code("""
FORCE_REEXTRACT = False
!FORCE_REEXTRACT={int(FORCE_REEXTRACT)} MR_ACT_DIR={MR_ACT} python minirocket_scan/extract_minirocket_features.py 2>&1 | grep -E "Windows|HC=|FOLD|HC train|Features|Already|Saved|done|Error|Traceback" | tail -40
""")

md("## 3. DeepScan on MiniROCKET features (same script and settings as PatchTST)")
code("""
!ACT_DIR={MR_ACT} RESULTS_DIR={MR_DIR} LAYERS=minirocket python scripts/run_deepscan.py 2>&1 | grep -vE "^Fold [0-9] —|^fatal" | tail -40
""")

md("## 4. Federated pseudo-site simulation")
code("""
!ACT_DIR={MR_ACT} RESULTS_DIR={MR_DIR} LAYER=minirocket python scripts/run_federated_sim.py 2>&1 | grep -E "FOLD|Central|SUMMARY|transmits|Error|Traceback"
""")

md("## 5. Privacy — re-identification sweep")
code("""
!ACT_DIR={MR_ACT} RESULTS_DIR={MR_DIR} LAYER=minirocket python scripts/privacy_analysis_bins.py 2>&1 | grep -E "Raw|N_BINS|ReID|DeepScan|Error|Traceback"
""")

md("## 6. Privacy — membership inference on shared histograms (+ DP)")
code("""
!ACT_DIR={MR_ACT} RESULTS_DIR={MR_DIR} LAYER=minirocket python scripts/privacy_membership.py 2>&1 | grep -E "FOLD|Membership|Method A|SUMMARY|Error|Traceback" | tail -40
""")

md("""
## 7. Comparison with PatchTST and decision
Paired on the same subjects. Decision rule: Method A AUC difference first; if its 95% CI includes 0,
then membership leakage, utility under ε = 1 noise, and histogram size, in that order.
""")
code("""
!MR_RESULTS_DIR={MR_DIR} python minirocket_scan/compare_models.py
""")

md("## 8. Backup")
code("""
import shutil, datetime
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
files = [f for f in os.listdir(MR_DIR) if f.endswith('.json')]
bdir = f'/content/drive/MyDrive/results_backup_minirocket_{stamp}'
os.makedirs(bdir, exist_ok=True)
for f in files:
    shutil.copy(os.path.join(MR_DIR, f), bdir)
print(f'Copied {len(files)} result files to {bdir}')
""")

nb = {"cells": cells, "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("cells:", len(cells))
