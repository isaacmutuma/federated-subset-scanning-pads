# MiniROCKET side pipeline

Scans MiniROCKET features instead of PatchTST activations, with everything else held fixed:
the same 355 PD/HC subjects, windows, folds, held-out HC reference, N = 10 bins, and the same
analysis scripts (`scripts/run_deepscan.py`, `run_federated_sim.py`, `privacy_analysis_bins.py`,
`privacy_membership.py`). Only the representation changes.

| File | Role |
|---|---|
| `extract_minirocket_features.py` | Per fold: fit MiniRocket (1,000 kernels, seeded, no labels) on the training windows and save features for every split in the PatchTST activation layout, layer name `minirocket` |
| `compare_models.py` | PatchTST Layer 2 vs MiniROCKET side by side, paired on the same subjects, and the decision rule |
| `build_notebook.py` | Builds `minirocket_scan_pipeline.ipynb` |
| `minirocket_scan_pipeline.ipynb` | The Colab notebook: run top to bottom |

**Prerequisite:** `notebooks/000_full_pipeline.ipynb` has been run, so the PatchTST results and
`minirocket_results.json` are in `RESULTS_DIR`.

**Where things go:** MiniROCKET produces 5,544 features per window (924 per channel × 6), about
2.2 GB per run, so features stay on the Colab runtime's local disk (`/content/minirocket_scan/`).
Result JSONs go to `RESULTS_DIR/minirocket_scan/` on Drive. The shared scripts are pointed there with
the `ACT_DIR`, `RESULTS_DIR` and `LAYER`/`LAYERS` environment variables; their defaults are unchanged,
so the main pipeline is not affected.

**Decision rule** (fixed in `compare_models.py` before the results were seen): the model whose Method A
pooled AUC is higher with a paired 95% CI excluding 0. If the CI includes 0, detection is tied and the
choice goes, in order, to lower membership leakage on unnoised site histograms, higher Method A AUC
under ε = 1 noise, and the smaller site histogram.
