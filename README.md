# Federated Subset Scanning for Parkinson's Screening from Smartwatches

Privacy-preserving anomaly detection for Parkinson's disease (PD) screening from smartwatch movement data,
designed for low-resource settings where neurologists are scarce and patient data cannot be pooled centrally.

Isaac Mutuma (Pusan National University), with Tanya Akumu (Universitat de Barcelona) and
Celia Cintas (IBM Research Africa). Manuscript in preparation (target: MICCAI 2027).

## The idea

Subset scanning finds people whose neural-network activations look unusual compared with a reference set of
healthy activations. That reference normally has to sit in one place, which exposes the healthy volunteers
who provide it. This project asks whether the reference can be built from **histogram counts** that volunteers
contribute separately, and what those counts reveal about the people behind them.

```
 healthy volunteers (per site)                     screened person's watch
 ─────────────────────────────                     ───────────────────────
 encoder → layer-2 activations                     encoder → activations
 → histogram counts (128 neurons × 10 bins)        → p-values against the shared reference
 → + Laplace noise (person-level ε-DP)             → subset-scan score → refer / don't refer
        │                                                    ▲
        └──── summed counts = shared healthy reference ──────┘
              (recordings, activations and scores never leave the watch)
```

- **Data:** PADS (Varghese et al., 2024), 355 people: 276 with PD and 79 healthy controls, bilateral wrist
  IMU recordings during short assessment tasks. Five-fold subject-level cross-validation.
- **Encoder:** PatchTST, trained once centrally on labelled data.
- **Scoring:** histogram p-values, then Method A (mean p-value), B (per-window subset scan) or C (group subset scan).

## Main results (layer 2, PD vs healthy, 5-fold)

| | AUC (mean ± SD over folds) | Pooled AUC [95% CI] |
|---|---|---|
| Supervised PatchTST classifier (reference point) | 0.904 ± 0.054 | |
| Scan, Method A | 0.860 ± 0.067 | 0.854 [0.804, 0.899] |
| Scan, Method B | 0.851 ± 0.063 | 0.852 [0.806, 0.895] |
| Scan, Method C | 0.839 ± 0.065 | 0.837 [0.788, 0.882] |

- **Federation is exact.** Histograms from separate sites sum to the central reference, so the federated
  and central AUC are identical (0.860).
- **Unprotected histograms leak.** With no noise, a single volunteer's histogram gives membership-inference
  AUC 0.99, a classifier names the right volunteer 44% of the time (chance 3.1%), and a decoder reconstructs
  the released mean activation almost exactly.
- **Person-level differential privacy fixes this.** Each volunteer's contribution is capped and Laplace noise
  is added to the counts. At ε ≤ 10 every attack tested falls to chance. We recommend **ε = 1**, where the DP
  guarantee bounds *any* membership attack at AUC 0.731. Detection AUC is unchanged (0.860–0.865 at all ε);
  specificity at 90% sensitivity falls from 0.54 to 0.44.
- **Under the same ε = 1,** a histogram reference keeps detection (pooled AUC 0.852), while diagonal and
  full Gaussian summaries do not (0.595 and 0.493).

## What limits the claims

These came out of an October 2026 audit of the full pipeline and follow-up control experiments.
We report them rather than drop them.

- **The ranking signal comes from the encoder and the tail directions, not from the shared counts.**
  Replacing the healthy counts with a flat histogram leaves AUC almost unchanged (difference +0.016,
  CI [−0.009, +0.041]). The counts mainly calibrate the referral threshold, which is why detection survives
  heavy DP noise.
- **The pipeline is label-free only at deployment.** The encoder and the tail direction are set with labelled
  data centrally. A fully self-supervised encoder loses more than 0.10 AUC.
- **The scan does not beat the classifier** on disorders it never trained on (difference −0.010,
  CI [−0.038, +0.019]) or under simulated population shift. Both need local recalibration with healthy volunteers.
- **Histograms alone are not private.** An earlier claim that binning protects volunteers by itself did not
  hold up. Formal protection needs DP.

The direction we are taking from this is **personal monitoring**: comparing a person with their own baseline
on the watch, which a diagnostic classifier does not do (`classifier_vs_scan/exp5_monitoring.py`, simulated).

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | data loading and folds, histogram code (`scanning/histograms.py`), metrics with bootstrap CIs and specificity at 90% sensitivity (`training/metrics.py`) |
| `scripts/` | training, activation extraction, scanning, federation, privacy attacks and figures |
| `notebooks/000_full_pipeline.ipynb` | the notebook that produces every reported number |
| `classifier_vs_scan/` | scan vs classifier: unseen disorders, population shift, private references, label-free encoder, monitoring ([README](classifier_vs_scan/README.md)) |
| `minirocket_scan/` | the same detection and privacy pipeline on MiniROCKET features |
| `paper/isbi/` | manuscript source |

## Reproducing the results

Everything runs on Google Colab, with PADS and the results on Google Drive (`scripts/config.py`, `ENV='colab'`).
PADS is not redistributed here: download it from
[PhysioNet](https://physionet.org/content/parkinsons-disease-smartwatch/) under its licence.
Open `notebooks/000_full_pipeline.ipynb` and run all cells. The **Results summary** cell prints every number
above from the JSON files the scripts write. Each result file stores its configuration and git commit.

| Step | Script | Output (in `RESULTS_DIR`) |
|---|---|---|
| Base models | `train_patchtst.py`, `train_minirocket.py`, `train_inceptiontime.py`, `train_boss_svm.py` | `*_results.json` |
| Activations | `extract_activations.py` | `activations/` (layers 0–2, per-window metadata) |
| Detection (A/B/C, CIs, layers, variants) | `run_deepscan.py` | `deepscan_results_layer_*.json` |
| Merged (federated) reference | `run_federated_sim.py` | `federated_sim_layer_2.json` |
| Re-identification vs bin count | `privacy_analysis_bins.py` | `privacy_bins_sweep.json` |
| Membership inference + DP | `privacy_membership.py`, `privacy_membership_v2.py` | `privacy_membership*_layer_2.json` |
| Stronger attacks and ε sweep | `privacy_attacks_v3.py` | `privacy_attacks_v3_layer_2.json` |
| What carries the signal (control) | `control_reference_counts.py` | `control_reference_counts_layer_2.json` |
| Compressed-input baselines | `compressed_input_baseline.py` | `compressed_input_baseline.json` |
| Figures | `plot_results_v2.py`, `analyze_layers.py`, `plot_shared_doc_figures.py` | `figures_v2/`, `figures_shared_doc/` |

Older scripts and notebooks (`privacy_analysis.py`, `plot_privacy_utility*.py`, `plot_model_comparison.py`,
`train_tspulse.py`, `notebooks/00_full_pipeline.ipynb` and `01`–`06`) belong to earlier iterations and do not
reproduce the reported results.

## References

- Varghese, J., Brenner, A., Fujarski, M., van Alen, C. M., Plagwitz, L. and Warnecke, T. Machine learning in the
  Parkinson's disease smartwatch (PADS) dataset. *npj Parkinson's Disease* 10, 9 (2024).
  [doi:10.1038/s41531-023-00625-7](https://doi.org/10.1038/s41531-023-00625-7)
- The subset-scanning methods build on work by Celia Cintas and colleagues at IBM Research Africa and use the
  [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

## Contact

Isaac Mutuma, isaacmutuma@pusan.ac.kr
