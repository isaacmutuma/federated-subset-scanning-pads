# classifier_vs_scan — can a PD classifier do the screening job?

Four experiments that test the histogram scan against the strongest **label-free** uses of the
PatchTST classifier and against simple distance trackers. The main pipeline is not changed:
checkpoints, config and `src/` helpers are only read, and every output goes to
`RESULTS_DIR/classifier_vs_scan/` (override with `CVS_DIR`).

## Competitors (no labels from the screened population in any of them)

| Name | What it is |
|---|---|
| `clf_K1` | PatchTST P(PD), subject mean; referral threshold from local healthy volunteers |
| `clf_K2` | same, after re-normalising the 768-d encoder features so local volunteers match the training healthy statistics |
| `maha` | Mahalanobis distance of the 128-d layer-2 activations from local volunteers |
| `knn` | mean distance to the 10 nearest local volunteer windows (standardised) |
| `scan_A` | histogram scan, Method A, directed by the server's tail directions |
| `scan_A_2t` | Method A, two-tailed (no PD information beyond the encoder) |
| `scan_C` | group subset scan (Method C), ART, seeded per subject |

Referral rule everywhere: refer anyone above the 90th percentile of the local volunteers' subject
scores (10% of healthy people referred). Volunteers are scored leave-one-subject-out.

## Experiments and decision rules (fixed before the run)

| Script | Question | Rule |
|---|---|---|
| `exp1_unseen_disorders.py` | Do methods flag the 114 other-disorder (DD) patients the classifier never saw? | `scan_A − clf_K1` on HC vs DD: paired-bootstrap AUC CI above 0 |
| `exp2_population_shift.py` | Shifted population (amplitude, noise, 25 Hz, rotated watch, slower movement, combined) with local re-referencing | per shift: `scan_A` beats both `clf_K1` and `clf_K2` on sensitivity at 10% referral (HC vs PD+DD), CIs above 0 |
| `exp3_private_references.py` | Histograms vs diagonal / full Gaussian references under equal person-level ε-DP | at ε = 1: `hist − diag_gauss` and `hist − full_gauss` pooled AUC CIs above 0 |
| `exp4_label_free_encoder.py` | Encoder trained without labels (masked-patch reconstruction; `ssl_hc` = healthy only, `ssl_all` = all windows, labels dropped) | no pass/fail: reports the AUC cost of a label-free pipeline |

Note: the main paper is PD vs HC only. DD subjects appear here **only** in this side study
(exp 1, and as part of the screened population in exp 2/4).

## Running (Colab, GPU)

Build and open the notebook:

```bash
python classifier_vs_scan/build_notebook.py classifier_vs_scan/classifier_vs_scan_pipeline.ipynb
```

Order: setup → data → `SHIFTS=none extract_features.py` → exp 1 → shifted features → exp 2 → exp 3 →
exp 4 → summary + backup (`MyDrive/results_backup_cvs_<stamp>`). Approximate times are in the
notebook's first cell. Exp 4 saves encoder checkpoints, so a restarted run resumes.

Useful environment variables: `SHIFTS`, `FORCE` (re-extract), `RAW_CACHE`, `RUN_C` (Method C in
exp 1/2), `N_DRAWS`, `N_REPEATS` (exp 3), `VARIANTS`, `EPOCHS` (exp 4).

## Outputs

`features/<shift>/fold<k>.npz`, `features/dd_diagnosis.json`, `exp1_unseen_disorders.json`,
`exp2_population_shift.json`, `exp3_private_references.json`, `exp4_label_free_encoder.json`,
`ssl_ckpt/`, `features_ssl/`. Each JSON stores its config and git commit.
