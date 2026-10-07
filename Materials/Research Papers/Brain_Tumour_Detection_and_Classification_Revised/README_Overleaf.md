# Brain Tumour Detection and Classification: Overleaf project

Elsevier `elsarticle` (ScienceDirect) submission package for the nine-backbone benchmark of 2D tumour detection and subtype classification. Every number, table and figure comes from the executed notebooks and their downloaded outputs in `Materials/Notebooks/binary-detection/` and `Materials/Notebooks/classification/` (see `Materials/Notebooks/README.md`).

## Files

| File                     | Purpose                                                                         |
| ------------------------ | ------------------------------------------------------------------------------- |
| `main.tex`               | The paper.                                                                      |
| `supplementary.tex`      | Supplementary Material: Tables S1 to S8, Figures S1 to S26. Compile separately. |
| `highlights.tex`         | Standalone highlights file. Compile separately.                                 |
| `graphical_abstract.tex` | Standalone graphical abstract (TikZ only, no images). Compile separately.       |
| `main.pdf`               | `main.tex` compiled with tectonic after this rewrite.                           |
| `supplementary.pdf`      | `supplementary.tex` compiled with tectonic after this rewrite.                  |

## Figures

`main.tex` and `supplementary.tex` include the notebook plots by their notebook path, for example `binary-detection/plots/09_confusion_matrices.png`, and set

```latex
\graphicspath{{figures/}{../../Notebooks/}}
```

- **Local build** (`Materials/build-latex.ps1` or any TeX installation run from this folder): the plots are read in place from `Materials/Notebooks/<task>/plots/`; nothing needs copying.
- **Overleaf**: create a `figures/` folder in the project and upload the two plot folders so that they sit at `figures/binary-detection/plots/` and `figures/classification/plots/`.

Plots used by `main.tex`: `02_samples_crop_mask.png` (both tasks), `09_confusion_matrices.png`, `18_accuracy_vs_latency.png` (binary), `13_gradcam_efficientnetb0.png` and `17_shortcut_test.png` (both tasks). The supplement uses every other plot of both folders, so each of the 25 binary and 26 classification plots appears in at least one of the two documents.

The figure files of the previous version (`figures/*.png`, `figures/supp/`, `figures/audit/`) came from the per-model notebooks and the revision audit, which no longer exist, and were removed; they remain in Git history.

## Where each result comes from

| Paper item                                                  | Source file in `Materials/Notebooks/<task>/`                                                                                     |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Provenance, crop fraction, native sizes (Table 2)           | `metrics/provenance.csv`, `metrics/deduplication_by_source.csv`                                                                  |
| Deduplication (Table 3, Table S1)                           | `metrics/deduplication_summary.csv`, `metrics/deduplication_threshold_sweep.csv`                                                 |
| Cross-label pairs and label noise                           | `metrics/cross_label_pairs.csv`, `logs/notebook.log`                                                                             |
| Groups, splits, folds, leakage audit (Table 4)              | `logs/notebook.log`, `metrics/split_roles.csv`, `metrics/fold_distribution.csv`, `metrics/leakage_audit.csv`                     |
| Training settings (Table 5)                                 | Configuration cell, `metrics/worker_config.json`, `metrics/run_metrics.csv`                                                      |
| Run environment                                             | `metrics/run_summary.json`                                                                                                       |
| Cross-validation (Table 6)                                  | `metrics/fold_summary.csv`                                                                                                       |
| Fold-ensemble test results (Tables 7 and 8, Table S4)       | `metrics/test_leaderboard.csv`, `metrics/test_per_class.csv`                                                                     |
| McNemar and gap to leader                                   | `metrics/mcnemar_holm_p.csv`, `metrics/gap_to_leader.csv`                                                                        |
| Calibration (Table 9), per-run temperatures (Tables S2, S3) | `metrics/calibration_ece.csv`, `metrics/run_metrics.csv`                                                                         |
| Efficiency (Table 10, Table S6)                             | `metrics/efficiency.csv`, `metrics/model_profiles.csv`                                                                           |
| Attention, masking, shortcut (Table 11, Tables S7, S8)      | `metrics/outside_head_energy.csv`, `metrics/occlusion_outside_head.csv`, `metrics/masking_test.csv`, `metrics/shortcut_test.csv` |
| Accuracy by distance (Table S5)                             | `metrics/accuracy_by_nearest_dev_distance.csv`                                                                                   |

Table numbers above are predicted from the order of the floats; confirm them after compiling.

## What changed from the previous version of the paper

The previous version reported 15 separate per-model notebooks (nine binary, six multiclass), a binary pool that was not deduplicated before training, slice-level stratified splits, one run per model, and a post-hoc revision audit. All of that was replaced by the two unified notebooks:

- Both pools are deduplicated before splitting, label-aware, with cross-label label noise excluded (binary 1,185 of 3,253 kept; multiclass 4,730 of 5,828).
- Group-aware five-fold cross-validation plus a held-out test set replaces the 70/15/15 slice split; the leakage audit finds no test image near a development image.
- All nine backbones run on both tasks from one model registry, with an equal number of optimiser steps, 20 epochs and checkpoint selection on validation log-loss.
- Fold ensembles, bootstrap confidence intervals, Holm-corrected McNemar tests, temperature scaling, Grad-CAM, attention rollout, occlusion, masking and background-only shortcut tests are computed for every model.
- The Implementation Differences section and table, the leakage-free subset analysis, the image-dimension table and the EfficientNetB0-only audit figures were removed because the protocol they described no longer exists.
- The reviewer-mapped supplement numbering (Comments-V2, Table M1) no longer applies; supplementary items are numbered S1 onwards in order of appearance.
- Four references were added for methods the notebooks use: temperature scaling (Guo et al., 2017), Grad-CAM (Selvaraju et al., 2017), attention rollout (Abnar and Zuidema, 2020) and the Holm procedure (Holm, 1979).

Headline results in this version: binary fold-ensemble accuracy 99.42% for CoAtNet-0, EfficientNetB0 and ViT-B/16 on 171 test images, with no significant pairwise difference; multiclass test set of only 45 images (one group of 1,431 images could not be split), on which six models are error-free; background-only shortcut models above chance for 7 of 9 binary and 9 of 9 multiclass backbones.

## Build

`main.tex` uses a manual `thebibliography` block, so no BibTeX pass is needed; two LaTeX runs resolve the cross-references. Under XeTeX (tectonic) the preamble loads the TeX Gyre Termes and Heros fonts through `fontspec`; under pdfLaTeX it uses the `times` option of `elsarticle`.

## Verification status

- Every number was transcribed from the files listed above and cross-checked against them.
- All four documents compile with `Materials/tectonic.exe` with no undefined references, citations or missing figures; `main.pdf` has 22 pages. Not checked: a pdfLaTeX build on Overleaf.
