# Overleaf Build Notes

Elsevier `elsarticle` (`final,3p,times`) manuscript for the six-network 3D segmentation benchmark. Every number, table and figure comes from the executed segmentation notebook and its outputs in `Materials/Notebooks/segmentation/` (see `Materials/Notebooks/README.md`).

## Files

| File                     | Purpose                                             |
| ------------------------ | --------------------------------------------------- |
| `main.tex`               | The paper; set it as the main document.             |
| `highlights.tex`         | Separate highlights file.                           |
| `graphical_abstract.tex` | Graphical abstract (TikZ only); compile separately. |

## Figures

`main.tex` includes the notebook plots by their notebook path (for example `segmentation/plots/04_training_curves.png`) and sets `\graphicspath{{figures/}{../../Notebooks/}}`.

- **Local build** (`Materials/build-latex.ps1`, run from the repository): the plots are read in place from `Materials/Notebooks/segmentation/plots/`.
- **Overleaf**: create a `figures/` folder in the project and upload the plot folder so that it sits at `figures/segmentation/plots/`.

All ten plots are used: `01_region_volumes`, `02_modalities_and_masks`, `03_augmentation`, `04_training_curves`, `05_wilcoxon`, `06_dice_by_grade`, `07_per_case_distributions`, `08_qualitative_comparison`, `09_uncertainty_vs_dice`, `10_dice_vs_latency`.

The three figure files of the previous version (`figures/segmentation_*.png`) came from the per-model notebooks and were removed; they remain in Git history.

## Sources of the results

| Paper item                          | Source file in `Materials/Notebooks/segmentation/`                              |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| Cases, grades, run environment      | `logs/notebook.log`, `metrics/run_summary.json`                                 |
| Region volumes, cases without ET    | `metrics/region_volumes_by_grade.csv`, `metrics/cases_without_et.csv`           |
| Split                               | `metrics/split.csv`, `metrics/cases.csv`                                        |
| Architectures and training settings | `src/volume_lib.py`, `metrics/model_profiles.csv`, `metrics/worker_config.json` |
| Model selection, ET thresholds      | `metrics/et_threshold_and_selection.csv`                                        |
| Test results                        | `metrics/test_results.csv`, `metrics/test_per_case.csv`                         |
| Wilcoxon tests                      | `metrics/wilcoxon_tests.csv`                                                    |
| Results by grade                    | `metrics/test_results_by_grade.csv`                                             |
| Ensembles                           | `metrics/ensemble_results.csv`                                                  |
| Uncertainty                         | `metrics/uncertainty_correlation.csv`                                           |
| Efficiency                          | `metrics/efficiency.csv`                                                        |

## Changes from the previous version

The previous text reported seven per-model notebooks trained for 10 epochs, 368 usable cases, a test result for five models only (the Swin UNETR test cell had failed), Dice at the resampled 128^3 grid, and significance claims that were not computed. It now reports the unified notebook: 369 cases, a grade-stratified 257/56/56 split, 50 epochs and 3,250 optimiser steps for every model, all six models evaluated per case at native resolution with TTA, bootstrap confidence intervals, Holm-corrected Wilcoxon tests, results by grade, ensembles, TTA uncertainty and measured latency. Descriptions that did not match the code were corrected (the nnU-Net model is a fixed nnU-Net-style network, VNet does not use gradient checkpointing, Swin UNETR has 10.52 million parameters and a full-resolution skip). The uncited DETR reference was removed, TransUNet is now cited, and the Holm (1979) reference was added. The licence statement was removed because the repository contains no licence file.

## Verification status

Every number was transcribed from the files above and cross-checked against them. `main.tex`, `highlights.tex` and `graphical_abstract.tex` compile with `Materials/tectonic.exe` with no undefined references, citations or missing figures (14 pages). Not checked: a pdfLaTeX build on Overleaf.
