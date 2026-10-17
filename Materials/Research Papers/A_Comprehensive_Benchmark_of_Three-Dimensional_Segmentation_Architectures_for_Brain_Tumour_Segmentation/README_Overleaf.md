# Overleaf Build Notes

Elsevier CAS single-column (`cas-sc`) manuscript for _Engineering Applications of Artificial Intelligence_: the six-network 3D segmentation benchmark. Every number, table and figure comes from the executed segmentation notebook and its outputs in `Materials/Notebooks/segmentation/` (see `Materials/Notebooks/README.md`).

## Files

| File                                                   | Purpose                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `main.tex`                                             | The paper; set it as the main document.                                           |
| `refs.bib`                                             | References (BibTeX), checked against Crossref and arXiv records.                  |
| `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` | Elsevier CAS template files, version 2.4 (single column, author-year references). |
| `thumbnails/cas-email.jpeg`                            | E-mail icon the CAS front matter prints beside each author address.               |
| `main.pdf`                                             | `main.tex` compiled with Tectonic (`npm run papers`).                             |
| `highlights.tex`                                       | Separate highlights file.                                                         |
| `graphical_abstract.tex`                               | Graphical abstract (TikZ only); compile separately.                               |

## Figures

`main.tex` includes the notebook plots by their notebook path (for example `segmentation/plots/04_training_curves.png`) and sets `\graphicspath{{figures/}{../../Notebooks/}}`.

- **Local build** (`npm run papers -- Segmentation` from the repository root): the plots are read in place from `Materials/Notebooks/segmentation/plots/`.
- **Overleaf**: create a `figures/` folder in the project and upload the plot folder so that it sits at `figures/segmentation/plots/`.

Figures and tables are not floats: `herefigure` and `heretable` keep the CAS caption style but set each item directly after the paragraph that first cites it, and every plot is capped at a fraction of the text height so nothing overflows. All ten plots are used: `01_region_volumes`, `02_modalities_and_masks`, `03_augmentation`, `04_training_curves`, `05_wilcoxon`, `06_dice_by_grade`, `07_per_case_distributions`, `08_qualitative_comparison`, `09_uncertainty_vs_dice`, `10_dice_vs_latency`.

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

## Changes for the EAAI template

- Moved from `elsarticle` to the Elsevier CAS single-column class, with author-year citations from `refs.bib` in place of the numbered manual bibliography.
- Title: "3D" is spelled out ("Three-Dimensional"), since EAAI forbids undefined acronyms in the title; the abstract is 234 words and states the AI contribution and the engineering application.
- Reference corrections found while checking every entry against Crossref and arXiv: UNETR pages 1748--1758 (previously 574--584), Swin UNETR has six authors (Myronenko removed) and appeared in BrainLes 2021 (LNCS 12962), not MICCAI; Swin-Unet cited in its published ECCV 2022 Workshops version; TransUNet is cited in its peer-reviewed Medical Image Analysis (2024) version in place of the 2021 arXiv preprint; Bakas et al. (2017) cited by article number 170117; DOIs added. The BraTS citation request Bakas et al. (2018) was added; it exists only as an arXiv preprint and is kept because the BraTS data-use terms ask for it. The ViT entry is formatted as `@misc` because ICLR publishes no page numbers. The CAS keywords-box overfull warning and hyperref's empty footnote anchors at `\maketitle` are fixed in the preamble.
- The augmentation figure moved from the appendix to the paragraph that cites it; the training curves are cited in Section 4.1, where they are discussed. The Kaggle dataset path is now a link.
- Added the Declaration of competing interest (same wording as the companion paper) and a Data availability statement; the Kaggle GPU note moved to Acknowledgements. No numbers changed.

## Verification status

Every number was transcribed from the files above and cross-checked against them. `main.pdf` (16 pages) compiles with Tectonic 0.17.0 (`npm run papers`) with no undefined references, citations or missing figures, and with no TeX, hyperref or BibTeX warnings; `highlights.tex` and `graphical_abstract.tex` compile separately. A PDF check confirmed every figure and table appears on the page of its first mention or the next, nothing lies outside the text block, and every internal link resolves. Not checked: a pdfLaTeX build on Overleaf.
