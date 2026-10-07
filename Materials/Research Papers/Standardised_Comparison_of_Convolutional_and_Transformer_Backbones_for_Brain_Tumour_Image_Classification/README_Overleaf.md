# Brain Tumour Detection and Classification: Overleaf project

Elsevier CAS single-column (`cas-sc`) manuscript for _Engineering Applications of Artificial Intelligence_: the nine-backbone benchmark of 2D tumour detection and subtype classification. Every number, table and figure comes from the executed notebooks and their downloaded outputs in `Materials/Notebooks/binary-detection/` and `Materials/Notebooks/classification/` (see `Materials/Notebooks/README.md`).

## Files

| File                                                   | Purpose                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `main.tex`                                             | The paper; set it as the main document.                                           |
| `refs.bib`                                             | References (BibTeX), checked against Crossref, DataCite and arXiv records.        |
| `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` | Elsevier CAS template files, version 2.4 (single column, author-year references). |
| `thumbnails/cas-email.jpeg`                            | E-mail icon the CAS front matter prints beside the corresponding author.          |
| `highlights.tex`                                       | Standalone highlights file (5 items, each at most 85 characters).                 |
| `graphical_abstract.tex`                               | Standalone graphical abstract (TikZ only, no images). Compile separately.         |
| `main.pdf`                                             | `main.tex` compiled with Tectonic (`npm run papers`).                             |

## Figures

`main.tex` includes the notebook plots by their notebook path, for example `binary-detection/plots/09_confusion_matrices.png`, and sets

```latex
\graphicspath{{figures/}{../../Notebooks/}}
```

- **Local build** (`npm run papers -- Standardised` from the repository root, or any TeX installation run from this folder): the plots are read in place from `Materials/Notebooks/<task>/plots/`; nothing needs copying.
- **Overleaf**: create a `figures/` folder in the project and upload the two plot folders so that they sit at `figures/binary-detection/plots/` and `figures/classification/plots/`.

All 25 binary and 26 classification plots appear in `main.tex`. The former separate supplement (`supplementary.tex`, Tables S1 to S8, Figures S1 to S26) was merged into the paper, because references from the paper into a separate PDF cannot be links; it remains in Git history.

## Layout

- Figures and tables are not floats. `main.tex` defines `herefigure` and `heretable`, which keep the CAS caption style but set each item exactly where it is written, directly after the paragraph that first cites it. A page can therefore end short when the next item does not fit.
- Every plot is scaled to the text width and capped at a fraction of the text height (`\plot`), so no figure overflows the page.
- Citations are author-year (natbib with `cas-model2-names.bst`), as the EAAI guide asks; citations, cross-references, DOIs and URLs are links.

## Where each result comes from

| Paper item                                     | Source file in `Materials/Notebooks/<task>/`                                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Provenance, crop fraction, native sizes        | `metrics/provenance.csv`, `metrics/deduplication_by_source.csv`                                                                  |
| Deduplication and threshold sweep              | `metrics/deduplication_summary.csv`, `metrics/deduplication_threshold_sweep.csv`                                                 |
| Cross-label pairs and label noise              | `metrics/cross_label_pairs.csv`, `logs/notebook.log`                                                                             |
| Groups, splits, folds, leakage audit           | `logs/notebook.log`, `metrics/split_roles.csv`, `metrics/fold_distribution.csv`, `metrics/leakage_audit.csv`                     |
| Training settings                              | Configuration cell, `metrics/worker_config.json`, `metrics/run_metrics.csv`                                                      |
| Run environment                                | `metrics/run_summary.json`                                                                                                       |
| Cross-validation                               | `metrics/fold_summary.csv`                                                                                                       |
| Per-run temperature, threshold and test result | `metrics/run_metrics.csv`                                                                                                        |
| Fold-ensemble test results, per-class metrics  | `metrics/test_leaderboard.csv`, `metrics/test_per_class.csv`                                                                     |
| McNemar and gap to leader                      | `metrics/mcnemar_holm_p.csv`, `metrics/gap_to_leader.csv`                                                                        |
| Accuracy by distance to the development pool   | `metrics/accuracy_by_nearest_dev_distance.csv`                                                                                   |
| Calibration                                    | `metrics/calibration_ece.csv`, `metrics/run_metrics.csv`                                                                         |
| Efficiency                                     | `metrics/efficiency.csv`, `metrics/model_profiles.csv`                                                                           |
| Attention, masking, shortcut                   | `metrics/outside_head_energy.csv`, `metrics/occlusion_outside_head.csv`, `metrics/masking_test.csv`, `metrics/shortcut_test.csv` |

## What changed from the previous version of the paper

The previous version reported 15 separate per-model notebooks (nine binary, six multiclass), a binary pool that was not deduplicated before training, slice-level stratified splits, one run per model, and a post-hoc revision audit. All of that was replaced by the two unified notebooks:

- Both pools are deduplicated before splitting, label-aware, with cross-label label noise excluded (binary 1,185 of 3,253 kept; multiclass 4,730 of 5,828).
- Group-aware five-fold cross-validation plus a held-out test set replaces the 70/15/15 slice split; the leakage audit finds no test image near a development image.
- All nine backbones run on both tasks from one model registry, with an equal number of optimiser steps, 20 epochs and checkpoint selection on validation log-loss.
- Fold ensembles, bootstrap confidence intervals, Holm-corrected McNemar tests, temperature scaling, Grad-CAM, attention rollout, occlusion, masking and background-only shortcut tests are computed for every model.
- The Implementation Differences section and table, the leakage-free subset analysis, the image-dimension table and the EfficientNetB0-only audit figures were removed because the protocol they described no longer exists.
- The reviewer-mapped supplement numbering (Comments-V2, Table M1) no longer applies; the former supplementary items are now numbered with the paper's own tables and figures.
- Four references were added for methods the notebooks use: temperature scaling (Guo et al., 2017), Grad-CAM (Selvaraju et al., 2017), attention rollout (Abnar and Zuidema, 2020) and the Holm procedure (Holm, 1979).

Headline results in this version: binary fold-ensemble accuracy 99.42% for CoAtNet-0, EfficientNetB0 and ViT-B/16 on 171 test images, with no significant pairwise difference; multiclass test set of only 45 images (one group of 1,431 images could not be split), on which six models are error-free; background-only shortcut models above chance for 7 of 9 binary and 9 of 9 multiclass backbones.

## Changes for the EAAI template

- Moved from `elsarticle` to the Elsevier CAS single-column class, with author-year citations from `refs.bib` in place of the numbered manual bibliography.
- Title and abstract no longer use the undefined acronym "MRI" in the title (now "Magnetic Resonance Image"); the abstract is 243 words (EAAI asks for about 250) and states the AI contribution and the engineering application.
- Reference corrections found while checking every entry against Crossref, DataCite and arXiv: full author lists in place of "et al.", DenseNet pages 2261--2269 and Swin Transformer pages 9992--10002 (IEEE Xplore pagination of the DOI version), the full page range of Seetha and Raja (2018), the issue number of Kapoor and Narayanan (2023), and DOIs for every entry that has one.
- The supplement was merged into the paper (see Figures). No numbers changed.

## Build

`npm run papers` (repository root) compiles `main.tex` of every paper project with Tectonic; `npm run papers -- Standardised` compiles only this one. Tectonic runs BibTeX itself; with another TeX installation, run `pdflatex main`, `bibtex main`, then `pdflatex main` twice.

## Verification status

- `main.pdf` (40 pages) compiles with Tectonic 0.15.0 with no undefined references or citations, no missing figures and no overfull lines in the body. The one overfull-box warning comes from an empty box inside the CAS `\maketitle` and also appears in Elsevier's own sample.
- A PDF check confirmed every figure and table appears on the page of its first mention or the next, every image and text line lies inside the text block, and every internal link resolves.
- Not checked: a pdfLaTeX build on Overleaf.
