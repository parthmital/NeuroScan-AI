# Brain Tumour Detection and Classification: Overleaf project

Elsevier CAS single-column (`cas-sc`) manuscript for _Engineering Applications of Artificial Intelligence_: the nine-backbone benchmark of 2D tumour detection and subtype classification. Every number, table and figure comes from the executed notebooks and their downloaded outputs in `Materials/Notebooks/binary-detection/` and `Materials/Notebooks/classification/` (see `Materials/Notebooks/README.md`).

## Files

| File                                                   | Purpose                                                                             |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| `main.tex`                                             | The paper; set it as the main document.                                             |
| `supplementary.tex`                                    | The supplementary material, a separate document (Sections S1 to S7), captions only. |
| `preamble.tex`                                         | Packages, plot helpers and template patches shared by both documents.               |
| `refs.bib`                                             | References (BibTeX) of both documents, checked against Crossref and DataCite.       |
| `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` | Elsevier CAS template files, version 2.4 (single column, author-year references).   |
| `thumbnails/cas-email.jpeg`                            | E-mail icon the CAS front matter prints beside the corresponding author.            |
| `highlights.tex`                                       | Standalone highlights file (5 items, each at most 85 characters).                   |
| `graphical_abstract.tex`                               | Standalone graphical abstract (TikZ only, no images). Compile separately.           |
| `main.pdf`, `supplementary.pdf`                        | Both documents compiled with Tectonic (`npm run papers`).                           |

## Figures

`main.tex` and `supplementary.tex` include the notebook plots by their notebook path, for example `binary-detection/plots/09_confusion_matrices.png`, and sets

```latex
\graphicspath{{figures/}{../../Notebooks/}}
```

- **Local build** (`npm run papers -- Standardised` from the repository root, or any TeX installation run from this folder): the plots are read in place from `Materials/Notebooks/<task>/plots/`; nothing needs copying.
- **Overleaf**: create a `figures/` folder in the project and upload the two plot folders so that they sit at `figures/binary-detection/plots/` and `figures/classification/plots/`.

24 of the 25 binary detection and 25 of the 26 subtype classification plots appear in the two documents; the dataset overview plots (`01_dataset_overview.png`) are not used, since Table 2 of the paper gives the same counts and image sizes. The paper (`main.pdf`, 19 pages including references) keeps Figures 1 to 3 and Tables 1 to 11; Figure 2 shows only the class-distribution panel of each task's `03_deduplication.png`, cropped in LaTeX (`\dedupbars`, `trim` and `clip`) so the plot file is unchanged. Every other plot and table is in the separate supplementary material (`supplementary.pdf`, 24 pages: Sections S1 to S7, Supplementary F1 to F26, Supplementary T1 to T9). The Introduction's last paragraph lists what each supplementary section holds.

## Cross-references between the two PDFs

Each document loads `xr-hyper` and reads the other's labels from its `.aux` file (`\externaldocument`), so "Supplementary F3" in the paper and "Figure 3 of the paper" in the supplement are numbered automatically and link into the other PDF. Three details make this work:

- `xr-hyper` is loaded with `\RequirePackage` before `\documentclass`, because the CAS class loads hyperref.
- `\externaldocument` runs inside a group with `\bibcite` disabled, so only labels are imported and the two documents' citation keys do not clash.
- The documents must be compiled in the order main, supplementary, main. `npm run papers` does this and keeps the `.aux` files (Git-ignored) between runs.

On Overleaf, compile `supplementary.tex` once as the main document, then `main.tex`; for automatic rebuilds, add the `latexmkrc` from Overleaf's guide [Cross referencing with the xr package in Overleaf](https://www.overleaf.com/learn/how-to/Cross_referencing_with_the_xr_package_in_Overleaf).

## Layout

- Figure 1, the methodology diagram, is drawn in TikZ in `main.tex` as a top-down flowchart: nodes coloured by role (grey data, pink processing, blue alternatives, yellow partitions, green decisions, lavender evaluation), solid arrows for the main flow and dashed arrows for the curation outcomes, the background-only shortcut arm and the cost measurements.
- Supplementary sections are numbered S1 to S7, and figures and tables F1, F2, ... and T1, T2, ...; their captions read "Supplementary F1: ..." and "Supplementary T1: ...". `\supp{label}` (in `preamble.tex`) prints "Supplementary F<n>" or "Supplementary T<n>".
- The supplement extends the paper rather than standing alone: it has no author block, corresponding author, abstract or reference list, and holds only section headings, figures and tables, with every detail in the captions or in the settings table (Supplementary T2). It patches two CAS macros in its own preamble: `\MaketitleBox` prints only the title, and table captions run on after "Supplementary T<n>:" like figure captions.
- `preamble.tex` patches two CAS template quirks without editing the class: the keywords box is set with `\hss` so `\maketitle` no longer reports a 117pt overfull box, and hyperref footnote links are off (`hyperfootnotes=false`) because the unmarked corresponding-author and e-mail notes cannot be anchored. It also resets hyperref's remote page number between links into the other PDF, and loads `xurl` so long URLs break cleanly.
- Figures and tables are not floats. `preamble.tex` defines `herefigure` and `heretable`, which keep the CAS caption style but set each item exactly where it is written, directly after the paragraph that first cites it. A page can therefore end short when the next item does not fit.
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

## Changes in the 2026-10-19 revision

- Figure 1 redrawn as a branching top-down flowchart with role-coloured nodes and dashed side arms, in place of the five-column stage grid.
- Figure 2 shows only the class distribution before and after deduplication, binary detection left and subtype classification right; the Hamming tolerance sweep stays in Supplementary T1.
- The subtype classification fold-ensemble table moved from the supplement into Section 4.3 (Table 7).
- Section 4.6 is a bulleted list: one bullet per metric definition and one per finding (Grad-CAM maps, masking test, shortcut test).
- Gneiting and Raftery (2007) is now cited in the paper's training paragraph for the strictly proper scoring rule, since the supplement no longer cites anything.
- Supplement: shorter title; author block, corresponding author, abstract, dataset overview figure (former Figure S1) and reference list removed; all body text folded into captions or the settings table; items numbered Supplementary F1 to F26 and T1 to T9. A corrupted `\ref` in the Grad-CAM paragraph went with the body text.

## Changes in the 2026-10-17 restructuring

- Main paper cut from 40 pages to 18 (text and references); per-run tables, per-model figures, the dataset overview, sample images, threshold sweep, cross-label pairs, group audit, augmentation examples, training settings, the fold-ensemble result tables of both tasks, confusion matrices, curves, McNemar, distance, reliability, the summary efficiency table, accuracy--latency, per-class attention, occlusion, eight of the nine Grad-CAM figures, masking and shortcut results moved to the supplementary material. Binary detection and subtype classification items always move together.
- Section 3 restructured: two-sentence introduction, Data Collection in two paragraphs, Preprocessing and Deduplication with Brain cropping and Label-aware deduplication subsections, a one-paragraph Group-Aware Splitting section, and Training Configuration and Evaluation Protocol merged (one training paragraph, one sentence per evaluation item).
- Section 4.1 renamed "Five-Fold Cross-Validation Spread of the Fold Models" with one paragraph per task; every results section opens by defining its terms; Section 4.6 defines each interpretability metric separately and explains the EfficientNetB0 Grad-CAM figure and the masking findings.
- Terminology standardised to "binary detection" and "subtype classification" in text, tables, captions, highlights and the graphical abstract.
- The efficiency table in the paper gives each model's parameter count in brackets; the literature comparison table has "Our work" rows instead of a column.
- Data availability lists every dataset link and the repository link directly; the former Appendix A (code repository) was folded into it.
- 34 references added for architectural, methodological and background claims (for example He et al. 2016 for ResNet50's 50 layers, Otsu 1979, McNemar 1947, Efron 1979, Youden 1950, AdamW, mixed precision, Grad-CAM, occlusion, shortcut learning); VGG now cites its ICLR 2015 publication instead of the arXiv preprint. Each added entry was checked on Crossref or the publisher's proceedings page. Uncited claims about prevalence documentation and radiologist supply in India were removed.
- The supplementary material became a separate document and PDF (`supplementary.tex`), linked to the paper with `xr-hyper`; it now cites the methods it describes.
- The arXiv-only audit of Vangala et al. (2026) was removed: its test-set duplicate percentages were dropped, the leakage context now cites the peer-reviewed BRISC dataset paper (Fateh et al., 2026, Scientific Data), the NCC definition cites Zitova and Flusser (2003), and the inflated-score remark cites Kapoor and Narayanan (2023) and Roberts et al. (2021). Gneiting and Raftery (2007) was added for proper scoring rules.
- Entries for venues that publish no page numbers (ICLR, NeurIPS 2017 and 2019, AAAI-15) are `@misc` with the venue in `howpublished`, because `cas-model2-names.bst` warns about missing pages for `@inproceedings`.

## Changes for the EAAI template

- Moved from `elsarticle` to the Elsevier CAS single-column class, with author-year citations from `refs.bib` in place of the numbered manual bibliography.
- Title and abstract no longer use the undefined acronym "MRI" in the title (now "Magnetic Resonance Image"); the abstract is 243 words (EAAI asks for about 250) and states the AI contribution and the engineering application.
- Reference corrections found while checking every entry against Crossref, DataCite and arXiv: full author lists in place of "et al.", DenseNet pages 2261--2269 and Swin Transformer pages 9992--10002 (IEEE Xplore pagination of the DOI version), the full page range of Seetha and Raja (2018), the issue number of Kapoor and Narayanan (2023), and DOIs for every entry that has one.
- The supplement was merged into the paper at the time (see Figures); it was split out again into `supplementary.tex` in the 2026-10-17 restructuring. No numbers changed.

## Build

`npm run papers` (repository root) compiles every paper project with Tectonic; `npm run papers -- Standardised` compiles only this one, in the order `main.tex`, `supplementary.tex`, `main.tex`. Tectonic runs BibTeX itself; use Tectonic 0.17.0 or later, because 0.15.0 reruns TeX six times with "internal consistency problem when checking if main.bbl changed". With another TeX installation, run for each document in the same order: `pdflatex <doc>`, `bibtex <doc>`, then `pdflatex <doc>` twice.

## Verification status

- `main.pdf` (19 pages) and `supplementary.pdf` (24 pages) compile with Tectonic 0.17.0 with no TeX, LaTeX, hyperref, natbib or BibTeX warnings, and no overfull or underfull boxes.
- The paper check (`paper_check.py` of the `latex-research-papers` skill, with link checks) reports CLEAN for `main.tex`. For `supplementary.tex` its only finding is that the 35 figures and tables are not referenced in body text, which is by design since the supplement has no body text; most are referenced from the paper. The 2026-10-17 check found, for both documents: every figure and table sits on the page of its first mention or the next, every cross-reference into the other document resolves, every reference is cited, peer-reviewed and has a DOI or URL, and every DOI is registered. Publishers that refuse automated requests (403 or 406) were confirmed through the DOI handle API.
- The repository linked in Data availability is private, so its link does not resolve for readers until it is made public.
- Not checked: a pdfLaTeX build on Overleaf.
