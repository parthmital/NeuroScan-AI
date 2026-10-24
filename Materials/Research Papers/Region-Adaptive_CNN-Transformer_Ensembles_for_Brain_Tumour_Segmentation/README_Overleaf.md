# Overleaf Build Notes

Elsevier CAS single-column (`cas-sc`) manuscript for _Engineering Applications of Artificial Intelligence_: a region-adaptive hybrid ensemble of nnU-Net, SegResNet and Swin UNETR-v2 for brain tumour segmentation, validated internally on BraTS 2020 and externally on BraTS-Africa, with disagreement-based quality control and quantitative Seg-Grad-CAM. Every number, table and figure comes from the executed segmentation notebook and its outputs in `Materials/Notebooks/segmentation/` (Kaggle run of 2026-10-24; see `Materials/Notebooks/README.md`).

## Files

| File                                                   | Purpose                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `main.tex`                                             | The paper; set it as the main document.                                           |
| `refs.bib`                                             | References (BibTeX), each checked against its Crossref record; no preprints.      |
| `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` | Elsevier CAS template files, version 2.4 (single column, author-year references). |
| `thumbnails/cas-email.jpeg`                            | E-mail icon the CAS front matter prints beside each author address.               |
| `main.pdf`                                             | `main.tex` compiled with Tectonic (`npm run papers`).                             |
| `highlights.tex`                                       | Separate highlights file (five items, at most 85 characters each).                |
| `graphical_abstract.tex`                               | Graphical abstract (TikZ only); compile separately.                               |

## Figures

`main.tex` includes the notebook plots by their notebook path (for example `segmentation/plots/04_training_curves.png`) and sets `\graphicspath{{figures/}{../../Notebooks/}}`.

- **Local build** (`npm run papers -- Segmentation` from the repository root): the plots are read in place from `Materials/Notebooks/segmentation/plots/`.
- **Overleaf**: create a `figures/` folder in the project and upload the plot folder so that it sits at `figures/segmentation/plots/`.

Figures and tables are not floats: `herefigure` and `heretable` keep the CAS caption style but set each item directly after the paragraph that first cites it, and every plot is capped at a fraction of the text height so nothing overflows. Sixteen of the seventeen plots are used. `08_protocol_comparison.png` is left out because it shows the published GCSER-UNet values, which exist only in an arXiv preprint; the paper reports the slice-wise protocol for its own models in Table 11 instead.

## Sources of the results

| Paper item                                 | Source file in `Materials/Notebooks/segmentation/`                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Cases, roles, crops, run environment       | `logs/notebook.log`, `metrics/cases.csv`, `metrics/split.csv`, `metrics/run_summary.json`                                |
| Region volumes, cases without ET           | `metrics/region_volumes_by_cohort.csv`, `metrics/cases_without_et.csv`                                                   |
| Members and training settings              | `src/volume_lib.py`, `metrics/model_profiles.csv`, `metrics/worker_config.json`                                          |
| Model selection, ET filters, training time | `metrics/et_threshold_and_selection.csv`, `metrics/runs/<run>/summary.json`, `metrics/runs/<run>/history.csv`            |
| Fusion weights and thresholds              | `metrics/fusion_weights.csv`, `metrics/run_summary.json`                                                                 |
| Internal and external results              | `metrics/test_results.csv`, `metrics/external_results.csv`, `metrics/test_per_case.csv`, `metrics/external_per_case.csv` |
| Generalisation gap                         | `metrics/generalisation_gap.csv`                                                                                         |
| Wilcoxon tests                             | `metrics/wilcoxon_tests.csv`                                                                                             |
| Results by grade                           | `metrics/test_results_by_grade.csv`                                                                                      |
| Slice-wise protocol                        | `metrics/slice_protocol_comparison.csv`                                                                                  |
| Quality control                            | `metrics/quality_control.csv`, `metrics/qc_per_case.csv`                                                                 |
| Seg-Grad-CAM attribution                   | `metrics/gradcam_attribution.csv`, `metrics/gradcam_attribution_per_case.csv`                                            |
| Sequence-drop ablation                     | `metrics/modality_ablation_summary.csv`                                                                                  |
| Efficiency                                 | `metrics/efficiency.csv`                                                                                                 |

## Changes from the previous version

The folder was `A_Comprehensive_Benchmark_of_Three-Dimensional_Segmentation_Architectures_for_Brain_Tumour_Segmentation/`, a six-network benchmark (UNet, nnU-Net-style, VNet, UNETR, Swin UNETR, TransBTS) on 128^3 resampled volumes. The notebook behind it was replaced by the hybrid-ensemble notebook, whose run overwrote the old outputs, so the paper was rewritten around the new study and renamed. The old version is in Git history.

- New title, abstract (249 words), highlights, graphical abstract and every section; all tables and figures come from the new run.
- References: the VNet, ViT, DeiT, Medical Transformer and Swin-Unet entries were dropped as no longer cited. Added, each checked on Crossref: SegResNet (Myronenko 2019), SwinUNETR-V2 (He et al. 2023), BraTS-Africa (Adewole et al. 2025, Radiology: Artificial Intelligence), ensembles (Kamnitsas et al. 2018), uncertainty and quality control (Wang et al. 2019, Jungo and Reyes 2019, Roy et al. 2019, Mehrtash et al. 2020), Grad-CAM (Selvaraju et al. 2017, the same entry as the companion paper) and Seg-Grad-CAM (Vinogradova et al. 2020), external validity (Zech et al. 2018), Metrics Reloaded (Maier-Hein et al. 2024), and the statistics (Wilcoxon 1945, Efron 1979, Spearman 1904, Hanley and McNeil 1982).
- Bakas et al. (2018, arXiv 1811.02629) was removed because it is a preprint. The BraTS data-use terms ask users to cite it; add it back before submission if the journal accepts that preprint citation.
- PDF bookmarks now work (the sample file's `\let\WriteBookmarks\relax` is removed, as in the companion paper) and are numbered, including the abstract, keywords, title, the unnumbered declarations and the references.

## Verification status

Every number was transcribed from the files above and cross-checked against them. `main.pdf` (20 pages) compiles with Tectonic 0.17.0 (`npm run papers -- Region-Adaptive`) with no TeX, hyperref or BibTeX warnings, and the skill's `paper_check.py` reports `RESULT: CLEAN`: no undefined references, no uncited entries, every figure and table on the page of its first mention or the next, nothing outside the text block, and 16 bookmarks. `highlights.tex` and `graphical_abstract.tex` compile separately. Link check: every DOI resolves to its Crossref record; the publisher pages of Elsevier, RSNA and IEEE refuse automated requests (URL error, 403 or 202), Kaggle dataset pages return 404 without a browser session, and `https://github.com/parthmital/NeuroScan-AI` returns 404 while the repository is private. Not checked: a pdfLaTeX build on Overleaf.
