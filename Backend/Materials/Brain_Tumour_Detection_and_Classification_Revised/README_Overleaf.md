# Brain Tumour Detection and Classification — Revised Overleaf Project

Elsevier `elsarticle` (ScienceDirect) submission package, rebuilt from the proofread
`Detection & Classification #7.docx` and re-based on the **rerun** notebook outputs in
`Backend/Binary Detection/` and `Backend/Classification/`.

## Files

| File                     | Purpose                                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `main.tex`               | The paper. Compile with pdfLaTeX.                                                                                |
| `supplementary.tex`      | Supplementary Material (Tables S4–S12, Figures S1–S25); compile separately.                                      |
| `highlights.tex`         | Standalone highlights file (compile separately).                                                                 |
| `graphical_abstract.tex` | Standalone graphical abstract (compile separately).                                                              |
| `figures/`               | 30 PNGs copied from the rerun notebook outputs, plus 6 cropped main-text panels (`*_test.png`, `*_overlay.png`). |
| `figures/supp/`          | Per-model training, ROC/PR and calibration plots for all 15 runs, used by the supplement.                        |
| `figures/audit/`         | The 9 plots of `Backend/Revision Audit/plots/`, used by the supplement as Figures S17–S25.                         |

Supplementary item numbers follow the reviewer's mapping table (Comments-V2, Table M1), so
`supplementary.tex` sets each number explicitly; numbers the mapping does not assign
(Tables S1–S3 and S8, Figure S5) are unused. Items added for the revision audit continue
after the mapped range (Tables S11–S12, Figures S17–S25).

## Revision audit

`Backend/Revision Audit/revision-audit.ipynb` (run on Kaggle, outputs in `metrics/`, `plots/`
and `logs/` beside it) answers the Comments-V2 questions that need no retraining. Its results
were carried into the paper as follows; every number comes from `metrics/*.csv` or
`metrics/revision_audit_summary.json`.

| Finding                                                                                     | Where it now appears                                       |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Per-source survival, Bhuvaji training/testing folders, native resolution per source         | `tab:provenance`, Section 3.2, Table S11, Figure S17       |
| Multiclass removals: 3 cross-source, 1,120 within-source, 23 cross-label                    | Sections 3.2–3.3, Table S11                                |
| Cross-label pairs: 31 (8 label conflicts, 23 hash false positives); label-aware pool        | Section 3.6, Figure S18                                    |
| Binary pass: 1,187 kept, 2,066 removed (731 cross-source); 82.6% test images leaked         | Abstract, Sections 3.3, 4.1, 4.5, conclusion, Figure S20   |
| pHash vs NCC: recall 96.9% / 89.0% at tolerance 5                                           | Section 3.3, Figure S19                                    |
| Leakage-free test accuracy (binary 85 images; multiclass 265 and 703)                       | `tab:binary-results` "Acc. LF", Sections 4.1–4.2, Table S12, Figure S21 |
| EfficientNetB0 Grad-CAM outside-head fraction and masking test                              | Section 4.4, Figures S23–S25                               |
| Latency and throughput re-measured, mean ± SD over three repeats                            | `tab:efficiency`, Section 4.3, Tables S9–S10, Figure S22, abstract |

Not carried in because the audit could not produce it: retrained binary or label-aware
models, the background-only training ablation, and any Swin-Base or ViT-B/16 attention
measurement (only the EfficientNetB0 checkpoints were available to the audit).

`main.tex` uses a manual `thebibliography` block, so there is no `.bib` file and no BibTeX
pass is required. Two pdfLaTeX runs are enough to resolve all cross-references.

Required packages, all present in the standard Overleaf TeX Live image: `elsarticle`,
`graphicx`, `booktabs`, `array`, `tabularx`, `multirow`, `amsmath`, `amssymb`, `placeins`,
`float`, `caption`, `subcaption`, `microtype`, `tikz`, `url`.

## Figure provenance

Dataset-level figures are identical across runs of the same task and were taken from the
ResNet50 (binary) and InceptionV3 (multiclass) output folders. Model-specific figures come
from three runs, and every such caption names the model explicitly, as the reviewer required:
**EfficientNetB0** as the released checkpoint and efficiency candidate, **Swin-Base** as the
best multiclass model, and **ResNet50** as the binary contrast case.

| Figure file                                                                                                                                       | Source                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `binary_class_distribution`, `binary_samples`, `binary_split`, `binary_augmentation`, `binary_image_properties`, `binary_resnet_confusion`        | `binary-detection-resnet50/plots/`       |
| `binary_training_curves`, `binary_confusion`, `binary_roc_pr`, `binary_calibration`, `binary_per_class`, `binary_predictions`, `binary_gradcam_*` | `binary-detection-efficientnetb0/plots/` |
| `cls_samples`, `cls_deduplication`, `cls_split`, `cls_augmentation`, `cls_image_properties`                                                       | `classification-inceptionv3/plots/`      |
| `cls_training_curves`, `cls_confusion`, `cls_roc_pr`, `cls_calibration`, `cls_per_class`, `cls_predictions`, `cls_gradcam_*`                      | `classification-efficientnetb0/plots/`   |
| `cls_swin_confusion`, `cls_swin_roc`, `cls_swin_gradcam`                                                                                          | `classification-swinbase/plots/`         |
| `*_confusion_test` (test-count panel), `*_gradcam_correct_overlay` (header and overlay rows)                                                      | Cropped from the files above             |
| `supp/<run>_{training_curves,roc_pr_curves,calibration}`                                                                                          | `<run>/plots/07_*`, `09_*`, `10_*`       |

### What was deliberately left out

The two folders hold roughly 200 plots across 15 runs. Including all of them would bury the
argument, so each added figure had to do work the existing ones did not:

- **Superseded (Comments-V2):** training curves, ROC/PR curves and calibration plots for every
  run are now in the supplement (Figures S7–S11). Per-class metric plots and prediction
  samples remain EfficientNetB0-only.
- **Not included:** `calibration_bins_*.csv` and `predictions_*.csv`. The first is summarised
  by the ECE column in `tab:calibration`; the second is raw per-image output, not paper
  material.
- **Not included:** `classification_report_*.txt`, which duplicates the per-class tables.
- **Used instead of plotted:** `benchmark_summary.csv`, `model_profile.json`,
  `training_summary.json`, `inference_benchmark.json` and `per_class_metrics_test.csv` from
  **all 15 runs** feed the tables directly, so every run is represented numerically even where
  it contributes no figure.

## How the reviewer comments were addressed

| #                     | Comment                                                 | Resolution                                                                                                                                                    |
| --------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C0                    | Add overall methodology with a flowchart                | New Section 3.1 plus a five-stage TikZ flowchart (Figure 1)                                                                                                   |
| C1–C4                 | Add dataset references                                  | Four dataset entries added (`ds1`–`ds4`) and cited in Section 3.2                                                                                             |
| C5                    | Deduplication not done for binary                       | Stated explicitly in Section 3.3; the revision audit runs the same pass post hoc and reports its outcome; retraining remains future work                      |
| C6                    | Write hyperparameter full names                         | Section 3.5 spells every term out; new Table 4 (`tab:hyperparams`) lists them                                                                                 |
| C7                    | Restructure Results into subsections                    | Results now 4.1–4.5 in the requested order                                                                                                                    |
| C8                    | Inconsistent model naming                               | One convention throughout: ResNet50, VGG16, DenseNet121, EfficientNetB0, MobileNetV3-Large, InceptionV3, ViT-B/16, Swin-Base, CoAtNet-0                       |
| C9, C11–C14, C16, C17 | Name the model behind each figure                       | Every model-specific caption bolds the model name; all figures numbered and referenced in text                                                                |
| C10                   | Table numbers, split used, per-class metrics            | Every table numbered and cited; "held-out test split" stated in each caption; new per-class Tables 7 and 9 (`tab:binary-perclass`, `tab:multiclass-perclass`) |
| C15                   | Parameter counts inconsistent between tables            | All counts now taken from `model_profile.json` and identical everywhere; Section 3.4 notes the CoAtNet-0 discrepancy against the commonly quoted figure       |
| C18                   | Remove informal phrase                                  | The conversational recommendation paragraph rewritten in formal register                                                                                      |
| C19                   | Remove Limitations and Future Work section              | Removed; content folded into the conclusion                                                                                                                   |
| C20                   | Conclusion as one paragraph including future directions | Section 5 is a single paragraph covering results, limitations, and future work                                                                                |
| C21                   | Reference [8] unused                                    | Removed, along with every other uncited entry (old refs 8, 10, 11, 17, 22, 27, 31, 33). No unused bibitems remain.                                            |

## Verified against all 15 notebooks

A fourth pass read the notebook source itself, treating the notebooks as ground truth rather
than the folder READMEs. All 15 were extracted cell by cell, diffed against a per-task
reference, and every number in every table was re-checked programmatically against the
`metrics/` artefacts. **723 automated checks now pass.**

Confirmed identical across all 15 notebooks: seed 42, 10 epochs, LR 3e-4, AdamW with the
library-default weight decay, `CosineAnnealingLR(T_max=10)`, stratified 0.3/0.5 splitting,
`Resize → RandomHorizontalFlip → RandomRotation(10) → ColorJitter(0.2, 0.2) → ToTensor →
Normalize`, `drop_last=True` on the training loader only, `num_workers=4`,
`cudnn.benchmark=True`, and reloading the best checkpoint before the test pass. The only
substantive per-model differences are batch size, input resolution, backbone construction and
head replacement, the Grad-CAM target layer, and the InceptionV3 auxiliary-head handling —
exactly the set listed in Table 5.

Also verified directly: `deduplication_summary.csv` and `split_distribution.csv` are
byte-identical across all runs of each task, so the splits really are shared; and only
`binary-detection-inceptionv3` records 0 MB on the second GPU, so the single-GPU anomaly is
correctly scoped to that one run.

**Three further errors found and fixed in this pass:**

- **Swin-Base is not the same model across tasks.** The binary notebook uses the `timm` model
  `swin_base_patch4_window7_224`; the multiclass notebook uses `torchvision`'s `swin_b`.
  Different implementations, different pretrained checkpoints. I had written that both
  CoAtNet-0 and Swin-Base came from `timm`. Section 3.6 and Table 5 now record the
  inconsistency, and the conclusion notes that Swin-Base results are not comparable between
  the two tasks.
- **The Grad-CAM "misclassified" figures show the _most confident_ errors** (up to six,
  ranked by probability assigned to the wrong class), not all errors and not representative
  ones. I had written that activation in these cases was more diffuse and consistent with
  "lower predicted confidence" — the opposite of how they are selected. Section 4.4 is
  rewritten, and all five Grad-CAM captions now state their selection rule (the "correct"
  panels show the two most confident correct predictions per class).
- **Two parameter counts were rounded wrongly.** VGG16 is 134,264,641 → 134.26 M, not 134.27;
  DenseNet121 is 6,954,881 → 6.95 M, not 6.96. Fixed in the prose and in all tables. Dependent
  claims were re-derived and still hold.

Precision fixes: AdamW's weight decay is the library default, not an explicitly configured
value; and the reported test loss is accumulated in double precision from a half-precision
forward pass, which the text now states exactly. The InceptionV3 auxiliary-head claim was
verified to the digit — the binary/multiclass parameter delta is 772,329, exactly
$(768{\times}1000 + 1000 - 769) + (2048{\times}3 + 3 - 2049)$.

## Notebook-level methodology, captured from the folder READMEs

The first two passes were built from the notebook _outputs_ only. A third pass read
`Backend/Binary Detection/README.md` and `Backend/Classification/README.md`, which document
the pipeline, the per-notebook deviations and the known anomalies. That surfaced material
the paper had been describing incorrectly or not at all.

**Two factual errors in the paper were corrected:**

- The appendix figure `cls_class_distribution` was captioned "after deduplication". It is
  **before** deduplication: the exploratory cells (plots `01`–`03` and
  `image_dimension_stats.csv`) run on the raw 5,828-image pool, not the 4,705 used for
  training. Caption fixed, and the image-properties caption qualified the same way.
- The Swin-Base Grad-CAM reading was wrong and too charitable. I had claimed its wider
  activation plausibly helped meningioma via meningeal context. The actual finding is the
  opposite in emphasis: activation sits on the lesion **only** for meningioma, and for
  glioma and pituitary it falls largely on image corners and borders outside the brain.
  Section 4.4 now reports this as a possible shortcut, and it is carried into the
  conclusion's limitations and into a caveat on the transformer-paradox argument.

**Methodology now documented that previously was not:**

| Item                                                                                                                                                                   | Where it now appears                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Optimisation budget is **not** controlled: batch size 32–256 at fixed LR and epochs gives 80–350 optimiser steps                                                       | Section 3.5, new Table 5 (`tab:impl-diffs`), conclusion limitations |
| `drop_last=True` discards 229 of 2,277 binary training images per epoch (reshuffled each epoch)                                                                        | Section 3.5                                                         |
| InceptionV3 binary adds `0.4 × auxiliary loss`; no other model does                                                                                                    | Sections 3.6, Table 5                                               |
| InceptionV3 binary ran on **one** GPU despite the DataParallel wrapper (peak memory on the second device is 0 MB), so its 4.9-minute training time is not comparable   | Section 3.6, Section 4.3, `tab:efficiency` caption                  |
| InceptionV3 multiclass retains a disabled 1000-class auxiliary head, adding 769k parameters — this is why its binary and multiclass counts differ (24.35 M vs 25.12 M) | Section 3.6                                                         |
| Deduplication is order-dependent and label-blind; glioma always wins cross-class ties (23 such removals, counted by the revision audit)                                | Section 3.6                                                         |
| The Bhuvaji official test split was merged into the training pool before resplitting, so multiclass results are not comparable with studies using that partition       | Sections 3.6 and 4.5                                                |
| Top four binary models are statistically indistinguishable (one error = 0.20 pp; single seed, no confidence intervals)                                                 | Section 4.1, conclusion                                             |
| MobileNetV3-Large's Youden-optimal threshold is 0.00044, three orders of magnitude below the 0.5 decision point actually used                                          | Section 4.1                                                         |
| Reproducibility is approximate, not exact: `cudnn.benchmark`, TF32 and AMP admit non-deterministic kernels                                                             | Section 3.5                                                         |
| Reported "test loss" is sklearn log loss from a float64 single-device re-inference, not the fp16 per-batch mean printed in the notebook                                | Section 3.7                                                         |
| Multiclass checkpoints are selected on validation **macro AUC**, which saturates; several models kept improving in accuracy afterwards                                 | Table 4, Appendix B                                                 |
| Grad-CAM target layer differs per architecture; for VGG16 it resolves to a max-pool rather than a conv layer                                                           | Section 3.6, Table 5                                                |
| Swin-Base and CoAtNet-0 come from `timm` rather than `torchvision` (CoAtNet has no torchvision implementation)                                                         | Section 3.6, Table 5                                                |

The main structural addition is **Section 3.6, "Implementation Differences Between Models"**,
with Table 5 listing each deviation, its scope and its reason, separating those that are
unavoidable (head replacement, source library, input resolution) from those that are
unintended (single-GPU InceptionV3, retained auxiliary head).

Not carried into the paper, as they concern the artefact folder rather than the experiments:
the unknown provenance of the two root-level `EfficientNetB0.pth` files, hard-coded Kaggle
paths, the missing `torch.no_grad()` in evaluation, and minor code drift such as unused
imports.

## Content corrections beyond the marked comments

- Removed the stray sentence naming **Wav2Vec2 XLSR-53 and Whisper-small**, which belonged
  to an unrelated speech paper.
- Removed the Results preamble claiming the tasks involved **strokes**.
- Corrected the input-resolution claim: InceptionV3 uses 299×299, not 224×224.
- Attributed the $3\times3$-filter depth finding to Simonyan and Zisserman rather than
  leaving it attached to the ResNet citation.
- Dropped the unsupported "10×" EfficientNet parameter claim.
- Fixed the truncated final sentence of the old conclusion.

## All results are updated to the rerun

Every number in the paper was regenerated from the rerun `metrics/` JSON and CSV files.
The headline results changed materially from the previous version:

| Claim                     | Old paper                          | Rerun                                                                      |
| ------------------------- | ---------------------------------- | -------------------------------------------------------------------------- |
| Best binary accuracy      | 99.59% (CoAtNet-0, EfficientNetB0) | **99.39%**, four-way tie: CoAtNet-0, DenseNet121, EfficientNetB0, ResNet50 |
| Best multiclass accuracy  | 98.58% (InceptionV3 and Swin-Base) | **98.58% Swin-Base alone**; InceptionV3 98.30%                             |
| InceptionV3 binary        | 99.39%, AUC 1.0000                 | 98.77%, AUC 0.9986                                                         |
| VGG16 binary              | 83.20%                             | 88.32%                                                                     |
| MobileNetV3 binary        | 91.39%                             | 90.78%                                                                     |
| ViT-B/16 multiclass       | 89.52%                             | 90.51%                                                                     |
| EfficientNetB0 parameters | 5.3M                               | 4.01M                                                                      |
| CoAtNet-0 parameters      | ~10M                               | 26.67M                                                                     |

New material the rerun made possible: per-class precision, sensitivity, specificity and F1
for every model; expected calibration error, MCC and Cohen's kappa; measured GFLOPs,
latency and throughput for both tasks; native image-geometry characterisation; and Grad-CAM
interpretability (Section 4.4).

Sections added in the second pass, to use the outputs more fully:

Numbers below are as predicted from document order; confirm them after the first compile.

- **`tab:image-dims` (Table 2), `fig:binary-imgprops`, `fig:cls-imgprops`** — native image
  dimensions from `image_dimension_stats.csv`, substantiating the new claim in Section 3.2
  that the binary collection is far more heterogeneous (150–1,920 px wide) than the
  multiclass one (median 512×512 px).
- **`fig:binary-resnet-confusion` (Figure 9)** — ResNet50 binary confusion matrix beside
  EfficientNetB0's, showing the mirror-image error profile at identical 99.39% accuracy.
  This is the paper's key clinical point and it previously had no figure.
- **`fig:binary-perclass-fig` (Figure 11)** — per-class metric plot for the binary task,
  which had none.
- **`fig:cls-swin-confusion`, `fig:cls-swin-roc` (Figures 14–15)** — the best multiclass
  model was not visualised at all in the first pass.
- **`tab:efficiency-multiclass` (Table 11)** — multiclass efficiency comparison. The
  classification `benchmark_summary.csv` files were entirely unused before; they show
  Swin-Base needs 35.1 ms per image against 8.4 ms for ViT-B/16 despite near-identical size.
- **`fig:cls-swin-gradcam` (Figure 24)** — Swin-Base Grad-CAM, enabling a CNN-versus-
  transformer attention comparison in Section 4.4.

## Verification status

- Structural validation passed: balanced environments and braces, no undefined `\ref` or
  `\cite`, no unused `\bibitem`, all 31 referenced figure files present with none left
  unused, and every table row matching its column specification.
- **723 automated content checks pass**, covering every cell of every results table against
  the `metrics/` artefacts, the protocol constants in all 15 notebooks, and the derived
  claims in the prose. The single remaining flag is a tolerance artifact: ViT-B/16's
  multiclass throughput is exactly 289.5 img/s, reported as 290.
- All derived numerical claims in the prose (error counts, accuracy-per-parameter ratios,
  percentage-point gaps, throughput ratios) were checked against the source metrics files.
- **Not verified:** the document has not been compiled. No LaTeX toolchain was available in
  the environment where it was written, so please run one pdfLaTeX pass on Overleaf and
  check for layout issues, particularly float placement of the wide `table*` environments.
