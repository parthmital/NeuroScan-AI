# Presentation Content

Content of `Presentation.pptx`, the Review 3 deck built from a duplicate of `Presentation Template.pptx`, slide by slide. Each slide gives its title, then its body. Lines starting with `>` describe the figures, charts, tables and diagrams or give notes for the presenter; they are not slide text. Citations [n] refer to the reference list on slides 19 and 20, numbered in order of first citation in the deck.

> Footer on slides 2 to 20: the review date 28/10/26, "SCOPE" and the slide number.

> Design: slide titles in Times New Roman 28 pt bold; all other text in Calibri (lead 16, subheadings 13, body 12, dense body 11, tables 10.5, captions 10, references 7.8 pt); navy is the only accent colour; model families are coloured only in charts (CNN navy, transformer brick, hybrid teal).

## Slide 1

> Layout: Title Slide. VIT logo kept at the top.

**Title:** NeuroScan AI: AI-Powered Brain Tumour Detection, Classification and Segmentation Platform

B.Tech BCSE497J – Project-I

**Team members**

Parth Mital (23BCE0842)

Ved Lawand (23BDS0141)

**Faculty guide**

Dr. Biji C. L.

Associate Professor Grade 2, SCOPE

Based on two research manuscripts: a leakage-controlled benchmark of 2D backbones for detection and subtype classification, and a region-adaptive CNN-transformer ensemble for 3D segmentation with external validation

> Speaker notes: Introduce the team and the guide. The project has two research outputs and one product: a 2D classification benchmark, a 3D segmentation study with external validation, and the NeuroScan AI web application that deploys the models those studies selected.

## Slide 2

> The guide's approval mail for Review 3 has not been received yet. When it arrives, replace the template's sample mail image with a screenshot of it and keep the line below.

**Title:** Approval Mail From Guide

All the above aspects are addressed and updated by the project team.

> Speaker notes: Replace the sample image with the guide's approval mail before the review.

## Slide 3

**Title:** Aim

To build and rigorously evaluate an AI system that **detects** brain tumours in MRI, **classifies** them as glioma, meningioma or pituitary tumour, and **segments** the whole tumour, tumour core and enhancing tumour in 3D, selecting every deployed model through a controlled, leakage-aware benchmark.

- **453,623**: primary brain and other CNS tumours diagnosed in the United States, 2016 to 2020 [1]
- **26.3%**: of these tumours are gliomas [1]
- **About 80%**: of malignant brain tumours worldwide are gliomas [2]

**1. Detect and classify in 2D**

Benchmark nine ImageNet-pretrained CNN, transformer and hybrid backbones for tumour detection and glioma, meningioma or pituitary classification on deduplicated public MRI slices (Paper 1).

**2. Segment in 3D**

Fuse two 3D CNNs and a transformer per tumour region for whole tumour, tumour core and enhancing tumour, and test them on BraTS 2020 and on external BraTS-Africa cases (Paper 2).

**3. Deploy**

Serve the most efficient models, EfficientNetB0 and SegResNet, in a secure web application with a linked multi-planar and 3D viewer and PDF and CSV reports.

**Motivation.** MRI is the main non-invasive modality for characterising brain tumours [3], which are classified by cell of origin and molecular features [4]. Manual reading is slow and varies between and within readers [5], and reliable automation can support faster, more consistent assessment [6, 7].

> Speaker notes: The aim has three parts that map to the two papers and the application. The statistics motivate the problem: brain tumours are common, and gliomas dominate the malignant group.

## Slide 4

**Title:** Abstract

**Problem.** Published brain tumour MRI results cannot be compared directly: studies differ in data, splits, training budgets and metrics, and near-duplicate images leak between training and test partitions [8, 9].

**Paper 1: 2D detection and subtype classification**

- Nine backbones (six CNNs, two transformers, one hybrid), one code path and an equal number of optimiser steps
- Label-aware perceptual-hash deduplication: 3,253 to 1,185 detection images and 5,828 to 4,730 subtype images; group-aware five-fold split with a held-out test set
- Fold ensembles, bootstrap 95% CIs, exact McNemar tests with Holm correction, calibration, Grad-CAM and background-only shortcut audits

**Result: 99.42%** detection accuracy of CoAtNet-0, EfficientNetB0 and ViT‑B/16 on 171 test images (95% CI 98.25 to 100%); no pair of models differs significantly; background-only models beat the majority-class rate for 7 of 9 detection and 9 of 9 subtype backbones

**Paper 2: 3D segmentation with external validation**

- nnU-Net and SegResNet (CNNs) and Swin UNETR-v2 (transformer) at native 1 mm resolution [10–12]
- Region-adaptive ensemble: one weight vector and threshold per region, chosen on 56 validation cases
- 56 internal BraTS 2020 test cases and 60 external BraTS-Africa cases [13]; member disagreement tested as a quality-control signal

**Result: 0.993** AUC of member disagreement for flagging failed cases internally (0.870 externally); the ensemble (Dice 0.8530) beats two members but not SegResNet alone (0.8717)

**Deployment.** EfficientNetB0 (detection and classification) and SegResNet (segmentation) run in a FastAPI and React application; 64 backend and 39 frontend automated tests pass.

> Speaker notes: One problem, two controlled studies, one deployed system. The 2D headline accuracy is measured after deduplication and qualified by the shortcut audit; in 3D the honest result is that the ensemble does not beat SegResNet, while the disagreement between its members is a strong quality signal.

## Slide 5

> The template shows this table as a picture; it is a native table here.

**Title:** Literature Review

| Study                                                   | Task and data                                        | Method                                                                    | Reported result             | Limitation addressed here                        |
| ------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------ |
| Zacharaki et al., 2009 [14]                             | Glioma vs metastasis, glioma grade; 102 tumours      | Shape, intensity and texture features; SVM with feature elimination       | 85% and 88% (leave-one-out) | Hand-crafted features transfer poorly [15, 16]   |
| Cheng et al., 2015 [17]                                 | Three-class subtype; 3,064 T1CE slices, 233 patients | Augmented tumour ROI, ring partition; intensity, GLCM, bag-of-words + SVM | Up to 91.28%                | Needs a tumour ROI; features redesigned per task |
| Seetha and Raja, 2018 [18]                              | Tumour vs non-tumour; Radiopaedia, BRATS 2015        | ImageNet-pretrained CNN, last layer trained                               | 97.5% (training)            | No held-out estimate                             |
| Talo et al., 2019 [19]                                  | Normal vs abnormal; 613 MR images                    | ResNet34 transfer learning [20, 21]                                       | 100% (five-fold)            | One small collection                             |
| Deepak and Ameer, 2019 [22]; Ghassemi et al., 2020 [23] | Three-class subtype; figshare 3,064 slices           | GoogLeNet features + classifiers; CNN pretrained as a GAN discriminator   | 98% mean (Deepak)           | One architecture each                            |
| Rehman et al., 2020 [24]; Noreen et al., 2020 [25]      | Three-class subtype; figshare 3,064 slices           | Fine-tuned VGG16 [26]; Inception-v3 [27] and DenseNet201 [28] features    | 98.69%; 99.34% and 99.51%   | No deduplication, CIs or tests                   |
| Khan et al., 2020 [29]                                  | Tumour classification; BraTS 2015, 2017, 2018        | VGG16 and VGG19 features, ELM-based selection                             | 97.8%, 96.9%, 92.5%         | Multi-stage tuned pipeline                       |
| Tummala et al., 2022 [30]; Asiri et al., 2024 [31]      | Three-class (3,064); four-class (2,870)              | ViT ensemble [32]; Swin Transformer [33]                                  | 98.7%; 97%                  | No CNN baseline under one protocol               |
| 3D CNNs [10, 11, 34, 35]                                | 3D medical segmentation                              | U-Net, 3D U-Net, SegResNet, self-configuring nnU-Net                      | Strong CNN baselines        | Mostly tested on internal splits                 |
| 3D transformers [12, 36–39]                             | 3D brain tumour segmentation                         | UNETR, Swin UNETR (v2), TransBTS, TransUNet                               | Long-range context          | Fused with CNNs by plain averaging [40]          |
| Kapoor and Narayanan, 2023 [8]; Fateh et al., 2026 [41] | Reliability of ML results; BRISC curation            | Leakage audit across 17 fields; duplicates removed before splitting       | Leakage inflates scores     | No architecture benchmark                        |

> Speaker notes: Early work used hand-crafted features; transfer learning pushed reported accuracy to 97 to 100%, but each study uses its own data, split and single model. Figures are as reported by each study; the figshare three-class studies are subtype classification, not detection.

## Slide 6

**Title:** Research Gap

1. **No controlled comparison.** Most studies test two or three models with their own data, splits and budgets [42, 43]. **Addressed by:** Nine 2D backbones and three 3D networks, each set under one code path and identical splits.
2. **Data leakage.** Combined public collections hold exact and near-duplicate images that are rarely removed before splitting [8, 9, 41]. **Addressed by:** Label-aware perceptual-hash deduplication, similarity groups, group-aware split and a leakage audit.
3. **Single-run leaderboards.** Results are quoted from one run, without confidence intervals, tests, per-class metrics or calibration. **Addressed by:** Five-fold CV, fold ensembles, bootstrap CIs, McNemar and Wilcoxon tests with Holm correction, ECE, per-grade results.
4. **Shortcuts unchecked.** Models can exploit borders, hospital markers or scanner cues instead of pathology [44–46]. **Addressed by:** Grad-CAM, occlusion, head and background masking, and background-only retraining.
5. **No external check in 3D.** 3D results are mostly internal, models degrade on new populations and scanners [45], and per-model uncertainty is a weak error signal [47]. **Addressed by:** External BraTS-Africa test set, per-region fusion, and member disagreement as a quality check.
6. **No path to use.** Benchmarked models are rarely turned into tools that accept MRI volumes and protect patient data. **Addressed by:** A secure web application with a multi-planar and 3D viewer and reports.

> Speaker notes: Each gap on the left is answered by a specific design decision on the right; these decisions are the contributions of the two papers.

## Slide 7

> SDG and TRL images kept from the template; SDG 3, SDG 9 and TRL 4 are outlined.

**Title:** Objectives

1. Benchmark nine 2D backbones for tumour detection and subtype classification with identical splits and an equal number of optimiser steps.
2. Remove leakage before splitting through label-aware perceptual-hash deduplication, similarity grouping and a group-aware split, verified by a leakage audit.
3. Evaluate with five-fold cross-validation, fold ensembles, bootstrap CIs, Holm-corrected McNemar tests, calibration, efficiency and shortcut audits.
4. Fuse two 3D CNNs and a transformer per tumour region on BraTS 2020, and test every model internally, externally on BraTS-Africa, for failure detection and for cost.
5. Deploy the most efficient models in a secure, tested web application with a multi-planar and 3D viewer and reports.

- **SDG 3:** Good Health and Well-being; **SDG 9:** Industry, Innovation and Infrastructure
- **TRL 4:** technology validated in the laboratory
- **Outcomes:** two journal manuscripts (Scopus) prepared for _Engineering Applications of Artificial Intelligence_ (Elsevier); an open-source web application

> Speaker notes: Objectives 1 to 3 belong to Paper 1, objective 4 to Paper 2 and objective 5 to the application.

## Slide 8

**Title:** Framework/Architecture/Block Diagram

> Diagram, research lane 1 (2D): 2D MRI collections (Br35H, Chakrabarty; Bhuvaji, Cheng) → curation (brain crop, perceptual-hash deduplication) → group-aware split (five folds, held-out test, leakage audit) → 9 backbones × 5 folds, two tasks, equal optimiser steps → evaluation (ensembles, CIs, McNemar, shortcut audit) → checkpoints.
> Diagram, research lane 2 (3D): BraTS 2020 and BraTS-Africa (369 + 60 cases, four sequences) → preprocessing (brain crop, foreground z-score, native 1 mm) → split 257 / 56 / 56 by grade plus 60 external cases → three networks and a per-region fusion → evaluation (internal, external, Wilcoxon, quality control) → checkpoints.
> Diagram, application: Browser (React, vtk.js) → reverse proxy (same origin, HttpOnly session cookie) → FastAPI (23 routes) → PipelineRunner (one background worker) → BrainProcessor (EfficientNetB0 detection on the middle axial slice, EfficientNetB0 classification, SegResNet segmentation with 128³ sliding windows into WT, TC and ET). The backend and the worker use SQLite with Alembic migrations and a per-scan upload store; results return to the user as probabilities, volumes, linked views and PDF and CSV reports.

> Speaker notes: Top: the two research pipelines that produced the checkpoints. Bottom: the application that loads them. The application only reads the exported checkpoints; training stays in the notebooks.

## Slide 9

**Title:** Functional Requirements and Modules

**Functional requirements**

- Register, sign in and out, change the password; optional TOTP two-factor authentication
- Upload one to four NIfTI volumes (FLAIR, T1, T1CE, T2) per case; modality detected from the file name
- Process each scan in the background with live status, step and progress; retry on failure
- Detect a tumour, classify its type with class probabilities, and segment WT, TC and ET with volumes
- View linked axial, coronal, sagittal and 3D panes with overlay, contrast, zoom and ruler
- Search, rename and delete scans; dashboard with counts, recent scans and a volume chart
- PDF and CSV report per scan; CSV export of all scans
- Operator account recovery from the command line

**Modules**

- **Data curation:** brain crop, perceptual-hash deduplication, label-noise exclusion, group split
- **2D benchmark:** 9 backbones × 5 folds plus a background-only run; 54 runs per task
- **3D study:** three networks, per-region fusion, internal and external tests, disagreement QC
- **Inference engine:** BrainProcessor with safe weight loading, the training crop and the mask in FLAIR space
- **Backend and pipeline:** FastAPI (23 routes), one worker, SQLite in WAL mode, Alembic
- **Security:** bcrypt, revocable JWT cookies, CSRF header, lockout, rate limits, TOTP
- **Viewer:** React and vtk.js; linked multi-planar and 3D views, slab modes, ruler
- **Reports:** PDF MRI report with key images; CSV export with formula escaping

> Image: `scan-viewer-panes.png`. Cropped from Screenshots/scan-viewer.png.
> Caption: NeuroScan AI viewer: linked axial, 3D, coronal and sagittal panes with the WT, TC and ET overlay on a BraTS case

> Speaker notes: Requirements are what the user can do; modules are how the project is built. The first three modules are the research notebooks; the rest form the application.

## Slide 10

> Section label: Experiments and Results: Paper 1 (2D).

**Title:** Data Curation and Leakage Control

> Diagram (pipeline, left to right):

- **1. Brain crop:** greyscale, 5×5 Gaussian, threshold 45, largest contour [48]
- **2. Perceptual hash:** 64-bit pHash [49] and a 64×64 thumbnail per image
- **3. Deduplicate:** same label, Hamming ≤ 5 [50]: keep one; cross-label NCC ≥ 0.98 [51]: drop both
- **4. Group:** Hamming ≤ 8 or NCC ≥ 0.98: 821 and 1,867 groups as a patient proxy
- **5. Split and audit:** stratified group k-fold [52, 53]: 15% test, five folds; no test image near development

| Task      | Sources                          | Images        | After deduplication | Dev.  | Test |
| --------- | -------------------------------- | ------------- | ------------------- | ----- | ---- |
| Detection | Br35H [54], Chakrabarty [55]     | 3,000 + 253   | 1,185 (−63.57%)     | 1,014 | 171  |
| Subtype   | Bhuvaji [56], Cheng [17, 57, 58] | 2,764 + 3,064 | 4,730 (−18.84%)     | 4,685 | 45   |

> Image: `detection-samples-crop-mask.png`. Left half of binary-detection/plots/02_samples_crop_mask.png.
> Caption: Original image, brain crop used for training, and head mask used by the attention audit

> Chart (column): Images per class before and after deduplication (Hamming ≤ 5)

| Category        | Before | After |
| --------------- | ------ | ----- |
| Tumour-negative | 1598   | 455   |
| Tumour          | 1655   | 730   |
| Glioma          | 2352   | 2008  |
| Meningioma      | 1645   | 1314  |
| Pituitary       | 1831   | 1408  |

- Eight of 31 cross-label pairs in the subtype pool were label noise; the detection pool had none.
- One subtype group held 1,431 images, so the subtype test set has only 45 images (1 glioma, 13 meningioma, 31 pituitary).
- The all-pairs search grows with the square of the pool; multi-index hashing [59] would scale it to larger collections.

> Speaker notes: Deduplication removed 63.57% of the detection pool, mostly Br35H copies, and 18.84% of the subtype pool. The brain crop is the same one the application applies. Grouping by similarity is a proxy for patient-level splitting because no source records patient identifiers.

## Slide 11

> Section label: Experiments and Results: Paper 1 (2D).

**Title:** Backbones, Training and Evaluation

| Backbone                   | Family      | Params (M) | GFLOPs | Latency (ms) | Images/s |
| -------------------------- | ----------- | ---------- | ------ | ------------ | -------- |
| ResNet50 [20]              | CNN         | 23.51      | 8.17   | 9.57         | 777      |
| VGG16 [26]                 | CNN         | 134.26     | 30.93  | 7.72         | 387      |
| DenseNet121 [28]           | CNN         | 6.95       | 5.67   | 23.58        | 657      |
| EfficientNetB0 [60]        | CNN         | 4.01       | 0.77   | 13.13        | 1,393    |
| MobileNetV3-Large [61, 62] | CNN         | 4.20       | 0.43   | 10.55        | 2,301    |
| InceptionV3 [27]           | CNN         | 21.79      | 11.42  | 17.63        | 512      |
| ViT-B/16 [32, 63]          | Transformer | 85.80      | 22.54  | 10.95        | 272      |
| Swin-Base [33]             | Transformer | 86.74      | 30.86  | 36.29        | 198      |
| CoAtNet-0 [64]             | Hybrid      | 26.67      | 8.42   | 16.53        | 513      |

CNNs stack learned convolutional filters [65, 66]; transformers relate all image patches through self-attention. All models are ImageNet-pretrained [67, 68] torchvision or timm models in PyTorch [69], with a one-output (detection) or three-output (subtype) head.

Cost was measured on one Tesla T4 in half precision: latency at batch size 1, throughput at batch size 32. **It spans a factor of about 72** (0.43 to 30.93 GFLOPs) while detection accuracy spans 2.34 points, and latency does not follow size: VGG16 (134 M parameters) takes 7.7 ms, DenseNet121 (6.95 M) 23.6 ms.

**Identical training: 54 runs per task**

- AdamW [70], learning rate 3×10⁻⁴, weight decay 0.01
- 20 epochs; effective batch 64 by gradient accumulation, so every model takes the same optimiser steps
- One-epoch warm-up and cosine decay [71]; half precision [72]; gradient clipping at 1.0 [73]
- Flip, ±10° rotation, ±20% brightness and contrast; checkpoint at the lowest validation log-loss [74]

**Evaluation protocol**

- Temperature scaling [75] and Youden threshold [76] fitted on each validation fold
- Five fold models averaged into one fold-ensemble prediction per test image
- Accuracy, macro F1, AUC [77, 78], MCC [79, 80] and ECE with 15 bins [81]
- Bootstrap 95% CIs from 2,000 resamples [82]; exact McNemar tests [83], Holm-corrected over 36 pairs [84]

> Speaker notes: Every row of every results table differs only in the architecture. Two Tesla T4 GPUs on Kaggle, PyTorch 2.11.0. Each backbone was also trained once on fold 0 with the head set to zero for the shortcut test, giving 54 runs per task, all completed. EfficientNetB0, shaded, is the deployed model.

## Slide 12

> Section label: Experiments and Results: Paper 1 (2D).

**Title:** Detection and Subtype Results

> Chart (bar): Detection: fold-ensemble test accuracy (%), 171 images

| Category          | Accuracy (%) |
| ----------------- | ------------ |
| CoAtNet-0         | 99.42        |
| EfficientNetB0    | 99.42        |
| ViT-B/16          | 99.42        |
| Swin-Base         | 98.83        |
| InceptionV3       | 98.25        |
| ResNet50          | 98.25        |
| DenseNet121       | 97.08        |
| MobileNetV3‑Large | 97.08        |
| VGG16             | 97.08        |

> Colours: CNN navy; Transformer brick; Hybrid teal.

- CoAtNet-0 missed one tumour; EfficientNetB0 and ViT-B/16 detected all 107 tumours and raised one false alarm.
- Every AUC lies between 0.9974 and 0.9997; the fold-to-fold standard deviation of accuracy, 0.32 to 2.20 points, matches the gaps between backbones.
- On the test images furthest from development (Hamming 17 to 20), every model is correct.

- **99.42%**: best accuracy, 95% CI 98.25 to 100%
- **0 of 36**: model pairs significant after Holm
- **4.01 M**: parameters in EfficientNetB0

| Subtype (45 test images)                                           | Accuracy [95% CI]  | Macro F1 | MCC    |
| ------------------------------------------------------------------ | ------------------ | -------- | ------ |
| ResNet50, DenseNet121, InceptionV3, ViT-B/16, Swin-Base, CoAtNet-0 | 100 [100, 100]     | 100      | 1.0000 |
| VGG16, EfficientNetB0, MobileNetV3-Large                           | 97.78 [93.33, 100] | 87.56    | 0.9515 |

The single subtype error is a meningioma predicted as glioma; with one glioma among 45 test images, this test set cannot rank the models.

**Calibration**

- ECE is 0.0146 to 0.0326 (detection) and 0.0044 to 0.0371 (subtype).
- Temperature scaling lowered ECE only for ResNet50 and ViT-B/16 (detection) and CoAtNet-0 (subtype).
- Near-separable validation folds push the fitted temperature to its lower bound, e⁻³ ≈ 0.0498, which sharpens the test probabilities.

> Speaker notes: No pair of models differs significantly (all Holm-adjusted p = 1.0), so the ordering on the left is not a ranking of architectures. The bootstrap gap to CoAtNet-0 excludes zero only for VGG16 and MobileNetV3-Large, both 2.34 points behind (0.58 to 4.68).

## Slide 13

> Section label: Experiments and Results: Paper 1 (2D).

**Title:** Interpretability and Shortcut Audit

> Image: `Notebooks\binary-detection\plots\13_gradcam_efficientnetb0.png`. Grad-CAM maps of EfficientNetB0 (detection), most confident correct predictions on the left and most confident errors on the right.

**Where the models look**

- Grad-CAM [85]: correct tumour predictions peak on the lesion; a missed temporal lesion peaks at the lower border
- ViT-B/16 attention rollout [86] puts 42.1% of its weight outside the head on tumour-negative images
- Occlusion sensitivity [87] outside the head: 0.7% to 19.4% (detection)

> Chart (bar): Shortcut test: accuracy (%) of models trained and tested on background-only images

| Category          | Detection (majority-class rate 62.6%) | Subtype (majority-class rate 68.9%) |
| ----------------- | ------------------------------------- | ----------------------------------- |
| ResNet50          | 81.29                                 | 93.33                               |
| VGG16             | 74.85                                 | 97.78                               |
| DenseNet121       | 79.53                                 | 86.67                               |
| EfficientNetB0    | 77.78                                 | 88.89                               |
| MobileNetV3‑Large | 67.84                                 | 93.33                               |
| InceptionV3       | 79.53                                 | 97.78                               |
| ViT-B/16          | 65.5                                  | 91.11                               |
| Swin-Base         | 83.04                                 | 91.11                               |
| CoAtNet-0         | 79.53                                 | 82.22                               |

> Grey bars: MobileNetV3-Large and ViT-B/16 detection, not significant after Holm correction.

**Masking and shortcut findings**

- Head-only inputs (Otsu head mask [88]) keep 93.7% to 96.8% (detection) and 94.2% to 99.6% (subtype) accuracy
- Without the head: 44.4% to 53.3% (detection; InceptionV3 65.7%) and 11.6% to 33.8% (subtype)
- Retrained on background only, 7 of 9 detection and 9 of 9 subtype backbones beat the majority-class rate (binomial test, Holm) [44]; grey bars are not significant
- **Borders and framing carry label information, so high accuracy on these collections does not prove that pathology was learned.**

> Speaker notes: Models trained on full images depend on the head, but the data do not stop them from using the background. Swin-Base reaches 83.0% detection accuracy without seeing any brain; subtype background-only models reach up to 97.8%.

## Slide 14

> Section label: Experiments and Results: Paper 2 (3D).

**Title:** Data, Networks and Region-Adaptive Fusion

> Image: `Notebooks\segmentation\plots\02_modalities_and_masks.png`. One BraTS 2020 case and one BraTS-Africa case: four sequences and three regions.

- BraTS 2020 [2, 89]: 369 cases (293 HGG, 76 LGG); split 257 / 56 / 56 by grade
- BraTS-Africa [13]: 60 external cases, never used for training or selection
- Nested regions: WT (all labels), TC (core and ET), ET; 27 of 76 LGG cases have no ET
- Brain crop and foreground z-score at native 1 mm; 128³ patches, one third on tumour; rotation, scaling, mirroring, noise, brightness, contrast and gamma
- Gaussian 128³ sliding windows with eight-flip TTA; ET size filter tuned on validation

| Network                | Type        | Design (MONAI 1.6.1)                      | Params (M) | Epochs |
| ---------------------- | ----------- | ----------------------------------------- | ---------- | ------ |
| nnU-Net [10]           | CNN         | Residual DynUNet, 6 stages, 32 to 320 ch. | 31.42      | 60     |
| SegResNet [11]         | CNN         | SegResNetDS, 32 filters, blocks 1-2-2-4   | 19.92      | 60     |
| Swin UNETR-v2 [12, 37] | Transformer | Swin encoder, feature size 48             | 72.77      | 30     |

**Region-adaptive hybrid ensemble**

- For each region, a convex weight vector over the three networks (0.1 grid) and a threshold (0.3 to 0.7) maximise validation Dice
- Selected: WT 0.4 nnU-Net + 0.6 SegResNet (threshold 0.4); TC 0.4 / 0.5 / 0.1 Swin (0.3); ET 0.5 / 0.5 (0.4)
- Ablations: each network, mean of all, mean of the CNNs, the weights at threshold 0.5
- Soft Dice plus cross-entropy with deep supervision; 528 minutes of training on two Tesla T4 GPUs

> Speaker notes: The search gave the transformer weight only for the tumour core. SegResNet, shaded, is the deployed model; the web application loads its checkpoint with the same MONAI version and runs it without TTA.

## Slide 15

> Section label: Experiments and Results: Paper 2 (3D).

**Title:** Internal and External Results

| Model             | Internal mean Dice [95% CI] | WT     | TC     | ET     | External mean Dice [95% CI] |
| ----------------- | --------------------------- | ------ | ------ | ------ | --------------------------- |
| nnU-Net           | 0.8404 [0.8008, 0.8760]     | 0.9139 | 0.8493 | 0.7580 | 0.7765 [0.7224, 0.8203]     |
| SegResNet         | 0.8717 [0.8389, 0.8987]     | 0.9264 | 0.8791 | 0.8095 | 0.7967 [0.7417, 0.8400]     |
| Swin UNETR-v2     | 0.8360 [0.7923, 0.8742]     | 0.9162 | 0.8427 | 0.7490 | 0.7888 [0.7355, 0.8325]     |
| CNN mean          | 0.8512 [0.8150, 0.8838]     | 0.9225 | 0.8630 | 0.7680 | 0.8005 [0.7484, 0.8419]     |
| Mean of all three | 0.8494 [0.8115, 0.8834]     | 0.9213 | 0.8601 | 0.7669 | 0.7971 [0.7456, 0.8400]     |
| Proposed ensemble | 0.8530 [0.8204, 0.8836]     | 0.9256 | 0.8639 | 0.7694 | 0.7966 [0.7430, 0.8391]     |

56 internal BraTS 2020 and 60 external BraTS-Africa test cases, eight-flip TTA; bootstrap CIs [82].

**Statistical comparison**

- Wilcoxon tests with Holm correction [84, 90]: the ensemble beats nnU-Net (p = 0.0001) and Swin UNETR (p = 0.007) internally
- **SegResNet alone beats the ensemble internally (p = 0.018) and equals it externally**
- Every model loses 0.047 to 0.075 Dice externally, mostly in TC

> Image: `Notebooks\segmentation\plots\11_qualitative_comparison.png`. Median and lowest internal and external cases: reference, the three networks and the proposed ensemble on FLAIR.

**Qualitative comparison**

- Rows: median and lowest internal case, median and lowest external case
- Lowest internal case: SegResNet 0.878, the other models 0.542 to 0.569
- Lowest external case: a small tumour that every model misses (Dice 0)
- Overlay: WT green, TC yellow, ET red

> Speaker notes: Best validation Dice: nnU-Net 0.8415 (epoch 55), SegResNet 0.8390 (50), Swin UNETR 0.8191 (30). The ensemble has the highest sensitivity in every region, from thresholds below 0.5, but not the highest Dice. Swin UNETR loses the least externally (0.047) and SegResNet the most (0.075).

## Slide 16

> Section label: Experiments and Results: Paper 2 (3D).

**Title:** Tumour Grade, Quality Control and Efficiency

> Chart (column): Enhancing-tumour Dice by tumour grade (internal)

| Category   | HGG (44 cases) | LGG (12 cases) |
| ---------- | -------------- | -------------- |
| nnU-Net    | 0.8531         | 0.4089         |
| SegResNet  | 0.8715         | 0.5822         |
| Swin UNETR | 0.8529         | 0.3681         |
| CNN mean   | 0.8622         | 0.4223         |
| Proposed   | 0.8648         | 0.4197         |

> Chart (bubble, area proportional to parameters): Internal mean Dice vs latency with TTA (bubble area: parameters)

| Model      | Latency per case with TTA (s) | Mean Dice | Parameters (M) |
| ---------- | ----------------------------- | --------- | -------------- |
| nnU-Net    | 8.2                           | 0.8404    | 31.42          |
| SegResNet  | 8.08                          | 0.8717    | 19.92          |
| Swin UNETR | 31.92                         | 0.836     | 72.77          |
| CNN mean   | 16.28                         | 0.8512    | 51.34          |
| Proposed   | 48.2                          | 0.853     | 124.11         |

**Disagreement flags failures**

- 1 minus the mean pairwise Dice of the three networks' masks; failure: mean Dice below 0.80
- AUC 0.993 internal, 0.870 external [77]; Spearman −0.80 and −0.82 [91]
- TTA spread: AUC 0.716 and 0.791 [92–94]

**Explanations, scored**

- Seg-Grad-CAM [85, 95]: CNN maps 4 to 23 times enriched in the tumour; Swin UNETR 1.1 to 1.7 times
- Dropping FLAIR hurts WT, dropping T1CE hurts TC and ET, for every network
- Slice-wise Dice differs from per-patient 3D Dice [96]

**SegResNet: the efficiency choice**

- Smallest network (19.92 M) and fastest: 1.01 s per case, 8.08 s with TTA on a T4
- The ensemble needs all three networks: 124.11 M parameters, 48.20 s with TTA, six times SegResNet, for a lower internal Dice

> Speaker notes: LGG cases are the weak spot of every model, because many have no enhancing tumour; SegResNet's internal lead comes mostly from them. Disagreement cannot flag a tumour that all networks miss. In the bubble chart, navy marks the CNNs, brick the transformer and teal the ensembles.

## Slide 17

**Title:** Conclusion and Future Work

1. **Cost decides in 2D.** All nine backbones reach 97.08% to 99.42% detection accuracy with no significant difference; EfficientNetB0 matches the leaders with 4.01 M parameters and 0.77 GFLOPs.
2. **Leakage and shortcuts are real.** Deduplication removed 63.57% and 18.84% of the images, and background-only models still beat chance: public 2D collections carry non-anatomical label cues.
3. **Fusion does not beat SegResNet in 3D.** The region-adaptive ensemble (0.8530 internal, 0.7966 external) beats two of its networks but not SegResNet alone (0.8717, 0.7967), at six times its cost.
4. **Disagreement is the useful product.** Disagreement between the networks flags failed cases with AUC 0.993 internally and 0.870 externally; every model loses 0.047 to 0.075 Dice on BraTS-Africa, and LGG ET Dice is 0.37 to 0.58.
5. **From benchmark to tool.** NeuroScan AI deploys EfficientNetB0 and SegResNet in a secure, tested application; it is a research prototype, not a diagnostic device.

**Future work**

- Patient-level splits and external, multi-centre validation
- Grouping that keeps every subtype in the test set
- Standardise image borders, then repeat the shortcut test
- 3D multi-sequence detection and classification
- Several seeds and cross-validated fusion for the 3D study
- More external cohorts with raw clinical preprocessing
- Show member disagreement in the application; grade-aware ET post-processing
- Calibration on larger held-out sets; DICOM input and physical volumes in the application

**Recommendation**

- Deploy SegResNet (3D) and EfficientNetB0 (2D); use a second, different network to measure disagreement rather than to fuse
- Validate externally before trusting an internal ranking: it changed on BraTS-Africa

> Speaker notes: Recommendation: SegResNet and EfficientNetB0 for deployment; in 3D, an architecturally different second network is most useful as a disagreement check. External validation changed the 3D ranking.

## Slide 18

> The template's line about uploading proof by 27 Oct 2026 (https://forms.gle/B9B71VPy5b9bZ9dJ8) is an instruction to the team, not slide content. Upload the proof there and update the status lines below (for example "Submitted" or "Under review") before the review.

**Title:** Outcome Achieved

**Journal publication:** two manuscripts prepared for _Engineering Applications of Artificial Intelligence_ (Elsevier, Scopus indexed)

**Manuscript 1**

**“Brain Tumour Magnetic Resonance Image Classification with Perceptual-Hash Deduplication: A Comparative Study Using Advanced Deep Learning Models”**

P. Mital, V. Lawand, Biji C. L., M. A. Dar and G. P. Doss C.

**Status:** prepared for submission

**Manuscript 2**

**“Region-Adaptive CNN-Transformer Ensembles for Brain Tumour Segmentation with External Validation and Disagreement-Based Quality Control”**

P. Mital, V. Lawand and Biji C. L.

**Status:** prepared for submission

**Others**

- Open-source NeuroScan AI web application (TRL 4) with 64 backend and 39 frontend automated tests
- Research notebooks with all outputs: splits, per-run metrics, plots and the trained checkpoints
- Repository: https://github.com/parthmital/NeuroScan-AI

> Speaker notes: Update the status before the review and upload the proof to the form given in the template.

## Slide 19

**Title:** References (1/2)

- [1] Q. T. Ostrom et al., “CBTRUS statistical report: Primary brain and other central nervous system tumors diagnosed in the United States in 2016–2020,” _Neuro-Oncology_, vol. 25, suppl. 4, pp. iv1–iv99, 2023.
- [2] S. Bakas et al., “Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features,” _Scientific Data_, vol. 4, Art. no. 170117, 2017.
- [3] J. E. Villanueva-Meyer, M. C. Mabray, and S. Cha, “Current clinical brain tumor imaging,” _Neurosurgery_, vol. 81, no. 3, pp. 397–415, 2017.
- [4] D. N. Louis et al., “The 2021 WHO classification of tumors of the central nervous system: A summary,” _Neuro-Oncology_, vol. 23, no. 8, pp. 1231–1251, 2021.
- [5] S. Bauer, R. Wiest, L.-P. Nolte, and M. Reyes, “A survey of MRI-based medical image analysis for brain tumor studies,” _Physics in Medicine and Biology_, vol. 58, no. 13, pp. R97–R129, 2013.
- [6] A. L. Beam and I. S. Kohane, “Big data and machine learning in health care,” _JAMA_, vol. 319, no. 13, pp. 1317–1318, 2018.
- [7] J. De Fauw et al., “Clinically applicable deep learning for diagnosis and referral in retinal disease,” _Nature Medicine_, vol. 24, no. 9, pp. 1342–1350, 2018.
- [8] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” _Patterns_, vol. 4, no. 9, Art. no. 100804, 2023.
- [9] M. Roberts et al., “Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans,” _Nature Machine Intelligence_, vol. 3, no. 3, pp. 199–217, 2021.
- [10] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation,” _Nature Methods_, vol. 18, no. 2, pp. 203–211, 2021.
- [11] A. Myronenko, “3D MRI brain tumor segmentation using autoencoder regularization,” in _Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2018)_, 2019, pp. 311–320.
- [12] Y. He, V. Nath, D. Yang, Y. Tang, A. Myronenko, and D. Xu, “SwinUNETR-V2: Stronger Swin transformers with stagewise convolutions for 3D medical image segmentation,” in _Medical Image Computing and Computer Assisted Intervention (MICCAI 2023)_, 2023, pp. 416–426.
- [13] M. Adewole et al., “The BraTS-Africa dataset: Expanding the brain tumor segmentation data to capture African populations,” _Radiology: Artificial Intelligence_, vol. 7, no. 4, Art. no. e240528, 2025.
- [14] E. I. Zacharaki et al., “Classification of brain tumor type and grade using MRI texture and shape in a machine learning scheme,” _Magnetic Resonance in Medicine_, vol. 62, no. 6, pp. 1609–1618, 2009.
- [15] N. Gordillo, E. Montseny, and P. Sobrevilla, “State of the art survey on MRI brain tumor segmentation,” _Magnetic Resonance Imaging_, vol. 31, no. 8, pp. 1426–1438, 2013.
- [16] G. Litjens et al., “A survey on deep learning in medical image analysis,” _Medical Image Analysis_, vol. 42, pp. 60–88, 2017.
- [17] J. Cheng et al., “Enhanced performance of brain tumor classification via tumor region augmentation and partition,” _PLoS ONE_, vol. 10, no. 10, Art. no. e0140381, 2015.
- [18] J. Seetha and S. S. Raja, “Brain tumor classification using convolutional neural networks,” _Biomedical and Pharmacology Journal_, vol. 11, no. 3, pp. 1457–1461, 2018.
- [19] M. Talo, U. B. Baloglu, Ö. Yıldırım, and U. R. Acharya, “Application of deep transfer learning for automated brain abnormality classification using MR images,” _Cognitive Systems Research_, vol. 54, pp. 176–188, 2019.
- [20] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016, pp. 770–778.
- [21] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” _Communications of the ACM_, vol. 60, no. 6, pp. 84–90, 2017.
- [22] S. Deepak and P. M. Ameer, “Brain tumor classification using deep CNN features via transfer learning,” _Computers in Biology and Medicine_, vol. 111, Art. no. 103345, 2019.
- [23] N. Ghassemi, A. Shoeibi, and M. Rouhani, “Deep neural network with generative adversarial networks pre-training for brain tumor classification based on MR images,” _Biomedical Signal Processing and Control_, vol. 57, Art. no. 101678, 2020.
- [24] A. Rehman, S. Naz, M. I. Razzak, F. Akram, and M. Imran, “A deep learning-based framework for automatic brain tumors classification using transfer learning,” _Circuits, Systems, and Signal Processing_, vol. 39, no. 2, pp. 757–775, 2020.
- [25] N. Noreen, S. Palaniappan, A. Qayyum, I. Ahmad, M. Imran, and M. Shoaib, “A deep learning model based on concatenation approach for the diagnosis of brain tumor,” _IEEE Access_, vol. 8, pp. 55135–55144, 2020.
- [26] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in _Proc. International Conference on Learning Representations (ICLR)_, 2015.
- [27] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016, pp. 2818–2826.
- [28] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2017, pp. 2261–2269.
- [29] M. A. Khan et al., “Multimodal brain tumor classification using deep learning and robust feature selection: A machine learning application for radiologists,” _Diagnostics_, vol. 10, no. 8, Art. no. 565, 2020.
- [30] S. Tummala, S. Kadry, S. A. C. Bukhari, and H. T. Rauf, “Classification of brain tumor from magnetic resonance imaging using vision transformers ensembling,” _Current Oncology_, vol. 29, no. 10, pp. 7498–7511, 2022.
- [31] A. A. Asiri et al., “Advancing brain tumor detection: Harnessing the Swin Transformer's power for accurate classification and performance analysis,” _PeerJ Computer Science_, vol. 10, Art. no. e1867, 2024.
- [32] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in _Proc. International Conference on Learning Representations (ICLR)_, 2021.
- [33] Z. Liu et al., “Swin transformer: Hierarchical vision transformer using shifted windows,” in _Proc. IEEE/CVF International Conference on Computer Vision (ICCV)_, 2021, pp. 9992–10002.
- [34] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in _Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)_, 2015, pp. 234–241.
- [35] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning dense volumetric segmentation from sparse annotation,” in _Medical Image Computing and Computer-Assisted Intervention (MICCAI 2016)_, 2016, pp. 424–432.
- [36] A. Hatamizadeh et al., “UNETR: Transformers for 3D medical image segmentation,” in _Proc. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_, 2022, pp. 1748–1758.
- [37] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, “Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images,” in _Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2021)_, 2022, pp. 272–284.
- [38] W. Wang, C. Chen, M. Ding, H. Yu, S. Zha, and J. Li, “TransBTS: Multimodal brain tumor segmentation using transformer,” in _Medical Image Computing and Computer Assisted Intervention (MICCAI 2021)_, 2021, pp. 109–119.
- [39] J. Chen et al., “TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers,” _Medical Image Analysis_, vol. 97, Art. no. 103280, 2024.
- [40] K. Kamnitsas et al., “Ensembles of multiple models and architectures for robust brain tumour segmentation,” in _Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2017)_, 2018, pp. 450–462.
- [41] A. Fateh et al., “BRISC: Annotated dataset for brain tumor segmentation and classification,” _Scientific Data_, vol. 13, Art. no. 361, 2026.
- [42] S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah, “Transformers in vision: A survey,” _ACM Computing Surveys_, vol. 54, no. 10s, pp. 1–41, 2022.
- [43] F. Shamshad et al., “Transformers in medical imaging: A survey,” _Medical Image Analysis_, vol. 88, Art. no. 102802, 2023.
- [44] R. Geirhos et al., “Shortcut learning in deep neural networks,” _Nature Machine Intelligence_, vol. 2, no. 11, pp. 665–673, 2020.
- [45] J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, “Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study,” _PLOS Medicine_, vol. 15, no. 11, Art. no. e1002683, 2018.
- [46] A. J. DeGrave, J. D. Janizek, and S.-I. Lee, “AI for radiographic COVID-19 detection selects shortcuts over signal,” _Nature Machine Intelligence_, vol. 3, no. 7, pp. 610–619, 2021.
- [47] A. Jungo and M. Reyes, “Assessing reliability and challenges of uncertainty estimations for medical image segmentation,” in _Medical Image Computing and Computer Assisted Intervention (MICCAI 2019)_, 2019, pp. 48–56.

> Speaker notes: IEEE-style references, numbered in order of first citation in this deck; all 96 references cited by the two manuscripts are included.

## Slide 20

**Title:** References (2/2)

- [48] S. Suzuki and K. Abe, “Topological structural analysis of digitized binary images by border following,” _Computer Vision, Graphics, and Image Processing_, vol. 30, no. 1, pp. 32–46, 1985.
- [49] V. Monga and B. L. Evans, “Perceptual image hashing via feature points: Performance evaluation and tradeoffs,” _IEEE Transactions on Image Processing_, vol. 15, no. 11, pp. 3452–3465, 2006.
- [50] R. W. Hamming, “Error detecting and error correcting codes,” _Bell System Technical Journal_, vol. 29, no. 2, pp. 147–160, 1950.
- [51] B. Zitová and J. Flusser, “Image registration methods: A survey,” _Image and Vision Computing_, vol. 21, no. 11, pp. 977–1000, 2003.
- [52] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” _Journal of Machine Learning Research_, vol. 12, no. 85, pp. 2825–2830, 2011.
- [53] M. Stone, “Cross-validatory choice and assessment of statistical predictions,” _Journal of the Royal Statistical Society: Series B (Methodological)_, vol. 36, no. 2, pp. 111–133, 1974.
- [54] A. Hamada, “Br35H :: Brain tumor detection 2020,” IEEE DataPort, also available at https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection, 2025.
- [55] N. Chakrabarty, “Brain MRI images for brain tumor detection,” Kaggle dataset, version 1, 2019.
- [56] S. Bhuvaji, A. Kadam, P. Bhumkar, S. Dedge, and S. Kanchan, “Brain tumor classification (MRI),” Kaggle dataset, 2020.
- [57] J. Cheng, “Brain tumor dataset,” figshare, version 5, 2017.
- [58] MoundirB, “Jun Cheng brain tumor MRI dataset JPG,” Kaggle dataset, version 1, 2025.
- [59] M. Norouzi, A. Punjani, and D. J. Fleet, “Fast exact search in Hamming space with multi-index hashing,” _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 36, no. 6, pp. 1107–1119, 2014.
- [60] M. Tan and Q. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in _Proc. 36th International Conference on Machine Learning (ICML)_, 2019, pp. 6105–6114.
- [61] A. Howard et al., “Searching for MobileNetV3,” in _Proc. IEEE/CVF International Conference on Computer Vision (ICCV)_, 2019, pp. 1314–1324.
- [62] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in _Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2018, pp. 4510–4520.
- [63] A. Vaswani et al., “Attention is all you need,” in _Advances in Neural Information Processing Systems (NeurIPS) 30_, 2017.
- [64] Z. Dai, H. Liu, Q. V. Le, and M. Tan, “CoAtNet: Marrying convolution and attention for all data sizes,” in _Advances in Neural Information Processing Systems (NeurIPS)_, 2021, pp. 3965–3977.
- [65] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” _Proceedings of the IEEE_, vol. 86, no. 11, pp. 2278–2324, 1998.
- [66] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” _Nature_, vol. 521, no. 7553, pp. 436–444, 2015.
- [67] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2009, pp. 248–255.
- [68] O. Russakovsky et al., “ImageNet large scale visual recognition challenge,” _International Journal of Computer Vision_, vol. 115, no. 3, pp. 211–252, 2015.
- [69] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in _Advances in Neural Information Processing Systems (NeurIPS) 32_, 2019.
- [70] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in _Proc. International Conference on Learning Representations (ICLR)_, 2019.
- [71] I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in _Proc. International Conference on Learning Representations (ICLR)_, 2017.
- [72] P. Micikevicius et al., “Mixed precision training,” in _Proc. International Conference on Learning Representations (ICLR)_, 2018.
- [73] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training recurrent neural networks,” in _Proc. 30th International Conference on Machine Learning (ICML)_, 2013, pp. 1310–1318.
- [74] T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” _Journal of the American Statistical Association_, vol. 102, no. 477, pp. 359–378, 2007.
- [75] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in _Proc. 34th International Conference on Machine Learning (ICML)_, 2017, pp. 1321–1330.
- [76] W. J. Youden, “Index for rating diagnostic tests,” _Cancer_, vol. 3, no. 1, pp. 32–35, 1950.
- [77] J. A. Hanley and B. J. McNeil, “The meaning and use of the area under a receiver operating characteristic (ROC) curve,” _Radiology_, vol. 143, no. 1, pp. 29–36, 1982.
- [78] T. Fawcett, “An introduction to ROC analysis,” _Pattern Recognition Letters_, vol. 27, no. 8, pp. 861–874, 2006.
- [79] B. W. Matthews, “Comparison of the predicted and observed secondary structure of T4 phage lysozyme,” _Biochimica et Biophysica Acta (BBA) – Protein Structure_, vol. 405, no. 2, pp. 442–451, 1975.
- [80] D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,” _BMC Genomics_, vol. 21, Art. no. 6, 2020.
- [81] M. Pakdaman Naeini, G. Cooper, and M. Hauskrecht, “Obtaining well calibrated probabilities using Bayesian binning,” in _Proc. AAAI Conference on Artificial Intelligence 29(1)_, 2015.
- [82] B. Efron, “Bootstrap methods: Another look at the jackknife,” _The Annals of Statistics_, vol. 7, no. 1, pp. 1–26, 1979.
- [83] Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” _Psychometrika_, vol. 12, no. 2, pp. 153–157, 1947.
- [84] S. Holm, “A simple sequentially rejective multiple test procedure,” _Scandinavian Journal of Statistics_, vol. 6, no. 2, pp. 65–70, 1979.
- [85] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in _Proc. IEEE International Conference on Computer Vision (ICCV)_, 2017, pp. 618–626.
- [86] S. Abnar and W. Zuidema, “Quantifying attention flow in transformers,” in _Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)_, 2020, pp. 4190–4197.
- [87] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in _Computer Vision – ECCV 2014_, 2014, pp. 818–833.
- [88] N. Otsu, “A threshold selection method from gray-level histograms,” _IEEE Transactions on Systems, Man, and Cybernetics_, vol. 9, no. 1, pp. 62–66, 1979.
- [89] B. H. Menze et al., “The multimodal brain tumor image segmentation benchmark (BRATS),” _IEEE Transactions on Medical Imaging_, vol. 34, no. 10, pp. 1993–2024, 2015.
- [90] F. Wilcoxon, “Individual comparisons by ranking methods,” _Biometrics Bulletin_, vol. 1, no. 6, pp. 80–83, 1945.
- [91] C. Spearman, “The proof and measurement of association between two things,” _The American Journal of Psychology_, vol. 15, no. 1, pp. 72–101, 1904.
- [92] G. Wang, W. Li, M. Aertsen, J. Deprest, S. Ourselin, and T. Vercauteren, “Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks,” _Neurocomputing_, vol. 338, pp. 34–45, 2019.
- [93] A. G. Roy, S. Conjeti, N. Navab, and C. Wachinger, “Bayesian QuickNAT: Model uncertainty in deep whole-brain segmentation for structure-wise quality control,” _NeuroImage_, vol. 195, pp. 11–22, 2019.
- [94] A. Mehrtash, W. M. Wells, C. M. Tempany, P. Abolmaesumi, and T. Kapur, “Confidence calibration and predictive uncertainty estimation for deep medical image segmentation,” _IEEE Transactions on Medical Imaging_, vol. 39, no. 12, pp. 3868–3878, 2020.
- [95] K. Vinogradova, A. Dibrov, and G. Myers, “Towards interpretable semantic segmentation via gradient-weighted class activation mapping (student abstract),” _Proceedings of the AAAI Conference on Artificial Intelligence_, vol. 34, no. 10, pp. 13943–13944, 2020.
- [96] L. Maier-Hein et al., “Metrics reloaded: Recommendations for image analysis validation,” _Nature Methods_, vol. 21, no. 2, pp. 195–212, 2024.

> Speaker notes: IEEE-style references, numbered in order of first citation in this deck; all 96 references cited by the two manuscripts are included.
