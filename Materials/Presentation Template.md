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

Based on two research manuscripts: a leakage-controlled benchmark of 2D backbones for detection and subtype classification, and a benchmark of 3D segmentation architectures on BraTS 2020

> Speaker notes: Introduce the team and the guide. The project has two research outputs and one product: a 2D classification benchmark, a 3D segmentation benchmark, and the NeuroScan AI web application that deploys the models those benchmarks selected.

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

Benchmark six 3D architectures for whole tumour, tumour core and enhancing tumour segmentation on BraTS 2020 under one training budget (Paper 2).

**3. Deploy**

Serve the most efficient models, EfficientNetB0 and VNet, in a secure web application with a linked multi-planar and 3D viewer and PDF and CSV reports.

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

**Paper 2: 3D segmentation on BraTS 2020**

- Six networks implemented from scratch: UNet, nnU-Net-style, VNet, UNETR, Swin UNETR and TransBTS
- 369 cases in a grade-stratified 257 / 56 / 56 split; 50 epochs and 3,250 optimiser steps per model
- Native-resolution evaluation with eight-flip test-time augmentation, bootstrap CIs, Wilcoxon tests with Holm correction and results by tumour grade

**Result: 0.8199** mean Dice of UNet (95% CI 0.7826 to 0.8533); VNet 0.7965 with 1.40 M parameters; transformer-based models 0.7056 to 0.7462; every pairwise difference significant

**Deployment.** EfficientNetB0 (detection and classification) and VNet (segmentation) run in a FastAPI and React application; 61 backend and 39 frontend automated tests pass.

> Speaker notes: One problem, two controlled benchmarks, one deployed system. The headline accuracy is measured after deduplication, and the shortcut audit qualifies it.

## Slide 5

> The template shows this table as a picture; it is a native table here.

**Title:** Literature Review

| Study                                                   | Task and data                                        | Method                                                                    | Reported result             | Limitation addressed here                        |
| ------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------ |
| Zacharaki et al., 2009 [10]                             | Glioma vs metastasis, glioma grade; 102 tumours      | Shape, intensity and texture features; SVM with feature elimination       | 85% and 88% (leave-one-out) | Hand-crafted features transfer poorly [11, 12]   |
| Cheng et al., 2015 [13]                                 | Three-class subtype; 3,064 T1CE slices, 233 patients | Augmented tumour ROI, ring partition; intensity, GLCM, bag-of-words + SVM | Up to 91.28%                | Needs a tumour ROI; features redesigned per task |
| Seetha and Raja, 2018 [14]                              | Tumour vs non-tumour; Radiopaedia, BRATS 2015        | ImageNet-pretrained CNN, last layer trained                               | 97.5% (training)            | No held-out estimate                             |
| Talo et al., 2019 [15]                                  | Normal vs abnormal; 613 MR images                    | ResNet34 transfer learning [16, 17]                                       | 100% (five-fold)            | One small collection                             |
| Deepak and Ameer, 2019 [18]; Ghassemi et al., 2020 [19] | Three-class subtype; figshare 3,064 slices           | GoogLeNet features + classifiers; CNN pretrained as a GAN discriminator   | 98% mean (Deepak)           | One architecture each                            |
| Rehman et al., 2020 [20]; Noreen et al., 2020 [21]      | Three-class subtype; figshare 3,064 slices           | Fine-tuned VGG16 [22]; Inception-v3 [23] and DenseNet201 [24] features    | 98.69%; 99.34% and 99.51%   | No deduplication, CIs or tests                   |
| Khan et al., 2020 [25]                                  | Tumour classification; BraTS 2015, 2017, 2018        | VGG16 and VGG19 features, ELM-based selection                             | 97.8%, 96.9%, 92.5%         | Multi-stage tuned pipeline                       |
| Tummala et al., 2022 [26]; Asiri et al., 2024 [27]      | Three-class (3,064); four-class (2,870)              | ViT ensemble [28]; Swin Transformer [29]                                  | 98.7%; 97%                  | No CNN baseline under one protocol               |
| 3D CNNs [30–33]                                         | 3D medical segmentation                              | U-Net, 3D U-Net, V-Net, self-configuring nnU-Net                          | Strong CNN baselines        | Compared under different pipelines               |
| 3D transformers [34–37]                                 | 3D brain tumour segmentation                         | UNETR, Swin UNETR, TransBTS, TransUNet                                    | Long-range context          | Data-hungry from scratch [38]                    |
| Kapoor and Narayanan, 2023 [8]; Fateh et al., 2026 [39] | Reliability of ML results; BRISC curation            | Leakage audit across 17 fields; duplicates removed before splitting       | Leakage inflates scores     | No architecture benchmark                        |

> Speaker notes: Early work used hand-crafted features; transfer learning pushed reported accuracy to 97 to 100%, but each study uses its own data, split and single model. Figures are as reported by each study; the figshare three-class studies are subtype classification, not detection.

## Slide 6

**Title:** Research Gap

1. **No controlled comparison.** Most studies test two or three models with their own data, splits and budgets [40, 41]. **Addressed by:** Nine backbones and six 3D networks, one code path, identical splits, equal optimiser steps.
2. **Data leakage.** Combined public collections hold exact and near-duplicate images that are rarely removed before splitting [8, 9, 39]. **Addressed by:** Label-aware perceptual-hash deduplication, similarity groups, group-aware split and a leakage audit.
3. **Single-run leaderboards.** Results are quoted from one run, without confidence intervals, tests, per-class metrics or calibration. **Addressed by:** Five-fold CV, fold ensembles, bootstrap CIs, McNemar and Wilcoxon tests with Holm correction, ECE, per-grade results.
4. **Shortcuts unchecked.** Models can exploit borders, hospital markers or scanner cues instead of pathology [42–44]. **Addressed by:** Grad-CAM, occlusion, head and background masking, and background-only retraining.
5. **Transformer cost in 3D.** Transformers need large data, distillation or modified attention [38, 45, 46], and their cost is rarely reported. **Addressed by:** Parameters, GFLOPs, latency, training time and memory measured for every model.
6. **No path to use.** Benchmarked models are rarely turned into tools that accept MRI volumes and protect patient data. **Addressed by:** A secure web application with a multi-planar and 3D viewer and reports.

> Speaker notes: Each gap on the left is answered by a specific design decision on the right; these decisions are the contributions of the two papers.

## Slide 7

> SDG and TRL images kept from the template; SDG 3, SDG 9 and TRL 4 are outlined.

**Title:** Objectives

1. Benchmark nine 2D backbones for tumour detection and subtype classification with identical splits and an equal number of optimiser steps.
2. Remove leakage before splitting through label-aware perceptual-hash deduplication, similarity grouping and a group-aware split, verified by a leakage audit.
3. Evaluate with five-fold cross-validation, fold ensembles, bootstrap CIs, Holm-corrected McNemar tests, calibration, efficiency and shortcut audits.
4. Implement six 3D segmentation networks from scratch and compare them on BraTS 2020 per case, region, tumour grade and computational cost.
5. Deploy the most efficient models in a secure, tested web application with a multi-planar and 3D viewer and reports.

- **SDG 3:** Good Health and Well-being; **SDG 9:** Industry, Innovation and Infrastructure
- **TRL 4:** technology validated in the laboratory
- **Outcomes:** two journal manuscripts (Scopus) prepared for _Engineering Applications of Artificial Intelligence_ (Elsevier); an open-source web application

> Speaker notes: Objectives 1 to 3 belong to Paper 1, objective 4 to Paper 2 and objective 5 to the application.

## Slide 8

**Title:** Framework/Architecture/Block Diagram

> Diagram, research lane 1 (2D): 2D MRI collections (Br35H, Chakrabarty; Bhuvaji, Cheng) → curation (brain crop, perceptual-hash deduplication) → group-aware split (five folds, held-out test, leakage audit) → 9 backbones × 5 folds, two tasks, equal optimiser steps → evaluation (ensembles, CIs, McNemar, shortcut audit) → checkpoints.
> Diagram, research lane 2 (3D): BraTS 2020 (369 cases, four sequences) → preprocessing (foreground z-score, 128³ resampling) → stratified split 257 / 56 / 56 by grade → 6 3D networks, 3,250 optimiser steps each → evaluation (native grid, eight-flip TTA, Wilcoxon) → checkpoints.
> Diagram, application: Browser (React, vtk.js) → reverse proxy (same origin, HttpOnly session cookie) → FastAPI (23 routes) → PipelineRunner (one background worker) → BrainProcessor (EfficientNetB0 detection on the middle axial slice, EfficientNetB0 classification, VNet segmentation at 128³ into WT, TC and ET). The backend and the worker use SQLite with Alembic migrations and a per-scan upload store; results return to the user as probabilities, volumes, linked views and PDF and CSV reports.

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
- **3D benchmark:** six networks, 3,250 steps each, eight-flip TTA, Dice, HD95, Wilcoxon
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

- **1. Brain crop:** greyscale, 5×5 Gaussian, threshold 45, largest contour [47]
- **2. Perceptual hash:** 64-bit pHash [48] and a 64×64 thumbnail per image
- **3. Deduplicate:** same label, Hamming ≤ 5 [49]: keep one; cross-label NCC ≥ 0.98 [50]: drop both
- **4. Group:** Hamming ≤ 8 or NCC ≥ 0.98: 821 and 1,867 groups as a patient proxy
- **5. Split and audit:** stratified group k-fold [51, 52]: 15% test, five folds; no test image near development

| Task      | Sources                          | Images        | After deduplication | Dev.  | Test |
| --------- | -------------------------------- | ------------- | ------------------- | ----- | ---- |
| Detection | Br35H [53], Chakrabarty [54]     | 3,000 + 253   | 1,185 (−63.57%)     | 1,014 | 171  |
| Subtype   | Bhuvaji [55], Cheng [13, 56, 57] | 2,764 + 3,064 | 4,730 (−18.84%)     | 4,685 | 45   |

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

> Speaker notes: Deduplication removed 63.57% of the detection pool, mostly Br35H copies, and 18.84% of the subtype pool. The brain crop is the same one the application applies. Grouping by similarity is a proxy for patient-level splitting because no source records patient identifiers.

## Slide 11

> Section label: Experiments and Results: Paper 1 (2D).

**Title:** Backbones, Training and Evaluation

| Backbone                   | Family      | Params (M) | GFLOPs | Latency (ms) | Images/s |
| -------------------------- | ----------- | ---------- | ------ | ------------ | -------- |
| ResNet50 [16]              | CNN         | 23.51      | 8.17   | 9.57         | 777      |
| VGG16 [22]                 | CNN         | 134.26     | 30.93  | 7.72         | 387      |
| DenseNet121 [24]           | CNN         | 6.95       | 5.67   | 23.58        | 657      |
| EfficientNetB0 [58]        | CNN         | 4.01       | 0.77   | 13.13        | 1,393    |
| MobileNetV3-Large [59, 60] | CNN         | 4.20       | 0.43   | 10.55        | 2,301    |
| InceptionV3 [23]           | CNN         | 21.79      | 11.42  | 17.63        | 512      |
| ViT-B/16 [28, 61]          | Transformer | 85.80      | 22.54  | 10.95        | 272      |
| Swin-Base [29]             | Transformer | 86.74      | 30.86  | 36.29        | 198      |
| CoAtNet-0 [62]             | Hybrid      | 26.67      | 8.42   | 16.53        | 513      |

CNNs stack learned convolutional filters [63, 64]; transformers relate all image patches through self-attention. All models are ImageNet-pretrained [65, 66] torchvision or timm models in PyTorch [67], with a one-output (detection) or three-output (subtype) head.

Cost was measured on one Tesla T4 in half precision: latency at batch size 1, throughput at batch size 32. **It spans two orders of magnitude** (0.43 to 30.93 GFLOPs) while detection accuracy spans 2.34 points, and latency does not follow size: VGG16 (134 M parameters) takes 7.7 ms, DenseNet121 (6.95 M) 23.6 ms.

**Identical training: 54 runs per task**

- AdamW [68], learning rate 3×10⁻⁴, weight decay 0.01
- 20 epochs; effective batch 64 by gradient accumulation, so every model takes the same optimiser steps
- One-epoch warm-up and cosine decay [69]; half precision [70]; gradient clipping at 1.0 [71]
- Flip, ±10° rotation, ±20% brightness and contrast; checkpoint at the lowest validation log-loss [72]

**Evaluation protocol**

- Temperature scaling [73] and Youden threshold [74] fitted on each validation fold
- Five fold models averaged into one fold-ensemble prediction per test image
- Accuracy, macro F1, AUC [75, 76], MCC [77, 78] and ECE with 15 bins [79]
- Bootstrap 95% CIs from 2,000 resamples [80]; exact McNemar tests [81], Holm-corrected over 36 pairs [82]

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

- Grad-CAM [83]: correct tumour predictions peak on the lesion; a missed temporal lesion peaks at the lower border
- ViT-B/16 attention rollout [84] puts 42.1% of its weight outside the head on tumour-negative images
- Occlusion sensitivity [85] outside the head: 0.7% to 19.4% (detection)

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

- Head-only inputs (Otsu head mask [86]) keep 93.7% to 96.8% (detection) and 94.2% to 99.6% (subtype) accuracy
- Without the head: 44.4% to 53.3% (detection; InceptionV3 65.7%) and 11.6% to 33.8% (subtype)
- Retrained on background only, 7 of 9 detection and 9 of 9 subtype backbones beat the majority-class rate (binomial test, Holm) [42]; grey bars are not significant
- **Borders and framing carry label information, so high accuracy on these collections does not prove that pathology was learned.**

> Speaker notes: Models trained on full images depend on the head, but the data do not stop them from using the background. Swin-Base reaches 83.0% detection accuracy without seeing any brain; subtype background-only models reach up to 97.8%.

## Slide 14

> Section label: Experiments and Results: Paper 2 (3D).

**Title:** BraTS 2020 Data and 3D Architectures

> Image: `Notebooks\segmentation\plots\02_modalities_and_masks.png`. One BraTS 2020 case: the four sequences and the three nested regions.

- BraTS 2020 [2, 87, 88]: 369 cases (293 HGG, 76 LGG) with FLAIR, T1, T1CE and T2
- Labels 1, 2 and 4 give nested regions: WT (all labels), TC (1 and 4), ET (4)
- Median ET volume 0.82 cm³ in LGG vs 19.46 cm³ in HGG; 27 of 76 LGG cases have no ET
- Foreground z-score per sequence; 240×240×155 resampled to 128³ (trilinear; masks nearest); flips, intensity scale 0.9 to 1.1 and shift ±0.1
- Grade-stratified split with seed 42: 257 training, 56 validation, 56 test (44 HGG, 12 LGG)

| Model              | Type        | Design                                         | Params (M) | GFLOPs  |
| ------------------ | ----------- | ---------------------------------------------- | ---------- | ------- |
| UNet [30, 31]      | CNN         | Four levels, 32 to 256 channels, LeakyReLU     | 5.60       | 907.0   |
| nnU-Net-style [33] | CNN         | Five stages, 32 to 320 channels                | 16.55      | 956.3   |
| VNet [32]          | CNN         | Residual blocks, 16 to 128 channels, PReLU     | 1.40       | 149.9   |
| UNETR [34]         | Transformer | ViT-Base on 16³ patches: 512 tokens            | 137.60     | 1,761.7 |
| Swin UNETR [35]    | Transformer | Swin encoder, 2³ patches, 7³ windows           | 10.52      | 797.1   |
| TransBTS [36]      | Hybrid      | CNN encoder, 4 transformer layers on 8³ tokens | 35.81      | 996.2   |

**Identical budget: 50 epochs, effective batch 4, 3,250 optimiser steps**

- Loss per region: soft Dice plus binary cross-entropy, with deep supervision in UNet, nnU-Net-style, VNet and TransBTS
- CNNs: SGD with Nesterov momentum 0.99, learning rate 0.01, polynomial decay (power 0.9)
- Transformer-based models: AdamW, learning rate 10⁻⁴, three-epoch warm-up, cosine decay
- Half precision, clipping at 1.0; gradient checkpointing for UNETR and Swin UNETR
- Checkpoint at the best validation mean Dice; 515 minutes of training on two Tesla T4 GPUs

> Speaker notes: All six networks are implemented from scratch in one notebook and take the same four-channel 128³ input. VNet, shaded, is the deployed model; its module names match the application's network class, so its weights load unchanged.

## Slide 15

> Section label: Experiments and Results: Paper 2 (3D).

**Title:** 3D Segmentation Test Results

| Model         | Type        | Mean Dice [95% CI]      | WT     | TC     | ET     | HD95 TC (mm) | ET sens. |
| ------------- | ----------- | ----------------------- | ------ | ------ | ------ | ------------ | -------- |
| UNet          | CNN         | 0.8199 [0.7826, 0.8533] | 0.9012 | 0.8300 | 0.7286 | 3.24         | 0.8486   |
| nnU-Net-style | CNN         | 0.8192 [0.7837, 0.8511] | 0.9002 | 0.8309 | 0.7264 | 3.46         | 0.8538   |
| VNet          | CNN         | 0.7965 [0.7497, 0.8340] | 0.8865 | 0.7972 | 0.7058 | 4.18         | 0.8278   |
| Swin UNETR    | Transformer | 0.7462 [0.7134, 0.7772] | 0.8714 | 0.7495 | 0.6176 | 6.70         | 0.9407   |
| TransBTS      | Hybrid      | 0.7291 [0.6876, 0.7661] | 0.8743 | 0.7488 | 0.5642 | 6.58         | 0.9475   |
| UNETR         | Transformer | 0.7056 [0.6700, 0.7384] | 0.8758 | 0.6692 | 0.5720 | 11.29        | 0.9520   |

56 test cases at native resolution with eight-flip TTA and validation-tuned ET filtering; HD95 is the median over cases.

**What decides the ranking**

- WT is easy for every model (Dice 0.8714 to 0.9012); TC and ET separate them
- Transformer-based models are more sensitive but over-segment; specificity exceeds 0.997 for all
- All 15 Wilcoxon pairs are significant after Holm; UNet and nnU-Net-style differ by a median 0.0023 (p = 0.042)

> Image: `Notebooks\segmentation\plots\08_qualitative_comparison.png`. Median HGG case (top) and hardest LGG case (bottom): reference and every model on FLAIR at the slice with the largest reference tumour.

**Qualitative comparison**

- Top: median HGG case; every model recovers the tumour (mean Dice 0.790 to 0.869)
- Bottom: hardest LGG case, whose reference has no enhancing tumour; the transformer-based models add ET and fall to mean Dice 0.329 to 0.347
- Overlay: WT green, TC yellow, ET red

> Speaker notes: UNet leads VNet by a median 0.0119 Dice and every transformer-based model by 0.0738 to 0.1139. Best validation Dice: UNet 0.7845 (epoch 45), nnU-Net-style 0.7824 (50), VNet 0.7598 (44), Swin UNETR 0.7235 (33), TransBTS 0.7102 (45), UNETR 0.6960 (39).

## Slide 16

> Section label: Experiments and Results: Paper 2 (3D).

**Title:** Tumour Grade, Ensembles and Efficiency

> Chart (column): Enhancing-tumour Dice by tumour grade

| Category      | HGG (44 cases) | LGG (12 cases) |
| ------------- | -------------- | -------------- |
| UNet          | 0.8117         | 0.4237         |
| nnU-Net-style | 0.8124         | 0.4112         |
| VNet          | 0.7923         | 0.3885         |
| Swin UNETR    | 0.7262         | 0.2193         |
| TransBTS      | 0.674          | 0.1613         |
| UNETR         | 0.6773         | 0.1857         |

> Chart (bubble, area proportional to parameters): Mean Dice vs latency per volume (bubble area: parameters)

| Model         | Latency per volume (s, no TTA) | Mean Dice | Parameters (M) |
| ------------- | ------------------------------ | --------- | -------------- |
| UNet          | 0.1076                         | 0.8199    | 5.6            |
| nnU-Net-style | 0.1112                         | 0.8192    | 16.55          |
| VNet          | 0.0375                         | 0.7965    | 1.4            |
| UNETR         | 0.2094                         | 0.7056    | 137.6          |
| Swin UNETR    | 0.4893                         | 0.7462    | 10.52          |
| TransBTS      | 0.1291                         | 0.7291    | 35.81          |

**Tumour grade**

- LGG mean Dice 0.5616 to 0.6624 vs HGG 0.7434 to 0.8632; WT Dice is slightly higher on LGG
- One false-positive ET voxel on a case without ET gives that case an ET Dice of 0

**Ensembles and uncertainty**

- CNN ensemble 0.8178 and six-model ensemble 0.7991: neither beats UNet alone
- TTA uncertainty vs Dice: Spearman −0.30 to −0.40 (p ≤ 0.023) for CNNs; not significant for the transformer-based models

**VNet: the efficiency choice**

- 97% of the UNet Dice with 1.40 M parameters, 149.9 GFLOPs and 0.53 training hours
- 0.0375 s per volume (UNet 0.1076 s) and 0.98 GB at inference; best Dice per parameter (0.5694) and per training hour (1.4939)

> Speaker notes: Swin UNETR is the slowest model (0.4893 s per volume, 5.29 training hours) despite 10.52 M parameters, because its first stage attends over 64³ tokens. TTA multiplies latency by about eight for every model. In the bubble chart, navy marks CNNs, brick transformers and teal the hybrid.

## Slide 17

**Title:** Conclusion and Future Work

1. **Cost decides in 2D.** All nine backbones reach 97.08% to 99.42% detection accuracy with no significant difference; EfficientNetB0 matches the leaders with 4.01 M parameters and 0.77 GFLOPs.
2. **Leakage and shortcuts are real.** Deduplication removed 63.57% and 18.84% of the images, and background-only models still beat chance: public 2D collections carry non-anatomical label cues.
3. **CNNs lead in 3D.** UNet (0.8199) and nnU-Net-style (0.8192) lead, every CNN beats every from-scratch transformer-based model, and VNet gives 97% of the UNet Dice at about a third of its latency.
4. **The weak spot is known.** Whole-tumour Dice is 0.87 to 0.90 for every model, but LGG enhancing-tumour Dice is 0.16 to 0.42, a bias that must be disclosed to users.
5. **From benchmark to tool.** NeuroScan AI deploys EfficientNetB0 and VNet in a secure, tested application; it is a research prototype, not a diagnostic device.

**Future work**

- Patient-level splits and external, multi-centre validation
- Grouping that keeps every subtype in the test set
- Standardise image borders, then repeat the shortcut test
- 3D multi-sequence detection and classification
- Several seeds and cross-validation for the 3D benchmark
- Pretrained transformer encoders; higher-resolution or patch-based training
- Grade-aware ET post-processing for LGG cases
- Calibration on larger held-out sets; DICOM input and physical volumes in the application

**Recommendation**

- Highest 3D accuracy: UNet; limited compute and deployment: VNet (3D) and EfficientNetB0 (2D)
- Revisit transformers with pretraining, larger datasets or higher-resolution training

> Speaker notes: Recommendation: UNet for the highest accuracy; VNet and EfficientNetB0 for deployment with limited compute. Transformers should be revisited with pretraining, more data or higher resolution.

## Slide 18

> The template's line about uploading proof by 27 Oct 2026 (https://forms.gle/B9B71VPy5b9bZ9dJ8) is an instruction to the team, not slide content. Upload the proof there and update the status lines below (for example "Submitted" or "Under review") before the review.

**Title:** Outcome Achieved

**Journal publication:** two manuscripts prepared for _Engineering Applications of Artificial Intelligence_ (Elsevier, Scopus indexed)

**Manuscript 1**

**“Standardised Comparison of Convolutional and Transformer Backbones for Brain Tumour Magnetic Resonance Image Classification under Perceptual-Hash Deduplication”**

P. Mital, V. Lawand, Biji C. L., M. A. Dar and G. P. Doss C.

**Status:** prepared for submission

**Manuscript 2**

**“A Comprehensive Benchmark of Three-Dimensional Segmentation Architectures for Brain Tumour Segmentation”**

P. Mital, V. Lawand and Biji C. L.

**Status:** prepared for submission

**Others**

- Open-source NeuroScan AI web application (TRL 4) with 61 backend and 39 frontend automated tests
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
- [10] E. I. Zacharaki et al., “Classification of brain tumor type and grade using MRI texture and shape in a machine learning scheme,” _Magnetic Resonance in Medicine_, vol. 62, no. 6, pp. 1609–1618, 2009.
- [11] N. Gordillo, E. Montseny, and P. Sobrevilla, “State of the art survey on MRI brain tumor segmentation,” _Magnetic Resonance Imaging_, vol. 31, no. 8, pp. 1426–1438, 2013.
- [12] G. Litjens et al., “A survey on deep learning in medical image analysis,” _Medical Image Analysis_, vol. 42, pp. 60–88, 2017.
- [13] J. Cheng et al., “Enhanced performance of brain tumor classification via tumor region augmentation and partition,” _PLoS ONE_, vol. 10, no. 10, Art. no. e0140381, 2015.
- [14] J. Seetha and S. S. Raja, “Brain tumor classification using convolutional neural networks,” _Biomedical and Pharmacology Journal_, vol. 11, no. 3, pp. 1457–1461, 2018.
- [15] M. Talo, U. B. Baloglu, Ö. Yıldırım, and U. R. Acharya, “Application of deep transfer learning for automated brain abnormality classification using MR images,” _Cognitive Systems Research_, vol. 54, pp. 176–188, 2019.
- [16] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016, pp. 770–778.
- [17] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” _Communications of the ACM_, vol. 60, no. 6, pp. 84–90, 2017.
- [18] S. Deepak and P. M. Ameer, “Brain tumor classification using deep CNN features via transfer learning,” _Computers in Biology and Medicine_, vol. 111, Art. no. 103345, 2019.
- [19] N. Ghassemi, A. Shoeibi, and M. Rouhani, “Deep neural network with generative adversarial networks pre-training for brain tumor classification based on MR images,” _Biomedical Signal Processing and Control_, vol. 57, Art. no. 101678, 2020.
- [20] A. Rehman, S. Naz, M. I. Razzak, F. Akram, and M. Imran, “A deep learning-based framework for automatic brain tumors classification using transfer learning,” _Circuits, Systems, and Signal Processing_, vol. 39, no. 2, pp. 757–775, 2020.
- [21] N. Noreen, S. Palaniappan, A. Qayyum, I. Ahmad, M. Imran, and M. Shoaib, “A deep learning model based on concatenation approach for the diagnosis of brain tumor,” _IEEE Access_, vol. 8, pp. 55135–55144, 2020.
- [22] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in _Proc. International Conference on Learning Representations (ICLR)_, 2015.
- [23] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016, pp. 2818–2826.
- [24] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2017, pp. 2261–2269.
- [25] M. A. Khan et al., “Multimodal brain tumor classification using deep learning and robust feature selection: A machine learning application for radiologists,” _Diagnostics_, vol. 10, no. 8, Art. no. 565, 2020.
- [26] S. Tummala, S. Kadry, S. A. C. Bukhari, and H. T. Rauf, “Classification of brain tumor from magnetic resonance imaging using vision transformers ensembling,” _Current Oncology_, vol. 29, no. 10, pp. 7498–7511, 2022.
- [27] A. A. Asiri et al., “Advancing brain tumor detection: Harnessing the Swin Transformer's power for accurate classification and performance analysis,” _PeerJ Computer Science_, vol. 10, Art. no. e1867, 2024.
- [28] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in _Proc. International Conference on Learning Representations (ICLR)_, 2021.
- [29] Z. Liu et al., “Swin transformer: Hierarchical vision transformer using shifted windows,” in _Proc. IEEE/CVF International Conference on Computer Vision (ICCV)_, 2021, pp. 9992–10002.
- [30] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in _Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)_, 2015, pp. 234–241.
- [31] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning dense volumetric segmentation from sparse annotation,” in _Medical Image Computing and Computer-Assisted Intervention (MICCAI 2016)_, 2016, pp. 424–432.
- [32] F. Milletari, N. Navab, and S.-A. Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” in _Proc. Fourth International Conference on 3D Vision (3DV)_, 2016, pp. 565–571.
- [33] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, “nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation,” _Nature Methods_, vol. 18, no. 2, pp. 203–211, 2021.
- [34] A. Hatamizadeh et al., “UNETR: Transformers for 3D medical image segmentation,” in _Proc. IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_, 2022, pp. 1748–1758.
- [35] A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, “Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images,” in _Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2021)_, 2022, pp. 272–284.
- [36] W. Wang, C. Chen, M. Ding, H. Yu, S. Zha, and J. Li, “TransBTS: Multimodal brain tumor segmentation using transformer,” in _Medical Image Computing and Computer Assisted Intervention (MICCAI 2021)_, 2021, pp. 109–119.
- [37] J. Chen et al., “TransUNet: Rethinking the U-Net architecture design for medical image segmentation through the lens of transformers,” _Medical Image Analysis_, vol. 97, Art. no. 103280, 2024.
- [38] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, “Training data-efficient image transformers and distillation through attention,” in _Proc. 38th International Conference on Machine Learning (ICML)_, 2021, pp. 10347–10357.
- [39] A. Fateh et al., “BRISC: Annotated dataset for brain tumor segmentation and classification,” _Scientific Data_, vol. 13, Art. no. 361, 2026.
- [40] S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah, “Transformers in vision: A survey,” _ACM Computing Surveys_, vol. 54, no. 10s, pp. 1–41, 2022.
- [41] F. Shamshad et al., “Transformers in medical imaging: A survey,” _Medical Image Analysis_, vol. 88, Art. no. 102802, 2023.
- [42] R. Geirhos et al., “Shortcut learning in deep neural networks,” _Nature Machine Intelligence_, vol. 2, no. 11, pp. 665–673, 2020.

> Speaker notes: IEEE-style references, numbered in order of first citation in this deck; all 88 references cited by the two manuscripts are included.

## Slide 20

**Title:** References (2/2)

- [43] J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, “Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study,” _PLOS Medicine_, vol. 15, no. 11, Art. no. e1002683, 2018.
- [44] A. J. DeGrave, J. D. Janizek, and S.-I. Lee, “AI for radiographic COVID-19 detection selects shortcuts over signal,” _Nature Machine Intelligence_, vol. 3, no. 7, pp. 610–619, 2021.
- [45] J. M. J. Valanarasu, P. Oza, I. Hacihaliloglu, and V. M. Patel, “Medical transformer: Gated axial-attention for medical image segmentation,” in _Medical Image Computing and Computer Assisted Intervention (MICCAI 2021)_, 2021, pp. 36–46.
- [46] H. Cao et al., “Swin-Unet: Unet-like pure transformer for medical image segmentation,” in _Computer Vision – ECCV 2022 Workshops_, 2023, pp. 205–218.
- [47] S. Suzuki and K. Abe, “Topological structural analysis of digitized binary images by border following,” _Computer Vision, Graphics, and Image Processing_, vol. 30, no. 1, pp. 32–46, 1985.
- [48] V. Monga and B. L. Evans, “Perceptual image hashing via feature points: Performance evaluation and tradeoffs,” _IEEE Transactions on Image Processing_, vol. 15, no. 11, pp. 3452–3465, 2006.
- [49] R. W. Hamming, “Error detecting and error correcting codes,” _Bell System Technical Journal_, vol. 29, no. 2, pp. 147–160, 1950.
- [50] B. Zitová and J. Flusser, “Image registration methods: A survey,” _Image and Vision Computing_, vol. 21, no. 11, pp. 977–1000, 2003.
- [51] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” _Journal of Machine Learning Research_, vol. 12, no. 85, pp. 2825–2830, 2011.
- [52] M. Stone, “Cross-validatory choice and assessment of statistical predictions,” _Journal of the Royal Statistical Society: Series B (Methodological)_, vol. 36, no. 2, pp. 111–133, 1974.
- [53] A. Hamada, “Br35H :: Brain tumor detection 2020,” IEEE DataPort, 2025.
- [54] N. Chakrabarty, “Brain MRI images for brain tumor detection,” Kaggle dataset, version 1, 2019.
- [55] S. Bhuvaji, A. Kadam, P. Bhumkar, S. Dedge, and S. Kanchan, “Brain tumor classification (MRI),” Kaggle dataset, 2020.
- [56] J. Cheng, “Brain tumor dataset,” figshare, version 5, 2017.
- [57] MoundirB, “Jun Cheng brain tumor MRI dataset JPG,” Kaggle dataset, version 1, 2025.
- [58] M. Tan and Q. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in _Proc. 36th International Conference on Machine Learning (ICML)_, 2019, pp. 6105–6114.
- [59] A. Howard et al., “Searching for MobileNetV3,” in _Proc. IEEE/CVF International Conference on Computer Vision (ICCV)_, 2019, pp. 1314–1324.
- [60] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in _Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, 2018, pp. 4510–4520.
- [61] A. Vaswani et al., “Attention is all you need,” in _Advances in Neural Information Processing Systems (NeurIPS) 30_, 2017.
- [62] Z. Dai, H. Liu, Q. V. Le, and M. Tan, “CoAtNet: Marrying convolution and attention for all data sizes,” in _Advances in Neural Information Processing Systems (NeurIPS)_, 2021, pp. 3965–3977.
- [63] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” _Proceedings of the IEEE_, vol. 86, no. 11, pp. 2278–2324, 1998.
- [64] Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” _Nature_, vol. 521, no. 7553, pp. 436–444, 2015.
- [65] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in _Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, 2009, pp. 248–255.
- [66] O. Russakovsky et al., “ImageNet large scale visual recognition challenge,” _International Journal of Computer Vision_, vol. 115, no. 3, pp. 211–252, 2015.
- [67] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in _Advances in Neural Information Processing Systems (NeurIPS) 32_, 2019.
- [68] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in _Proc. International Conference on Learning Representations (ICLR)_, 2019.
- [69] I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in _Proc. International Conference on Learning Representations (ICLR)_, 2017.
- [70] P. Micikevicius et al., “Mixed precision training,” in _Proc. International Conference on Learning Representations (ICLR)_, 2018.
- [71] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training recurrent neural networks,” in _Proc. 30th International Conference on Machine Learning (ICML)_, 2013, pp. 1310–1318.
- [72] T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” _Journal of the American Statistical Association_, vol. 102, no. 477, pp. 359–378, 2007.
- [73] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in _Proc. 34th International Conference on Machine Learning (ICML)_, 2017, pp. 1321–1330.
- [74] W. J. Youden, “Index for rating diagnostic tests,” _Cancer_, vol. 3, no. 1, pp. 32–35, 1950.
- [75] J. A. Hanley and B. J. McNeil, “The meaning and use of the area under a receiver operating characteristic (ROC) curve,” _Radiology_, vol. 143, no. 1, pp. 29–36, 1982.
- [76] T. Fawcett, “An introduction to ROC analysis,” _Pattern Recognition Letters_, vol. 27, no. 8, pp. 861–874, 2006.
- [77] B. W. Matthews, “Comparison of the predicted and observed secondary structure of T4 phage lysozyme,” _Biochimica et Biophysica Acta (BBA) – Protein Structure_, vol. 405, no. 2, pp. 442–451, 1975.
- [78] D. Chicco and G. Jurman, “The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation,” _BMC Genomics_, vol. 21, Art. no. 6, 2020.
- [79] M. Pakdaman Naeini, G. Cooper, and M. Hauskrecht, “Obtaining well calibrated probabilities using Bayesian binning,” in _Proc. AAAI Conference on Artificial Intelligence 29(1)_, 2015.
- [80] B. Efron, “Bootstrap methods: Another look at the jackknife,” _The Annals of Statistics_, vol. 7, no. 1, pp. 1–26, 1979.
- [81] Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” _Psychometrika_, vol. 12, no. 2, pp. 153–157, 1947.
- [82] S. Holm, “A simple sequentially rejective multiple test procedure,” _Scandinavian Journal of Statistics_, vol. 6, no. 2, pp. 65–70, 1979.
- [83] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in _Proc. IEEE International Conference on Computer Vision (ICCV)_, 2017, pp. 618–626.
- [84] S. Abnar and W. Zuidema, “Quantifying attention flow in transformers,” in _Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)_, 2020, pp. 4190–4197.
- [85] M. D. Zeiler and R. Fergus, “Visualizing and understanding convolutional networks,” in _Computer Vision – ECCV 2014_, 2014, pp. 818–833.
- [86] N. Otsu, “A threshold selection method from gray-level histograms,” _IEEE Transactions on Systems, Man, and Cybernetics_, vol. 9, no. 1, pp. 62–66, 1979.
- [87] B. H. Menze et al., “The multimodal brain tumor image segmentation benchmark (BRATS),” _IEEE Transactions on Medical Imaging_, vol. 34, no. 10, pp. 1993–2024, 2015.
- [88] S. Bakas et al., “Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the BRATS challenge,” arXiv:1811.02629, 2018.

> Speaker notes: IEEE-style references, numbered in order of first citation in this deck; all 88 references cited by the two manuscripts are included.
