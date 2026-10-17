# Presentation Content

Content for `Presentation Template.pptx`, slide by slide, ready to paste over the template's placeholder text. Each slide gives its title, then its body. Lines starting with `>` are notes for whoever builds the slides (images, footer); they are not slide text. Citations [n] refer to the reference list on slide 14.

> Footer on slides 2 to 14: change the template date "05/10/26" to the review date, 28/10/26; keep "SCOPE" and the slide number.

## Slide 1

> Layout: Title Slide. Keep the VIT logo at the top.

**Title:** B.Tech BCSE497J – Project-I: NeuroScan AI: AI-Powered Brain Tumour Detection, Classification and Segmentation Platform

Team members:

- Parth Mital (23BCE0842)
- Ved Lawand (23BDS0141)

Faculty guide:

- Dr. Biji C. L., Associate Professor Grade 2, SCOPE

## Slide 2

**Title:** Approval Mail From Guide

> The guide's approval mail for Review 3 has not been received yet. When it arrives, paste a screenshot of it here in place of the template's sample image, and keep the line below.

- All the above aspects are addressed and updated by the project team.

## Slide 3

**Title:** Aim

- To build and evaluate an AI system that detects brain tumours in MRI, classifies them as glioma, meningioma or pituitary tumour, and segments the whole tumour, tumour core and enhancing tumour in 3D.
- To choose its models through a fair, leakage-controlled benchmark of CNN and transformer architectures.
- To deploy the most efficient models in a secure web application with a multi-planar and 3D viewer and PDF and CSV reports.

## Slide 4

**Title:** Abstract

- Published brain tumour MRI results are hard to compare: studies differ in data, splits and evaluation, and near-duplicate images leak between training and test sets.
- 2D benchmark: 9 ImageNet-pretrained backbones for tumour detection and subtype classification under one protocol. Perceptual-hash deduplication, a group-aware split, 5-fold cross-validation, McNemar tests with Holm correction, and shortcut audits.
- Detection: CoAtNet-0, EfficientNetB0 and ViT-B/16 reached 99.42% on 171 held-out images (95% CI 98.25% to 100%); no pair of models differed significantly.
- 3D benchmark: 6 networks on BraTS 2020 with an equal budget. UNet had the best mean Dice (0.8199); VNet reached 0.7965 with only 1.40 M parameters.
- Deployment: EfficientNetB0 and VNet in a FastAPI and React application with secure sessions, a linked axial, coronal, sagittal and 3D viewer, and PDF and CSV reports. 100 automated tests pass.

## Slide 5

**Title:** Literature Review

| Paper                                                    | Objective                                     | Methodology                                                              | Pros/Cons                                                            | Findings                                    |
| -------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------- |
| Cheng et al., 2015 [1]                                   | Classify glioma, meningioma, pituitary        | Intensity, texture and shape features with SVM on 3,064 T1CE slices      | Pro: interpretable features. Con: hand-crafted, poor generalisation  | 90.0% accuracy                              |
| Deepak and Ameer, 2019 [2]                               | Subtype classification                        | Transfer learning with InceptionV3 features                              | Pro: no manual features. Con: single model, single split             | 97.5% accuracy                              |
| Tan and Le, 2019 [3]                                     | Efficient CNN scaling                         | EfficientNet compound scaling of depth, width and resolution             | Pro: high accuracy with few parameters. Con: tuned on natural images | EfficientNetB0 baseline, 0.77 GFLOPs        |
| Dosovitskiy et al., 2021 [4]; Liu et al., 2021 [5]       | Transformers for images                       | ViT on 16 x 16 patches; Swin with shifted windows                        | Pro: global context. Con: data-hungry, costly                        | Match CNNs with large-scale pretraining     |
| Milletari et al., 2016 [6]; Isensee et al., 2021 [7]     | 3D medical segmentation                       | VNet residual 3D CNN; self-configuring nnU-Net                           | Pro: strong on small datasets. Con: nnU-Net is compute-heavy         | CNN baselines remain strong                 |
| Hatamizadeh et al., 2022 [8]; Wang et al., 2021 [9]      | Transformers for 3D brain tumour segmentation | UNETR, Swin UNETR and TransBTS encoders                                  | Pro: long-range modelling. Con: many parameters, need large data     | Mixed gains over CNNs                       |
| Kapoor and Narayanan, 2023 [10]; Fateh et al., 2026 [11] | Reliability of ML benchmarks                  | Leakage audit across fields; deduplication of public brain MRI aggregate | Pro: exposes inflated results. Con: no architecture comparison       | Duplicates must be removed before splitting |

> The template shows this table as a picture. Replace the picture with a real table, or render this table and paste it as an image.

## Slide 6

**Title:** Research Gap

- No controlled comparison: most studies test 2 or 3 models, each with its own data, splits and training budget.
- Data leakage: combined public collections contain exact and near-duplicate images that are rarely removed before splitting.
- Single-run results: no confidence intervals, significance tests, per-class metrics or calibration.
- Cost and shortcuts ignored: parameters, latency and reliance on image borders are rarely measured.
- No path to use: models are not turned into tools that accept MRI volumes, show results and protect patient data.

## Slide 7

**Title:** Objectives

- Benchmark 9 2D backbones for detection and subtype classification with identical splits and an equal number of optimiser steps.
- Control leakage with label-aware perceptual-hash deduplication and a group-aware split, then audit it.
- Evaluate with 5-fold cross-validation, fold ensembles, bootstrap confidence intervals, Holm-corrected McNemar tests, calibration, efficiency and shortcut audits.
- Implement 6 3D segmentation networks from scratch and compare them on BraTS 2020 per case, by tumour grade and by cost.
- Deploy the most efficient models in a secure web application with a multi-planar and 3D viewer and reports.

Identified: Sustainable Development Goals | SDG 3: Good Health and Well-being; SDG 9: Industry, Innovation and Infrastructure

Outcomes: Journal (Scopus): two manuscripts prepared for _Engineering Applications of Artificial Intelligence_ (Elsevier); Product: open-source web application at TRL 4 (technology validated in lab)

> Keep the SDG and TRL images. Highlight SDG 3 and SDG 9 on the SDG chart and level 4 on the TRL chart.

## Slide 8

**Title:** Framework/Architecture/Block Diagram

- Browser (React, vtk.js viewer) talks to the backend over the same origin through a reverse proxy, with an HttpOnly session cookie.
- FastAPI backend: authentication, scan upload and validation, slices, reports.
- PipelineRunner: one background worker processes each scan.
- BrainProcessor runs three models:
  - Detection: EfficientNet-B0 on the middle axial slice (brain crop, 224 x 224).
  - Classification: EfficientNet-B0 into glioma, meningioma or pituitary.
  - Segmentation: VNet on FLAIR, T1, T1CE and T2 at 128³, giving WT, TC and ET.
- Storage: SQLite (users, scans) with Alembic migrations; an uploads folder per scan for the volumes and segmentation mask.
- Research notebooks on Kaggle (2 x Tesla T4) train and export the three checkpoints.

> Insert Figure 4.1 of the report (system architecture, rendered from its Mermaid code) on this slide.

## Slide 9

**Title:** Functional Requirements

- Register, sign in and sign out; change password; optional two-factor authentication (TOTP).
- Upload one to four NIfTI volumes (FLAIR, T1, T1CE, T2) per case; modality detected from the file name.
- Process the scan in the background with live status, step and progress, and retry on failure.
- Detect a tumour, classify its type with class probabilities, and segment WT, TC and ET with their volumes.
- View the case in linked axial, coronal, sagittal and 3D panes with the overlay, contrast, zoom and ruler.
- Search, rename and delete scans; dashboard with counts, recent scans and a volume chart.
- PDF and CSV report per scan; export of all scans as CSV.
- Operator account recovery from the command line.

## Slide 10

**Title:** Modules

- **Data curation:** brain crop, perceptual-hash deduplication (Hamming ≤ 5), label-noise exclusion, group-aware split and leakage audit.
- **2D benchmark:** 9 backbones, 5 folds plus a background-only run (54 runs per task), temperature scaling, fold ensembles, statistical tests, Grad-CAM and shortcut tests.
- **3D benchmark:** UNet, nnU-Net-style, VNet, UNETR, Swin UNETR and TransBTS; 3,250 optimiser steps each; 8-flip TTA; Dice, HD95 and Wilcoxon tests.
- **Inference engine:** BrainProcessor with safe weight loading, the same crop as training, and the VNet label mask saved in FLAIR space.
- **Backend and pipeline:** FastAPI (23 routes), single background worker, SQLite in WAL mode, Alembic migrations.
- **Security:** bcrypt, revocable JWT cookie sessions, CSRF header, lockout and rate limits, TOTP, per-user isolation, validated uploads.
- **Frontend viewer:** React with vtk.js; linked multi-planar and 3D views; slab modes; measurement tools.
- **Reports:** PDF MRI brain report with key images and sign-off block; CSV exports with formula escaping.

## Slide 11

**Title:** Experiments and Results

| Task           | Data                                              | Best result                                                                                           | Deployed model                                    |
| -------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Detection      | 3,253 images, 1,185 after deduplication; 171 test | 99.42% (CoAtNet-0, EfficientNetB0, ViT-B/16); no significant differences                              | EfficientNetB0, 4.01 M parameters, 1,393 images/s |
| Classification | 5,828 images, 4,730 after deduplication; 45 test  | 6 models 100%, 3 models 97.78%; test set too small to rank                                            | EfficientNetB0                                    |
| Segmentation   | BraTS 2020 [12], 369 cases (257/56/56)            | Mean Dice: UNet 0.8199, nnU-Net 0.8192, VNet 0.7965, Swin UNETR 0.7462, TransBTS 0.7291, UNETR 0.7056 | VNet, 1.40 M parameters, 0.0375 s per volume      |

- Background-only models still beat chance (detection 7 of 9, classification 9 of 9 backbones): public collections carry label information outside the brain.
- Low-grade gliomas: enhancing-tumour Dice drops to 0.16 to 0.42 for every 3D model.
- Application: one BraTS case processed in about 4 s on a CPU; 61 backend and 39 frontend tests pass.

> Optional images: `Materials/Notebooks/segmentation/plots/10_dice_vs_latency.png` and `Materials/Screenshots/scan-viewer.png`.

## Slide 12

**Title:** Conclusion and Future Work

- Under one fair protocol, all 2D backbones reached 97% to 99% detection accuracy with no significant differences, so cost decides: EfficientNetB0 is accurate and fast.
- In 3D, CNNs beat transformers trained from scratch; VNet gives 97% of UNet's Dice at a third of its latency.
- Deduplication removed 63.57% (detection) and 18.84% (classification) of the images; the shortcut test shows the limits of public data.
- NeuroScan AI turns these models into a secure, tested application with a multi-planar and 3D viewer and reports.
- Future work:
  - Patient-level splits and external validation.
  - 3D multi-sequence detection and classification.
  - Several seeds and pretrained transformers.
  - Calibration and physical volumes in the app.
  - DICOM support.

## Slide 13

**Title:** Outcome Achieved

- Journal Publication: two manuscripts prepared for _Engineering Applications of Artificial Intelligence_ (Elsevier, Scopus Indexed):
  - "Standardised Comparison of Convolutional and Transformer Backbones for Brain Tumour Magnetic Resonance Image Classification under Perceptual-Hash Deduplication"
  - "A Comprehensive Benchmark of Three-Dimensional Segmentation Architectures for Brain Tumour Segmentation"
- Others: open-source NeuroScan AI web application, research notebooks with all outputs, and trained checkpoints at https://github.com/parthmital/NeuroScan-AI

> The template's line about uploading proof by 27 Oct 2026 (https://forms.gle/B9B71VPy5b9bZ9dJ8) is an instruction to the team, not slide content. Upload the proof there and update the status above (for example "Submitted" or "Under review") before the review.

## Slide 14

**Title:** References

- [1] J. Cheng _et al._, "Enhanced performance of brain tumor classification via tumor region augmentation and partition," _PLoS ONE_, vol. 10, no. 10, p. e0140381, 2015.
- [2] S. Deepak and P. M. Ameer, "Brain tumor classification using deep CNN features via transfer learning," _Computers in Biology and Medicine_, vol. 111, p. 103345, 2019.
- [3] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in _Proc. ICML_, 2019, pp. 6105–6114.
- [4] A. Dosovitskiy _et al._, "An image is worth 16x16 words: Transformers for image recognition at scale," in _Proc. ICLR_, 2021.
- [5] Z. Liu _et al._, "Swin transformer: Hierarchical vision transformer using shifted windows," in _Proc. ICCV_, 2021, pp. 9992–10002.
- [6] F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in _Proc. 3DV_, 2016, pp. 565–571.
- [7] F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation," _Nature Methods_, vol. 18, no. 2, pp. 203–211, 2021.
- [8] A. Hatamizadeh _et al._, "UNETR: Transformers for 3D medical image segmentation," in _Proc. WACV_, 2022, pp. 1748–1758.
- [9] W. Wang, C. Chen, M. Ding, H. Yu, S. Zha, and J. Li, "TransBTS: Multimodal brain tumor segmentation using transformer," in _Proc. MICCAI_, 2021, pp. 109–119.
- [10] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," _Patterns_, vol. 4, no. 9, p. 100804, 2023.
- [11] A. Fateh _et al._, "BRISC: Annotated dataset for brain tumor segmentation and classification," _Scientific Data_, vol. 13, p. 361, 2026.
- [12] B. H. Menze _et al._, "The multimodal brain tumor image segmentation benchmark (BRATS)," _IEEE Transactions on Medical Imaging_, vol. 34, no. 10, pp. 1993–2024, 2015.
