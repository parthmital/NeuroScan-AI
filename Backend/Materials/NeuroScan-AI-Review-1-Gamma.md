# NeuroScan AI

## AI powered brain tumour detection, classification and segmentation platform

**B.Tech | BCSE497J Project I**

**Review #1**

**Team members:** Parth Mital, Ved Lawand

**Faculty guide:** Biji C. L.

**School of Computer Science and Engineering, Vellore Institute of Technology**

**Date:** 21 July 2026

---

# Aim

To design and implement a full stack medical imaging platform that assists radiologists in analysing brain MRI scans through automated tumour detection, tumour type classification, and volumetric tumour segmentation.

## Project focus

- Detect whether a brain MRI scan shows tumour presence.
- Classify tumour positive scans into glioma, meningioma, or pituitary tumour.
- Segment tumour regions from multimodal MRI volumes into Whole Tumour, Tumour Core, and Enhancing Tumour.
- Provide a secure web interface for MRI upload, scan history, visual review, and report generation.
- Select deployment suitable models through fair comparison of CNN, transformer, and hybrid architectures.

---

# Abstract

Brain tumour diagnosis from MRI is clinically important, but manual interpretation is time consuming, specialist dependent, and variable across readers. NeuroScan AI addresses this by combining research grade deep learning models with a usable web based workflow for MRI analysis.

The project follows a three stage pipeline. First, a binary detector identifies tumour presence. Second, a multiclass classifier predicts glioma, meningioma, or pituitary tumour for positive cases. Third, a 3D segmentation model estimates tumour sub regions and volumetric measures when all required MRI modalities are available.

The research component benchmarked multiple architectures under standardised preprocessing and training conditions. EfficientNetB0 was selected for practical detection and classification deployment because it gives strong accuracy with only 5.3 million parameters. VNet was selected for segmentation because it provides the best efficiency profile among the evaluated 3D models, with 1.4 million parameters and strong Dice performance.

The implementation combines a FastAPI backend, PyTorch inference, SQLModel persistence, JWT based authentication, and a React TypeScript frontend with scan library, upload, viewer, reports, and settings workflows.

---

# Literature Review

## Detection and classification studies

| Paper                                         | Objective                                           | Methodology                                                   | Findings                                                                                            |
| --------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Zacharaki et al., 2009                        | Classify brain tumour type and grade from MRI       | Texture and shape features with SVM                           | Early ML was feasible but depended heavily on handcrafted features and small datasets               |
| Cheng et al., 2015                            | Improve tumour classification performance           | Tumour region augmentation and partitioning                   | Region focused preprocessing improved classification, but feature engineering remained a limitation |
| Szegedy et al., 2016                          | Improve image classification using multi scale CNNs | Inception modules with parallel convolution paths             | Multi scale feature extraction is useful for varied tumour size and appearance                      |
| Tan and Le, 2019                              | Build accurate and efficient CNNs                   | EfficientNet compound scaling of depth, width, and resolution | High accuracy can be achieved with fewer parameters, supporting resource constrained deployment     |
| Dosovitskiy et al., 2021 and Liu et al., 2021 | Apply transformers to vision tasks                  | ViT patch attention and Swin shifted window attention         | Transformers capture global context but need larger data and careful optimisation                   |

---

# Literature Review (Contd.)

## Segmentation and clinical imaging studies

| Paper                                          | Objective                                              | Methodology                                            | Findings                                                                                          |
| ---------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Ronneberger et al., 2015                       | Segment biomedical images accurately with limited data | UNet encoder decoder with skip connections             | Skip connections preserve localisation detail and remain a strong medical segmentation baseline   |
| Cicek et al., 2016                             | Extend UNet to volumetric medical data                 | 3D convolutional UNet                                  | 3D context improves volumetric segmentation compared with isolated 2D slices                      |
| Milletari et al., 2016                         | Improve volumetric segmentation efficiency             | VNet with residual 3D blocks and Dice focused learning | Residual volumetric designs can produce strong performance with fewer parameters                  |
| Isensee et al., 2021                           | Reduce manual architecture tuning                      | nnU-Net self configuring pipeline                      | Preprocessing, patch size, and training configuration can matter as much as model novelty         |
| Hatamizadeh et al., 2022 and Wang et al., 2021 | Use transformers for 3D medical segmentation           | Swin UNETR and TransBTS hybrid designs                 | Global context is promising, but compute cost and data requirements remain deployment constraints |

---

# Research Gap

Most existing work proves isolated model performance, but the clinical workflow needs an integrated and validated pipeline from upload to visual decision support.

## Gaps identified

- Many studies evaluate only one task, such as detection or classification, instead of connecting detection, subtype classification, and segmentation.
- Cross paper comparison is weak because datasets, preprocessing, splits, metrics, and training budgets differ.
- Public MRI datasets can contain duplicates or near duplicates, which may inflate results if deduplication is not performed before splitting.
- Transformer models are often discussed for global context, but their cost, convergence behaviour, and data needs are not always compared fairly against CNN baselines.
- Clinical usability is under addressed: secure access, scan history, visual overlays, report export, and human review are often outside the model benchmark.

## Project response

NeuroScan AI combines controlled architecture comparison with a working full stack prototype, so model selection is tied to actual deployment constraints rather than accuracy alone.

---

# Objectives

## Research objectives

- Curate and preprocess brain MRI datasets for binary detection, multiclass classification, and 3D segmentation.
- Apply deduplication where required to reduce leakage risk before training and evaluation.
- Benchmark CNN, transformer, and hybrid architectures under comparable experimental settings.
- Select practical models using both accuracy and efficiency, not accuracy alone.

## System objectives

- Build a secure FastAPI inference service for MRI upload, scan storage, model execution, slice retrieval, segmentation retrieval, and file download.
- Build a React TypeScript user interface for login, dashboard, upload, scan library, scan detail, reports, and settings.
- Provide clinically useful outputs: tumour presence, tumour type, confidence, WT volume, TC volume, ET volume, and visual segmentation masks.
- Keep the system assistive, with radiologist review and future external validation as mandatory requirements before clinical use.

## Review milestone

The project has moved from literature identification and model benchmarking to a functioning prototype architecture that can be demonstrated and hardened.

---

# Proposed System and Current Implementation

## Workflow

MRI files are uploaded through the frontend and sent to the backend as authenticated requests. The backend stores the scan, identifies available modalities from filenames, extracts a representative 2D slice, runs tumour detection, runs tumour classification when tumour probability is positive, and runs 3D VNet segmentation when FLAIR, T1, T1CE, and T2 are all available.

## Implemented stack

| Layer              | Implementation                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| Frontend           | React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, React Query, VTK.js                            |
| Backend            | FastAPI, Uvicorn, SQLModel, SQLite, JWT auth, bcrypt password hashing                               |
| AI inference       | PyTorch, TorchVision EfficientNetB0, custom VNet3D, OpenCV, NiBabel                                 |
| User workflows     | Register, login, upload MRI, view scan library, inspect slices, view segmentation, generate reports |
| Deployment support | Dockerfiles, Docker Compose, environment based API configuration                                    |

## Model choices for prototype

- Detection: EfficientNetB0, selected for high binary detection accuracy and low parameter count.
- Classification: EfficientNetB0, selected for strong multiclass performance with efficient deployment.
- Segmentation: VNet3D, selected for efficient 3D tumour volume estimation and fast training behaviour.

---

# References

## Detection and classification references

1. E. I. Zacharaki, S. Wang, S. Chawla, et al., "Classification of brain tumour type and grade using MRI texture and shape in a machine learning scheme," Magnetic Resonance in Medicine, 2009.
2. J. Cheng, W. Huang, S. Cao, et al., "Enhanced performance of brain tumour classification via tumour region augmentation and partition," PLoS ONE, 2015.
3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," CVPR, 2016.
4. C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the inception architecture for computer vision," CVPR, 2016.
5. M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," ICML, 2019.
6. A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al., "An image is worth 16x16 words: Transformers for image recognition at scale," ICLR, 2021.
7. Z. Liu, Y. Lin, Y. Cao, et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," ICCV, 2021.
8. W. Xu, Y. Xu, T. Chang, and Z. Tu, "Co-scale conv-attentional image transformers," ICCV, 2021.

---

# References (Contd.)

## Segmentation, dataset, and project references

1. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," MICCAI, 2015.
2. B. H. Menze et al., "The multimodal brain tumour image segmentation benchmark," IEEE Transactions on Medical Imaging, 2015.
3. O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, "3D U-Net: Learning dense volumetric segmentation from sparse annotation," MICCAI, 2016.
4. F. Milletari, N. Navab, and S. A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," 3DV, 2016.
5. F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnU-Net: A self-configuring method for deep learning based biomedical segmentation," Nature Methods, 2021.
6. A. Hatamizadeh et al., "UNETR: Transformers for 3D medical image segmentation," WACV, 2022.
7. A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, A. Myronenko, H. R. Roth, and D. Xu, "Swin UNETR: Swin transformers for semantic segmentation of brain tumours in MRI images," MICCAI, 2022.
8. W. Wang, C. Chen, M. Ding, H. Yu, Z. Sha, and L. Li, "TransBTS: Multimodal brain tumour segmentation using transformer," MICCAI, 2021.
9. NeuroScan AI repository source material: README, backend API implementation, processor implementation, frontend routes, and Overleaf research reports.
