# Brain MRI Binary Tumour Detection: Deep Learning Benchmark and Comparative Study

A rigorous, empirical comparative evaluation of 9 deep learning architectures spanning classical convolutional neural networks, modern lightweight CNNs, vision transformers, and hybrid conv-attention networks for binary brain tumour detection from Magnetic Resonance Imaging (MRI) scans.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Abstract and Executive Summary](#abstract-and-executive-summary)
- [Dataset Architecture and Preprocessing](#dataset-architecture-and-preprocessing)
  - [Data Provenance and Cohort Summary](#data-provenance-and-cohort-summary)
  - [Image Dimension Distribution](#image-dimension-distribution)
  - [Stratified Partitioning](#stratified-partitioning)
  - [Data Augmentation and Pipeline Normalisation](#data-augmentation-and-pipeline-normalisation)
- [Evaluated Model Architectures and Profiles](#evaluated-model-architectures-and-profiles)
  - [Architectural Taxonomy](#architectural-taxonomy)
  - [Parameter and Computational Complexity Profile](#parameter-and-computational-complexity-profile)
  - [Network Adaptation and Layer Targets](#network-adaptation-and-layer-targets)
- [Standardised Training Protocol](#standardised-training-protocol)
  - [Optimisation and Hyperparameter Setup](#optimisation-and-hyperparameter-setup)
  - [Hardware and Distributed Acceleration](#hardware-and-distributed-acceleration)
- [Training Dynamics and Convergence Analysis](#training-dynamics-and-convergence-analysis)
  - [Epoch Progression and Throughput](#epoch-progression-and-throughput)
  - [Training and Validation Loss Trajectories](#training-and-validation-loss-trajectories)
- [Comprehensive Experimental Evaluation](#comprehensive-experimental-evaluation)
  - [Validation Set Metrics](#validation-set-metrics)
  - [Test Set Metrics](#test-set-metrics)
  - [Confusion Matrices and Diagnostic Accuracy](#confusion-matrices-and-diagnostic-accuracy)
  - [ROC and Precision-Recall Dynamics](#roc-and-precision-recall-dynamics)
  - [Probability Calibration and Confidence Reliability](#probability-calibration-and-confidence-reliability)
  - [Per-Class Performance Profiling](#per-class-performance-profiling)
  - [Representative Qualitative Predictions](#representative-qualitative-predictions)
- [Visual Explainability: Grad-CAM Interpretability](#visual-explainability-grad-cam-interpretability)
  - [Interpretability Methodology across CNNs and Transformers](#interpretability-methodology-across-cnns-and-transformers)
  - [Attribution on Correct Classifications](#attribution-on-correct-classifications)
  - [Attribution on Misclassifications and Failure Analysis](#attribution-on-misclassifications-and-failure-analysis)
- [Inference Latency and Throughput Benchmarks](#inference-latency-and-throughput-benchmarks)
  - [Benchmarking Setup on Tesla T4](#benchmarking-setup-on-tesla-t4)
  - [Latency and Throughput Profile](#latency-and-throughput-profile)
  - [Efficiency vs Accuracy Trade-off Matrix](#efficiency-vs-accuracy-trade-off-matrix)
- [Comparative Synthesis and Deployment Recommendations](#comparative-synthesis-and-deployment-recommendations)
  - [Top-Tier Production Candidates](#top-tier-production-candidates)
  - [Architectural Insights and Inductive Bias](#architectural-insights-and-inductive-bias)
  - [Failure Modes and Threshold Sensitivity](#failure-modes-and-threshold-sensitivity)
- [Directory Layout and Generated Artefacts](#directory-layout-and-generated-artefacts)
- [Quick Start and Execution Guide](#quick-start-and-execution-guide)
  - [Prerequisites](#prerequisites)
  - [Executing Individual Notebooks](#executing-individual-notebooks)
  - [Loading Pretrained Checkpoints for Inference](#loading-pretrained-checkpoints-for-inference)
- [Troubleshooting and Operational Notes](#troubleshooting-and-operational-notes)

---

## Project Overview

This subproject provides a standardised, end-to-end medical deep learning benchmarking pipeline designed to evaluate 9 neural network architectures on binary classification of brain MRI scans into `No Tumor` (Class 0) versus `Tumor` (Class 1).

The 9 evaluated architectures represent 3 distinct structural families:

1. **Classical Convolutional Networks:** VGG-16, ResNet-50, Inception-V3, DenseNet-121
2. **Lightweight and Efficient CNNs:** EfficientNet-B0, MobileNetV3-Large
3. **Vision Transformers and Hybrid Models:** ViT-B/16, Swin-Base, CoaTNet-0

Each model is evaluated under an identical experimental protocol, using the same dataset splits, image transformations, mixed-precision routines, evaluation metrics, calibration calculations, inference speed profiling on NVIDIA Tesla T4 GPUs, and gradient-weighted class activation mapping (Grad-CAM) interpretability visualisations.

---

## Abstract and Executive Summary

Automated detection of brain tumours from Magnetic Resonance Imaging (MRI) demands high sensitivity to prevent false negatives while maintaining high specificity to prevent unnecessary clinical interventions. In this study, 9 distinct deep learning models were pretrained on ImageNet and fine-tuned for 10 epochs on a curated dataset of 3,253 brain MRI scans using a 70/15/15 stratified train/validation/test protocol.

### Key Benchmark Findings

- **Top Test Accuracy:** **DenseNet-121**, **ResNet-50**, **CoaTNet-0**, and **EfficientNet-B0** all achieved **99.39% test accuracy** (485 correct out of 488 test images).
- **Zero False Negatives (100.00% Sensitivity):** **CoaTNet-0** and **EfficientNet-B0** achieved 100.00% test recall (248/248 tumours detected, 0 missed tumours), making them suitable for clinical triage scenarios.
- **Zero False Positives (100.00% Precision):** **DenseNet-121** and **ResNet-50** achieved 100.00% test precision (0 false alarms, 240/240 healthy scans correctly identified).
- **Discrimination Capability:** **DenseNet-121** and **ResNet-50** attained a test **ROC-AUC of 1.0000** and **PR-AUC of 1.0000**, followed by **CoaTNet-0** (0.9999) and **EfficientNet-B0** (0.9998).
- **Best Parameter Efficiency:** **EfficientNet-B0** delivered **99.39% test accuracy** and 100.00% sensitivity with only **4.01M parameters** and **0.769 GFLOPs**, delivering **1,364.24 images/s** batched throughput.
- **Highest Inference Throughput:** **MobileNetV3-Large** achieved **2,408.52 images/s** batched throughput at 0.431 GFLOPs, but required threshold calibration to compensate for lower recall under the default 0.50 threshold.
- **Architectural Observations:** Vision Transformer (**ViT-B/16**, 94.88% test acc) and Hierarchical Transformer (**Swin-Base**, 95.90% test acc) exhibited lower sample efficiency compared to inductive-bias-rich CNNs and hybrid Conv-Attention networks (**CoaTNet-0**, 99.39% test acc) on this medium-scale dataset. **VGG-16** yielded the lowest test accuracy (88.32%) despite possessing the largest footprint (134.26M parameters, 30.93 GFLOPs).

---

## Dataset Architecture and Preprocessing

### Data Provenance and Cohort Summary

The experimental dataset is compiled from two established open-access medical imaging cohorts:

- `navoneel/brain-mri-images-for-brain-tumor-detection`
- `ahmedhamada0/brain-tumor-detection`

The consolidated dataset comprises **3,253 axial and coronal brain MRI scans** across 2 classes:

- **No Tumor (Negative Class, Label 0):** 1,598 scans (49.12%)
- **Tumor (Positive Class, Label 1):** 1,655 scans (50.88%)

![Class Distribution](binary-detection-resnet50/plots/01_class_distribution.png)

Representative raw MRI samples illustrate distinct tumour morphologies, varying slice orientations, contrast levels, and skull-stripping differences:

![Sample Images](binary-detection-resnet50/plots/02_sample_images.png)

### Image Dimension Distribution

Raw images exhibit significant variance in native resolutions across the consolidated cohorts:

| Class        | Metric            | Width (px)              | Height (px)             |
| :----------- | :---------------- | :---------------------- | :---------------------- |
| **No Tumor** | Mean +/- Std      | 302.63 +/- 154.89       | 308.88 +/- 144.90       |
|              | Median [IQR]      | 227.00 [214.00, 380.00] | 240.00 [225.00, 400.00] |
|              | Range (Min - Max) | 150 - 1920              | 168 - 1080              |
| **Tumor**    | Mean +/- Std      | 351.47 +/- 195.21       | 400.24 +/- 208.08       |
|              | Median [IQR]      | 283.00 [248.00, 353.00] | 342.00 [294.00, 380.00] |
|              | Range (Min - Max) | 175 - 1275              | 167 - 1427              |

![Image Properties](binary-detection-resnet50/plots/03_image_properties.png)

### Stratified Partitioning

To ensure reproducible evaluation and prevent target distribution skew, the 3,253 images are partitioned using a **70% Train, 15% Validation, 15% Test** stratified split fixed with `random_state=42`:

| Subset             | No Tumor (Class 0) | Tumor (Class 1) | Total Scans | Proportion |
| :----------------- | :----------------- | :-------------- | :---------- | :--------- |
| **Training Set**   | 1,119              | 1,158           | **2,277**   | 70.00%     |
| **Validation Set** | 239                | 249             | **488**     | 15.00%     |
| **Test Set**       | 240                | 248             | **488**     | 15.00%     |
| **Total Cohort**   | **1,598**          | **1,655**       | **3,253**   | 100.00%    |

![Split Distribution](binary-detection-resnet50/plots/05_split_distribution.png)

### Data Augmentation and Pipeline Normalisation

To regularise deep architectures against overfitting on medical MRI variations, training loaders apply stochastic data augmentations:

- **Spatial Resizing:** Resized to standardized input dimensions (224 x 224 pixels, except InceptionV3 auxiliary pathways adapted internally).
- **Random Horizontal Flip:** p = 0.50 (exploits bilateral cerebral hemisphere symmetry).
- **Random Affine Rotation:** Range [-10 deg, +10 deg] (accommodates patient head tilt in the MRI bore).
- **Colour Jitter:** Brightness jitter factor 0.20, contrast jitter factor 0.20 (simulates varying MRI sequence gains and scanner calibrations).
- **Tensor Normalisation:** Pixel channels standardised via standard ImageNet statistics:
  - Mean: [0.485, 0.456, 0.406]
  - Standard Deviation: [0.229, 0.224, 0.225]

Validation and test sets bypass stochastic augmentations, using deterministic resizing and normalisation only.

![Augmentation Before and After](binary-detection-resnet50/plots/06_augmentation_before_after.png)

---

## Evaluated Model Architectures and Profiles

### Architectural Taxonomy

The benchmark evaluates 9 models representing four major paradigms in computer vision:

```text
+-------------------------------------------------------------------------+
|                        Evaluated Architectures                          |
+--------------------+-------------------+--------------------------------+
|  Classical CNNs    |  Efficient CNNs   |  Transformers & Hybrids        |
+--------------------+-------------------+--------------------------------+
|  - VGG-16          |  - EfficientNet-B0|  - Vision Transformer (ViT-B16)|
|  - ResNet-50       |  - MobileNetV3-L  |  - Swin Transformer Base       |
|  - Inception-V3    |                   |  - CoaTNet-0 (Conv + Attn)     |
|  - DenseNet-121    |                   |                                |
+--------------------+-------------------+--------------------------------+
```

1. **VGG-16:** Deep sequential network with homogeneous 3x3 convolutional kernels and large fully connected output heads.
2. **ResNet-50:** 50-layer network with bottleneck residual skip connections to eliminate vanishing gradients.
3. **Inception-V3:** Multi-scale feature extraction network with factorised convolutions and parallel asymmetric receptive fields.
4. **DenseNet-121:** Densely connected convolutional network where every layer receives concatenated feature maps from all preceding layers.
5. **EfficientNet-B0:** Compound-scaled mobile architecture employing MBConv (Inverted Residuals) and Squeeze-and-Excitation blocks.
6. **MobileNetV3-Large:** Mobile architecture optimised with neural architecture search (NAS), Hard-Swish activations, and depthwise separable convolutions.
7. **ViT-B/16:** Pure vision transformer operating on non-overlapping 16x16 image patches with global multi-head self-attention.
8. **Swin-Base:** Hierarchical vision transformer using shifted local windows to limit self-attention complexity to linear runtime.
9. **CoaTNet-0:** Hybrid conv-attention network combining depthwise convolutions with self-attention across multi-stage feature pyramids.

### Parameter and Computational Complexity Profile

All models were profiled on input tensors of dimension 3x224x224. Complexity statistics were extracted directly from the compiled weights:

| Model Architecture    | Total Parameters | Trainable Parameters | Parameter Memory (FP32) | Serialised Checkpoint | FLOPs per Image | GFLOPs    |
| :-------------------- | :--------------- | :------------------- | :---------------------- | :-------------------- | :-------------- | :-------- |
| **MobileNetV3-Large** | **4,203,137**    | 4,203,137            | **16.03 MB**            | **16.27 MB**          | **431,211,840** | **0.431** |
| **EfficientNet-B0**   | **4,009,489**    | 4,009,489            | **15.29 MB**            | **15.59 MB**          | **769,173,088** | **0.769** |
| **DenseNet-121**      | 6,954,881        | 6,954,881            | 26.53 MB                | 26.83 MB              | 5,666,023,424   | 5.666     |
| **ResNet-50**         | 23,510,081       | 23,510,081           | 89.68 MB                | 89.99 MB              | 8,174,481,408   | 8.174     |
| **CoaTNet-0**         | 26,666,745       | 26,666,745           | 101.73 MB               | 102.05 MB             | 8,415,815,680   | 8.416     |
| **Inception-V3**      | 24,346,241       | 24,346,241           | 92.87 MB                | 93.30 MB              | 11,421,790,080  | 11.422    |
| **ViT-B/16**          | 85,799,425       | 85,799,425           | 327.30 MB               | 327.36 MB             | 22,540,715,520  | 22.541    |
| **Swin-Base**         | 86,744,249       | 86,744,249           | 330.90 MB               | 331.03 MB             | 30,860,199,424  | 30.860    |
| **VGG-16**            | 134,264,641      | 134,264,641          | 512.18 MB               | 512.19 MB             | 30,932,344,832  | 30.932    |

### Network Adaptation and Layer Targets

Each network replaces its original 1000-class ImageNet linear projection head with a single output logit for binary classification, trained with Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).

Target layers used for extracting activation feature maps and backpropagated gradients for Grad-CAM explainability are:

| Model                 | Library Source                                      | Adapted Head Definition                   | Grad-CAM Target Layer             |
| :-------------------- | :-------------------------------------------------- | :---------------------------------------- | :-------------------------------- |
| **CoaTNet-0**         | `timm.create_model("coatnet_0_rw_224")`             | `model.reset_classifier(num_classes=1)`   | `net.stages[-1]`                  |
| **DenseNet-121**      | `torchvision.models.densenet121`                    | `nn.Linear(classifier.in_features, 1)`    | `net.features[-1]`                |
| **EfficientNet-B0**   | `torchvision.models.efficientnet_b0`                | `nn.Linear(classifier[1].in_features, 1)` | `net.features[-1]`                |
| **Inception-V3**      | `torchvision.models.inception_v3`                   | `nn.Linear(fc.in_features, 1)`            | `net.Mixed_7c`                    |
| **MobileNetV3-Large** | `torchvision.models.mobilenet_v3_large`             | `nn.Linear(classifier[3].in_features, 1)` | `net.features[-1]`                |
| **ResNet-50**         | `torchvision.models.resnet50`                       | `nn.Linear(fc.in_features, 1)`            | `net.layer4[-1]`                  |
| **Swin-Base**         | `timm.create_model("swin_base_patch4_window7_224")` | `num_classes=1`                           | `net.layers[-1].blocks[-1].norm1` |
| **VGG-16**            | `torchvision.models.vgg16`                          | `nn.Linear(classifier[6].in_features, 1)` | `net.features[-1]`                |
| **ViT-B/16**          | `torchvision.models.vit_b_16`                       | `nn.Linear(heads.head.in_features, 1)`    | `net.encoder.layers[-1].ln_1`     |

---

## Standardised Training Protocol

### Optimisation and Hyperparameter Setup

To isolate architectural contributions from hyperparameter variability, all 9 models were trained using a uniform optimisation protocol:

- **Epochs:** 10 full epochs over the training split (2,277 images).
- **Optimizer:** `AdamW` (decay-rate decoupled adaptive moment estimation).
- **Initial Learning Rate:** 3e-4 (0.0003).
- **Learning Rate Scheduler:** `CosineAnnealingLR` with T_max = 10, decaying the learning rate monotonically towards 0.
- **Loss Criterion:** `torch.nn.BCEWithLogitsLoss()` combining a Sigmoid activation layer and standard binary cross-entropy into a numerically stable single operation.
- **Precision:** Automated Mixed Precision (`torch.amp.GradScaler`) utilizing FP16 Tensor Cores.
- **Data Loaders:** `num_workers=4`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`.
- **Batch Size:** Configured according to memory footprint constraints:
  - Batch Size 64: CoaTNet-0, Swin-Base
  - Batch Size 128: ViT-B/16
  - Batch Size 256: DenseNet-121, EfficientNet-B0, Inception-V3, MobileNetV3-Large, ResNet-50, VGG-16

### Hardware and Distributed Acceleration

All training and inference routines were executed in a high-performance environment:

- **GPUs:** 2 x NVIDIA Tesla T4 (16 GB GDDR6 VRAM each) accessed via `torch.nn.DataParallel`.
- **Acceleration Settings:** `torch.backends.cudnn.benchmark = True`, `torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`.
- **Environment:** Linux / Kaggle GPU Kernel (Python 3.10+, PyTorch 2.x, Torchvision, TIMM, Scikit-Learn).

---

## Training Dynamics and Convergence Analysis

### Epoch Progression and Throughput

Detailed resource utilization, total training duration, and training throughput metrics recorded across the 10 epochs:

| Model               | Total Training Time     | Mean Epoch Duration | Train Images / sec | Best Val Epoch | Peak VRAM GPU 0 | Peak VRAM GPU 1 | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc |
| :------------------ | :---------------------- | :------------------ | :----------------- | :------------- | :-------------- | :-------------- | :--------------- | :------------- | :-------------- | :------------ |
| **ViT-B/16**        | **280.77 s** (4m 41s)   | **27.62 s**         | **90.93 img/s**    | Epoch 10       | 10,462 MB       | 9,768 MB        | 0.1649           | 0.1449         | 94.39%          | 94.06%        |
| **Inception-V3**    | **291.30 s** (4m 51s)   | **29.13 s**         | **84.80 img/s**    | Epoch 3        | 13,518 MB       | 0 MB            | 0.0014           | 0.0202         | 99.95%          | 99.39%        |
| **ResNet-50**       | **323.78 s** (5m 24s)   | **32.27 s**         | **77.52 img/s**    | Epoch 7        | 10,839 MB       | 10,592 MB       | 0.0025           | 0.0156         | 99.95%          | 99.59%        |
| **Swin-Base**       | **393.68 s** (6m 34s)   | **38.95 s**         | **66.01 img/s**    | Epoch 6        | 10,802 MB       | 10,060 MB       | 0.0077           | 0.0539         | 99.78%          | 98.77%        |
| **DenseNet-121**    | **507.50 s** (8m 27s)   | **50.70 s**         | **52.80 img/s**    | Epoch 4        | 4,267 MB        | 4,198 MB        | 0.0028           | 0.0049         | 99.95%          | 99.80%        |
| **MobileNetV3-L**   | **718.20 s** (11m 58s)  | **71.80 s**         | **35.20 img/s**    | Epoch 10       | 5,416 MB        | 5,318 MB        | 0.0023           | 0.3995         | 99.90%          | 88.11%        |
| **VGG-16**          | **747.94 s** (12m 28s)  | **74.06 s**         | **33.73 img/s**    | Epoch 10       | 13,314 MB       | 14,588 MB       | 0.2590           | 0.2421         | 88.72%          | 89.75%        |
| **EfficientNet-B0** | **977.20 s** (16m 17s)  | **97.70 s**         | **25.80 img/s**    | Epoch 4        | 10,560 MB       | 10,463 MB       | 0.0065           | 0.0142         | 99.90%          | 99.39%        |
| **CoaTNet-0**       | **1748.00 s** (29m 08s) | **174.70 s**        | **15.60 img/s**    | Epoch 6        | 5,921 MB        | 5,694 MB        | 0.0010           | 0.0404         | 100.00%         | 98.98%        |

### Training and Validation Loss Trajectories

Convergence behaviour demonstrates that modern residual and dense CNN architectures (ResNet-50, DenseNet-121, EfficientNet-B0) reach near-zero training loss within 4 to 6 epochs without overfitting.

#### Training Curves per Architecture

|                                    DenseNet-121                                    |                                  ResNet-50                                   |
| :--------------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
| ![DenseNet121 Training](binary-detection-densenet121/plots/07_training_curves.png) | ![ResNet50 Training](binary-detection-resnet50/plots/07_training_curves.png) |

|                                     EfficientNet-B0                                      |                                  CoaTNet-0                                   |
| :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
| ![EfficientNetB0 Training](binary-detection-efficientnetb0/plots/07_training_curves.png) | ![CoaTNet0 Training](binary-detection-coatnet0/plots/07_training_curves.png) |

|                                    Inception-V3                                    |                                  Swin-Base                                   |
| :--------------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
| ![InceptionV3 Training](binary-detection-inceptionv3/plots/07_training_curves.png) | ![SwinBase Training](binary-detection-swinbase/plots/07_training_curves.png) |

|                                ViT-B/16                                |                                    MobileNetV3-Large                                     |
| :--------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
| ![ViT Training](binary-detection-vit-b16/plots/07_training_curves.png) | ![MobileNetV3 Training](binary-detection-mobilenetv3-large/plots/07_training_curves.png) |

|                                 VGG-16                                 |
| :--------------------------------------------------------------------: |
| ![VGG16 Training](binary-detection-vgg16/plots/07_training_curves.png) |

---

## Comprehensive Experimental Evaluation

### Validation Set Metrics

Performance on the held-out validation set (N = 488, 239 No Tumor, 249 Tumor) evaluating intermediate model selection checkpoints:

| Model Architecture  | Val Accuracy | Balanced Acc | Precision   | Recall (Sens.) | Specificity | F1-Score    | ROC-AUC    | PR-AUC     | MCC        | Log Loss   | ECE        | Val TP | Val FP | Val FN | Val TN |
| :------------------ | :----------- | :----------- | :---------- | :------------- | :---------- | :---------- | :--------- | :--------- | :--------- | :--------- | :--------- | :----- | :----- | :----- | :----- |
| **DenseNet-121**    | **100.00%**  | **100.00%**  | **100.00%** | **100.00%**    | **100.00%** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **0.0043** | **0.0036** | 249    | 0      | 0      | 239    |
| **EfficientNet-B0** | **99.59%**   | 99.58%       | 99.20%      | **100.00%**    | 99.16%      | 99.60%      | 0.9999     | **1.0000** | 0.9918     | 0.0180     | 0.0089     | 249    | 2      | 0      | 237    |
| **Inception-V3**    | **99.59%**   | 99.58%       | 99.20%      | **100.00%**    | 99.16%      | 99.60%      | **1.0000** | **1.0000** | 0.9918     | 0.0109     | 0.0092     | 249    | 2      | 0      | 237    |
| **ResNet-50**       | **99.39%**   | 99.39%       | 99.60%      | 99.20%         | 99.58%      | 99.40%      | 0.9999     | 0.9999     | 0.9877     | 0.0193     | 0.0058     | 247    | 1      | 2      | 238    |
| **CoaTNet-0**       | **98.98%**   | 98.95%       | 98.03%      | **100.00%**    | 97.91%      | 99.01%      | 0.9999     | 0.9999     | 0.9797     | 0.0337     | 0.0098     | 249    | 5      | 0      | 234    |
| **Swin-Base**       | **96.93%**   | 96.98%       | 99.58%      | 94.38%         | 99.58%      | 96.91%      | 0.9973     | 0.9979     | 0.9399     | 0.0943     | 0.0198     | 235    | 1      | 14     | 238    |
| **ViT-B/16**        | **94.06%**   | 93.98%       | 91.35%      | 97.59%         | 90.38%      | 94.37%      | 0.9900     | 0.9915     | 0.8831     | 0.1455     | 0.0294     | 243    | 23     | 6      | 216    |
| **VGG-16**          | **89.75%**   | 89.82%       | 92.70%      | 86.75%         | 92.89%      | 89.63%      | 0.9687     | 0.9705     | 0.7970     | 0.2419     | 0.0503     | 216    | 17     | 33     | 222    |
| **MobileNetV3-L**   | **88.11%**   | 88.35%       | **100.00%** | 76.71%         | **100.00%** | 86.82%      | 0.9993     | 0.9993     | 0.7857     | 0.4001     | 0.0855     | 191    | 0      | 58     | 239    |

### Test Set Metrics

Final unbiased evaluation on the isolated test set (N = 488, 240 No Tumor, 248 Tumor) containing never-before-seen MRI scans:

| Model Architecture | Test Accuracy | Balanced Acc | Precision | Recall (Sens.) | Specificity | F1-Score | F2-Score | ROC-AUC | PR-AUC | MCC | Log Loss | ECE | Brier Score | TP | FP | FN | TN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DenseNet-121** | **99.39%** | **99.40%** | **100.00%** | 98.79% | **100.00%** | 99.39% | 99.03% | **1.0000** | **1.0000** | **0.9878** | **0.0189** | **0.0044** | **0.0097** | 245 | **0** | 3 | **240** |
| **ResNet-50** | **99.39%** | **99.40%** | **100.00%** | 98.79% | **100.00%** | 99.39% | 99.03% | **1.0000** | **1.0000** | **0.9878** | 0.0217 | 0.0057 | 0.0113 | 245 | **0** | 3 | **240** |
| **CoaTNet-0** | **99.39%** | 99.38% | 98.80% | **100.00%** | 98.75% | **99.40%** | **99.76%** | 0.9999 | 0.9999 | **0.9878** | 0.0253 | 0.0084 | 0.0119 | **248** | 3 | **0** | 237 |
| **EfficientNet-B0** | **99.39%** | 99.38% | 98.80% | **100.00%** | 98.75% | **99.40%** | **99.76%** | 0.9998 | 0.9998 | **0.9878** | 0.0233 | 0.0049 | 0.0125 | **248** | 3 | **0** | 237 |
| **Inception-V3** | **98.77%** | 98.78% | 99.59% | 97.98% | 99.58% | 98.78% | 98.30% | 0.9986 | 0.9988 | 0.9755 | 0.0419 | 0.0078 | 0.0204 | 243 | 1 | 5 | 239 |
| **Swin-Base** | **95.90%** | 95.94% | 98.31% | 93.55% | 98.33% | 95.87% | 94.46% | 0.9961 | 0.9966 | 0.9192 | 0.1117 | 0.0235 | 0.0552 | 232 | 4 | 16 | 236 |
| **ViT-B/16** | **94.88%** | 94.85% | 93.73% | 96.37% | 93.33% | 95.03% | 95.83% | 0.9862 | 0.9889 | 0.8978 | 0.1525 | 0.0323 | 0.0813 | 239 | 16 | 9 | 224 |
| **MobileNetV3-L** | **90.78%** | 90.93% | **100.00%** | 81.85% | **100.00%** | 90.02% | 84.94% | 0.9981 | 0.9981 | 0.8302 | 0.3698 | 0.0685 | 0.1345 | 203 | **0** | 45 | **240** |
| **VGG-16** | **88.32%** | 88.35% | 90.30% | 86.29% | 90.42% | 88.25% | 87.06% | 0.9676 | 0.9699 | 0.7673 | 0.2448 | 0.0437 | 0.1518 | 214 | 23 | 34 | 217 |

### Confusion Matrices and Diagnostic Accuracy

Confusion matrices computed on the test set provide transparent insight into diagnostic errors (False Positives vs False Negatives):

|                                  DenseNet-121                                   |                                 ResNet-50                                 |                                    EfficientNet-B0                                    |
| :-----------------------------------------------------------------------------: | :-----------------------------------------------------------------------: | :-----------------------------------------------------------------------------------: |
| ![DenseNet121 CM](binary-detection-densenet121/plots/08_confusion_matrices.png) | ![ResNet50 CM](binary-detection-resnet50/plots/08_confusion_matrices.png) | ![EfficientNetB0 CM](binary-detection-efficientnetb0/plots/08_confusion_matrices.png) |

|                                 CoaTNet-0                                 |                                  Inception-V3                                   |                                 Swin-Base                                 |
| :-----------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :-----------------------------------------------------------------------: |
| ![CoaTNet0 CM](binary-detection-coatnet0/plots/08_confusion_matrices.png) | ![InceptionV3 CM](binary-detection-inceptionv3/plots/08_confusion_matrices.png) | ![SwinBase CM](binary-detection-swinbase/plots/08_confusion_matrices.png) |

|                              ViT-B/16                               |                                   MobileNetV3-Large                                   |                               VGG-16                                |
| :-----------------------------------------------------------------: | :-----------------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![ViT CM](binary-detection-vit-b16/plots/08_confusion_matrices.png) | ![MobileNetV3 CM](binary-detection-mobilenetv3-large/plots/08_confusion_matrices.png) | ![VGG16 CM](binary-detection-vgg16/plots/08_confusion_matrices.png) |

### ROC and Precision-Recall Dynamics

Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves demonstrate superior class separation and high area-under-the-curve metrics across modern architectures:

|                         DenseNet-121 (AUC: 1.0000)                          |                        ResNet-50 (AUC: 1.0000)                        |                           EfficientNet-B0 (AUC: 0.9998)                           |
| :-------------------------------------------------------------------------: | :-------------------------------------------------------------------: | :-------------------------------------------------------------------------------: |
| ![DenseNet121 ROC](binary-detection-densenet121/plots/09_roc_pr_curves.png) | ![ResNet50 ROC](binary-detection-resnet50/plots/09_roc_pr_curves.png) | ![EfficientNetB0 ROC](binary-detection-efficientnetb0/plots/09_roc_pr_curves.png) |

|                        CoaTNet-0 (AUC: 0.9999)                        |                         Inception-V3 (AUC: 0.9986)                          |                        Swin-Base (AUC: 0.9961)                        |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------------: | :-------------------------------------------------------------------: |
| ![CoaTNet0 ROC](binary-detection-coatnet0/plots/09_roc_pr_curves.png) | ![InceptionV3 ROC](binary-detection-inceptionv3/plots/09_roc_pr_curves.png) | ![SwinBase ROC](binary-detection-swinbase/plots/09_roc_pr_curves.png) |

### Probability Calibration and Confidence Reliability

Reliability diagrams and Expected Calibration Error (ECE) verify whether predicted model confidences reflect real-world probabilities. DenseNet-121, EfficientNet-B0, ResNet-50, and CoaTNet-0 exhibit calibration curves that align closely with the diagonal (ECE < 0.01):

|                        DenseNet-121 (ECE: 0.0044)                         |                          EfficientNet-B0 (ECE: 0.0049)                          |                       ResNet-50 (ECE: 0.0057)                       |
| :-----------------------------------------------------------------------: | :-----------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![DenseNet121 Cal](binary-detection-densenet121/plots/10_calibration.png) | ![EfficientNetB0 Cal](binary-detection-efficientnetb0/plots/10_calibration.png) | ![ResNet50 Cal](binary-detection-resnet50/plots/10_calibration.png) |

|                       CoaTNet-0 (ECE: 0.0084)                       |                    ViT-B/16 (ECE: 0.0323)                     |
| :-----------------------------------------------------------------: | :-----------------------------------------------------------: |
| ![CoaTNet0 Cal](binary-detection-coatnet0/plots/10_calibration.png) | ![ViT Cal](binary-detection-vit-b16/plots/10_calibration.png) |

### Per-Class Performance Profiling

Decomposed precision, recall, and F1 metrics for both `No Tumor` and `Tumor` classes verify consistent diagnostic balance:

|                                     DenseNet-121                                     |                                      EfficientNet-B0                                       |                                   ResNet-50                                    |
| :----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
| ![DenseNet121 PerClass](binary-detection-densenet121/plots/11_per_class_metrics.png) | ![EfficientNetB0 PerClass](binary-detection-efficientnetb0/plots/11_per_class_metrics.png) | ![ResNet50 PerClass](binary-detection-resnet50/plots/11_per_class_metrics.png) |

### Representative Qualitative Predictions

Sample qualitative predictions on unseen test cases illustrate confident probability scores on diverse slice orientations:

|                               DenseNet-121 Predictions                               |                             ResNet-50 Predictions                              |
| :----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
| ![DenseNet121 Samples](binary-detection-densenet121/plots/12_prediction_samples.png) | ![ResNet50 Samples](binary-detection-resnet50/plots/12_prediction_samples.png) |

|                                EfficientNet-B0 Predictions                                 |                             CoaTNet-0 Predictions                              |
| :----------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
| ![EfficientNetB0 Samples](binary-detection-efficientnetb0/plots/12_prediction_samples.png) | ![CoaTNet0 Samples](binary-detection-coatnet0/plots/12_prediction_samples.png) |

---

## Visual Explainability: Grad-CAM Interpretability

### Interpretability Methodology across CNNs and Transformers

To ensure that models base their predictions on clinically relevant pathology rather than spurious background artefacts, Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented.

For standard convolutional architectures, gradients flow directly into the final 2D convolutional feature maps. For Vision Transformers and Hierarchical Transformers (ViT-B/16, Swin-Base, CoaTNet-0), activation tokens are dynamically extracted, the classification token is removed, and 1D token sequences are unflattened into a spatial grid (B, H, W, C) before computing activation heatmaps.

### Attribution on Correct Classifications

Heatmap attributions for correctly classified test cases confirm that high-performing networks accurately localise intracranial tumour masses, hyperintense lesion margins, and structural tissue disruptions:

#### Top-Performing CNN and Hybrid Attributions

| Model               | Correct Classification Grad-CAM Heatmaps                                                        |
| :------------------ | :---------------------------------------------------------------------------------------------- |
| **DenseNet-121**    | ![DenseNet121 GradCAM Correct](binary-detection-densenet121/plots/13_gradcam_correct.png)       |
| **ResNet-50**       | ![ResNet50 GradCAM Correct](binary-detection-resnet50/plots/13_gradcam_correct.png)             |
| **EfficientNet-B0** | ![EfficientNetB0 GradCAM Correct](binary-detection-efficientnetb0/plots/13_gradcam_correct.png) |
| **CoaTNet-0**       | ![CoaTNet0 GradCAM Correct](binary-detection-coatnet0/plots/13_gradcam_correct.png)             |
| **Inception-V3**    | ![InceptionV3 GradCAM Correct](binary-detection-inceptionv3/plots/13_gradcam_correct.png)       |
| **Swin-Base**       | ![SwinBase GradCAM Correct](binary-detection-swinbase/plots/13_gradcam_correct.png)             |
| **ViT-B/16**        | ![ViT GradCAM Correct](binary-detection-vit-b16/plots/13_gradcam_correct.png)                   |

### Attribution on Misclassifications and Failure Analysis

Inspecting Grad-CAM heatmaps on misclassified test instances highlights specific failure mechanisms:

- **Low-Contrast / Diffuse Tumours:** Small, low-grade lesions with low signal intensity relative to surrounding parenchyma occasionally fail to trigger activation thresholds.
- **Anatomical Boundary Confusion:** Ventricular asymmetry, skull contours, or prominent dural structures can mislead models that lack local convolutional inductive biases (e.g. ViT and Swin).
- **Under-confidence in Lightweight Models:** MobileNetV3 activations often localize the tumour correctly in the feature map, but produce logits slightly below the default 0.50 threshold due to conservative output calibration.

| Model                 | Misclassification Grad-CAM Heatmaps                                                                         |
| :-------------------- | :---------------------------------------------------------------------------------------------------------- |
| **DenseNet-121**      | ![DenseNet121 GradCAM Misclassified](binary-detection-densenet121/plots/14_gradcam_misclassified.png)       |
| **ResNet-50**         | ![ResNet50 GradCAM Misclassified](binary-detection-resnet50/plots/14_gradcam_misclassified.png)             |
| **EfficientNet-B0**   | ![EfficientNetB0 GradCAM Misclassified](binary-detection-efficientnetb0/plots/14_gradcam_misclassified.png) |
| **CoaTNet-0**         | ![CoaTNet0 GradCAM Misclassified](binary-detection-coatnet0/plots/14_gradcam_misclassified.png)             |
| **Inception-V3**      | ![InceptionV3 GradCAM Misclassified](binary-detection-inceptionv3/plots/14_gradcam_misclassified.png)       |
| **MobileNetV3-Large** | ![MobileNetV3 GradCAM Misclassified](binary-detection-mobilenetv3-large/plots/14_gradcam_misclassified.png) |
| **ViT-B/16**          | ![ViT GradCAM Misclassified](binary-detection-vit-b16/plots/14_gradcam_misclassified.png)                   |
| **VGG-16**            | ![VGG16 GradCAM Misclassified](binary-detection-vgg16/plots/14_gradcam_misclassified.png)                   |

---

## Inference Latency and Throughput Benchmarks

### Benchmarking Setup on Tesla T4

Inference benchmarking was conducted under strict, repeatable conditions:

- **Hardware Target:** Dedicated NVIDIA Tesla T4 GPU (16 GB VRAM).
- **Precision:** `fp16 autocast` (FP16 half-precision inference with PyTorch AMP).
- **Warm-up:** 20 iterations prior to timing.
- **Single-Image Latency (B=1):** Profiled over 100 repeated single-sample iterations (`torch.cuda.Event` synchronization).
- **Batched Throughput (B=32):** Profiled over 50 iterations with batch size 32.
- **Full Test Set Pipeline:** End-to-end evaluation time over the entire 488-image test set including data loading, preprocessing, tensor conversion, forward pass, and metric reduction.

### Latency and Throughput Profile

| Model Architecture    | Batch 1 Mean Latency | Batch 1 Median | Batch 1 P95 Latency | Batch 1 Throughput | Batch 32 Latency | Batch 32 Throughput | Full Test Set (488 img) Time | Test Set Processing Speed |
| :-------------------- | :------------------- | :------------- | :------------------ | :----------------- | :--------------- | :------------------ | :--------------------------- | :------------------------ |
| **ResNet-50**         | **7.35 ms**          | **7.18 ms**    | **8.47 ms**         | **136.10 img/s**   | 41.49 ms         | 771.21 img/s        | 3.24 s                       | 150.72 img/s              |
| **MobileNetV3-Large** | 8.08 ms              | 7.95 ms        | 9.26 ms             | 123.71 img/s       | **13.29 ms**     | **2,408.52 img/s**  | 2.71 s                       | 180.31 img/s              |
| **VGG-16**            | 8.19 ms              | 7.99 ms        | 9.72 ms             | 122.18 img/s       | 94.63 ms         | 338.15 img/s        | 3.84 s                       | 127.14 img/s              |
| **ViT-B/16**          | 8.80 ms              | 8.61 ms        | 9.82 ms             | 113.68 img/s       | 111.54 ms        | 286.88 img/s        | 3.05 s                       | 160.06 img/s              |
| **EfficientNet-B0**   | 10.76 ms             | 10.60 ms       | 11.78 ms            | 92.96 img/s        | **23.46 ms**     | **1,364.24 img/s**  | 2.86 s                       | 170.59 img/s              |
| **CoaTNet-0**         | 13.91 ms             | 13.67 ms       | 15.16 ms            | 71.90 img/s        | 61.50 ms         | 520.34 img/s        | **2.31 s**                   | **211.02 img/s**          |
| **Inception-V3**      | 14.05 ms             | 13.62 ms       | 15.84 ms            | 71.18 img/s        | 62.24 ms         | 514.15 img/s        | 3.20 s                       | 152.48 img/s              |
| **DenseNet-121**      | 19.43 ms             | 19.28 ms       | 21.16 ms            | 51.47 img/s        | 50.81 ms         | 629.80 img/s        | **2.13 s**                   | **228.95 img/s**          |
| **Swin-Base**         | 28.99 ms             | 28.77 ms       | 31.39 ms            | 34.50 img/s        | 163.57 ms        | 195.63 img/s        | 3.87 s                       | 126.07 img/s              |

### Efficiency vs Accuracy Trade-off Matrix

```text
   Test Accuracy (%)
      ^
100%  |     [DenseNet-121]  [ResNet-50]  [CoaTNet-0]  [EfficientNet-B0]
      |     (5.67G, 99.39%) (8.17G, 99.39%) (8.42G, 99.39%) (0.77G, 99.39%)  <-- Pareto Frontier
 98%  |                                [Inception-V3] (11.42G, 98.77%)
      |
 96%  |                 [Swin-Base] (30.86G, 95.90%)
      |
 94%  |                 [ViT-B/16] (22.54G, 94.88%)
      |
 90%  |  [MobileNetV3-L] (0.43G, 90.78%)
      |
 88%  |                                      [VGG-16] (30.93G, 88.32%)
      +-------------------------------------------------------------------->
         0.4G     1.0G     5.0G     10.0G    20.0G    30.0G      GFLOPs
```

---

## Comparative Synthesis and Deployment Recommendations

### Top-Tier Production Candidates

1. **Best Overall Diagnostic and Efficient Model: EfficientNet-B0**
   - **Performance:** 99.39% Test Accuracy, 100.00% Sensitivity (0 False Negatives), 0.9998 ROC-AUC.
   - **Footprint:** 4.01M parameters (15.29 MB), 0.769 GFLOPs.
   - **Throughput:** 1,364.24 images/s at batch size 32.
   - **Clinical Recommendation:** Ideal for primary diagnostic triage in hospital Picture Archiving and Communication Systems (PACS) and cloud deployment due to zero false negative rate and minimal memory footprint.

2. **Best Zero-False-Positive Precision Model: DenseNet-121**
   - **Performance:** 99.39% Test Accuracy, 100.00% Precision (0 False Positives), 1.0000 ROC-AUC.
   - **Footprint:** 6.95M parameters (26.53 MB), 5.666 GFLOPs.
   - **Clinical Recommendation:** Ideal for secondary verification where false alarms must be strictly prevented.

3. **Best High-Throughput Edge Candidate: MobileNetV3-Large**
   - **Performance:** 90.78% Test Accuracy at default 0.50 threshold; 0.9981 ROC-AUC.
   - **Throughput:** 2,408.52 images/s (8.08 ms single-sample latency).
   - **Calibration Note:** When configured with optimal Youden J operating threshold (threshold = 0.052), sensitivity increases significantly from 81.85% to 98.39%.

### Architectural Insights and Inductive Bias

- **Inductive Bias Matters in Medical Imaging:** Convolutions exhibit translation equivariance and localized receptive fields that are well-matched to detecting localized focal lesions in brain MRI. DenseNet-121 and ResNet-50 outperformed pure Vision Transformers (ViT-B/16) despite having 12x and 3.6x fewer parameters.
- **Hybrid Conv-Attention (CoaTNet):** CoaTNet-0 successfully combined the early spatial inductive bias of convolutions with late-stage global self-attention, achieving zero false negatives (100% recall) and 99.39% accuracy.
- **Transformer Sample Efficiency:** Pure Vision Transformers (ViT-B/16, 94.88%) and Swin Transformers (95.90%) require larger dataset scales or stronger regularization to match CNN sample efficiency on small-to-medium medical cohorts (N approx 3,000).
- **Obsolescence of Deep Dense Heads (VGG-16):** VGG-16 required 134.26M parameters and 30.93 GFLOPs, but yielded the lowest test accuracy (88.32%), demonstrating the inefficiency of large un-regularised fully connected heads compared to global average pooling.

### Failure Modes and Threshold Sensitivity

| Model Architecture  | Default Threshold (0.50) Recall | Optimal Youden Threshold | Youden Recall | Optimal Youden J Statistic |
| :------------------ | :------------------------------ | :----------------------- | :------------ | :------------------------- |
| **DenseNet-121**    | 98.79%                          | 0.448                    | 98.79%        | 0.9879                     |
| **ResNet-50**       | 98.79%                          | 0.070                    | 99.60%        | 0.9960                     |
| **CoaTNet-0**       | **100.00%**                     | 0.170                    | 100.00%       | 0.9917                     |
| **EfficientNet-B0** | **100.00%**                     | 0.281                    | 100.00%       | 0.9917                     |
| **Inception-V3**    | 97.98%                          | 0.046                    | 99.19%        | 0.9878                     |
| **Swin-Base**       | 93.55%                          | 0.238                    | 95.97%        | 0.9426                     |
| **ViT-B/16**        | 96.37%                          | 0.633                    | 96.37%        | 0.9137                     |
| **MobileNetV3-L**   | 81.85%                          | 0.052                    | 98.39%        | 0.9839                     |
| **VGG-16**          | 86.29%                          | 0.728                    | 86.29%        | 0.7813                     |

---

## Directory Layout and Generated Artefacts

The `Binary Detection` workspace contains all 9 runnable Jupyter notebooks along with isolated output directories for each architecture containing serialised PyTorch model checkpoints, full CSV/JSON metric exports, TensorBoard logs, and publication-ready plots:

```text
Backend/Binary Detection/
|-- binary-detection-coatnet0.ipynb
|-- binary-detection-densenet121.ipynb
|-- binary-detection-efficientnetb0.ipynb
|-- binary-detection-inceptionv3.ipynb
|-- binary-detection-mobilenetv3-large.ipynb
|-- binary-detection-resnet50.ipynb
|-- binary-detection-swinbase.ipynb
|-- binary-detection-vgg16.ipynb
|-- binary-detection-vit-b16.ipynb
|-- EfficientNetB0.pth
|-- README.md
|
|-- binary-detection-coatnet0/
|   |-- metrics/
|   |   |-- benchmark_summary.csv
|   |   |-- benchmark_summary.json
|   |   |-- calibration_bins_test.csv
|   |   |-- calibration_bins_val.csv
|   |   |-- classification_report_test.txt
|   |   |-- classification_report_val.txt
|   |   |-- confusion_matrix_test.csv
|   |   |-- confusion_matrix_val.csv
|   |   |-- image_dimension_stats.csv
|   |   |-- inference_benchmark.json
|   |   |-- metrics_summary.csv
|   |   |-- metrics_test.json
|   |   |-- metrics_val.json
|   |   |-- model_profile.json
|   |   |-- per_class_metrics_test.csv
|   |   |-- per_class_metrics_val.csv
|   |   |-- predictions_test.csv
|   |   |-- predictions_val.csv
|   |   |-- split_distribution.csv
|   |   |-- training_history.csv
|   |   `-- training_summary.json
|   |-- plots/
|   |   |-- 01_class_distribution.png
|   |   |-- 02_sample_images.png
|   |   |-- 03_image_properties.png
|   |   |-- 05_split_distribution.png
|   |   |-- 06_augmentation_before_after.png
|   |   |-- 07_training_curves.png
|   |   |-- 08_confusion_matrices.png
|   |   |-- 09_roc_pr_curves.png
|   |   |-- 10_calibration.png
|   |   |-- 11_per_class_metrics.png
|   |   |-- 12_prediction_samples.png
|   |   |-- 13_gradcam_correct.png
|   |   `-- 14_gradcam_misclassified.png
|   |-- tensorboard/
|   |   `-- 20260916-175559/
|   `-- weights/
|       |-- best_coatnet.pth
|       `-- final_coatnet_brain_tumor.pth
|
|-- binary-detection-densenet121/    [Same metrics, plots, tensorboard, weights structure]
|-- binary-detection-efficientnetb0/ [Same metrics, plots, tensorboard, weights structure]
|-- binary-detection-inceptionv3/    [Same metrics, plots, tensorboard, weights structure]
|-- binary-detection-mobilenetv3-large/ [Same metrics, plots, tensorboard, weights structure]
|-- binary-detection-resnet50/       [Same metrics, plots, tensorboard, weights structure]
|-- binary-detection-swinbase/       [Same metrics, plots, tensorboard, weights structure]
|-- binary-detection-vgg16/          [Same metrics, plots, tensorboard, weights structure]
`-- binary-detection-vit-b16/        [Same metrics, plots, tensorboard, weights structure]
```

---

## Quick Start and Execution Guide

### Prerequisites

Ensure the following dependencies are installed in your Python 3.10+ environment:

```bash
pip install torch torchvision timm scikit-learn pandas numpy matplotlib pillow tqdm tensorboard
```

### Executing Individual Notebooks

To reproduce training, validation, Grad-CAM generation, and inference benchmarking for any model:

```bash
# Example: Run ResNet50 pipeline via Jupyter nbconvert
jupyter nbconvert --to notebook --execute "binary-detection-resnet50.ipynb" --output "executed-resnet50.ipynb"
```

Each notebook executes independently and generates:

1. Complete training logs and loss curves.
2. Formatted test evaluation metrics.
3. Grad-CAM visualisations for both correct and misclassified MRI samples.
4. Model latency and throughput benchmarks.
5. Saved `.pth` model weights in `weights/` and metrics in `metrics/`.

### Loading Pretrained Checkpoints for Inference

To load a trained model weight for downstream clinical inference:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Instantiate architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

# 2. Load fine-tuned weights
checkpoint_path = "binary-detection-efficientnetb0/weights/best_efficientnet.pth"
state_dict = torch.load(checkpoint_path, map_location=device)

# Handle DataParallel prefix if present
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# 3. Preprocess scan
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("sample_mri.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# 4. Predict
with torch.no_grad():
    logit = model(input_tensor)
    probability = torch.sigmoid(logit).item()
    prediction = "Tumor" if probability >= 0.50 else "No Tumor"

print(f"Prediction: {prediction} (Tumour Probability: {probability:.4f})")
```

---

## Troubleshooting and Operational Notes

| Issue / Symptom                                   | Probable Cause                                                                                  | Diagnostic Command / Resolution                                                                                                      |
| :------------------------------------------------ | :---------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| **`RuntimeError: CUDA out of memory`**            | Batch size too large for available GPU VRAM.                                                    | Reduce `BATCH_SIZE` in the configuration cell (e.g. from 256 to 64 or 32). Clear cache via `torch.cuda.empty_cache()`.               |
| **`Missing key(s) in state_dict: module.`**       | Model was saved with `nn.DataParallel` wrapper but loaded into a single-GPU model instance.     | Strip the `module.` prefix from state dict keys using `{k.replace("module.", ""): v for k, v in state_dict.items()}` before loading. |
| **`ModuleNotFoundError: No module named 'timm'`** | PyTorch Image Models (`timm`) library not installed in environment.                             | Install TIMM via `pip install timm`. Required for CoaTNet-0 and Swin-Base.                                                           |
| **Grad-CAM shape mismatch error on Swin / ViT**   | Transformer feature map is 3D token tensor (B, N, C) instead of 4D spatial tensor (B, C, H, W). | Ensure `to_nchw()` reshaping helper is applied to remove class token and unflatten (B, H, W, C) into (B, C, H, W).                   |
| **Dataset path mismatch**                         | Local directory structure differs from the Kaggle dataset mount paths.                          | Modify `NO_PATHS` and `YES_PATHS` in Cell 2 of the respective notebook to point to your local dataset directory.                     |
