# Research notebooks and results

This folder holds the three Kaggle notebooks behind NeuroScan AI and the outputs downloaded from their executed runs. Every number and figure below is copied from those outputs (`metrics/`, `logs/`, `plots/`, `weights/deploy/metadata.json`); nothing here was re-run locally.

| Notebook                                           | Task                                                                   | Models          | Outputs folder                           | Notebook time |
| -------------------------------------------------- | ---------------------------------------------------------------------- | --------------- | ---------------------------------------- | ------------- |
| [`binary-detection.ipynb`](binary-detection.ipynb) | Tumour-positive vs tumour-negative 2D MRI slices                       | 9               | [`binary-detection/`](binary-detection/) | 71.8 min      |
| [`classification.ipynb`](classification.ipynb)     | Glioma vs meningioma vs pituitary 2D MRI slices                        | 9               | [`classification/`](classification/)     | 273.6 min     |
| [`segmentation.ipynb`](segmentation.ipynb)         | Whole tumour (WT), tumour core (TC), enhancing tumour (ET), BraTS 2020 | 3 + 4 ensembles | [`segmentation/`](segmentation/)         | 547.9 min     |

Notebook times are `notebook_minutes` in each `metrics/run_summary.json`. The segmentation notebook is also validated externally on 60 BraTS-Africa cases.

## Contents

- [How the folder is organised](#how-the-folder-is-organised)
- [Run environment](#run-environment)
- [Binary detection notebook](#binary-detection-notebook)
- [Classification notebook](#classification-notebook)
- [Segmentation notebook](#segmentation-notebook)
- [Checkpoints used by the web app](#checkpoints-used-by-the-web-app)
- [Limitations recorded by the notebooks](#limitations-recorded-by-the-notebooks)

## How the folder is organised

```text
Materials/Notebooks/
|-- binary-detection.ipynb       Executed notebook with all cell outputs (25 inline figures)
|-- classification.ipynb         Executed notebook with all cell outputs (26 inline figures)
|-- segmentation.ipynb           Executed notebook with all cell outputs (17 inline figures)
`-- <task>/                      Outputs downloaded from the executed Kaggle run
    |-- __notebook__.ipynb       Kaggle's copy of the same run (Git-ignored); same cells, without the last cell's output
    |-- metrics/                 CSV tables and JSON summaries; metrics/runs/<run>/ per-run history
    |-- plots/                   Every figure as PNG
    |-- logs/                    notebook.log, worker_gpu0/1.log, events_gpu0/1.jsonl, status_gpu0/1.json
    |-- predictions/             2D tasks: <run>.npz logits; segmentation: <run>/<val|test|external>/<case>.npz probabilities
    |-- weights/<model>/         Best-validation checkpoint of every run (*.pth, Git-ignored)
    |-- weights/deploy/          Checkpoint the web app loads plus metadata.json (segmentation: copied from this run by hand)
    `-- src/                     The library and worker modules each notebook writes and imports
```

The root `.gitignore` excludes `__notebook__.ipynb`, `*.pth`, `*.log` and `*.npy`, and `Materials/Notebooks/segmentation/predictions/`, so a fresh clone has the three executed notebooks, `deploy/metadata.json`, metrics, plots, JSON logs and `.npz` predictions, but not Kaggle's duplicate notebook copies, the weights, the text logs or the segmentation probability files.

Each notebook writes its shared code to `src/` in an early cell and runs training in one worker process per GPU: the 2D notebooks write `image_lib.py`, `image_worker.py` and `shared_lib.py`; the segmentation notebook writes `volume_lib.py`, `volume_worker.py` and `shared_lib.py`.

## Run environment

All three runs used Kaggle with two Tesla T4 GPUs (14.6 GB each as reported by the notebook), 4 CPU cores and 33.7 GB RAM (`logs/notebook.log`). From each `metrics/run_summary.json`:

| Item          | Binary detection | Classification | Segmentation |
| ------------- | ---------------- | -------------- | ------------ |
| Python        | 3.13.15          | 3.13.15        | 3.13.15      |
| PyTorch       | 2.11.0+cu128     | 2.11.0+cu128   | 2.11.0+cu128 |
| torchvision   | 0.26.0+cu128     | 0.26.0+cu128   | not recorded |
| timm          | 1.0.29           | 1.0.29         | not used     |
| MONAI         | not used         | not used       | 1.6.1        |
| CUDA          | 12.8             | 12.8           | 12.8         |
| Deterministic | False            | False          | False        |
| Training wall | 63.5 min         | 264.6 min      | 528.3 min    |
| Jobs finished | 54 of 54         | 54 of 54       | 3 of 3 runs  |

The 2D notebooks also print numpy 2.1.3 and pandas 2.3.3 in their first cell; the segmentation notebook prints numpy 2.1.3 and nibabel 5.4.2.

## Binary detection notebook

Source: [`binary-detection.ipynb`](binary-detection.ipynb). Outputs: [`binary-detection/`](binary-detection/).

### 1. Setup

- **Imports.** Installs `imagehash` and `timm` only if missing and prints the library versions listed above.
- **Configuration.** One cell holds every setting. Task `binary`, classes `Tumour-negative` and `Tumour`, nine models (ResNet50, VGG16, DenseNet121, EfficientNetB0, MobileNetV3-Large, InceptionV3, ViT-B/16, Swin-Base, CoAtNet-0), `SEED = 42`, `N_FOLDS = 5`, `TEST_FRACTION = 0.15`, `EPOCHS = 20`, `WARMUP_EPOCHS = 1`, `EFFECTIVE_BATCH = 64`, `LR = 3e-4`, `WEIGHT_DECAY = 0.01`, `GRAD_CLIP = 1.0`, `CACHE_SIZE = 320`, `DEDUP_TOLERANCE = 5`, `GROUP_TOLERANCE = 8`, `NCC_DUPLICATE = 0.98`, `SHORTCUT_TEST = True`, `BOOTSTRAP_SAMPLES = 2000`, `TRAINING_BUDGET_HOURS = 9.0`, `DEPLOY_MODEL = "EfficientNetB0"`. Output: `Task: binary | classes: ['Tumour-negative', 'Tumour'] | models: 9 | folds: 5`.
- **Hardware check.** Stops unless two GPUs are visible. Output: two Tesla T4 lines, 4 CPU cores, 33.7 GB RAM, 20.9 GB free disk.
- **Shared library, training worker, worker launcher.** Write `src/image_lib.py`, `src/image_worker.py` and `src/shared_lib.py`, then import them. Output: `Library loaded: 9 registered architectures`.

Model construction (`image_lib.build_model`): every model except CoAtNet-0 comes from torchvision with its `DEFAULT` ImageNet weights and a replaced final linear layer; CoAtNet-0 is timm `coatnet_0_rw_224`. InceptionV3's auxiliary classifier is removed. The worker trains with AdamW, a linear warm-up for one epoch then cosine decay per optimiser step, fp16 autocast with gradient scaling, gradient clipping at 1.0, binary cross-entropy with logits (cross-entropy for subtypes), and keeps the checkpoint with the lowest validation log-loss. Augmentation runs on the GPU: horizontal flip (p = 0.5), rotation up to 10 degrees, and brightness and contrast jitter of 20 percent. Every image is cropped with `crop_to_brain`, the same Gaussian blur (5 x 5), threshold 45 and largest-contour box the web app uses.

### 2. Data

- **Collect the image pool.** Output: `Pool: 3253 images | external: 0`.
- **Decode, hash, crop and cache.** Computes the native size, 64-bit perceptual hash and a 64 x 64 thumbnail of every original image and caches the brain crop. Output: `Cached 3253 images in 39s (1.33 GB); unreadable: 0`.
- **Dataset overview.** Provenance table (`metrics/provenance.csv`) and two figures.

| Source      | Tumour | Tumour-negative | Total | Distinct sizes | Modal size | Modal share | Median crop fraction |
| ----------- | ------ | --------------- | ----- | -------------- | ---------- | ----------- | -------------------- |
| Br35H       | 1,500  | 1,500           | 3,000 | 291            | 225x225    | 0.109       | 0.787                |
| Chakrabarty | 155    | 98              | 253   | 184            | 225x225    | 0.071       | 0.783                |
| Total       | 1,655  | 1,598           | 3,253 |                |            |             |                      |

![Binary dataset overview](binary-detection/plots/01_dataset_overview.png)

_Images per source and class and native image sizes of the binary pool._

![Binary samples, crops and head masks](binary-detection/plots/02_samples_crop_mask.png)

_Sample images, the brain crop the model sees, and the head masks used later by the attention audit._

### 3. Deduplication

- **Label-aware perceptual-hash deduplication.** 64-bit pHash, Hamming distance at most 5, applied per label before any split. Cross-label pairs within the tolerance are checked with thumbnail normalised cross-correlation (NCC). Output: `Cross-label pairs within Hamming 5: 0 (0 label noise, 0 hash false positives)`; `metrics/cross_label_pairs.csv` is empty.

| Class (`deduplication_summary.csv`) | Before | Duplicates removed | Label noise excluded | After | Removed |
| ----------------------------------- | ------ | ------------------ | -------------------- | ----- | ------- |
| Tumour-negative                     | 1,598  | 1,143              | 0                    | 455   | 71.53%  |
| Tumour                              | 1,655  | 925                | 0                    | 730   | 55.89%  |
| Total                               | 3,253  | 2,068              | 0                    | 1,185 | 63.57%  |

| Removed image source (`deduplication_by_source.csv`) | Kind          | Tumour | Tumour-negative | All   |
| ---------------------------------------------------- | ------------- | ------ | --------------- | ----- |
| Br35H                                                | cross-source  | 312    | 417             | 729   |
| Br35H                                                | within-source | 588    | 704             | 1,292 |
| Chakrabarty                                          | within-source | 25     | 22              | 47    |
| All                                                  |               | 925    | 1,143           | 2,068 |

- **Threshold sensitivity** (`deduplication_threshold_sweep.csv`). Images kept at each Hamming tolerance:

| Tolerance       | 0, 1  | 2, 3  | 4, 5  | 6, 7  | 8, 9 | 10, 11 | 12  |
| --------------- | ----- | ----- | ----- | ----- | ---- | ------ | --- |
| Kept            | 2,077 | 1,495 | 1,185 | 1,002 | 882  | 804    | 709 |
| Tumour-negative | 860   | 583   | 455   | 395   | 341  | 310    | 285 |
| Tumour          | 1,217 | 912   | 730   | 607   | 541  | 494    | 424 |

![Binary deduplication](binary-detection/plots/03_deduplication.png)

_Images kept at each tolerance, and the class distribution before and after the pass at tolerance 5._

- **Cross-label pairs.** Output: `No cross-label pairs within the tolerance.` (no gallery is saved for this task).

### 4. Group-aware split and leakage audit

Kept images linked by Hamming distance at most 8 or NCC at least 0.98 are joined into groups, a proxy for patient-level grouping. `StratifiedGroupKFold` draws a held-out test set and five development folds so no group crosses a partition. Output: `1185 images in 821 groups; largest group 7 images; 260 groups with more than one image`.

| Role (`split_roles.csv`) | Tumour | Tumour-negative | Total |
| ------------------------ | ------ | --------------- | ----- |
| Development (5 folds)    | 623    | 391             | 1,014 |
| Test                     | 107    | 64              | 171   |
| Removed duplicate        | 925    | 1,143           | 2,068 |
| Total                    | 1,655  | 1,598           | 3,253 |

| Fold (`fold_distribution.csv`) | 0   | 1   | 2   | 3   | 4   | All   |
| ------------------------------ | --- | --- | --- | --- | --- | ----- |
| Tumour                         | 110 | 123 | 133 | 133 | 124 | 623   |
| Tumour-negative                | 74  | 88  | 74  | 77  | 78  | 391   |
| All                            | 184 | 211 | 207 | 210 | 202 | 1,014 |

Leakage audit (`leakage_audit.csv`): 0 test images have a development image within Hamming 5, within Hamming 8, or at NCC 0.98 or more. `metrics/splits.csv` lists the role and fold of every image.

![Binary groups and leakage](binary-detection/plots/05_groups_and_leakage.png)

_Group sizes and the distance from each test image to its nearest development image._

- **Augmentation preview.**

![Binary augmentation](binary-detection/plots/06_augmentation.png)

_GPU training augmentation on a few training images, with the background-only and head-only inputs of the shortcut tests._

### 5. Training

- **Model profiles** (`model_profiles.csv`). Parameter count and GFLOPs at the native input and at 224 px.

| Model             | Family        | Input | Parameters (M) | GFLOPs (native) | GFLOPs (224) |
| ----------------- | ------------- | ----- | -------------- | --------------- | ------------ |
| ResNet50          | Classical CNN | 224   | 23.510         | 8.174           | 8.174        |
| VGG16             | Classical CNN | 224   | 134.265        | 30.932          | 30.932       |
| DenseNet121       | Classical CNN | 224   | 6.955          | 5.666           | 5.666        |
| EfficientNetB0    | Efficient CNN | 224   | 4.009          | 0.769           | 0.769        |
| MobileNetV3-Large | Efficient CNN | 224   | 4.203          | 0.431           | 0.431        |
| InceptionV3       | Efficient CNN | 299   | 21.788         | 11.422          | 5.672        |
| ViT-B/16          | Transformer   | 224   | 85.799         | 22.541          | 22.541       |
| Swin-Base         | Transformer   | 224   | 86.744         | 30.860          | 30.860       |
| CoAtNet-0         | Hybrid        | 224   | 26.667         | 8.416           | 8.416        |

- **Launch the two GPU workers.** 45 full-image runs (9 models x 5 folds) plus 9 background-only runs (fold 0) for the shortcut test. Output: `54 jobs, 0 already finished; launching 2 workers` and `Training stage finished in 63.5 min; 54/54 jobs complete`. Every run took 260 optimiser steps (20 epochs x 13 steps of 64 images); Swin-Base used a micro-batch of 32 with 2 accumulation steps, every other model a micro-batch of 64. Per-run details are in `metrics/run_metrics.csv` and `metrics/runs/<run>/history.csv`; the worker settings are in `metrics/worker_config.json`.

### 6. Results

- **Per-run calibration and metrics.** For each run, a temperature is fitted on its validation fold, the decision threshold is chosen there by Youden's J, and the test set is scored. Fold summary (`fold_summary.csv`, mean +/- standard deviation over 5 folds, calibrated):

| Model             | Accuracy          | Balanced acc.     | Macro F1          | ROC AUC           | ECE               | Sensitivity       | Specificity       |
| ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| InceptionV3       | 0.9789 +/- 0.0032 | 0.9775 +/- 0.0042 | 0.9775 +/- 0.0034 | 0.9966 +/- 0.0042 | 0.0195 +/- 0.0033 | 0.9832 +/- 0.0078 | 0.9719 +/- 0.0131 |
| Swin-Base         | 0.9778 +/- 0.0162 | 0.9785 +/- 0.0161 | 0.9764 +/- 0.0172 | 0.9974 +/- 0.0026 | 0.0182 +/- 0.0117 | 0.9757 +/- 0.0194 | 0.9812 +/- 0.0204 |
| EfficientNetB0    | 0.9778 +/- 0.0049 | 0.9747 +/- 0.0053 | 0.9762 +/- 0.0052 | 0.9968 +/- 0.0034 | 0.0197 +/- 0.0045 | 0.9869 +/- 0.0051 | 0.9625 +/- 0.0086 |
| CoAtNet-0         | 0.9754 +/- 0.0172 | 0.9760 +/- 0.0183 | 0.9739 +/- 0.0183 | 0.9966 +/- 0.0038 | 0.0216 +/- 0.0125 | 0.9738 +/- 0.0154 | 0.9781 +/- 0.0237 |
| ViT-B/16          | 0.9754 +/- 0.0096 | 0.9722 +/- 0.0128 | 0.9737 +/- 0.0104 | 0.9970 +/- 0.0013 | 0.0175 +/- 0.0044 | 0.9850 +/- 0.0156 | 0.9594 +/- 0.0324 |
| DenseNet121       | 0.9719 +/- 0.0076 | 0.9663 +/- 0.0077 | 0.9698 +/- 0.0082 | 0.9927 +/- 0.0048 | 0.0259 +/- 0.0048 | 0.9888 +/- 0.0078 | 0.9438 +/- 0.0086 |
| ResNet50          | 0.9684 +/- 0.0135 | 0.9635 +/- 0.0166 | 0.9660 +/- 0.0146 | 0.9876 +/- 0.0105 | 0.0226 +/- 0.0060 | 0.9832 +/- 0.0078 | 0.9438 +/- 0.0305 |
| MobileNetV3-Large | 0.9637 +/- 0.0220 | 0.9635 +/- 0.0188 | 0.9616 +/- 0.0230 | 0.9946 +/- 0.0033 | 0.0274 +/- 0.0084 | 0.9645 +/- 0.0320 | 0.9625 +/- 0.0086 |
| VGG16             | 0.9591 +/- 0.0083 | 0.9585 +/- 0.0102 | 0.9565 +/- 0.0088 | 0.9943 +/- 0.0023 | 0.0335 +/- 0.0075 | 0.9607 +/- 0.0167 | 0.9562 +/- 0.0280 |

The file also holds uncalibrated ECE, log-loss and threshold-tuned accuracy, sensitivity and specificity for every model.

- **Training curves.**

![Binary training curves](binary-detection/plots/07_training_curves.png)

_Mean +/- one standard deviation across folds of training and validation loss, validation accuracy and learning rate, per model._

- **Fold-ensemble test results.** Calibrated test probabilities are averaged over each model's five folds and scored on the 171 test images, with bootstrap 95% confidence intervals (`test_leaderboard.csv`):

| Model             | Accuracy (95% CI)       | Balanced acc. | Macro F1 | MCC    | ROC AUC | PR AUC | Log-loss | Brier  | ECE    | Sens.  | Spec.  | FN  | FP  |
| ----------------- | ----------------------- | ------------- | -------- | ------ | ------- | ------ | -------- | ------ | ------ | ------ | ------ | --- | --- |
| CoAtNet-0         | 0.9942 [0.9825, 1.0000] | 0.9953        | 0.9938   | 0.9876 | 0.9994  | 0.9997 | 0.0412   | 0.0200 | 0.0275 | 0.9907 | 1.0000 | 1   | 0   |
| EfficientNetB0    | 0.9942 [0.9825, 1.0000] | 0.9922        | 0.9937   | 0.9876 | 0.9991  | 0.9995 | 0.0420   | 0.0241 | 0.0243 | 1.0000 | 0.9844 | 0   | 1   |
| ViT-B/16          | 0.9942 [0.9825, 1.0000] | 0.9922        | 0.9937   | 0.9876 | 0.9984  | 0.9990 | 0.0508   | 0.0215 | 0.0207 | 1.0000 | 0.9844 | 0   | 1   |
| Swin-Base         | 0.9883 [0.9708, 1.0000] | 0.9907        | 0.9876   | 0.9755 | 0.9997  | 0.9998 | 0.0348   | 0.0183 | 0.0152 | 0.9813 | 1.0000 | 2   | 0   |
| InceptionV3       | 0.9825 [0.9591, 1.0000] | 0.9797        | 0.9812   | 0.9625 | 0.9991  | 0.9995 | 0.0407   | 0.0245 | 0.0228 | 0.9907 | 0.9688 | 1   | 2   |
| ResNet50          | 0.9825 [0.9591, 1.0000] | 0.9828        | 0.9813   | 0.9627 | 0.9988  | 0.9993 | 0.0474   | 0.0276 | 0.0157 | 0.9813 | 0.9844 | 2   | 1   |
| DenseNet121       | 0.9708 [0.9415, 0.9942] | 0.9641        | 0.9685   | 0.9376 | 0.9984  | 0.9991 | 0.0534   | 0.0346 | 0.0252 | 0.9907 | 0.9375 | 1   | 4   |
| MobileNetV3-Large | 0.9708 [0.9415, 0.9942] | 0.9704        | 0.9689   | 0.9378 | 0.9978  | 0.9987 | 0.0714   | 0.0415 | 0.0326 | 0.9720 | 0.9688 | 3   | 2   |
| VGG16             | 0.9708 [0.9415, 0.9942] | 0.9704        | 0.9689   | 0.9378 | 0.9974  | 0.9985 | 0.0724   | 0.0433 | 0.0296 | 0.9720 | 0.9688 | 3   | 2   |

Sensitivity is the recall of the tumour class and specificity the recall of the tumour-negative class, at a threshold of 0.5. At the mean validation-tuned thresholds (0.4289 to 0.8384, column `threshold`) sensitivity and specificity are given in the `tuned_*` columns; for example EfficientNetB0 0.9626 and 0.9844. `test_per_class.csv` gives precision, sensitivity, specificity, F1, ROC AUC and average precision per class and model.

- **Statistical comparison.** Exact McNemar tests between every pair of fold ensembles, Holm-corrected: every adjusted p-value is 1.0 (`mcnemar_holm_p.csv`). Bootstrap 95% CI of each model's accuracy gap to CoAtNet-0 (`gap_to_leader.csv`): only VGG16 and MobileNetV3-Large (gap 0.0234, CI [0.0058, 0.0468]) are separable from the leader; the others' intervals include zero.

![Binary McNemar p-values](binary-detection/plots/08_mcnemar.png)

_Holm-corrected McNemar p-values between models._

- **Confusion matrices, ROC and PR curves.**

![Binary confusion matrices](binary-detection/plots/09_confusion_matrices.png)

_Test confusion matrices of every fold ensemble, counts with row percentages._

![Binary ROC and PR curves](binary-detection/plots/10_roc_pr_curves.png)

_ROC and precision-recall curves of every fold ensemble for the tumour class._

- **Calibration before and after temperature scaling** (`calibration_ece.csv`, test ECE of the fold ensembles):

| Model             | Temperature-scaled | Uncalibrated |
| ----------------- | ------------------ | ------------ |
| CoAtNet-0         | 0.0275             | 0.0207       |
| DenseNet121       | 0.0252             | 0.0233       |
| EfficientNetB0    | 0.0243             | 0.0231       |
| InceptionV3       | 0.0228             | 0.0151       |
| MobileNetV3-Large | 0.0326             | 0.0146       |
| ResNet50          | 0.0157             | 0.0173       |
| Swin-Base         | 0.0152             | 0.0148       |
| VGG16             | 0.0296             | 0.0179       |
| ViT-B/16          | 0.0207             | 0.0222       |

Temperature scaling lowered ECE only for ResNet50 and ViT-B/16. In fold 4, the fitted temperature of CoAtNet-0, EfficientNetB0, InceptionV3, MobileNetV3-Large, ResNet50 and Swin-Base is between 0.0498 and 0.0682 (`run_metrics.csv`), at or near the lower bound exp(-3), about 0.0498, of the temperature search. For the first five of these, fold 4 has the highest test log-loss of the model's five folds (for example EfficientNetB0 0.543 against 0.038 to 0.088 for its other folds).

![Binary calibration](binary-detection/plots/11_calibration.png)

_Reliability diagrams of every fold ensemble, uncalibrated and temperature-scaled._

- **Accuracy against distance to the development pool** (`accuracy_by_nearest_dev_distance.csv`, by Hamming distance of each test image's nearest development image):

| Model             | 9-12   | 13-16  | 17-20  |
| ----------------- | ------ | ------ | ------ |
| CoAtNet-0         | 0.9886 | 1.0000 | 1.0000 |
| DenseNet121       | 0.9773 | 0.9589 | 1.0000 |
| EfficientNetB0    | 1.0000 | 0.9863 | 1.0000 |
| InceptionV3       | 1.0000 | 0.9589 | 1.0000 |
| MobileNetV3-Large | 0.9545 | 0.9863 | 1.0000 |
| ResNet50          | 0.9886 | 0.9726 | 1.0000 |
| Swin-Base         | 1.0000 | 0.9726 | 1.0000 |
| VGG16             | 0.9659 | 0.9726 | 1.0000 |
| ViT-B/16          | 1.0000 | 0.9863 | 1.0000 |

![Binary accuracy by distance](binary-detection/plots/12_accuracy_by_distance.png)

_Test accuracy by distance to the nearest development image._

### 7. Interpretability

- **Grad-CAM, attention rollout and outside-head energy.** For each model, the fold with the lowest validation loss is reloaded and Grad-CAM is computed for every test image; ViT-B/16 also gets attention rollout. The share of map energy outside the head mask (Otsu threshold, closing, largest component, hole filling), mean +/- sd (`outside_head_energy.csv`):

| Model             | Tumour-negative   | Tumour            |
| ----------------- | ----------------- | ----------------- |
| ResNet50          | 0.1215 +/- 0.2192 | 0.0787 +/- 0.2002 |
| VGG16             | 0.1526 +/- 0.2115 | 0.0737 +/- 0.1839 |
| DenseNet121       | 0.1897 +/- 0.2293 | 0.1764 +/- 0.1834 |
| EfficientNetB0    | 0.2868 +/- 0.2107 | 0.0720 +/- 0.1800 |
| MobileNetV3-Large | 0.1424 +/- 0.2305 | 0.0820 +/- 0.1630 |
| InceptionV3       | 0.1458 +/- 0.2342 | 0.1474 +/- 0.1873 |
| ViT-B/16 Grad-CAM | 0.4255 +/- 0.2375 | 0.2435 +/- 0.2643 |
| ViT-B/16 rollout  | 0.4211 +/- 0.1786 | 0.2850 +/- 0.1626 |
| Swin-Base         | 0.1586 +/- 0.2461 | 0.1845 +/- 0.1949 |
| CoAtNet-0         | 0.1391 +/- 0.2479 | 0.1882 +/- 0.2004 |
| Uniform map       | 0.3558            | 0.3278            |

![Binary outside-head energy](binary-detection/plots/14_outside_head_energy.png)

_Share of Grad-CAM energy outside the head per model and class; the black mark is a uniform map._

Grad-CAM galleries (most confident correct and most confident wrong test predictions per model):

![Binary Grad-CAM ResNet50](binary-detection/plots/13_gradcam_resnet50.png)
![Binary Grad-CAM VGG16](binary-detection/plots/13_gradcam_vgg16.png)
![Binary Grad-CAM DenseNet121](binary-detection/plots/13_gradcam_densenet121.png)
![Binary Grad-CAM EfficientNetB0](binary-detection/plots/13_gradcam_efficientnetb0.png)
![Binary Grad-CAM MobileNetV3-Large](binary-detection/plots/13_gradcam_mobilenetv3-large.png)
![Binary Grad-CAM InceptionV3](binary-detection/plots/13_gradcam_inceptionv3.png)
![Binary Grad-CAM ViT-B/16](binary-detection/plots/13_gradcam_vit-b-16.png)
![Binary Grad-CAM Swin-Base](binary-detection/plots/13_gradcam_swin-base.png)
![Binary Grad-CAM CoAtNet-0](binary-detection/plots/13_gradcam_coatnet-0.png)

- **Occlusion maps.** Model-agnostic occlusion sensitivity (32 px patch, stride 16) on the same 16 test images for every model; share outside the head (`occlusion_outside_head.csv`):

| Model             | Outside-head share |
| ----------------- | ------------------ |
| ResNet50          | 0.1161 +/- 0.1324  |
| VGG16             | 0.0072 +/- 0.0162  |
| DenseNet121       | 0.1347 +/- 0.1601  |
| EfficientNetB0    | 0.1942 +/- 0.1974  |
| MobileNetV3-Large | 0.1109 +/- 0.1700  |
| InceptionV3       | 0.1674 +/- 0.2052  |
| ViT-B/16          | 0.1547 +/- 0.1597  |
| Swin-Base         | 0.1820 +/- 0.1539  |
| CoAtNet-0         | 0.0888 +/- 0.1451  |

![Binary occlusion maps](binary-detection/plots/15_occlusion_maps.png)

_Occlusion maps for every model on the same test images._

- **Masking test (inference only).** Each fold's checkpoint scored on test images with the head set to zero (background only) or the background set to zero (head only); mean over folds (`masking_test.csv`):

| Model             | Full image | Head only | Background only |
| ----------------- | ---------- | --------- | --------------- |
| CoAtNet-0         | 0.9754     | 0.9637    | 0.4444          |
| DenseNet121       | 0.9719     | 0.9439    | 0.5181          |
| EfficientNetB0    | 0.9778     | 0.9556    | 0.4982          |
| InceptionV3       | 0.9789     | 0.9450    | 0.6573          |
| MobileNetV3-Large | 0.9637     | 0.9368    | 0.5333          |
| ResNet50          | 0.9684     | 0.9602    | 0.4924          |
| Swin-Base         | 0.9778     | 0.9684    | 0.5170          |
| VGG16             | 0.9591     | 0.9474    | 0.4994          |
| ViT-B/16          | 0.9754     | 0.9567    | 0.4503          |

![Binary masking test](binary-detection/plots/16_masking_test.png)

- **Shortcut test: retraining on background-only images.** Each model was trained (fold 0) and tested on images whose head region is zero. Majority-class rate 0.6257; one-sided exact binomial test, Holm-corrected (`shortcut_test.csv`):

| Model             | Background-only accuracy | Background-only AUC | Holm p | Shortcut detected |
| ----------------- | ------------------------ | ------------------- | ------ | ----------------- |
| ResNet50          | 0.8129                   | 0.8657              | 0.0000 | Yes               |
| VGG16             | 0.7485                   | 0.8359              | 0.0013 | Yes               |
| DenseNet121       | 0.7953                   | 0.8578              | 0.0000 | Yes               |
| EfficientNetB0    | 0.7778                   | 0.8584              | 0.0001 | Yes               |
| MobileNetV3-Large | 0.6784                   | 0.7821              | 0.1772 | No                |
| InceptionV3       | 0.7953                   | 0.8606              | 0.0000 | Yes               |
| ViT-B/16          | 0.6550                   | 0.7411              | 0.2396 | No                |
| Swin-Base         | 0.8304                   | 0.8893              | 0.0000 | Yes               |
| CoAtNet-0         | 0.7953                   | 0.8756              | 0.0000 | Yes               |

![Binary shortcut test](binary-detection/plots/17_shortcut_test.png)

_Accuracy of models trained and tested on background-only images; red bars are above chance._

### 8. Efficiency

Measured on one T4 under fp16 autocast: latency is 100 timed single-image passes after 10 warm-up passes, throughput is batch 32 over 20 passes after 3 warm-up, both repeated three times. Training cost is the mean over folds (`efficiency.csv`):

| Model             | Latency (ms)     | Throughput (img/s) | Train min/run | Train img/s | Peak train memory (GB) | Accuracy | Acc. per M params |
| ----------------- | ---------------- | ------------------ | ------------- | ----------- | ---------------------- | -------- | ----------------- |
| ResNet50          | 9.573 +/- 0.376  | 777.2 +/- 6.1      | 1.150         | 235.2       | 5.616                  | 0.982    | 4.179             |
| VGG16             | 7.722 +/- 0.422  | 386.5 +/- 4.8      | 2.173         | 124.4       | 9.876                  | 0.971    | 0.723             |
| DenseNet121       | 23.584 +/- 1.534 | 656.6 +/- 22.9     | 1.458         | 185.4       | 4.478                  | 0.971    | 13.958            |
| EfficientNetB0    | 13.129 +/- 0.550 | 1393.3 +/- 115.8   | 1.994         | 136.4       | 3.093                  | 0.994    | 24.799            |
| MobileNetV3-Large | 10.552 +/- 1.090 | 2301.1 +/- 292.9   | 1.488         | 182.3       | 2.088                  | 0.971    | 23.095            |
| InceptionV3       | 17.632 +/- 0.788 | 511.6 +/- 5.8      | 1.735         | 155.8       | 5.849                  | 0.982    | 4.509             |
| ViT-B/16          | 10.952 +/- 1.039 | 272.2 +/- 2.8      | 2.908         | 93.0        | 5.960                  | 0.994    | 1.159             |
| Swin-Base         | 36.294 +/- 1.234 | 197.6 +/- 2.9      | 4.296         | 62.9        | 6.448                  | 0.988    | 1.139             |
| CoAtNet-0         | 16.533 +/- 1.065 | 513.1 +/- 6.9      | 2.281         | 118.7       | 11.386                 | 0.994    | 3.728             |

The two workers shared four CPU cores, so training throughput is a lower bound (notebook text).

![Binary accuracy against latency](binary-detection/plots/18_accuracy_vs_latency.png)

- **External validation.** Output: `No external evaluation: EXTERNAL_FOLDERS is empty or every external image duplicated the pool.`

### 9. Export and summary

- **Web-app checkpoint.** Exports the EfficientNetB0 fold with the lowest validation loss and checks that it reloads into the web app's architecture and reproduces the stored logits. Output: `Exported /kaggle/working/weights/deploy/EfficientNetB0.pth (16.3 MB); reload check passed`. See [Checkpoints used by the web app](#checkpoints-used-by-the-web-app).
- **Summary.** `summary_table.csv` (the headline columns of the tables above) and `run_summary.json`: 3,253 images in the pool, 1,185 after deduplication, 0 label-noise exclusions, 171 test images, leader CoAtNet-0, 54 of 54 jobs. Log: `Leader: CoAtNet-0; 54/54 jobs; notebook time 72 min`.
- **Package the outputs.** Zips `/kaggle/working/` (except the image cache) for download.

## Classification notebook

Source: [`classification.ipynb`](classification.ipynb). Outputs: [`classification/`](classification/).

The notebook runs the same code as the binary notebook with task `multiclass`, classes `Glioma`, `Meningioma`, `Pituitary`, and the same nine models and settings. Only the data, the outputs and the differences below are described.

### 1. Setup

Same cells and settings as the binary notebook. Output: `Task: multiclass | classes: ['Glioma', 'Meningioma', 'Pituitary'] | models: 9 | folds: 5`.

### 2. Data

Sources: Bhuvaji et al. `sartajbhuvaji/brain-tumor-classification-mri` (training and testing folders merged, tumour-negative folder excluded) and the Cheng et al. figshare collection in the JPEG conversion `moundir8/jun-cheng-brain-tumor-mri-dataset-jpg`. Output: `Pool: 5828 images | external: 0`, `Cached 5828 images in 68s (2.39 GB); unreadable: 0`.

| Source (`provenance.csv`) | Glioma | Meningioma | Pituitary | Total | Distinct sizes | Modal size | Modal share | Median crop fraction |
| ------------------------- | ------ | ---------- | --------- | ----- | -------------- | ---------- | ----------- | -------------------- |
| Bhuvaji testing           | 100    | 115        | 74        | 289   | 184            | 512x512    | 0.111       | 0.846                |
| Bhuvaji training          | 826    | 822        | 827       | 2,475 | 137            | 512x512    | 0.931       | 0.671                |
| Cheng                     | 1,426  | 708        | 930       | 3,064 | 2              | 512x512    | 0.995       | 0.635                |
| Total                     | 2,352  | 1,645      | 1,831     | 5,828 |                |            |             |                      |

![Classification dataset overview](classification/plots/01_dataset_overview.png)

![Classification samples, crops and head masks](classification/plots/02_samples_crop_mask.png)

### 3. Deduplication

Output: `Cross-label pairs within Hamming 5: 31 (8 label noise, 23 hash false positives)`. All 31 pairs are glioma against meningioma; the 8 label-noise pairs (NCC at least 0.9997) are all inside the Bhuvaji collection and both copies are excluded; the 23 hash false positives (NCC 0.67 to 0.86) are kept (`cross_label_pairs.csv`).

| Class (`deduplication_summary.csv`) | Before | Duplicates removed | Label noise excluded | After | Removed |
| ----------------------------------- | ------ | ------------------ | -------------------- | ----- | ------- |
| Glioma                              | 2,352  | 342                | 2                    | 2,008 | 14.63%  |
| Meningioma                          | 1,645  | 329                | 2                    | 1,314 | 20.12%  |
| Pituitary                           | 1,831  | 423                | 0                    | 1,408 | 23.10%  |
| Total                               | 5,828  | 1,094              | 4                    | 4,730 | 18.84%  |

`split_roles.csv` and `run_summary.json` count the same outcome per image instead: 1,088 images removed as duplicates and 10 excluded as label noise (2 glioma, 8 meningioma); both tables total 4,730 kept images.

| Removed image source (`deduplication_by_source.csv`) | Kind          | Glioma | Meningioma | Pituitary | All   |
| ---------------------------------------------------- | ------------- | ------ | ---------- | --------- | ----- |
| Bhuvaji                                              | within-source | 143    | 229        | 236       | 608   |
| Cheng                                                | cross-source  | 0      | 3          | 0         | 3     |
| Cheng                                                | within-source | 199    | 97         | 187       | 483   |
| All                                                  |               | 342    | 329        | 423       | 1,094 |

| Tolerance (`deduplication_threshold_sweep.csv`) | 0, 1  | 2, 3  | 4, 5  | 6, 7  | 8, 9  | 10, 11 | 12    |
| ----------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ------ | ----- |
| Kept                                            | 5,572 | 5,337 | 4,734 | 3,928 | 3,098 | 2,286  | 1,611 |
| Glioma                                          | 2,303 | 2,221 | 2,010 | 1,653 | 1,277 | 919    | 627   |
| Meningioma                                      | 1,518 | 1,453 | 1,316 | 1,130 | 935   | 704    | 506   |
| Pituitary                                       | 1,751 | 1,663 | 1,408 | 1,145 | 886   | 663    | 478   |

![Classification deduplication](classification/plots/03_deduplication.png)

![Classification cross-label pairs](classification/plots/04_cross_label_pairs.png)

_Cross-label pairs within the tolerance, label-noise pairs first._

### 4. Group-aware split and leakage audit

Output: `4730 images in 1867 groups; largest group 1431 images; 596 groups with more than one image` followed by `Warning: one group holds more than 5% of the pool; consider a smaller GROUP_TOLERANCE.` The run kept `GROUP_TOLERANCE = 8`. The large group makes the split very uneven:

| Role (`split_roles.csv`) | Glioma | Meningioma | Pituitary | Total |
| ------------------------ | ------ | ---------- | --------- | ----- |
| Development (5 folds)    | 2,007  | 1,301      | 1,377     | 4,685 |
| Test                     | 1      | 13         | 31        | 45    |
| Removed duplicate        | 342    | 323        | 423       | 1,088 |
| Label noise              | 2      | 8          | 0         | 10    |
| Total                    | 2,352  | 1,645      | 1,831     | 5,828 |

| Fold (`fold_distribution.csv`) | 0   | 1   | 2   | 3   | 4     | All   |
| ------------------------------ | --- | --- | --- | --- | ----- | ----- |
| Glioma                         | 61  | 275 | 246 | 285 | 1,140 | 2,007 |
| Meningioma                     | 40  | 220 | 204 | 225 | 612   | 1,301 |
| Pituitary                      | 53  | 247 | 247 | 366 | 464   | 1,377 |
| All                            | 154 | 742 | 697 | 876 | 2,216 | 4,685 |

The test set has 45 images, one of them glioma. Because the fold sizes differ, the number of optimiser steps per run differs by fold: 1,420 (fold 0), 1,240, 1,260, 1,200 and 780 (fold 4), the same for every model within a fold (`run_metrics.csv`). Leakage audit: 0 test images near a development image by any criterion.

![Classification groups and leakage](classification/plots/05_groups_and_leakage.png)

![Classification augmentation](classification/plots/06_augmentation.png)

### 5. Training

Model profiles match the binary table except for the three-output heads (for example EfficientNetB0 4.011 M, ResNet50 23.514 M, VGG16 134.273 M parameters; `model_profiles.csv`). Output: `54 jobs, 0 already finished; launching 2 workers`, `Training stage finished in 264.6 min; 54/54 jobs complete`.

### 6. Results

Fold summary (`fold_summary.csv`, calibrated, mean +/- sd over 5 folds):

| Model             | Accuracy          | Balanced acc.     | Macro F1          | ROC AUC           | ECE               | Log-loss          |
| ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| InceptionV3       | 0.9956 +/- 0.0099 | 0.9949 +/- 0.0115 | 0.9751 +/- 0.0557 | 1.0000 +/- 0.0000 | 0.0088 +/- 0.0097 | 0.0139 +/- 0.0207 |
| Swin-Base         | 0.9956 +/- 0.0099 | 0.9949 +/- 0.0115 | 0.9751 +/- 0.0557 | 1.0000 +/- 0.0000 | 0.0109 +/- 0.0123 | 0.0127 +/- 0.0157 |
| ViT-B/16          | 0.9911 +/- 0.0122 | 0.9897 +/- 0.0140 | 0.9502 +/- 0.0682 | 0.9995 +/- 0.0011 | 0.0147 +/- 0.0081 | 0.0397 +/- 0.0460 |
| DenseNet121       | 0.9911 +/- 0.0122 | 0.9927 +/- 0.0113 | 0.9715 +/- 0.0542 | 0.9998 +/- 0.0004 | 0.0207 +/- 0.0214 | 0.0342 +/- 0.0424 |
| CoAtNet-0         | 0.9867 +/- 0.0122 | 0.9876 +/- 0.0129 | 0.9467 +/- 0.0653 | 1.0000 +/- 0.0000 | 0.0211 +/- 0.0124 | 0.0319 +/- 0.0258 |
| ResNet50          | 0.9867 +/- 0.0122 | 0.9876 +/- 0.0129 | 0.9269 +/- 0.0668 | 0.9998 +/- 0.0004 | 0.0175 +/- 0.0108 | 0.0445 +/- 0.0376 |
| MobileNetV3-Large | 0.9778 +/- 0.0000 | 0.9744 +/- 0.0000 | 0.8756 +/- 0.0000 | 0.9998 +/- 0.0004 | 0.0236 +/- 0.0033 | 0.0756 +/- 0.0378 |
| VGG16             | 0.9733 +/- 0.0243 | 0.9752 +/- 0.0167 | 0.8846 +/- 0.0730 | 0.9989 +/- 0.0021 | 0.0415 +/- 0.0351 | 0.1106 +/- 0.0899 |
| EfficientNetB0    | 0.9733 +/- 0.0365 | 0.9782 +/- 0.0239 | 0.9061 +/- 0.0943 | 0.9989 +/- 0.0017 | 0.0245 +/- 0.0235 | 0.1652 +/- 0.2584 |

![Classification training curves](classification/plots/07_training_curves.png)

Fold-ensemble test results on the 45 test images (`test_leaderboard.csv`):

| Model             | Accuracy (95% CI)       | Balanced acc. | Macro precision | Macro F1 (95% CI)       | MCC    | ROC AUC | Log-loss | ECE    |
| ----------------- | ----------------------- | ------------- | --------------- | ----------------------- | ------ | ------- | -------- | ------ |
| ResNet50          | 1.0000 [1.0000, 1.0000] | 1.0000        | 1.0000          | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000  | 0.0257   | 0.0222 |
| DenseNet121       | 1.0000 [1.0000, 1.0000] | 1.0000        | 1.0000          | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000  | 0.0237   | 0.0215 |
| InceptionV3       | 1.0000 [1.0000, 1.0000] | 1.0000        | 1.0000          | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000  | 0.0093   | 0.0088 |
| ViT-B/16          | 1.0000 [1.0000, 1.0000] | 1.0000        | 1.0000          | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000  | 0.0220   | 0.0182 |
| Swin-Base         | 1.0000 [1.0000, 1.0000] | 1.0000        | 1.0000          | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000  | 0.0114   | 0.0109 |
| CoAtNet-0         | 1.0000 [1.0000, 1.0000] | 1.0000        | 1.0000          | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000  | 0.0272   | 0.0253 |
| VGG16             | 0.9778 [0.9333, 1.0000] | 0.9744        | 0.8333          | 0.8756 [0.7565, 1.0000] | 0.9515 | 1.0000  | 0.0664   | 0.0371 |
| EfficientNetB0    | 0.9778 [0.9333, 1.0000] | 0.9744        | 0.8333          | 0.8756 [0.7565, 1.0000] | 0.9515 | 1.0000  | 0.0365   | 0.0277 |
| MobileNetV3-Large | 0.9778 [0.9333, 1.0000] | 0.9744        | 0.8333          | 0.8756 [0.7565, 1.0000] | 0.9515 | 1.0000  | 0.0583   | 0.0285 |

The single error of VGG16, EfficientNetB0 and MobileNetV3-Large is one of the 13 meningioma images predicted as glioma (`test_per_class.csv`: glioma precision 0.5, meningioma sensitivity 0.9231). Every Holm-corrected McNemar p-value is 1.0 and no model is separable from the leader, ResNet50 (`mcnemar_holm_p.csv`, `gap_to_leader.csv`).

![Classification McNemar p-values](classification/plots/08_mcnemar.png)

![Classification confusion matrices](classification/plots/09_confusion_matrices.png)

![Classification ROC and PR curves](classification/plots/10_roc_pr_curves.png)

| Model (`calibration_ece.csv`) | Temperature-scaled ECE | Uncalibrated ECE |
| ----------------------------- | ---------------------- | ---------------- |
| CoAtNet-0                     | 0.0253                 | 0.0259           |
| DenseNet121                   | 0.0215                 | 0.0165           |
| EfficientNetB0                | 0.0277                 | 0.0263           |
| InceptionV3                   | 0.0088                 | 0.0045           |
| MobileNetV3-Large             | 0.0285                 | 0.0204           |
| ResNet50                      | 0.0222                 | 0.0125           |
| Swin-Base                     | 0.0109                 | 0.0044           |
| VGG16                         | 0.0371                 | 0.0355           |
| ViT-B/16                      | 0.0182                 | 0.0114           |

![Classification calibration](classification/plots/11_calibration.png)

Accuracy by distance to the development pool (`accuracy_by_nearest_dev_distance.csv`): 1.0000 for every model in the 13-16 bin; in the 9-12 bin 0.9773 for EfficientNetB0, MobileNetV3-Large and VGG16 and 1.0000 for the others.

![Classification accuracy by distance](classification/plots/12_accuracy_by_distance.png)

### 7. Interpretability

Outside-head Grad-CAM energy, mean +/- sd (`outside_head_energy.csv`; the glioma column has one test image, so its sd is 0):

| Model             | Glioma | Meningioma        | Pituitary         |
| ----------------- | ------ | ----------------- | ----------------- |
| ResNet50          | 0.0979 | 0.0561 +/- 0.0738 | 0.1058 +/- 0.2021 |
| VGG16             | 0.0258 | 0.1212 +/- 0.0655 | 0.1148 +/- 0.1630 |
| DenseNet121       | 0.1081 | 0.1481 +/- 0.0416 | 0.1495 +/- 0.1525 |
| EfficientNetB0    | 0.0330 | 0.0673 +/- 0.0417 | 0.0905 +/- 0.1564 |
| MobileNetV3-Large | 0.0381 | 0.0576 +/- 0.0556 | 0.1343 +/- 0.1356 |
| InceptionV3       | 0.1018 | 0.1356 +/- 0.0615 | 0.1292 +/- 0.1776 |
| ViT-B/16 Grad-CAM | 0.3262 | 0.0773 +/- 0.0454 | 0.2184 +/- 0.2285 |
| ViT-B/16 rollout  | 0.2624 | 0.3501 +/- 0.0469 | 0.3579 +/- 0.1205 |
| Swin-Base         | 0.0793 | 0.1234 +/- 0.0456 | 0.1030 +/- 0.1583 |
| CoAtNet-0         | 0.4172 | 0.1520 +/- 0.0636 | 0.1223 +/- 0.1837 |
| Uniform map       | 0.2129 | 0.2679            | 0.2671            |

![Classification outside-head energy](classification/plots/14_outside_head_energy.png)

![Classification Grad-CAM ResNet50](classification/plots/13_gradcam_resnet50.png)
![Classification Grad-CAM VGG16](classification/plots/13_gradcam_vgg16.png)
![Classification Grad-CAM DenseNet121](classification/plots/13_gradcam_densenet121.png)
![Classification Grad-CAM EfficientNetB0](classification/plots/13_gradcam_efficientnetb0.png)
![Classification Grad-CAM MobileNetV3-Large](classification/plots/13_gradcam_mobilenetv3-large.png)
![Classification Grad-CAM InceptionV3](classification/plots/13_gradcam_inceptionv3.png)
![Classification Grad-CAM ViT-B/16](classification/plots/13_gradcam_vit-b-16.png)
![Classification Grad-CAM Swin-Base](classification/plots/13_gradcam_swin-base.png)
![Classification Grad-CAM CoAtNet-0](classification/plots/13_gradcam_coatnet-0.png)

Occlusion outside-head share on 17 test images (`occlusion_outside_head.csv`): ResNet50 0.0791, VGG16 0.0788, DenseNet121 0.1103, EfficientNetB0 0.0655, MobileNetV3-Large 0.0709, InceptionV3 0.0993, ViT-B/16 0.1066, Swin-Base 0.1219, CoAtNet-0 0.1241.

![Classification occlusion maps](classification/plots/15_occlusion_maps.png)

| Model (`masking_test.csv`) | Full image | Head only | Background only |
| -------------------------- | ---------- | --------- | --------------- |
| CoAtNet-0                  | 0.9867     | 0.9822    | 0.1556          |
| DenseNet121                | 0.9911     | 0.9600    | 0.2756          |
| EfficientNetB0             | 0.9733     | 0.9556    | 0.1333          |
| InceptionV3                | 0.9956     | 0.9467    | 0.2178          |
| MobileNetV3-Large          | 0.9778     | 0.9511    | 0.3378          |
| ResNet50                   | 0.9867     | 0.9467    | 0.2844          |
| Swin-Base                  | 0.9956     | 0.9956    | 0.1689          |
| VGG16                      | 0.9733     | 0.9422    | 0.2400          |
| ViT-B/16                   | 0.9911     | 0.9867    | 0.1156          |

![Classification masking test](classification/plots/16_masking_test.png)

Shortcut test (`shortcut_test.csv`, majority-class rate 0.6889): every model trained on background-only images scores above chance after Holm correction.

| Model             | Background-only accuracy | Background-only AUC | Holm p |
| ----------------- | ------------------------ | ------------------- | ------ |
| ResNet50          | 0.9333                   | 0.9968              | 0.0006 |
| VGG16             | 0.9778                   | 1.0000              | 0.0000 |
| DenseNet121       | 0.8667                   | 0.9904              | 0.0104 |
| EfficientNetB0    | 0.8889                   | 1.0000              | 0.0048 |
| MobileNetV3-Large | 0.9333                   | 1.0000              | 0.0006 |
| InceptionV3       | 0.9778                   | 0.9984              | 0.0000 |
| ViT-B/16          | 0.9111                   | 0.9717              | 0.0020 |
| Swin-Base         | 0.9111                   | 0.9976              | 0.0020 |
| CoAtNet-0         | 0.8222                   | 0.9752              | 0.0337 |

![Classification shortcut test](classification/plots/17_shortcut_test.png)

### 8. Efficiency

| Model (`efficiency.csv`) | Latency (ms)     | Throughput (img/s) | Train min/run | Peak train memory (GB) | Accuracy |
| ------------------------ | ---------------- | ------------------ | ------------- | ---------------------- | -------- |
| ResNet50                 | 9.074 +/- 0.363  | 716.8 +/- 6.9      | 5.319         | 4.076                  | 1.000    |
| VGG16                    | 8.015 +/- 0.395  | 332.1 +/- 6.2      | 9.496         | 9.869                  | 0.978    |
| DenseNet121              | 22.805 +/- 1.221 | 607.7 +/- 17.7     | 6.865         | 4.478                  | 1.000    |
| EfficientNetB0           | 12.805 +/- 0.796 | 1265.7 +/- 50.3    | 5.078         | 3.039                  | 0.978    |
| MobileNetV3-Large        | 9.985 +/- 0.923  | 2183.1 +/- 234.4   | 3.490         | 2.081                  | 0.978    |
| InceptionV3              | 16.690 +/- 1.251 | 438.2 +/- 6.6      | 7.974         | 5.263                  | 1.000    |
| ViT-B/16                 | 10.255 +/- 1.097 | 241.0 +/- 3.0      | 15.020        | 5.968                  | 1.000    |
| Swin-Base                | 34.649 +/- 1.513 | 173.1 +/- 2.8      | 21.427        | 6.446                  | 1.000    |
| CoAtNet-0                | 15.875 +/- 0.726 | 448.4 +/- 6.3      | 9.457         | 10.423                 | 1.000    |

![Classification accuracy against latency](classification/plots/18_accuracy_vs_latency.png)

External validation: none configured.

### 9. Export and summary

Output: `Exported /kaggle/working/weights/deploy/EfficientNetB0.pth (16.3 MB); reload check passed` and `Leader: ResNet50; 54/54 jobs; notebook time 274 min`. `run_summary.json`: 5,828 images in the pool, 4,730 after deduplication, 10 label-noise exclusions, 45 test images.

## Segmentation notebook

Source: [`segmentation.ipynb`](segmentation.ipynb). Outputs: [`segmentation/`](segmentation/).

The notebook trains three 3D segmentation networks on BraTS 2020, fuses them with a region-adaptive hybrid ensemble, and tests every model on the 56 held-out BraTS 2020 patients (internal test) and on 60 BraTS-Africa patients (external test, never used for training or selection). It replaced an earlier six-network benchmark (UNet, nnU-Net, VNet, UNETR, Swin UNETR, TransBTS at 128^3); that run's outputs were overwritten by this one and are only in Git history (commit `6902ac6` and earlier). The validation and test patients are the same 56 + 56 IDs as that benchmark.

The run documented below is the sixth Kaggle attempt (2026-10-24, notebook time 547.9 min). Earlier attempts stopped in the model-profile cell (MONAI's Swin UNETR cannot be built on the meta device), on an old pinned Kaggle image (PyTorch older than 2.8), in a silent preprocessing stretch (a fork-based pool, now `spawn`), on a Swin UNETR out-of-memory error at micro-batch 2 (now micro-batch 1 with 2 accumulation steps), and in the Seg-Grad-CAM attribution cell (now layer-only gradients with members spread over both GPUs). Their outputs were not kept.

### 1. Setup

- **Install and import.** Installs MONAI 1.6.1 (`--no-deps`), `einops` and `nibabel` only if missing and stops if PyTorch is older than 2.8. Output: `torch 2.11.0+cu128 | monai 1.6.1 | nibabel 5.4.2 | numpy 2.1.3`.
- **Configuration.** `SEEDS = [42]`, `PATCH = 128` (training patch and sliding window at native 1 mm), `SOURCE = 160` (cube read before rotation and zoom), `EPOCHS = {"nnU-Net": 60, "SegResNet": 60, "Swin UNETR": 30}`, `ITERS_PER_EPOCH = 250`, `BATCH = 2`, `FG_PROBABILITY = 0.33` (share of tumour-centred patches), `VAL_EVERY = 5`, `SW_OVERLAP = 0.5`, `TTA = True` (8 axis flips), `ET_CANDIDATES = [0, 50, 100, 200, 300, 500, 800]`, `FUSION_STEP = 0.1`, `FUSION_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]`, `QC_FAIL_DICE = 0.80`, `CAM_MARGIN = 5`, `SESSION_HOURS = 12`, `POST_TRAINING_HOURS = 1.0`, `SCORE_TIMEOUT = 900`. Output: `Members: ['nnU-Net', 'SegResNet', 'Swin UNETR'] | seeds: [42] | epochs: {...} x 250 steps | patch 128^3`.
- **Hardware check.** Two Tesla T4 GPUs (14.6 GB each), 4 CPU cores, 33.7 GB RAM (31.4 GB available), 20.9 GB free in `/kaggle/working`.
- **Shared library, worker and launcher.** The cells write `src/volume_lib.py` (case discovery for both naming conventions, brain cropping, foreground z-score, region bit mask, patch sampling, GPU augmentation, the three MONAI networks, Dice + BCE loss with deep supervision, Gaussian sliding-window inference with flip TTA, region hierarchy and ET filter, metrics, fusion search, Seg-Grad-CAM), `src/volume_worker.py` (one training worker per GPU) and `src/shared_lib.py` (launcher, live monitor, Holm correction).
- **Model profiles** (`model_profiles.csv`; GFLOPs for one 128^3 patch):

| Model      | Family      | Parameters (M) | GFLOPs per patch | Optimiser | LR   | Weight decay | Warm-up epochs | Grad. clip | Micro-batch | Deep-supervision weights | Seg-Grad-CAM layer   |
| ---------- | ----------- | -------------- | ---------------- | --------- | ---- | ------------ | -------------- | ---------- | ----------- | ------------------------ | -------------------- |
| nnU-Net    | CNN         | 31.422         | 968.1            | SGD       | 1e-2 | 3e-5         | 0              | 12.0       | 2           | 1.0, 0.5, 0.25           | `upsamples.2`        |
| SegResNet  | CNN         | 19.917         | 1218.2           | AdamW     | 2e-4 | 1e-5         | 2              | 1.0        | 2           | 1.0, 0.5, 0.25           | `up_layers.0.blocks` |
| Swin UNETR | Transformer | 72.767         | 1665.6           | AdamW     | 1e-4 | 1e-5         | 2              | 1.0        | 1           | 1.0                      | `decoder3`           |

nnU-Net is MONAI's residual `DynUNet` (six stages, 32 to 320 channels), SegResNet is `SegResNetDS` (32 initial filters, blocks 1-2-2-4), and Swin UNETR is Swin UNETR-v2 (feature size 48, gradient checkpointing).

### 2. Data

- **Find the cases.** BraTS 2020 (`awsaf49/brats20-dataset-training-validation`) and BraTS-Africa (`aiocta/brats2023-ssa-training-dataset`, ET label 3). Output: `429 complete cases: {'BraTS 2020': 369, 'BraTS-Africa': 60}`; grades HGG 293, LGG 76, BraTS-Africa 60 (no grade recorded).
- **Preprocess and cache.** Eight `spawn` processes crop each case to its brain box, z-score each sequence over its non-zero voxels and cache float16 images at native resolution. Output: `Cached 429 cases in 337s (12.45 GB)`, `Per case: 3.1s reading from /kaggle/input, 2.9s processing and writing`. Every case is 240x240x155 at 1x1x1 mm. Brain crops average 137.4 x 169.9 x 138.1 voxels (minimum 122 x 128 x 98, maximum 157 x 193 x 155). The check found 22 reference voxels outside the brain crop: 4 in training case 091 and 18 in validation case 344, none in a test or external case (`cases.csv`, `label_outside_crop`).
- **Dataset overview.** Region volumes in cm^3 per cohort (`region_volumes_by_cohort.csv`) and cases without ET (`cases_without_et.csv`: 0 of 293 HGG, 27 of 76 LGG (35.5%), 0 of 60 BraTS-Africa):

| Region | Cohort         | Mean   | SD    | Min   | Median | Max    |
| ------ | -------------- | ------ | ----- | ----- | ------ | ------ |
| WT     | BraTS 2020 HGG | 96.44  | 53.78 | 7.28  | 91.13  | 256.88 |
| WT     | BraTS 2020 LGG | 111.54 | 76.72 | 17.08 | 86.32  | 361.78 |
| WT     | BraTS-Africa   | 161.36 | 92.28 | 1.03  | 163.75 | 394.41 |
| TC     | BraTS 2020 HGG | 37.66  | 29.26 | 0.56  | 32.54  | 126.72 |
| TC     | BraTS 2020 LGG | 57.85  | 49.66 | 0.83  | 44.59  | 190.19 |
| TC     | BraTS-Africa   | 61.68  | 58.32 | 0.49  | 39.81  | 228.44 |
| ET     | BraTS 2020 HGG | 23.39  | 18.50 | 0.11  | 19.46  | 111.25 |
| ET     | BraTS 2020 LGG | 5.46   | 13.53 | 0.00  | 0.82   | 89.55  |
| ET     | BraTS-Africa   | 36.32  | 30.98 | 0.49  | 28.43  | 122.36 |

![Segmentation region volumes](segmentation/plots/01_region_volumes.png)

_Region volumes per cohort: BraTS-Africa tumours are larger on average than BraTS 2020 tumours._

![Segmentation modalities and masks](segmentation/plots/02_modalities_and_masks.png)

_One internal and one external case with their four sequences and three region masks._

- **Split** (`split.csv`; validation and test IDs fixed to those of the earlier benchmark; per-case roles in `cases.csv`):

| Role       | HGG | LGG | BraTS-Africa | Total |
| ---------- | --- | --- | ------------ | ----- |
| Train      | 204 | 53  | 0            | 257   |
| Validation | 45  | 11  | 0            | 56    |
| Test       | 44  | 12  | 0            | 56    |
| External   | 0   | 0   | 60           | 60    |
| Total      | 293 | 76  | 60           | 429   |

- **Augmentation preview.** Rotation up to 30 degrees and zoom 0.7 to 1.4 (p = 0.2 each), mirroring, noise, brightness, contrast and gamma, on the GPU.

![Segmentation augmentation](segmentation/plots/03_augmentation.png)

### 3. Training

Output: `3 jobs, 0 already finished; launching 2 workers; training ends by 21:16` and `Training stage finished in 528.3 min; pending runs: none; stopped early by the time budget: none`. GPU 1 trained Swin UNETR; GPU 0 trained nnU-Net, then SegResNet. Per run (`metrics/runs/<run>/summary.json`, `et_threshold_and_selection.csv`):

| Run               | Steps  | Patches seen | Throughput (patches/s) | Train hours | Best epoch | Best validation Dice | ET minimum voxels | Peak train memory (GB) |
| ----------------- | ------ | ------------ | ---------------------- | ----------- | ---------- | -------------------- | ----------------- | ---------------------- |
| nnu-net_seed42    | 15,000 | 30,000       | 2.24 to 2.28           | 3.69        | 55         | 0.8415               | 500               | 5.69                   |
| segresnet_seed42  | 15,000 | 30,000       | 2.19 to 2.23           | 3.77        | 50         | 0.8390               | 800               | 6.02                   |
| swin-unetr_seed42 | 7,500  | 15,000       | 0.62 to 0.63           | 6.65        | 30         | 0.8191               | 500               | 8.67                   |

Validation Dice is measured with sliding windows and no TTA every 5 epochs; the ET threshold is the one with the best validation mean Dice after TTA. No run was truncated by the time budget. Predicting the 56 validation, 56 test and 60 external cases with TTA took about 7 min per split for each CNN and about 28 min per split for Swin UNETR.

### 4. Internal validation of the members

![Segmentation training curves](segmentation/plots/04_training_curves.png)

_Training loss, validation Dice per region and throughput for each member._

Internal test results of the members on 56 cases with 8-flip TTA (`member_test_results.csv`). Dice is the mean over cases with a bootstrap 95% CI for the mean of the three regions; HD95 (mm) is the median over cases, because a missed or spurious region costs the fixed BraTS value of 373.13 mm:

| Model      | Mean Dice (95% CI)      | Dice WT (SD)    | Dice TC (SD)    | Dice ET (SD)    | HD95 WT | HD95 TC | HD95 ET | Sens. WT | Sens. TC | Sens. ET |
| ---------- | ----------------------- | --------------- | --------------- | --------------- | ------- | ------- | ------- | -------- | -------- | -------- |
| nnU-Net    | 0.8404 [0.8008, 0.8760] | 0.9139 (0.0629) | 0.8493 (0.2052) | 0.7580 (0.2735) | 3.39    | 3.00    | 2.34    | 0.9110   | 0.8838   | 0.8611   |
| SegResNet  | 0.8717 [0.8389, 0.8987] | 0.9264 (0.0437) | 0.8791 (0.1542) | 0.8095 (0.2317) | 3.08    | 2.64    | 1.57    | 0.9280   | 0.8900   | 0.8400   |
| Swin UNETR | 0.8360 [0.7923, 0.8742] | 0.9162 (0.0651) | 0.8427 (0.2066) | 0.7490 (0.2858) | 3.00    | 3.00    | 2.24    | 0.9151   | 0.8688   | 0.8562   |

Specificity is between 0.9992 and 0.9998 for every member and region. Per-case values are in `metrics/runs/<run>/test_cases.csv` and `test_per_case.csv`.

### 5. Region-adaptive hybrid ensemble

**Fusion search on the validation cases** (`fusion_weights.csv`). For each region, every weight vector on a 0.1 grid of the simplex (66 vectors plus the exact equal-weight vector) is combined with every threshold in `FUSION_THRESHOLDS`:

| Region | nnU-Net | SegResNet | Swin UNETR | Threshold | Val. Dice (proposed) | Val. Dice (weights only, threshold 0.5) | Val. Dice (mean ensemble) | Best single member (val. Dice) |
| ------ | ------- | --------- | ---------- | --------- | -------------------- | --------------------------------------- | ------------------------- | ------------------------------ |
| WT     | 0.4     | 0.6       | 0.0        | 0.4       | 0.8951               | 0.8935                                  | 0.8879                    | SegResNet (0.8913)             |
| TC     | 0.4     | 0.5       | 0.1        | 0.3       | 0.8552               | 0.8490                                  | 0.8413                    | nnU-Net (0.8455)               |
| ET     | 0.5     | 0.5       | 0.0        | 0.4       | 0.7974               | 0.7963                                  | 0.7916                    | nnU-Net (0.7929)               |

The search gives the transformer weight only for TC (0.1). Each ensemble then gets its own ET size threshold on validation: CNN mean ensemble 300 voxels, the other three ensembles 500 voxels.

![Segmentation fusion search](segmentation/plots/05_fusion_search.png)

_Left: the selected weights per region. Right: validation Dice against the threshold for the selected weights and the plain mean ensemble._

**Internal test results, members and ensembles** (`test_results.csv`, 56 cases, TTA):

| Model                                      | Mean Dice (95% CI)      | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET | Sens. WT | Sens. TC | Sens. ET |
| ------------------------------------------ | ----------------------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | -------- |
| nnU-Net                                    | 0.8404 [0.8008, 0.8760] | 0.9139  | 0.8493  | 0.7580  | 3.39    | 3.00    | 2.34    | 0.9110   | 0.8838   | 0.8611   |
| SegResNet                                  | 0.8717 [0.8389, 0.8987] | 0.9264  | 0.8791  | 0.8095  | 3.08    | 2.64    | 1.57    | 0.9280   | 0.8900   | 0.8400   |
| Swin UNETR                                 | 0.8360 [0.7923, 0.8742] | 0.9162  | 0.8427  | 0.7490  | 3.00    | 3.00    | 2.24    | 0.9151   | 0.8688   | 0.8562   |
| CNN mean ensemble                          | 0.8512 [0.8150, 0.8838] | 0.9225  | 0.8630  | 0.7680  | 3.39    | 2.53    | 2.24    | 0.9192   | 0.8858   | 0.8580   |
| Mean ensemble (all members)                | 0.8494 [0.8115, 0.8834] | 0.9213  | 0.8601  | 0.7669  | 3.24    | 2.91    | 2.24    | 0.9176   | 0.8816   | 0.8589   |
| Region-adaptive weights (threshold 0.5)    | 0.8488 [0.8121, 0.8819] | 0.9260  | 0.8563  | 0.7640  | 3.32    | 2.91    | 2.24    | 0.9248   | 0.8854   | 0.8599   |
| Region-adaptive hybrid ensemble (proposed) | 0.8530 [0.8204, 0.8836] | 0.9256  | 0.8639  | 0.7694  | 3.27    | 2.83    | 2.19    | 0.9345   | 0.9122   | 0.8727   |

Per-case Dice SD of the proposed ensemble: WT 0.0441, TC 0.1797, ET 0.2636 (the smallest of all models for TC). Specificity is 0.9992 to 0.9997 for every ensemble.

**Statistical comparison** (`wilcoxon_tests.csv`). Wilcoxon signed-rank tests of the proposed ensemble against each comparator on per-case mean Dice, Holm-corrected within each set; the difference is proposed minus comparator with a bootstrap 95% CI:

| Set           | Comparator                              | Mean difference | 95% CI              | Median difference | Holm p  |
| ------------- | --------------------------------------- | --------------- | ------------------- | ----------------- | ------- |
| Internal test | nnU-Net                                 | 0.01258         | [0.00400, 0.02761]  | 0.00328           | 0.00007 |
| Internal test | SegResNet                               | -0.01872        | [-0.04425, 0.00352] | -0.00342          | 0.01766 |
| Internal test | Swin UNETR                              | 0.01701         | [0.00410, 0.03639]  | 0.00447           | 0.00693 |
| Internal test | CNN mean ensemble                       | 0.00179         | [-0.00643, 0.01425] | -0.00208          | 0.06429 |
| Internal test | Mean ensemble (all members)             | 0.00352         | [-0.00610, 0.01838] | -0.00141          | 0.38369 |
| Internal test | Region-adaptive weights (threshold 0.5) | 0.00418         | [-0.00174, 0.01443] | -0.00129          | 0.38369 |
| External      | nnU-Net                                 | 0.02008         | [0.00987, 0.03196]  | 0.01075           | 0.00005 |
| External      | SegResNet                               | -0.00015        | [-0.01165, 0.01129] | -0.00032          | 1.0     |
| External      | Swin UNETR                              | 0.00775         | [-0.00207, 0.01915] | 0.00031           | 1.0     |
| External      | CNN mean ensemble                       | -0.00393        | [-0.01550, 0.00691] | 0.00161           | 1.0     |
| External      | Mean ensemble (all members)             | -0.00057        | [-0.00592, 0.00449] | -0.00128          | 1.0     |
| External      | Region-adaptive weights (threshold 0.5) | 0.00934         | [0.00056, 0.01992]  | -0.00014          | 1.0     |

On the internal test set the proposed ensemble beats nnU-Net and Swin UNETR but is significantly worse than SegResNet alone (Holm p = 0.018); it is not significantly different from the other ensembles. On the external set it differs significantly only from nnU-Net.

![Segmentation paired differences](segmentation/plots/06_paired_differences.png)

_Mean paired Dice difference (proposed minus comparator) with 95% CI and Holm p per set._

### 6. External validation and the slice-wise protocol

**External results** on 60 BraTS-Africa cases (`external_results.csv`, same models, weights and thresholds, TTA):

| Model                                      | Mean Dice (95% CI)      | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET | Sens. WT | Sens. TC | Sens. ET |
| ------------------------------------------ | ----------------------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | -------- |
| nnU-Net                                    | 0.7765 [0.7224, 0.8203] | 0.8892  | 0.7241  | 0.7162  | 5.05    | 8.70    | 5.92    | 0.8836   | 0.8172   | 0.6679   |
| SegResNet                                  | 0.7967 [0.7417, 0.8400] | 0.9005  | 0.7798  | 0.7099  | 3.16    | 5.57    | 4.79    | 0.9168   | 0.8705   | 0.6140   |
| Swin UNETR                                 | 0.7888 [0.7355, 0.8325] | 0.9012  | 0.7292  | 0.7361  | 3.93    | 6.77    | 4.58    | 0.9012   | 0.8416   | 0.6970   |
| CNN mean ensemble                          | 0.8005 [0.7484, 0.8419] | 0.9000  | 0.7686  | 0.7328  | 4.06    | 5.39    | 4.74    | 0.9016   | 0.8483   | 0.6411   |
| Mean ensemble (all members)                | 0.7971 [0.7456, 0.8400] | 0.9021  | 0.7553  | 0.7340  | 3.61    | 5.24    | 4.53    | 0.9029   | 0.8505   | 0.6690   |
| Region-adaptive weights (threshold 0.5)    | 0.7872 [0.7327, 0.8304] | 0.9002  | 0.7441  | 0.7174  | 3.46    | 7.60    | 5.24    | 0.9078   | 0.8306   | 0.6586   |
| Region-adaptive hybrid ensemble (proposed) | 0.7966 [0.7430, 0.8391] | 0.8979  | 0.7507  | 0.7411  | 3.53    | 6.52    | 5.00    | 0.9185   | 0.8998   | 0.6883   |

Per-case values are in `external_per_case.csv` and `metrics/runs/<run>/external_cases.csv`. The CNN mean ensemble has the best external mean Dice (0.8005); the proposed ensemble has the best external ET Dice (0.7411).

**Generalisation gap** (`generalisation_gap.csv`, external minus internal Dice):

| Model                                      | Mean    | WT      | TC      | ET      |
| ------------------------------------------ | ------- | ------- | ------- | ------- |
| nnU-Net                                    | -0.0639 | -0.0248 | -0.1252 | -0.0417 |
| SegResNet                                  | -0.0750 | -0.0260 | -0.0994 | -0.0996 |
| Swin UNETR                                 | -0.0471 | -0.0150 | -0.1135 | -0.0129 |
| CNN mean ensemble                          | -0.0507 | -0.0225 | -0.0944 | -0.0351 |
| Mean ensemble (all members)                | -0.0523 | -0.0192 | -0.1048 | -0.0329 |
| Region-adaptive weights (threshold 0.5)    | -0.0616 | -0.0258 | -0.1123 | -0.0466 |
| Region-adaptive hybrid ensemble (proposed) | -0.0564 | -0.0277 | -0.1132 | -0.0284 |

Swin UNETR loses the least (mean -0.0471, ET -0.0129); SegResNet loses the most (mean -0.0750, ET -0.0996).

![Segmentation internal against external](segmentation/plots/07_internal_vs_external.png)

_Internal test and external Dice per model and region._

**The 2D slice-wise protocol** (`slice_protocol_comparison.csv`). Per-patient 3D Dice (the BraTS protocol) against the mean Dice over tumour-containing axial slices and the Dice pooled over those slices. The GCSER-UNet row is the published slice-wise result of a 2D study on its own 75:15:10 slice split (an arXiv preprint, `REFERENCE_2D`); nothing of it is rerun:

| Set                   | Model                    | 3D WT  | Slice mean WT | Slice pooled WT | 3D TC  | Slice mean TC | Slice pooled TC | 3D ET  | Slice mean ET | Slice pooled ET |
| --------------------- | ------------------------ | ------ | ------------- | --------------- | ------ | ------------- | --------------- | ------ | ------------- | --------------- |
| Internal test         | nnU-Net                  | 0.9139 | 0.8246        | 0.9209          | 0.8493 | 0.7667        | 0.8698          | 0.7580 | 0.7543        | 0.8626          |
| Internal test         | SegResNet                | 0.9264 | 0.8424        | 0.9317          | 0.8791 | 0.7897        | 0.8864          | 0.8095 | 0.7639        | 0.8740          |
| Internal test         | Swin UNETR               | 0.9162 | 0.8218        | 0.9197          | 0.8427 | 0.7557        | 0.8616          | 0.7490 | 0.7511        | 0.8584          |
| Internal test         | CNN mean ensemble        | 0.9225 | 0.8346        | 0.9269          | 0.8630 | 0.7776        | 0.8805          | 0.7680 | 0.7623        | 0.8686          |
| Internal test         | Mean ensemble            | 0.9213 | 0.8302        | 0.9248          | 0.8601 | 0.7730        | 0.8775          | 0.7669 | 0.7617        | 0.8677          |
| Internal test         | Region-adaptive weights  | 0.9260 | 0.8396        | 0.9309          | 0.8563 | 0.7719        | 0.8757          | 0.7640 | 0.7593        | 0.8666          |
| Internal test         | Proposed                 | 0.9256 | 0.8443        | 0.9316          | 0.8639 | 0.7883        | 0.8856          | 0.7694 | 0.7683        | 0.8704          |
| External              | nnU-Net                  | 0.8892 | 0.8168        | 0.9217          | 0.7241 | 0.6975        | 0.8234          | 0.7162 | 0.6420        | 0.7758          |
| External              | SegResNet                | 0.9005 | 0.8520        | 0.9324          | 0.7798 | 0.7591        | 0.8747          | 0.7099 | 0.6166        | 0.7599          |
| External              | Swin UNETR               | 0.9012 | 0.8337        | 0.9308          | 0.7292 | 0.7160        | 0.8365          | 0.7361 | 0.6771        | 0.7938          |
| External              | CNN mean ensemble        | 0.9000 | 0.8379        | 0.9321          | 0.7686 | 0.7427        | 0.8629          | 0.7328 | 0.6391        | 0.7726          |
| External              | Mean ensemble            | 0.9021 | 0.8388        | 0.9321          | 0.7553 | 0.7349        | 0.8544          | 0.7340 | 0.6551        | 0.7837          |
| External              | Region-adaptive weights  | 0.9002 | 0.8438        | 0.9322          | 0.7441 | 0.7192        | 0.8418          | 0.7174 | 0.6414        | 0.7751          |
| External              | Proposed                 | 0.8979 | 0.8495        | 0.9312          | 0.7507 | 0.7502        | 0.8611          | 0.7411 | 0.6637        | 0.7919          |
| Published (own split) | GCSER-UNet ensemble (2D) |        | 0.9500        |                 |        | 0.9200        |                 |        | 0.9000        |                 |

Pooled slice Dice is higher than per-patient 3D Dice for every model, region and set; the slice mean is lower for WT and TC. Numbers from different protocols and splits are therefore not comparable.

![Segmentation protocol comparison](segmentation/plots/08_protocol_comparison.png)

_Dice of the proposed ensemble under each protocol, next to the published slice-wise GCSER-UNet results._

**Results by tumour grade** (`test_results_by_grade.csv`, 44 HGG and 12 LGG internal test cases):

| Model                                      | Grade | Dice WT | Dice TC | Dice ET | Mean Dice | 95% CI           |
| ------------------------------------------ | ----- | ------- | ------- | ------- | --------- | ---------------- |
| nnU-Net                                    | HGG   | 0.9147  | 0.8989  | 0.8531  | 0.8889    | [0.8531, 0.9154] |
| nnU-Net                                    | LGG   | 0.9110  | 0.6675  | 0.4089  | 0.6625    | [0.6068, 0.7232] |
| SegResNet                                  | HGG   | 0.9250  | 0.9237  | 0.8715  | 0.9067    | [0.8906, 0.9221] |
| SegResNet                                  | LGG   | 0.9317  | 0.7157  | 0.5822  | 0.7432    | [0.6465, 0.8298] |
| Swin UNETR                                 | HGG   | 0.9132  | 0.8913  | 0.8529  | 0.8858    | [0.8467, 0.9146] |
| Swin UNETR                                 | LGG   | 0.9270  | 0.6645  | 0.3681  | 0.6532    | [0.5833, 0.7237] |
| CNN mean ensemble                          | HGG   | 0.9211  | 0.9106  | 0.8622  | 0.8980    | [0.8669, 0.9212] |
| CNN mean ensemble                          | LGG   | 0.9277  | 0.6886  | 0.4223  | 0.6796    | [0.6257, 0.7375] |
| Mean ensemble (all members)                | HGG   | 0.9189  | 0.9068  | 0.8613  | 0.8957    | [0.8616, 0.9207] |
| Mean ensemble (all members)                | LGG   | 0.9298  | 0.6889  | 0.4210  | 0.6799    | [0.6243, 0.7392] |
| Region-adaptive weights (threshold 0.5)    | HGG   | 0.9247  | 0.9063  | 0.8589  | 0.8967    | [0.8671, 0.9198] |
| Region-adaptive weights (threshold 0.5)    | LGG   | 0.9308  | 0.6730  | 0.4159  | 0.6732    | [0.6211, 0.7306] |
| Region-adaptive hybrid ensemble (proposed) | HGG   | 0.9245  | 0.9142  | 0.8648  | 0.9012    | [0.8825, 0.9189] |
| Region-adaptive hybrid ensemble (proposed) | LGG   | 0.9297  | 0.6793  | 0.4197  | 0.6762    | [0.6250, 0.7320] |

SegResNet's lead over the ensembles comes mostly from LGG cases (LGG ET Dice 0.5822 against 0.4197 for the proposed ensemble).

![Segmentation Dice by grade](segmentation/plots/09_dice_by_grade.png)

![Segmentation per-case distributions](segmentation/plots/10_per_case_distributions.png)

_Box plots of per-case Dice and HD95 per region for every member and ensemble, internal and external._

![Segmentation qualitative comparison](segmentation/plots/11_qualitative_comparison.png)

_Internal and external cases with the median and the lowest proposed-ensemble Dice: reference, members and proposed ensemble on FLAIR (oedema-only WT green, TC yellow, ET red). Internal median BraTS20_Training_084 (0.897); internal lowest BraTS20_Training_335 (0.565; SegResNet 0.878); external median BraTS-SSA-00044-000 (0.854); external lowest BraTS-SSA-00215-000, a small tumour that every model misses completely (Dice 0.000)._

### 7. Uncertainty and quality control

Two per-case warning signals against the proposed ensemble's per-case mean Dice (`quality_control.csv`; per case in `qc_per_case.csv`); a failure is a case with mean Dice below 0.80:

| Set           | Signal                     | Cases | Failures | Spearman rho | p       | AUROC for failures |
| ------------- | -------------------------- | ----- | -------- | ------------ | ------- | ------------------ |
| Internal test | Architectural disagreement | 56    | 14       | -0.8018      | 1.1e-13 | 0.9932             |
| Internal test | TTA spread                 | 56    | 14       | -0.3894      | 0.0030  | 0.7160             |
| External      | Architectural disagreement | 60    | 18       | -0.8163      | 1.9e-15 | 0.8704             |
| External      | TTA spread                 | 60    | 18       | -0.6096      | 2.4e-7  | 0.7910             |

Disagreement is 1 minus the mean pairwise Dice between the members' masks, averaged over the three regions; TTA spread is the mean flip standard deviation inside the predicted WT, averaged over members. Disagreement flags failures better than TTA spread in both sets.

![Segmentation quality control](segmentation/plots/12_quality_control.png)

_Per-case mean Dice of the proposed ensemble against each signal; red points are failures, the dashed line is the 0.80 failure threshold._

### 8. Explainability

**Seg-Grad-CAM maps** on the internal (BraTS20_Training_084) and external (BraTS-SSA-00044-000) cases with the median proposed-ensemble Dice, a 128^3 crop around the tumour, one map per member and region:

![Segmentation Seg-Grad-CAM, internal](segmentation/plots/13_gradcam_test.png)

![Segmentation Seg-Grad-CAM, external](segmentation/plots/13_gradcam_external.png)

_FLAIR, the Seg-Grad-CAM map and the reference region outline (white). nnU-Net and Swin UNETR maps concentrate on the tumour; SegResNet's TC map in the internal case is diffuse, with a grid pattern._

**Quantitative attribution** (`gradcam_attribution.csv`; per map in `gradcam_attribution_per_case.csv`). Output: `Seg-Grad-CAM attribution: 1035 maps scored, 0 left out (out of memory)`. Share of the map inside the reference region dilated by 5 voxels, against the share expected by chance:

| Set           | Model      | Region | Inside | Chance | Enrichment |
| ------------- | ---------- | ------ | ------ | ------ | ---------- |
| Internal test | nnU-Net    | WT     | 0.9382 | 0.1145 | 9.90       |
| Internal test | nnU-Net    | TC     | 0.8280 | 0.0522 | 22.62      |
| Internal test | nnU-Net    | ET     | 0.2435 | 0.0387 | 6.42       |
| Internal test | SegResNet  | WT     | 0.9709 | 0.1145 | 10.40      |
| Internal test | SegResNet  | TC     | 0.1937 | 0.0522 | 5.16       |
| Internal test | SegResNet  | ET     | 0.5737 | 0.0387 | 15.88      |
| Internal test | Swin UNETR | WT     | 0.1604 | 0.1145 | 1.48       |
| Internal test | Swin UNETR | TC     | 0.0850 | 0.0522 | 1.50       |
| Internal test | Swin UNETR | ET     | 0.0737 | 0.0387 | 1.68       |
| External      | nnU-Net    | WT     | 0.9767 | 0.1572 | 8.65       |
| External      | nnU-Net    | TC     | 0.8804 | 0.0689 | 22.18      |
| External      | nnU-Net    | ET     | 0.4320 | 0.0629 | 7.64       |
| External      | SegResNet  | WT     | 0.9609 | 0.1572 | 8.38       |
| External      | SegResNet  | TC     | 0.2933 | 0.0689 | 4.25       |
| External      | SegResNet  | ET     | 0.7575 | 0.0629 | 18.85      |
| External      | Swin UNETR | WT     | 0.1942 | 0.1572 | 1.27       |
| External      | Swin UNETR | TC     | 0.1248 | 0.0689 | 1.49       |
| External      | Swin UNETR | ET     | 0.0824 | 0.0629 | 1.06       |

The CNN maps are 4 to 23 times enriched in the tumour region; the Swin UNETR maps, taken at `decoder3`, are close to chance (1.06 to 1.68) even though they look focused in the median cases, so they spread a large share of their mass over the rest of the crop.

![Segmentation Seg-Grad-CAM attribution](segmentation/plots/14_gradcam_attribution.png)

**Sequence-drop ablation** (`modality_ablation_summary.csv`; per case in `metrics/runs/<run>/modality_ablation.csv`). Dice drop on the 56 internal test cases when one sequence is replaced by its brain mean, without TTA:

| Model      | Dropped | Drop WT | Drop TC | Drop ET |
| ---------- | ------- | ------- | ------- | ------- |
| nnU-Net    | FLAIR   | 0.5945  | 0.4389  | 0.4128  |
| nnU-Net    | T1      | 0.0140  | 0.0587  | 0.0504  |
| nnU-Net    | T1CE    | 0.0225  | 0.3048  | 0.6388  |
| nnU-Net    | T2      | 0.0853  | 0.0788  | 0.0427  |
| SegResNet  | FLAIR   | 0.2448  | 0.0981  | 0.0865  |
| SegResNet  | T1      | 0.0078  | 0.0284  | 0.0400  |
| SegResNet  | T1CE    | 0.0401  | 0.3880  | 0.7554  |
| SegResNet  | T2      | 0.0238  | 0.0180  | -0.0129 |
| Swin UNETR | FLAIR   | 0.3611  | 0.1186  | 0.1037  |
| Swin UNETR | T1      | 0.0097  | 0.0461  | 0.0495  |
| Swin UNETR | T1CE    | 0.0135  | 0.3211  | 0.6920  |
| Swin UNETR | T2      | 0.0372  | 0.0468  | -0.0145 |

Every member relies on FLAIR for WT and on T1CE for TC and ET, the expected clinical pattern; nnU-Net depends on FLAIR far more than the others for all three regions.

![Segmentation sequence drop](segmentation/plots/15_sequence_drop.png)

### 9. Efficiency and summary

**Efficiency** (`efficiency.csv`). Latency per case is the sliding window over the median test crop (136 x 170 x 139) at fp16 on one T4, mean of 4 runs; an ensemble costs the sum of the members it uses:

| Model                                      | Parameters (M) | GFLOPs per patch | Train hours | Peak inference memory (GB) | Latency (s) | Latency with TTA (s) | Internal Dice | External Dice |
| ------------------------------------------ | -------------- | ---------------- | ----------- | -------------------------- | ----------- | -------------------- | ------------- | ------------- |
| nnU-Net                                    | 31.42          | 968.1            | 3.69        | 3.15                       | 1.03        | 8.20                 | 0.8404        | 0.7765        |
| SegResNet                                  | 19.92          | 1218.2           | 3.77        | 2.91                       | 1.01        | 8.08                 | 0.8717        | 0.7967        |
| Swin UNETR                                 | 72.77          | 1665.6           | 6.65        | 8.58                       | 3.99        | 31.92                | 0.8360        | 0.7888        |
| CNN mean ensemble                          | 51.34          | 2186.3           | 7.46        |                            | 2.04        | 16.28                | 0.8512        | 0.8005        |
| Mean ensemble (all members)                | 124.11         | 3851.9           | 14.11       |                            | 6.03        | 48.20                | 0.8494        | 0.7971        |
| Region-adaptive weights (threshold 0.5)    | 51.34          | 2186.3           | 7.46        |                            | 2.04        | 16.28                | 0.8488        | 0.7872        |
| Region-adaptive hybrid ensemble (proposed) | 124.11         | 3851.9           | 14.11       |                            | 6.03        | 48.20                | 0.8530        | 0.7966        |

Latency standard deviations are below 0.05 s. The proposed ensemble needs all three members because Swin UNETR carries weight 0.1 for TC.

![Segmentation Dice against latency](segmentation/plots/16_dice_vs_latency.png)

**Summary.** Output: `Proposed ensemble: internal mean Dice 0.8530, external 0.7966; pending runs: 0`. `summary_table.csv` collects the internal and external Dice and HD95 per model; `run_summary.json` records the case counts, fusion weights and thresholds, ET thresholds, quality-control results, environment, the slice-protocol Dice of the proposed ensemble and the limitations below. Its last limitation (that the web app still runs the earlier VNet) was true at run time; the web app has since switched to this run's SegResNet (see [Checkpoints used by the web app](#checkpoints-used-by-the-web-app)).

**Package the outputs.** Output: `Zipped 590 files (0.72 GB) into /kaggle/working/outputs.zip`. The zip holds `src/`, `metrics/`, `plots/`, `logs/`, `weights/<model>/<run>.pth` (best-validation checkpoints: nnU-Net 251.4 MB, SegResNet 79.7 MB, Swin UNETR 294.9 MB; Git-ignored) and `predictions/<run>/<val|test|external>/<case>.npz` (516 files, 86.6 MB in total: uint8 TTA probabilities on the brain crop with the case's TTA spread; Git-ignored).

**What the numbers support.** All three members and all ensembles are within overlapping confidence intervals. The region-adaptive ensemble beats nnU-Net and Swin UNETR internally and nnU-Net externally, but not SegResNet, which is the best single model internally (0.8717) and matches the proposed ensemble externally (0.7967 against 0.7966). The clearest positive results are the quality-control signal (architectural disagreement, AUROC 0.99 internal and 0.87 external) and the explainability checks; every result rests on one seed per member.

## Checkpoints used by the web app

The 2D notebooks export one checkpoint each to `weights/deploy/` with a `metadata.json`; the segmentation notebook does not, and `segmentation/weights/deploy/SegResNet.pth` is a byte-identical copy of its `weights/segresnet/segresnet_seed42.pth` (SHA-256 in `metadata.json`), made with `cp Materials/Notebooks/segmentation/weights/segresnet/segresnet_seed42.pth Materials/Notebooks/segmentation/weights/deploy/SegResNet.pth`. The backend loads these three files by default (see the root [README](../../README.md#configuration)).

| File                                                 | Run                    | Selection                                | Recorded result                                                                                   | Other metadata                                       |
| ---------------------------------------------------- | ---------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `binary-detection/weights/deploy/EfficientNetB0.pth` | `efficientnetb0_fold4` | Lowest validation loss of 5 folds        | Fold test accuracy 0.9766 (171 images)                                                            | Temperature 0.0498, decision threshold 1.0, 1 output |
| `classification/weights/deploy/EfficientNetB0.pth`   | `efficientnetb0_fold0` | Lowest validation loss of 5 folds        | Fold test accuracy 0.9778 (45 images)                                                             | Temperature 0.0498, no threshold, 3 outputs          |
| `segmentation/weights/deploy/SegResNet.pth`          | `segresnet_seed42`     | Best internal test Dice of the 3 members | Best validation Dice 0.8390 (epoch 50); with TTA: internal test mean Dice 0.8717, external 0.7967 | ET minimum 800 voxels, no TTA in the web app         |

The 2D metadata records the preprocessing `crop_to_brain (blur 5x5, threshold 45, largest contour box), resize, ImageNet normalisation`, which is the same crop as `Backend/neuroscan/inference/processor.py`. The backend does not read `metadata.json`: it applies no temperature and uses a fixed 0.5 detection threshold. It segments like the notebook (native-resolution brain crop, foreground z-score, 128^3 Gaussian sliding window with overlap 0.5, threshold 0.5, ET size filter of 800 voxels, region hierarchy) but without the 8-flip TTA, which would make CPU inference eight times slower; the published Dice values were measured with TTA. On an Intel Core i7-9750H (6 cores, CPU-only PyTorch) the sample case `Materials/Input/BraTS20_Training_001` took 83.5 s.

## Limitations recorded by the notebooks

From `run_summary.json`:

- 2D tasks: the MRI sequence of the images is not recorded by the sources; binary negatives are tumour-negative, not certified normal; there are no patient identifiers, so grouping by image similarity is a proxy for a patient-level split; only two-dimensional public slices are used, with no institutional or three-dimensional external data.
- Segmentation: one seed per member, so seed-to-seed variance of the ensemble is not measured; fusion weights, thresholds and the ET size filter are selected on 56 validation cases (the test and external sets are never used for selection); BraTS-Africa shares the BraTS annotation protocol and preprocessing (co-registration, skull stripping), so it tests acquisition and population shift, not protocol shift; the comparison with 2D studies uses their published numbers on their own splits; and "the web app still runs the earlier 128^3 VNet", which is no longer true (see above).

Found in the outputs and worth knowing before citing the results:

- The classification split is dominated by one group of 1,431 images, and its test set has 45 images (1 glioma, 13 meningioma, 31 pituitary). Six models score 45 of 45 and the other three 44 of 45, so the test set cannot rank the models.
- The background-only shortcut test is positive for 7 of 9 binary models and all 9 classification models.
- Every segmentation run used one seed, so seed-to-seed spread is not available. The proposed ensemble does not beat SegResNet alone (internal Holm p = 0.018 in SegResNet's favour; external difference -0.00015), and every model misses external case BraTS-SSA-00215-000 completely.
- The segmentation GCSER-UNet comparison values come from an arXiv preprint (arXiv:2605.30510), not a peer-reviewed paper.
- The markdown overview of each 2D notebook still says "Runtime. Not measured yet", and the export cells tell the reader to copy the checkpoint to `Backend/Binary Detection/`, `Backend/Classification/` or `Backend/Segmentation/`. Those folders no longer exist; the backend now reads the checkpoints from `weights/deploy/` directly.
