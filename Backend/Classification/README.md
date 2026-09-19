# Classification: Brain Tumour Type (Glioma / Meningioma / Pituitary)

This folder benchmarks **six pretrained image classifiers** on a three-class task: given a brain MRI slice that contains a tumour, predict whether it is a **Glioma**, **Meningioma** or **Pituitary** tumour. There is one notebook per architecture. All six share the same pipeline, the same deduplicated data and split, and the same hyperparameters (apart from batch size and input size), and they write the same artefacts.

Everything below comes from the executed notebook outputs and the exported artefacts in this folder. Where the evidence does not explain a choice, this is stated instead of inferred.

This pipeline is a direct adaptation of the binary pipeline in [../Binary Detection/README.md](../Binary%20Detection/README.md). This README documents the whole pipeline, and it lists every place where the classification pipeline departs from the binary one.

## Contents

- [Folder layout](#folder-layout)
- [Headline results](#headline-results)
- [End-to-end pipeline](#end-to-end-pipeline)
- [Dataset and deduplication](#dataset-and-deduplication)
- [Shared configuration](#shared-configuration)
- [Cell-by-cell walkthrough (shared template)](#cell-by-cell-walkthrough-shared-template)
- [Per-notebook differences](#per-notebook-differences)
- [Per-model results and training dynamics](#per-model-results-and-training-dynamics)
- [Visualisations](#visualisations)
- [Inconsistencies, anomalies and limitations](#inconsistencies-anomalies-and-limitations)
- [Reproducing a run](#reproducing-a-run)
- [How Classification relates to Binary Detection](#how-classification-relates-to-binary-detection)

## Folder layout

```
Classification/
├── EfficientNetB0.pth                       # standalone weights, not produced by these notebooks (see anomalies)
├── classification-<model>.ipynb             # 6 notebooks, 75 cells each
└── classification-<model>/                  # unzipped output of each notebook
    ├── metrics/     # 22 files (binary set + deduplication_summary.csv)
    ├── plots/       # 14 PNGs, 01–14 (includes 04_deduplication_before_after.png)
    ├── tensorboard/<YYYYMMDD-HHMMSS>/events.out.tfevents.*
    └── weights/     # best_<model>_multi_class.pth, final_<model>_brain_tumor_multi_class.pth
```

| Notebook                                                                   | Output folder                                                    | Weights                                                                                     |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [classification-coatnet0.ipynb](classification-coatnet0.ipynb)             | [classification-coatnet0/](classification-coatnet0/)             | `best_coatnet0_multi_class.pth`, `final_coatnet0_brain_tumor_multi_class.pth`               |
| [classification-efficientnetb0.ipynb](classification-efficientnetb0.ipynb) | [classification-efficientnetb0/](classification-efficientnetb0/) | `best_efficientnet_b0_multi_class.pth`, `final_efficientnet_b0_brain_tumor_multi_class.pth` |
| [classification-inceptionv3.ipynb](classification-inceptionv3.ipynb)       | [classification-inceptionv3/](classification-inceptionv3/)       | `best_inceptionv3_multi_class.pth`, `final_inceptionv3_brain_tumor_multi_class.pth`         |
| [classification-resnet50.ipynb](classification-resnet50.ipynb)             | [classification-resnet50/](classification-resnet50/)             | `best_resnet50_multi_class.pth`, `final_resnet50_brain_tumor_multi_class.pth`               |
| [classification-swinbase.ipynb](classification-swinbase.ipynb)             | [classification-swinbase/](classification-swinbase/)             | `best_swinbase_multi_class.pth`, `final_swinbase_brain_tumor_multi_class.pth`               |
| [classification-vit-b16.ipynb](classification-vit-b16.ipynb)               | [classification-vit-b16/](classification-vit-b16/)               | `best_vitb16_multi_class.pth`, `final_vitb16_brain_tumor_multi_class.pth`                   |

The `final_*` files are written after the best checkpoint has been reloaded. They hold the **best-validation-AUC weights**, not the weights from the last epoch.

## Headline results

Test set: 706 images (300 Glioma, 196 Meningioma, 210 Pituitary). Source: `metrics/benchmark_summary.json`. AUC is one-vs-rest macro.

| Model          | Test acc   | Balanced acc | Macro F1   | MCC        | κ          | ROC AUC (OvR macro) | PR AUC (macro) | Top-2 acc  | Log loss   | ECE        | Best epoch |
| -------------- | ---------- | ------------ | ---------- | ---------- | ---------- | ------------------- | -------------- | ---------- | ---------- | ---------- | ---------- |
| SwinBase       | **0.9858** | **0.9842**   | **0.9847** | **0.9784** | **0.9783** | 0.99929             | 0.99848        | 0.9986     | **0.0410** | **0.0076** | 7          |
| CoAtNet-0      | 0.9844     | 0.9821       | 0.9835     | 0.9762     | 0.9761     | 0.99892             | 0.99773        | **1.0000** | 0.0521     | 0.0084     | 9          |
| InceptionV3    | 0.9830     | 0.9829       | 0.9822     | 0.9741     | 0.9740     | **0.99938**         | **0.99872**    | 0.9986     | 0.0471     | 0.0077     | 8          |
| ResNet50       | 0.9759     | 0.9757       | 0.9746     | 0.9633     | 0.9632     | 0.99883             | 0.99752        | 0.9986     | 0.0593     | 0.0116     | 9          |
| EfficientNetB0 | 0.9759     | 0.9750       | 0.9745     | 0.9632     | 0.9632     | 0.99877             | 0.99719        | 0.9972     | 0.0588     | 0.0112     | 9          |
| ViT-B16        | 0.9051     | 0.9037       | 0.9034     | 0.8550     | 0.8550     | 0.97594             | 0.95159        | 0.9943     | 0.2637     | 0.0266     | 10         |

Cost and speed. Source: `model_profile.json`, `training_summary.json`, `inference_benchmark.json`. Hardware: 2x Tesla T4 (Kaggle). Inference used 1 GPU with fp16 autocast.

| Model          | Params (M) | GFLOPs/img | State dict (MB) | Input | Batch | Train time | Mean epoch (s) | Latency bs=1 (ms) | Throughput bs=32 (img/s) | Peak GPU0 alloc (MB) |
| -------------- | ---------- | ---------- | --------------- | ----- | ----- | ---------- | -------------- | ----------------- | ------------------------ | -------------------- |
| EfficientNetB0 | 4.011      | 0.769      | 15.59           | 224   | 256   | 0:26:53    | 161.20         | 13.85             | 1470.84                  | 11078.2              |
| ResNet50       | 23.514     | 8.174      | 90.00           | 224   | 256   | 0:08:07    | 48.52          | 7.82              | 791.58                   | 11374.0              |
| InceptionV3    | 25.118     | 11.422     | 96.15           | 299   | 128   | 0:12:14    | 73.26          | 14.15             | 532.64                   | 7104.8               |
| CoAtNet-0      | 26.669     | 8.416      | 101.87          | 224   | 128   | 0:28:01    | 167.99         | 13.23             | 498.58                   | 11257.0              |
| ViT-B16        | 85.801     | 22.541     | 327.36          | 224   | 32    | 0:07:23    | 43.79          | 8.39              | 289.50                   | 3875.8               |
| SwinBase       | 86.746     | 30.860     | 331.47          | 224   | 64    | 0:09:28    | 56.42          | 35.13             | 202.41                   | 10784.5              |

**Reading these results**

- SwinBase, CoAtNet-0 and InceptionV3 are within 0.3 percentage points of each other: 10, 11 and 12 test errors respectively. With a single seed and 706 test images, that gap is not meaningful. SwinBase is best on accuracy and calibration. InceptionV3 is best on ranking (AUC). CoAtNet-0 is the only model whose top-2 prediction is always correct.
- EfficientNetB0 matches ResNet50 (17 errors each) at about 1/10 of the FLOPs and 1/6 of the storage. It has the best batched throughput apart from MobileNet-class models, and MobileNet is not tested in this folder.
- ViT-B16 lags far behind (67 errors). Its best epoch is the last one, and validation AUC is still rising, so it is under-trained with this recipe.
- As in Binary Detection, wall-clock training time is dominated by data loading. EfficientNetB0 at 0.77 GFLOPs took longer than SwinBase at 30.9 GFLOPs, at 23.5 vs 67.4 train images/s. Timings come from separate Kaggle sessions.
- Meningioma is the hardest class for every model. It is the smallest class and absorbs most of the confusion with Glioma (see the confusion matrices below).

## End-to-end pipeline

```
Kaggle inputs (2 datasets, 3 tumour folders each; sartaj Training + Testing merged)
  └─ os.walk (.jpg only) → DataFrame(path, label)       5828 images
      ├─ EDA on the raw (pre-dedup) data: distribution, samples, sizes, intensities
      └─ pHash every image → greedy near-duplicate removal (Hamming ≤ 5)   → 4705 images
          ├─ dedup before/after plot + table
          └─ stratified 70/15/15 split (seed 42)        train 3293 / val 706 / test 706
              └─ BrainDataset (long labels) + same augmentation as binary
                  └─ pretrained backbone, head → 3 logits, DataParallel over 2 GPUs
                      └─ 10 epochs: AdamW 3e-4, cosine LR, CrossEntropy, AMP fp16
                          ├─ keep best macro OvR val AUC checkpoint
                          └─ reload best → test → save final
                              └─ single-GPU re-inference → metrics, CM, ROC/PR (per class + micro),
                                 calibration, per-class bars, samples, Grad-CAM, benchmark, zip
```

## Dataset and deduplication

**Sources.** From the data paths cell (cell 10):

| Label | Class      | Paths                                                                                                                                                                             |
| ----- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | Glioma     | `sartajbhuvaji/brain-tumor-classification-mri/Testing/glioma_tumor`, `…/Training/glioma_tumor`, `moundir8/jun-cheng-brain-tumor-mri-dataset-jpg/brain tumor mri dataset/2_glioma` |
| 1     | Meningioma | `…/Testing/meningioma_tumor`, `…/Training/meningioma_tumor`, `…/1_meningioma`                                                                                                     |
| 2     | Pituitary  | `…/Testing/pituitary_tumor`, `…/Training/pituitary_tumor`, `…/3_pituitary_tumor`                                                                                                  |

All paths are prefixed with `/kaggle/input/datasets/`. Only `.jpg` files are collected (`VALID_EXTENSIONS = (".jpg",)`). The binary pipeline also accepts `.jpeg`, `.png` and `.tif`. The sartajbhuvaji `no_tumor` folders are not used, and its official `Training`/`Testing` split is **discarded**: both halves go into one pool, which is then re-split.

**Class distribution before and after deduplication.** Source: `metrics/deduplication_summary.csv`. Identical in all six runs.

| Class      | Before   | Share | After    | Share | Removed  | Removed % |
| ---------- | -------- | ----- | -------- | ----- | -------- | --------- |
| Glioma     | 2352     | 40.4% | 2000     | 42.5% | 352      | 14.97     |
| Meningioma | 1645     | 28.2% | 1304     | 27.7% | 341      | 20.73     |
| Pituitary  | 1831     | 31.4% | 1401     | 29.8% | 430      | 23.48     |
| **Total**  | **5828** |       | **4705** |       | **1123** | **19.27** |

![Deduplication before vs after](classification-efficientnetb0/plots/04_deduplication_before_after.png)

**Deduplication algorithm** (cell 18):

1. Compute a 64-bit perceptual hash of every image (`imagehash.phash` on the RGB image). Unreadable files get `None` and are dropped.
2. Walk the DataFrame in collection order (all Glioma paths, then Meningioma, then Pituitary). Keep an image only if its Hamming distance to **every** previously kept hash is `> HAMMING_TOLERANCE = 5`. The code comment reads `0 = exact, higher = more lenient`.
3. Keep the surviving rows and drop the hash column.

Implications:

- About 19% of the images were near-duplicates. Most likely this is because the two sources overlap: the sartajbhuvaji dataset is commonly derived in part from the Jun Cheng figshare data. That overlap is **not verified** in the notebook, which only reports per-class counts. Either way, deduplicating **before** splitting is what stops the same scan from appearing in both train and test. The binary pipeline does not do this.
- Comparison runs across classes and is first-come-first-kept. If a Meningioma or Pituitary image is within distance 5 of an already kept Glioma image, the later one is silently removed. Label conflicts are not reported, so it is unknown how many removals crossed classes.
- The loop is O(n²) in Python (about 11 M hash comparisons).
- EDA cells 14 and 16 (`01`–`03` plots, `image_dimension_stats.csv`) run on the **pre-deduplication** data (`df_raw`, 5828 images). The split and all training use the deduplicated 4705 images.

![Class distribution (pre-dedup)](classification-efficientnetb0/plots/01_class_distribution.png)
![Sample images](classification-efficientnetb0/plots/02_sample_images.png)

**Image properties (pre-dedup).** Source: `metrics/image_dimension_stats.csv`. Most images are 512x512, which matches the Jun Cheng data. The outliers come from sartajbhuvaji.

| Class      | Width mean ± std (min–max) | Median | Height mean ± std (min–max) | Median |
| ---------- | -------------------------- | ------ | --------------------------- | ------ |
| Glioma     | 507.6 ± 33.0 (175–640)     | 512    | 508.4 ± 30.6 (167–665)      | 512    |
| Meningioma | 483.1 ± 90.4 (174–1275)    | 512    | 485.9 ± 85.1 (195–1427)     | 512    |
| Pituitary  | 512.5 ± 84.4 (201–1365)    | 512    | 512.2 ± 81.3 (202–1365)     | 512    |

![Image properties](classification-efficientnetb0/plots/03_image_properties.png)

The samples and Grad-CAM panels show axial, coronal and sagittal slices mixed together, mainly T1 contrast-enhanced scans.

**Split.** Stratified 70/15/15 with `random_state=42` on the deduplicated data. Source: `metrics/split_distribution.csv`.

|            | Train    | Validation | Test    |
| ---------- | -------- | ---------- | ------- |
| Glioma     | 1400     | 300        | 300     |
| Meningioma | 913      | 195        | 196     |
| Pituitary  | 980      | 211        | 210     |
| **Total**  | **3293** | **706**    | **706** |

![Split distribution](classification-efficientnetb0/plots/05_split_distribution.png)

The classes are moderately imbalanced: Glioma is 42.5% and Meningioma 27.7%. No class weighting is used.

## Shared configuration

| Setting         | Value                                                                                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Seed            | 42                                                                                                                                                                                          |
| Epochs / LR     | 10 / 3e-4                                                                                                                                                                                   |
| Optimiser       | `AdamW` (default weight decay 0.01). All parameters trainable.                                                                                                                              |
| Scheduler       | `CosineAnnealingLR(T_max=10)`, stepped per epoch. LR goes from 3.0e-4 to 7.34e-6.                                                                                                           |
| Loss            | `CrossEntropyLoss` on 3 logits. Labels are `torch.long`.                                                                                                                                    |
| Precision       | AMP autocast, `GradScaler("cuda")`, TF32, `cudnn.benchmark=True`                                                                                                                            |
| Image size      | 224 (299 for InceptionV3)                                                                                                                                                                   |
| Batch size      | 256 by default. Overridden per model, see below.                                                                                                                                            |
| Loaders         | `num_workers=4`, `pin_memory`, `persistent_workers`, `prefetch_factor=4`. Train loader shuffles with `drop_last=True`.                                                                      |
| Augmentation    | Identical to binary: `Resize` → `RandomHorizontalFlip` → `RandomRotation(10)` → `ColorJitter(0.2, 0.2)` → `ToTensor` → ImageNet `Normalize`. Eval uses `Resize` → `ToTensor` → `Normalize`. |
| Epoch metrics   | Accuracy = `argmax` of softmax. AUC = `roc_auc_score(np.eye(3)[labels], probs, multi_class="ovr")`. The one-hot target makes sklearn compute a macro-averaged per-class AUC.                |
| Checkpoint rule | Save on `val_auc > best_auc`                                                                                                                                                                |
| Environment     | Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, 2x Tesla T4. Requires `imagehash`. CoAtNet-0 also requires `timm`.                                                                           |

Train batches per epoch after `drop_last`: 12 at batch 256 (3072 images), 25 at batch 128 (3200), 51 at batch 64 (3264), 102 at batch 32 (3264).

## Cell-by-cell walkthrough (shared template)

Every notebook has 75 cells in markdown/code pairs. The original cells are tagged `# Cell 1` to `# Cell 14`. Most models also use `# ===== SECTION =====` banners.

### 1. Setup (cells 0–8)

| Cell | Content                                                                                                                                                |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0    | Title and the 9-step pipeline description (same wording as binary)                                                                                     |
| 2    | Imports only. In binary, imports and config share one cell. Here they are split.                                                                       |
| 4    | Config (seed, batch, epochs, LR, image size, loader flags, device, seeding, cuDNN/TF32). Prints `CUDA available: True`, `GPU count: 2`.                |
| 6    | `MODEL_NAME`, `TASK="classification"`, `CLASS_NAMES=["Glioma","Meningioma","Pituitary"]`, `RUN_NAME`, `get_gradcam_target_layer`                       |
| 8    | Output folders and helpers (same code as binary). `NUM_CLASSES=3`, so `IS_BINARY=False`, which switches every downstream cell to multi-class branches. |

### 2. Data (cells 9–32)

| Cell | Content                                              | Output                                                             |
| ---- | ---------------------------------------------------- | ------------------------------------------------------------------ |
| 10   | Data paths (separate cell, unlike binary)            |                                                                    |
| 12   | `collect()` and DataFrame                            | `Total Images before deduplication: 5828` (G 2352, M 1645, P 1831) |
| 14   | `df_raw` copy, class bar chart, 6 samples per class  | `01`, `02`                                                         |
| 16   | Image sizes and intensity histograms (100 per class) | `03`, `image_dimension_stats.csv`                                  |
| 18   | **pHash deduplication** (new)                        | `Total Images after deduplication: 4705`                           |
| 20   | **Before/after plot and table** (new, plot `04`)     | `04_deduplication_before_after.png`, `deduplication_summary.csv`   |
| 22   | Stratified split                                     | `Train: 3293 Val: 706 Test: 706`                                   |
| 24   | Split distribution plot and table                    | `05`, `split_distribution.csv`                                     |
| 26   | `BrainDataset`, returning `torch.long` labels        |                                                                    |
| 28   | Transforms (separate cell)                           |                                                                    |
| 30   | DataLoaders (separate cell)                          |                                                                    |
| 32   | Augmentation before/after (2 per class, 6 columns)   | `06`                                                               |

### 3. Model and training (cells 33–46)

| Cell | Content                                                                                                       |
| ---- | ------------------------------------------------------------------------------------------------------------- |
| 34   | Backbone with a 3-way head, DataParallel, `CrossEntropyLoss`, AdamW, cosine schedule, scaler                  |
| 36   | Model profile: parameters, size, FLOPs (`model_profile.json`)                                                 |
| 38   | `run_epoch`: `outputs = model(images)` (no squeeze), `softmax(dim=1)` probabilities, argmax accuracy, OvR AUC |
| 40   | Training tracker wrapper (identical to binary)                                                                |
| 42   | 10-epoch loop with TensorBoard and best-AUC checkpoint                                                        |
| 44   | Training summary (`training_history.csv`, `training_summary.json`)                                            |
| 46   | Training curves (`07_training_curves.png`). The AUC panel shows macro OvR.                                    |

### 4. Test, save, evaluate (cells 47–64)

| Cell | Content                                                                                                                                                                                                                                                                                                      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 48   | Reload best, run the test pass, print `FINAL UNSEEN TEST RESULTS`                                                                                                                                                                                                                                            |
| 50   | Save `final_*_multi_class.pth`                                                                                                                                                                                                                                                                               |
| 52   | Single-GPU re-inference and per-image predictions (`prob_glioma`, `prob_meningioma`, `prob_pituitary`)                                                                                                                                                                                                       |
| 54   | Metrics. The multi-class branch adds `roc_auc_ovr_macro`, `roc_auc_ovr_weighted`, `roc_auc_ovo_macro`, `pr_auc_macro`, `pr_auc_micro`, `specificity_macro` and `top_2_accuracy`. The binary-only TP/FP/Youden fields are absent. Per-class one-vs-rest metrics and the classification report are also saved. |
| 56   | 3x3 confusion matrices, counts and row-normalised (`08`)                                                                                                                                                                                                                                                     |
| 58   | One-vs-rest ROC and PR per class, plus the **micro-average** curve (`09`)                                                                                                                                                                                                                                    |
| 60   | Calibration on max-softmax confidence (`10`)                                                                                                                                                                                                                                                                 |
| 62   | Per-class metric bars on test (`11`)                                                                                                                                                                                                                                                                         |
| 64   | Correct and misclassified samples (`12`)                                                                                                                                                                                                                                                                     |

### 5. Explainability, benchmark, packaging (cells 65–74)

| Cell | Content                                                                                                                        |
| ---- | ------------------------------------------------------------------------------------------------------------------------------ |
| 66   | Generic Grad-CAM. For a multi-logit model, the score is the logit of the predicted class.                                      |
| 68   | Grad-CAM for the 2 most confident correct predictions per class (6 columns) and the 6 most confident errors (`13`, `14`)       |
| 70   | Latency (batch 1) and throughput (batch 32) benchmark                                                                          |
| 72   | `benchmark_summary.{json,csv}`. The headline set uses `test_roc_auc_ovr_macro`, `test_pr_auc_macro` and `test_top_2_accuracy`. |
| 74   | Zip of 39 files: the binary 37 plus the dedup plot and table                                                                   |

## Per-notebook differences

EfficientNetB0 is the reference. Differences that are only comments, `# ===== … =====` banners or print wording are left out.

| Notebook       | Backbone and head                                                                           | Batch   | Img     | Grad-CAM layer (resolved)                           | Other deviations                                                                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------- | ------- | ------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EfficientNetB0 | `torchvision.efficientnet_b0(DEFAULT)`, `classifier[1] → Linear(1280,3)`                    | 256     | 224     | `features[-1]` (Conv2dNormActivation)               | Reference. Unused `models` import.                                                                                                                                                              |
| ResNet50       | `torchvision.resnet50(DEFAULT)`, `fc → Linear(2048,3)`                                      | 256     | 224     | `layer4[-1]` (Bottleneck)                           | Unused `classification_report` import                                                                                                                                                           |
| InceptionV3    | `torchvision.inception_v3(DEFAULT)`, `fc → Linear(2048,3)`, then `model.aux_logits = False` | 128     | **299** | `Mixed_7c` (InceptionE)                             | **No auxiliary loss** (binary uses `main + 0.4·aux`). `AuxLogits` is left as an unused 1000-class head (see anomalies). `import imagehash` sits in the imports cell instead of the dedup cell.  |
| CoAtNet-0      | `timm.create_model("coatnet_0_rw_224", pretrained=True)`, `reset_classifier(3)`             | **128** | 224     | `stages[-1]` (MaxxVitStage)                         | Needs `timm`. Run a day later than the others (18 Sep vs 17 Sep 2026).                                                                                                                          |
| SwinBase       | **torchvision** `swin_b(Swin_B_Weights.DEFAULT)`, `head → Linear(1024,3)`                   | 64      | 224     | `features[-1][-1].norm1` (LayerNorm, channels-last) | Binary uses **timm** `swin_base_patch4_window7_224`, so the pretrained weights differ. Extra prints (`Model loaded successfully on cuda`, parameter count). A redundant second `swin_b` import. |
| ViT-B16        | `torchvision.vit_b_16(DEFAULT)`, `heads.head → Linear(768,3)`                               | **32**  | 224     | `encoder.layers[-1].ln_1` (LayerNorm, tokens)       | Replaces `heads.head`. Binary replaces the whole `heads` module. The result is equivalent.                                                                                                      |

**Reasons for the differences, where the notebooks give evidence:**

- **Image size 299 for InceptionV3.** This is the Inception default and follows the binary notebook's stated reasoning.
- **Batch sizes.** No rationale is written in any classification notebook. Compared with binary, CoAtNet-0 rose from 64 to 128 (peak 11.3 GB on GPU0) and ViT-B16 fell from 128 to 32 (peak 3.9 GB, far below the limit). Memory therefore does not explain the ViT change. The reason is **unknown**. Batch sizes vary from 32 to 256, so step counts range from 120 to 1020 over 10 epochs, and the comparison is not controlled for optimisation budget.
- **Dropping the Inception auxiliary loss.** Not explained. Setting `aux_logits=False` after construction makes torchvision return a plain tensor in train mode, so the shared `run_epoch` (which expects one output) works without the Inception-specific branch that binary needed.
- **torchvision vs timm Swin.** Not explained.

**Differences from the binary pipeline (all notebooks):** separate imports, config, paths, transforms and loader cells (hence 75 cells instead of 65); `.jpg`-only collection; pHash deduplication and plot `04`; `long` labels; a 3-way head; `CrossEntropyLoss`; softmax and argmax metrics; OvR AUC; multi-class metric branch; micro-average ROC/PR; `_multi_class` weight filenames; 39 zipped files.

## Per-model results and training dynamics

Validation macro OvR AUC per epoch (`training_history.csv`). **Bold** marks the checkpointed epoch.

| Epoch | EffNetB0   | ResNet50   | InceptionV3 | CoAtNet-0  | SwinBase   | ViT-B16    |
| ----- | ---------- | ---------- | ----------- | ---------- | ---------- | ---------- |
| 1     | 0.9382     | 0.9523     | 0.9738      | 0.8942     | 0.9853     | 0.8154     |
| 2     | 0.9687     | 0.9686     | 0.9914      | 0.9307     | 0.9937     | 0.8239     |
| 3     | 0.9801     | 0.9901     | 0.9950      | 0.9841     | 0.9943     | 0.8556     |
| 4     | 0.9900     | 0.9928     | 0.9912      | 0.9946     | 0.9970     | 0.8551     |
| 5     | 0.9929     | 0.9966     | 0.9952      | 0.9945     | 0.9973     | 0.8885     |
| 6     | 0.9939     | 0.9975     | 0.9906      | 0.9961     | 0.9966     | 0.9255     |
| 7     | 0.9952     | 0.9979     | 0.9961      | 0.9969     | **0.9974** | 0.9332     |
| 8     | 0.9957     | 0.9980     | **0.9963**  | 0.9969     | 0.9973     | 0.9567     |
| 9     | **0.9959** | **0.9981** | 0.9963      | **0.9975** | 0.9970     | 0.9644     |
| 10    | 0.9958     | 0.9981     | 0.9963      | 0.9973     | 0.9972     | **0.9689** |

Validation accuracy at the best epoch vs test accuracy: EfficientNetB0 0.9547 → 0.9759; ResNet50 0.9731 → 0.9759; InceptionV3 0.9745 → 0.9830; CoAtNet-0 0.9660 → 0.9844; SwinBase 0.9731 → 0.9858; ViT-B16 0.8810 → 0.9051. Every model scores **higher on test than on validation**. The notebooks give no explanation. With these sample sizes it is compatible with the validation split being slightly harder by chance.

Test confusion matrices (rows are true, columns are predicted, in the order Glioma, Meningioma, Pituitary), from `metrics/confusion_matrix_test.csv`:

| Model          | Glioma row | Meningioma row | Pituitary row | Meningioma recall | Errors |
| -------------- | ---------- | -------------- | ------------- | ----------------- | ------ |
| SwinBase       | 298, 2, 0  | 3, 188, 5      | 0, 0, 210     | 0.959             | 10     |
| CoAtNet-0      | 299, 1, 0  | 6, 188, 2      | 0, 2, 208     | 0.959             | 11     |
| InceptionV3    | 295, 5, 0  | 3, 192, 1      | 0, 3, 207     | 0.980             | 12     |
| ResNet50       | 293, 7, 0  | 3, 190, 3      | 0, 4, 206     | 0.969             | 17     |
| EfficientNetB0 | 294, 5, 1  | 4, 188, 4      | 0, 3, 207     | 0.959             | 17     |
| ViT-B16        | 272, 27, 1 | 22, 167, 7     | 4, 6, 200     | 0.852             | 67     |

Glioma and Pituitary are almost never confused with each other (at most 1 case, except ViT-B16). Nearly all errors involve Meningioma.

EfficientNetB0 per-class test metrics (`per_class_metrics_test.csv`):

| Class      | Support | Precision | Recall | Specificity | F1     | AUC    |
| ---------- | ------- | --------- | ------ | ----------- | ------ | ------ |
| Glioma     | 300     | 0.9866    | 0.9800 | 0.9901      | 0.9833 | 0.9989 |
| Meningioma | 196     | 0.9592    | 0.9592 | 0.9843      | 0.9592 | 0.9977 |
| Pituitary  | 210     | 0.9764    | 0.9857 | 0.9899      | 0.9810 | 0.9997 |

**Per-model notes**

- **SwinBase.** Strongest from the start: validation AUC 0.985 after epoch 1. The best epoch is 7, and validation accuracy keeps rising to 0.9816 at epoch 10. Pituitary recall on test is perfect (210/210).
- **CoAtNet-0.** Slow start (val accuracy 0.62 at epoch 1, 0.69 at epoch 2), then converges. Slowest wall-clock run (28 min). Top-2 accuracy is 1.0 on test.
- **InceptionV3.** Validation loss is noisy (0.218 at epoch 4, 0.253 at epoch 6) before settling at about 0.10. Best test ROC and PR AUC.
- **ResNet50.** Smooth and monotonic. Best validation AUC (0.9981) of all six, but only mid-table on test accuracy.
- **EfficientNetB0.** Lowest validation accuracy among the strong models (0.955). Validation AUC plateaus at about 0.996 from epoch 8.
- **ViT-B16.** Training accuracy only reaches 0.881 and validation AUC is still climbing at epoch 10. It has batch size 32, the smallest, so it takes the most steps, yet it still underfits. With the same LR and 10 epochs, this recipe does not suit ViT-B16 on this data.

## Visualisations

Each run folder contains 14 plots. Representative examples follow. Change the folder name to view any other model.

**Training curves.** SwinBase vs ViT-B16:

![SwinBase training curves](classification-swinbase/plots/07_training_curves.png)
![ViT-B16 training curves](classification-vit-b16/plots/07_training_curves.png)

**Confusion matrices.** SwinBase and ViT-B16:

![SwinBase confusion matrices](classification-swinbase/plots/08_confusion_matrices.png)
![ViT-B16 confusion matrices](classification-vit-b16/plots/08_confusion_matrices.png)

**One-vs-rest ROC / PR with micro-average.** InceptionV3:

![InceptionV3 ROC and PR](classification-inceptionv3/plots/09_roc_pr_curves.png)

**Calibration and per-class metrics.** CoAtNet-0:

![CoAtNet-0 calibration](classification-coatnet0/plots/10_calibration.png)
![CoAtNet-0 per-class metrics](classification-coatnet0/plots/11_per_class_metrics.png)

**Augmentation and prediction samples.** EfficientNetB0 and ResNet50:

![Augmentation](classification-efficientnetb0/plots/06_augmentation_before_after.png)
![ResNet50 prediction samples](classification-resnet50/plots/12_prediction_samples.png)

**Grad-CAM.** SwinBase, correct and misclassified:

![SwinBase Grad-CAM correct](classification-swinbase/plots/13_gradcam_correct.png)
![SwinBase Grad-CAM misclassified](classification-swinbase/plots/14_gradcam_misclassified.png)

In the SwinBase "most confident correct" panel, the Meningioma heatmaps sit on the visible lesion. The Glioma and Pituitary heatmaps mostly highlight image corners and borders outside the brain. This is a visual observation from one figure, not a measured result. It may point to background or acquisition cues that differ between sources or classes. Such cues are plausible because the classes come from datasets with different image sizes and framing, but the notebooks do not test this. Compare with the CNN Grad-CAMs:

![ResNet50 Grad-CAM correct](classification-resnet50/plots/13_gradcam_correct.png)

## Inconsistencies, anomalies and limitations

1. **EfficientNetB0: the test-cell accuracy disagrees with the metrics files.** The original test cell prints `Test Accuracy: 0.9773` (690/706). `metrics_test.json` and `benchmark_summary.json` report 0.9759 (689/706). The first number comes from the DataParallel model under fp16 autocast. The second comes from the single-GPU deep copy used in cell 52. Most likely a near-tied prediction flipped because of numerical differences, but the notebook does not show which image. The other five classification runs agree between the two paths. This README uses the metrics files.
2. **InceptionV3 keeps an unused auxiliary head.** Only `fc` is replaced. `AuxLogits.fc` still maps 768 → 1000 ImageNet classes. With `aux_logits=False` it is never called, but it is counted in the profile: 25.118 M parameters vs 24.346 M in binary, where the auxiliary head was replaced with a 1-output layer. The difference of 772,329 parameters is exactly (768·1000+1000) − (768+1) for the aux head, plus (2048·3+3) − (2048+1) for the main head. It is also stored in the checkpoint, so anyone loading the weights must construct the model with the same unmodified auxiliary head.
3. **Deduplication is cross-class and order-dependent.** Glioma images always win ties. Removals across labels are neither counted nor flagged, which could hide mislabelled duplicates.
4. **EDA describes pre-deduplication data.** Plots `01`–`03` and `image_dimension_stats.csv` cover 5828 images, not the 4705 used for training.
5. **The official sartajbhuvaji Testing split is merged into training data.** Results are not comparable with published numbers that use that split.
6. **Test > validation for every model.** This is unexplained. Single seed, no confidence intervals. At 706 images, 1 error equals 0.14 percentage points.
7. **Unequal batch sizes (32–256)** at the same LR and epoch count. The optimisation budget is not controlled, and ViT-B16's batch-size change from binary has no stated reason.
8. **Checkpoint selection uses validation macro AUC,** which saturates near 0.997. Selection is therefore insensitive to accuracy differences at the end of training.
9. **Grad-CAM for SwinBase focuses outside the brain** for some classes (see above). The model may be relying partly on non-anatomical cues.
10. **`Classification/EfficientNetB0.pth`** (16,359,363 bytes, dated 1 Mar 2026) is the same size as `classification-efficientnetb0/weights/final_efficientnet_b0_brain_tumor_multi_class.pth` but has a different MD5 (`15162bc9…` vs `4ae7790e…`). No notebook here writes it, so its provenance and performance are unknown.
11. **No "no tumour" class.** The model always outputs one of three tumour types, even for a healthy scan.
12. **Reproducibility is approximate** (`cudnn.benchmark`, TF32, AMP). Kaggle paths are hard-coded. Evaluation in `run_epoch` does not use `torch.no_grad()`.

## Reproducing a run

1. Create a Kaggle notebook with **GPU T4 x2** and attach `sartajbhuvaji/brain-tumor-classification-mri` and `moundir8/jun-cheng-brain-tumor-mri-dataset-jpg`, so that the paths in cell 10 resolve.
2. Install or verify `imagehash`, and `timm` for CoAtNet-0. Versions recorded in the runs: Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8.
3. Run all cells in order. The tracker (cell 40) must run before training (cell 42). The pHash pass over 5828 images and its O(n²) comparison run before training starts.
4. Download `/kaggle/working/<RUN_NAME>-outputs.zip`. Unzipped, it matches the run folders here.

Loading a model:

```python
import torch, torch.nn as nn
from torchvision.models import swin_b
m = swin_b()
m.head = nn.Linear(m.head.in_features, 3)
m.load_state_dict(torch.load("classification-swinbase/weights/best_swinbase_multi_class.pth", map_location="cpu"))
m.eval()
# RGB → Resize((224,224)) → ToTensor → ImageNet Normalize; softmax over [Glioma, Meningioma, Pituitary]
```

For InceptionV3, build `inception_v3()` with the default 1000-class auxiliary head, replace only `fc` with `Linear(2048, 3)`, load the checkpoint, and use a 299-px input.

## How Classification relates to Binary Detection

|                           | [Binary Detection](../Binary%20Detection/README.md)                      | Classification (this folder)                                                       |
| ------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Question                  | Tumour present?                                                          | Which tumour type?                                                                 |
| Classes                   | No Tumor / Tumor                                                         | Glioma / Meningioma / Pituitary                                                    |
| Data                      | navoneel + ahmedhamada0, 3253 images, no dedup                           | sartajbhuvaji + Jun Cheng, 5828 → 4705 after pHash dedup                           |
| Split sizes               | 2277 / 488 / 488                                                         | 3293 / 706 / 706                                                                   |
| Head / loss               | 1 logit, BCEWithLogits                                                   | 3 logits, CrossEntropy                                                             |
| Selection metric          | Val ROC AUC                                                              | Val macro OvR ROC AUC                                                              |
| Models                    | 9                                                                        | 6 (no DenseNet121, MobileNetV3-Large or VGG16)                                     |
| Best test accuracy        | 0.9939 (DenseNet121, ResNet50, EfficientNetB0, CoAtNet-0 tied)           | 0.9858 (SwinBase)                                                                  |
| Same-architecture changes | Swin via timm. Inception with aux loss. CoAtNet batch 64. ViT batch 128. | Swin via torchvision. Inception without aux loss. CoAtNet batch 128. ViT batch 32. |

The two tasks are designed to work together: binary detection first, then tumour typing for positive cases. That is the only sensible use of a classifier with no normal class. **No notebook in either folder chains the two models or evaluates the cascade.** They were trained on disjoint source datasets, so the classifier's accuracy on images flagged by the detector has not been measured.
