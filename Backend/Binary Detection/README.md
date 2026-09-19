# Binary Detection: Brain Tumour Present vs Absent

This folder benchmarks **nine pretrained image classifiers** on one task: deciding whether a brain MRI slice shows a tumour (`Tumor`) or not (`No Tumor`). Each architecture has its own notebook. All nine notebooks follow the same pipeline, use the same data, split and hyperparameters, and write the same set of artefacts. The only intended difference between them is the model.

Everything in this README comes from the executed notebook outputs and the exported artefacts in this folder (`metrics/*.csv|json|txt`, `plots/*.png`, `weights/*.pth`, `tensorboard/`). Where the evidence does not explain a choice, the README says so instead of guessing.

For the follow-on three-class tumour-type task, see [../Classification/README.md](../Classification/README.md).

## Contents

- [Folder layout](#folder-layout)
- [Headline results](#headline-results)
- [End-to-end pipeline](#end-to-end-pipeline)
- [Dataset](#dataset)
- [Shared configuration](#shared-configuration)
- [Cell-by-cell walkthrough (shared template)](#cell-by-cell-walkthrough-shared-template)
- [Per-notebook differences](#per-notebook-differences)
- [Per-model results and training dynamics](#per-model-results-and-training-dynamics)
- [Visualisations](#visualisations)
- [Inconsistencies, anomalies and limitations](#inconsistencies-anomalies-and-limitations)
- [Reproducing a run](#reproducing-a-run)
- [How Binary Detection relates to Classification](#how-binary-detection-relates-to-classification)

## Folder layout

```
Binary Detection/
├── EfficientNetB0.pth                         # standalone weights, not produced by these notebooks (see anomalies)
├── binary-detection-<model>.ipynb             # 9 notebooks, 65 cells each
└── binary-detection-<model>/                  # unzipped output of each notebook
    ├── metrics/     # 21 files: CSV / JSON / TXT
    ├── plots/       # 13 PNGs (01–03, 05–14; 04 is classification-only)
    ├── tensorboard/<YYYYMMDD-HHMMSS>/events.out.tfevents.*
    └── weights/     # best_<model>.pth and final_<model>_brain_tumor.pth
```

| Notebook                                                                             | Output folder                                                              | Weights                                                                 |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [binary-detection-coatnet0.ipynb](binary-detection-coatnet0.ipynb)                   | [binary-detection-coatnet0/](binary-detection-coatnet0/)                   | `best_coatnet.pth`, `final_coatnet_brain_tumor.pth`                     |
| [binary-detection-densenet121.ipynb](binary-detection-densenet121.ipynb)             | [binary-detection-densenet121/](binary-detection-densenet121/)             | `best_densenet121.pth`, `final_densenet121_brain_tumor.pth`             |
| [binary-detection-efficientnetb0.ipynb](binary-detection-efficientnetb0.ipynb)       | [binary-detection-efficientnetb0/](binary-detection-efficientnetb0/)       | `best_efficientnet.pth`, `final_efficientnet_brain_tumor.pth`           |
| [binary-detection-inceptionv3.ipynb](binary-detection-inceptionv3.ipynb)             | [binary-detection-inceptionv3/](binary-detection-inceptionv3/)             | `best_inception.pth`, `final_inception_brain_tumor.pth`                 |
| [binary-detection-mobilenetv3-large.ipynb](binary-detection-mobilenetv3-large.ipynb) | [binary-detection-mobilenetv3-large/](binary-detection-mobilenetv3-large/) | `best_mobilenetv3_large.pth`, `final_mobilenetv3_large_brain_tumor.pth` |
| [binary-detection-resnet50.ipynb](binary-detection-resnet50.ipynb)                   | [binary-detection-resnet50/](binary-detection-resnet50/)                   | `best_resnet50.pth`, `final_resnet50_brain_tumor.pth`                   |
| [binary-detection-swinbase.ipynb](binary-detection-swinbase.ipynb)                   | [binary-detection-swinbase/](binary-detection-swinbase/)                   | `best_swin.pth`, `final_swin_brain_tumor.pth`                           |
| [binary-detection-vgg16.ipynb](binary-detection-vgg16.ipynb)                         | [binary-detection-vgg16/](binary-detection-vgg16/)                         | `best_vgg16.pth`, `final_vgg16_brain_tumor.pth`                         |
| [binary-detection-vit-b16.ipynb](binary-detection-vit-b16.ipynb)                     | [binary-detection-vit-b16/](binary-detection-vit-b16/)                     | `best_vit.pth`, `final_vit_brain_tumor.pth`                             |

The `final_*.pth` file is saved after the best checkpoint has been reloaded, so it holds the **best-validation-AUC weights**, not the weights from the last epoch. The two files differ slightly in size and hash only because they are serialised separately.

## Headline results

Test set: 488 images (240 No Tumor, 248 Tumor). Source: `metrics/benchmark_summary.json` in each run folder. The positive class is `Tumor`. Metrics use `argmax`, which for the two-column probability vector is the same as a 0.5 threshold.

| Model             | Test acc   | Balanced acc | Sensitivity | Specificity | FN  | FP  | ROC AUC     | MCC        | Log loss   | ECE        | Best epoch |
| ----------------- | ---------- | ------------ | ----------- | ----------- | --- | --- | ----------- | ---------- | ---------- | ---------- | ---------- |
| DenseNet121       | **0.9939** | **0.9940**   | 0.9879      | 1.0000      | 3   | 0   | **0.99998** | **0.9878** | **0.0189** | **0.0044** | 4          |
| ResNet50          | **0.9939** | **0.9940**   | 0.9879      | 1.0000      | 3   | 0   | 0.99997     | **0.9878** | 0.0217     | 0.0057     | 7          |
| EfficientNetB0    | **0.9939** | 0.9938       | **1.0000**  | 0.9875      | 0   | 3   | 0.99983     | **0.9878** | 0.0233     | 0.0049     | 4          |
| CoAtNet-0         | **0.9939** | 0.9938       | **1.0000**  | 0.9875      | 0   | 3   | 0.99987     | **0.9878** | 0.0253     | 0.0084     | 6          |
| InceptionV3       | 0.9877     | 0.9878       | 0.9798      | 0.9958      | 5   | 1   | 0.99861     | 0.9755     | 0.0419     | 0.0078     | 3          |
| SwinBase          | 0.9590     | 0.9594       | 0.9355      | 0.9833      | 16  | 4   | 0.99614     | 0.9192     | 0.1117     | 0.0235     | 6          |
| ViT-B16           | 0.9488     | 0.9485       | 0.9637      | 0.9333      | 9   | 16  | 0.98616     | 0.8978     | 0.1525     | 0.0323     | 10         |
| MobileNetV3-Large | 0.9078     | 0.9093       | 0.8185      | 1.0000      | 45  | 0   | 0.99808     | 0.8302     | 0.3698     | 0.0685     | 10         |
| VGG16             | 0.8832     | 0.8835       | 0.8629      | 0.9042      | 34  | 23  | 0.96760     | 0.7673     | 0.2448     | 0.0437     | 10         |

Cost and speed. Source: `model_profile.json`, `training_summary.json`, `inference_benchmark.json`. All runs used 2x Tesla T4 on Kaggle. Inference was measured on one T4 with fp16 autocast.

| Model             | Params (M) | GFLOPs/img | State dict (MB) | Input | Batch | Train time | Mean epoch (s) | Latency bs=1 (ms) | Throughput bs=32 (img/s) | Peak GPU0 alloc (MB) |
| ----------------- | ---------- | ---------- | --------------- | ----- | ----- | ---------- | -------------- | ----------------- | ------------------------ | -------------------- |
| EfficientNetB0    | 4.009      | 0.769      | 15.58           | 224   | 256   | 0:16:17    | 97.69          | 10.76             | 1364.24                  | 10559.6              |
| MobileNetV3-Large | 4.203      | 0.431      | 16.23           | 224   | 256   | 0:11:58    | 71.77          | 8.08              | 2408.52                  | 5415.8               |
| DenseNet121       | 6.955      | 5.666      | 27.11           | 224   | 64    | 0:08:28    | 50.71          | 19.43             | 629.80                   | 4267.1               |
| ResNet50          | 23.510     | 8.174      | 89.99           | 224   | 256   | 0:05:24    | 32.27          | 7.35              | 771.21                   | 10838.9              |
| InceptionV3       | 24.346     | 11.422     | 93.20           | 299   | 128   | 0:04:51    | 29.07          | 14.05             | 514.15                   | 13518.5              |
| CoAtNet-0         | 26.667     | 8.416      | 101.86          | 224   | 64    | 0:29:08    | 174.68         | 13.91             | 520.34                   | 5921.2               |
| ViT-B16           | 85.799     | 22.541     | 327.36          | 224   | 128   | 0:04:41    | 27.62          | 8.80              | 286.88                   | 10462.3              |
| SwinBase          | 86.744     | 30.860     | 331.03          | 224   | 64    | 0:06:34    | 38.95          | 28.99             | 195.63                   | 10801.8              |
| VGG16             | 134.265    | 30.932     | 512.19          | 224   | 256   | 0:12:28    | 74.06          | 8.19              | 338.15                   | 13313.8              |

**Reading these results**

- Four models (DenseNet121, ResNet50, EfficientNetB0, CoAtNet-0) tie at 3 errors out of 488. They differ in which way they fail. DenseNet121 and ResNet50 miss 3 tumours (FN) and raise no false alarms. EfficientNetB0 and CoAtNet-0 catch every tumour but give 3 false alarms. If a missed tumour costs more than a false alarm, **EfficientNetB0** gives 100% sensitivity at the lowest compute (0.769 GFLOPs, 15.6 MB). On probability quality (log loss, ECE, ROC AUC), **DenseNet121** is best.
- Wall-clock training time mostly measures data loading, not model cost. EfficientNetB0 (0.77 GFLOPs) took 16 minutes while ResNet50 (8.2 GFLOPs) took 5. Throughput was 25.8 train images/s for EfficientNetB0 and 77.5 for ResNet50. The runs happened one after another on different Kaggle sessions (each TensorBoard event file has a different host id), so I/O caching conditions were not controlled. Compare training times with caution.
- MobileNetV3-Large reaches a ROC AUC of 0.998 but only 0.908 accuracy. Its ranking is good, but the 0.5 threshold is badly placed (see [anomalies](#inconsistencies-anomalies-and-limitations)).
- VGG16, ViT-B16 and MobileNetV3-Large had their best validation AUC at the final epoch (10). They were still improving when training stopped, so these results reflect an under-trained model with this budget, not the architecture's ceiling.

## End-to-end pipeline

```
Kaggle inputs (2 datasets, yes/no folders)
  └─ os.walk → DataFrame(path, label)            3253 images, 0 = No Tumor, 1 = Tumor
      ├─ EDA: class distribution, samples, image sizes, intensity histograms (no mutation)
      └─ stratified split 70/15/15 (seed 42)     train 2277 / val 488 / test 488
          └─ BrainDataset + transforms (resize, flip, rotate, jitter, ImageNet normalise)
              └─ pretrained backbone, head → 1 logit, nn.DataParallel over 2 GPUs
                  └─ 10 epochs: AdamW 3e-4, cosine LR, BCEWithLogits, AMP fp16
                      ├─ keep checkpoint with best val AUC (best_*.pth)
                      └─ reload best → test → save final_*.pth
                          └─ single-GPU re-inference on val + test → per-image predictions
                              ├─ metrics, confusion matrices, ROC/PR, calibration, per-class
                              ├─ Grad-CAM (correct and misclassified)
                              ├─ latency / throughput benchmark
                              └─ benchmark_summary.{json,csv} → zip of everything
```

The first markdown cell of each notebook states the design rule. The data, split, training and testing cells are the original cells. The cells added for observation "only observe the run" and wrap any randomness in `preserved_rng_state()`, so adding them does not change the training outcome.

## Dataset

**Sources.** Kaggle paths hard-coded in the data collection cell:

| Label        | Paths                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 `No Tumor` | `/kaggle/input/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/no`, `/kaggle/input/datasets/ahmedhamada0/brain-tumor-detection/no`   |
| 1 `Tumor`    | `/kaggle/input/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/yes`, `/kaggle/input/datasets/ahmedhamada0/brain-tumor-detection/yes` |

The notebooks accept extensions `.jpg`, `.jpeg`, `.png` and `.tif` (case-insensitive) and walk folders recursively.

**Class distribution.** 3253 images in total: 1598 No Tumor (49.1%) and 1655 Tumor (50.9%). The classes are close to balanced, so no class weighting or resampling is used.

![Class distribution](binary-detection-efficientnetb0/plots/01_class_distribution.png)

**Split.** Two chained `train_test_split` calls, stratified on `label` with `random_state=42`: 70% train, then the remaining 30% halved into validation and test. Source: `metrics/split_distribution.csv`. The split is identical in all nine runs.

|           | Train    | Validation | Test    |
| --------- | -------- | ---------- | ------- |
| No Tumor  | 1119     | 239        | 240     |
| Tumor     | 1158     | 249        | 248     |
| **Total** | **2277** | **488**    | **488** |

![Split distribution](binary-detection-efficientnetb0/plots/05_split_distribution.png)

**Image properties.** Source: `metrics/image_dimension_stats.csv`. Image sizes vary a lot, and the two classes have different size distributions:

| Class    | Width mean ± std (min–max) | Width median | Height mean ± std (min–max) | Height median |
| -------- | -------------------------- | ------------ | --------------------------- | ------------- |
| No Tumor | 302.6 ± 154.9 (150–1920)   | 227          | 308.9 ± 144.9 (168–1080)    | 240           |
| Tumor    | 351.5 ± 195.2 (175–1275)   | 283          | 400.2 ± 208.1 (167–1427)    | 342           |

All images are resized to a fixed square (224 or 299) **without keeping the aspect ratio**. Because size differs systematically between classes, the amount of resize distortion is partly class-dependent. The notebooks do not test whether models exploit this.

![Image properties](binary-detection-efficientnetb0/plots/03_image_properties.png)

![Sample images](binary-detection-efficientnetb0/plots/02_sample_images.png)

The sample grid shows mixed MRI sequences, contrasts, crops and borders across both sources. One `No Tumor` sample (third column, top row) has focal bright spots. This is a visual observation only. The notebooks do not audit labels.

**No deduplication.** Unlike [Classification](../Classification/README.md), this pipeline does **not** remove duplicate or near-duplicate images before splitting. Two public datasets are merged, and the notebooks do not check for overlap between them or within them. If duplicates exist, they could fall on both sides of the train/test split. The near-perfect scores of the top models should be read with that in mind.

## Shared configuration

These values are identical in every notebook unless listed under [per-notebook differences](#per-notebook-differences).

| Setting            | Value                                                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Seed               | `42` (Python `random`, NumPy, `torch.manual_seed`)                                                                                                                                     |
| Epochs             | `10`                                                                                                                                                                                   |
| Learning rate      | `3e-4`                                                                                                                                                                                 |
| Optimiser          | `AdamW(model.parameters(), lr=3e-4)`. Default weight decay 0.01. All layers trainable, nothing frozen.                                                                                 |
| Scheduler          | `CosineAnnealingLR(T_max=10)`, stepped once per epoch after validation. The logged LR runs from 3.0e-4 (epoch 1) to 7.34e-6 (epoch 10).                                                |
| Loss               | `BCEWithLogitsLoss` on a single logit. Labels are `float32`.                                                                                                                           |
| Precision          | `torch.amp.autocast("cuda")` with `GradScaler`. TF32 is enabled for matmul and cuDNN. `cudnn.benchmark = True`.                                                                        |
| Image size         | 224 (299 for InceptionV3)                                                                                                                                                              |
| Batch size         | 256 by default. Overridden per model, see below.                                                                                                                                       |
| DataLoader         | `num_workers=4`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`. The train loader uses `shuffle=True, drop_last=True`. Val and test loaders do not shuffle or drop. |
| Train augmentation | `Resize((S,S))` → `RandomHorizontalFlip()` → `RandomRotation(10)` → `ColorJitter(brightness=0.2, contrast=0.2)` → `ToTensor()` → `Normalize(ImageNet mean/std)`                        |
| Eval transform     | `Resize((S,S))` → `ToTensor()` → `Normalize(ImageNet mean/std)`                                                                                                                        |
| Multi-GPU          | `nn.DataParallel` when more than one GPU is present. Both runs had 2x T4.                                                                                                              |
| Checkpoint rule    | Save when `val_auc > best_auc`, so the first epoch reaching the maximum wins.                                                                                                          |
| Environment        | Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, 2x Tesla T4 (from `benchmark_summary.json`)                                                                                             |

With `drop_last=True`, part of the training set is skipped in every epoch. The skipped images differ from epoch to epoch because of shuffling. With batch 256, the notebook runs 8 batches (2048 images) and drops 229 images per epoch. With batch 128 it runs 17 batches (2176). With batch 64 it runs 35 batches (2240).

## Cell-by-cell walkthrough (shared template)

Every notebook has 65 cells, laid out as markdown/code pairs. Indices below are 0-based. Code comments such as `# CELL 1 … # CELL 10` mark the "original" training cells. The unnumbered code cells are the added observation, evaluation and benchmarking cells.

### 1. Setup (cells 0–6)

| Cell   | What it does                                                                                                                                                                                                                                                                                                                           | Why                                                                   |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 0 (md) | Title, the 9-step pipeline, and the statement that the added cells only observe the run                                                                                                                                                                                                                                                | Lets you compare notebooks directly                                   |
| 2      | Imports. Config (`SEED`, `BATCH_SIZE`, `EPOCHS`, `LR`, `IMG_SIZE`, loader settings). `DEVICE`. Seeding. cuDNN benchmark and TF32 flags. Prints CUDA availability and GPU count (output: `True`, `2`).                                                                                                                                  | One place for hyperparameters                                         |
| 4      | `MODEL_NAME`, `TASK="binary-detection"`, `CLASS_NAMES=["No Tumor","Tumor"]`, `RUN_NAME`, and `get_gradcam_target_layer(net)`                                                                                                                                                                                                           | The only identity settings the shared cells need                      |
| 6      | Creates `/kaggle/working/outputs/<RUN_NAME>/{plots,metrics}`. Defines `save_fig`, `save_json`, `save_table`, `save_text`, `unwrap_model` (removes the `DataParallel` wrapper), `denormalise`, `sync_cuda`, and the `preserved_rng_state()` context manager, which snapshots and restores the Python, NumPy, torch and CUDA RNG states. | Consistent artefacts. Added cells cannot disturb training randomness. |

### 2. Data (cells 7–22)

| Cell | What it does                                                                                                                                                                      | Output (identical for all nine)                        |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 8    | `collect(paths, label)` walks the folders and builds `df[path,label]`                                                                                                             | `Total Images: 3253`, label 1: 1655, label 0: 1598     |
| 10   | Copies `df` to `df_raw`. Bar chart of classes and 6 random samples per class (`random_state=SEED`).                                                                               | `01_class_distribution.png`, `02_sample_images.png`    |
| 12   | Reads every image size. Builds grayscale intensity histograms from 100 sampled images per class (51 bins, density).                                                               | `03_image_properties.png`, `image_dimension_stats.csv` |
| 14   | Stratified 70/15/15 split                                                                                                                                                         | `Train: 2277 Val: 488 Test: 488`                       |
| 16   | Per-split class counts (grouped bars) and proportions (stacked bars)                                                                                                              | `05_split_distribution.png`, `split_distribution.csv`  |
| 18   | `BrainDataset`: opens the image, converts to RGB (grayscale MRIs are replicated to 3 channels to match ImageNet backbones), applies the transform, returns `(img, float32 label)` |                                                        |
| 20   | Transforms and the three `DataLoader`s                                                                                                                                            |                                                        |
| 22   | For 3 train images per class, shows the eval transform (top) and one random train augmentation (bottom)                                                                           | `06_augmentation_before_after.png`                     |

Plot number `04` is skipped on purpose. It is the deduplication plot, which exists only in Classification.

### 3. Model (cells 23–26)

| Cell | What it does                                                                                                                                                                                                                                                            |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 24   | Loads the pretrained backbone, replaces the head with a 1-output `Linear`, wraps it in `DataParallel`, moves it to the device, and creates the criterion, optimiser, scheduler and `GradScaler`                                                                         |
| 26   | Model profile: total and trainable parameters (always all trainable), buffer count, serialised `state_dict` size (via `BytesIO`), and FLOPs for one `1x3xSxS` input using `torch.utils.flop_counter.FlopCounterMode` on a CPU deep copy. Saved as `model_profile.json`. |

### 4. Training (cells 27–36)

| Cell | What it does                                                                                                                                                                                                                                                                                                                                                                                               |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 28   | `run_epoch(loader, training)`: train or eval mode, autocast forward, `loss = BCEWithLogits(logits.squeeze(1), labels)`, scaled backward and step while training, collects sigmoid probabilities. Returns mean batch loss, accuracy at 0.5, and ROC AUC. The eval pass does **not** use `torch.no_grad()`. The optimiser step is simply skipped.                                                            |
| 30   | **Training tracker.** Rebinds `run_epoch` to a wrapper that identifies the phase (train, val or test) by loader identity, times it with CUDA syncs, records LR, loss, accuracy, AUC and timings per epoch, drives an epoch-level tqdm bar, and resets GPU peak-memory stats. A `_tracks_history` flag prevents double wrapping if the cell is re-run. This lets the original training cell stay unchanged. |
| 32   | Original training loop: 10 epochs of train → val → `scheduler.step()`, TensorBoard scalars (`Loss/Train`, `Loss/Val`, `AUC/Train`, `AUC/Val`) to `/kaggle/working/runs/<timestamp>`, and a checkpoint on improved validation AUC (`model.module.state_dict()` when more than one GPU)                                                                                                                      |
| 34   | Training summary: history DataFrame, cumulative time, best epoch (`idxmax`, the first maximum, matching the `>` rule), total and mean times, train images/s (`len(train_loader)*BATCH_SIZE / mean train time`), time to best epoch, and peak allocated/reserved memory per GPU. Saved as `training_history.csv` and `training_summary.json`.                                                               |
| 36   | 2x3 figure. Loss, accuracy and AUC (train vs val, best epoch marked), generalisation gap (val loss minus train loss), LR, and stacked train/val time per epoch. Saved as `07_training_curves.png`.                                                                                                                                                                                                         |

Training accuracy and AUC are measured on **augmented** batches in train mode (dropout and BatchNorm active). They are not directly comparable to the validation numbers.

### 5. Testing and saving (cells 37–40)

| Cell | What it does                                                                                                                                         |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 38   | Loads `best_*.pth` into the (DataParallel) model, runs `run_epoch(test_loader, False)`, and prints `FINAL UNSEEN TEST RESULTS` (loss, accuracy, AUC) |
| 40   | Saves the reloaded best weights again as `final_*_brain_tumor.pth`                                                                                   |

### 6. Detailed evaluation (cells 41–54)

| Cell | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Artefacts                                                                                                                    |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 42   | Deep-copies the unwrapped model onto one GPU and re-runs val and test under autocast. For one logit it builds `[1-σ, σ]`. Renormalises in float64. Stores per-image path, true and predicted class, confidence, per-class probabilities and correctness.                                                                                                                                                                                                                                                                                            | `predictions_val.csv`, `predictions_test.csv`                                                                                |
| 44   | `compute_metrics`: accuracy, balanced accuracy, error rate, precision/recall/F1 (macro, weighted, micro), MCC, Cohen's κ, macro Jaccard, log loss, multi-class Brier score, 15-bin ECE, mean confidence. For binary runs it adds TP/FP/FN/TN, precision, sensitivity, specificity, NPV, FPR, FNR, F1, F2, ROC AUC, average precision, positive-class Brier score, and the Youden J statistic with its optimal threshold, sensitivity and specificity. Also computes per-class one-vs-rest metrics and the sklearn classification report (4 digits). | `metrics_{val,test}.json`, `per_class_metrics_{val,test}.csv`, `classification_report_{val,test}.txt`, `metrics_summary.csv` |
| 46   | Confusion matrices: counts and row-normalised, validation (top) and test (bottom)                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `confusion_matrix_{val,test}.csv`, `08_confusion_matrices.png`                                                               |
| 48   | ROC and PR curves for the positive class, with prevalence lines on the PR plot                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `09_roc_pr_curves.png`                                                                                                       |
| 50   | Reliability diagram (15 bins of predicted-class confidence) and confidence histograms for correct vs incorrect predictions (log y-axis)                                                                                                                                                                                                                                                                                                                                                                                                             | `calibration_bins_{val,test}.csv`, `10_calibration.png`                                                                      |
| 52   | Per-class precision, recall, specificity, F1 and AUC on test                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `11_per_class_metrics.png`                                                                                                   |
| 54   | 6 random correct and 6 random misclassified test images (seeded `RandomState(42)`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | `12_prediction_samples.png`                                                                                                  |

### 7. Explainability (cells 55–58)

| Cell | What it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 56   | Generic `GradCAM` class. A forward hook on the target layer captures activations, and a tensor hook captures gradients. `to_nchw` converts transformer tokens `(B,T,C)` (dropping a class token when T is not a perfect square) and channels-last maps `(B,H,W,C)` to `(B,C,H,W)`. For a single-logit model the score is `+logit` for class 1 and `−logit` for class 0. The CAM is ReLU of the gradient-mean-weighted activation sum, bilinearly upsampled and min-max normalised. It runs on a float32 deep copy. |
| 58   | Grad-CAM (input / heatmap / overlay) for the 2 most confident correct predictions per class (`13_gradcam_correct.png`) and the 6 most confident misclassifications (`14_gradcam_misclassified.png`). Failures are caught and stored in `gradcam_status`. Every run reports `ok`.                                                                                                                                                                                                                                   |

### 8. Benchmark and packaging (cells 59–64)

| Cell | What it does                                                                                                                                                                                                                                                                                                                  |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 60   | Inference benchmark on synthetic `randn` input under fp16 autocast. Batch 1: 10 warm-up and 100 timed iterations (mean, median, p95, std). Batch 32: 3 warm-up and 20 timed iterations (throughput). Also reports end-to-end test-set time from cell 42 and the original test-cell time. Saved as `inference_benchmark.json`. |
| 62   | Flattens config, profile, training, inference, validation and test metrics, Grad-CAM status and environment into one record. Saved as `benchmark_summary.json` and `.csv`, with a headline table.                                                                                                                             |
| 64   | Zips `outputs/<RUN_NAME>`, every `*.pth` in `/kaggle/working`, and the TensorBoard run into `<RUN_NAME>-outputs.zip` (37 files each). The run folders in this repository are the unzipped contents of these archives.                                                                                                         |

## Per-notebook differences

EfficientNetB0 is used as the reference here. Its code is the most "canonical" version of the template. Differences that are only comments, cell titles (`# CELL 1` vs `# Cell 1: …`) or equivalent reformatting are left out.

| Notebook          | Backbone and head                                                                                           | Batch | Img     | Grad-CAM layer (resolved type)                           | Other code deviations                                                                                                                                                                                                                          |
| ----------------- | ----------------------------------------------------------------------------------------------------------- | ----- | ------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EfficientNetB0    | `torchvision.efficientnet_b0(DEFAULT)`, `classifier[1] → Linear(1280,1)`                                    | 256   | 224     | `features[-1]` (Conv2dNormActivation)                    | Reference                                                                                                                                                                                                                                      |
| MobileNetV3-Large | `torchvision.mobilenet_v3_large(IMAGENET1K_V1)`, `classifier[3] → Linear(·,1)`                              | 256   | 224     | `features[-1]` (Conv2dNormActivation)                    | Pins `IMAGENET1K_V1` explicitly. Every other torchvision model uses `.DEFAULT`.                                                                                                                                                                |
| ResNet50          | `torchvision.resnet50(DEFAULT)`, `fc → Linear(2048,1)`                                                      | 256   | 224     | `layer4[-1]` (Bottleneck)                                | Unused import of `classification_report`                                                                                                                                                                                                       |
| VGG16             | `torchvision.vgg16(DEFAULT)`, `classifier[6] → Linear(4096,1)`                                              | 256   | 224     | `features[-1]` (**MaxPool2d**)                           | `BrainDataset` has no `try/except` around image loading. Grad-CAM hooks the final max-pool, not a conv layer.                                                                                                                                  |
| DenseNet121       | `torchvision.densenet121(DEFAULT)`, `classifier → Linear(1024,1)`                                           | 64    | 224     | `features[-1]` (BatchNorm2d, i.e. `norm5`)               | No `try/except` in the dataset. Sigmoid is applied inline.                                                                                                                                                                                     |
| InceptionV3       | `torchvision.inception_v3(DEFAULT, aux_logits=True)`, `fc → Linear(2048,1)`, `AuxLogits.fc → Linear(768,1)` | 128   | **299** | `Mixed_7c` (InceptionE)                                  | **Auxiliary loss:** `loss = main + 0.4·aux` while training. **The forward call uses the unwrapped module `m`**, see anomalies. TF32 flags removed. `GradScaler()` without a device argument. No "Using 2 GPUs" print. No dataset `try/except`. |
| CoAtNet-0         | `timm.create_model("coatnet_0_rw_224", pretrained=True)`, `reset_classifier(1)`                             | 64    | 224     | `stages[-1]` (MaxxVitStage)                              | Needs `timm`. `GradScaler()` without a device. Tensors moved without `non_blocking`. No dataset `try/except`.                                                                                                                                  |
| SwinBase          | `timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=1)`                         | 64    | 224     | `layers[-1].blocks[-1].norm1` (LayerNorm, channels-last) | Uses **timm** (Classification uses torchvision `swin_b`). Weights come from the Hugging Face Hub (output warns about an unauthenticated HF request). `GradScaler()` without a device.                                                          |
| ViT-B16           | `torchvision.vit_b_16(DEFAULT)`, `heads → Linear(768,1)` (replaces the whole `heads` Sequential)            | 128   | 224     | `encoder.layers[-1].ln_1` (LayerNorm, tokens)            | Unused `classification_report` import                                                                                                                                                                                                          |

**Reasons for the differences, where the notebooks give evidence:**

- **Batch size.** The only explicit rationale in the code is InceptionV3's comment `# smaller batch due to inception memory`, together with `IMG_SIZE = 299  # Inception default input size`. For the others (64 for DenseNet121, CoAtNet-0 and SwinBase; 128 for ViT-B16) no reason is written down. Recorded peak memory shows batch 256 already reaching 13.3 GB on GPU0 and 14.6 GB on GPU1 for VGG16, close to the T4's 16 GB. Memory limits are therefore a plausible but **unstated** reason. Because batch size changes both the number of optimiser steps per epoch (8 vs 17 vs 35) and gradient noise at a fixed LR, **this comparison is not perfectly controlled.**
- **Head replacement.** Each architecture exposes its classifier under a different attribute, so the code must differ.
- **Grad-CAM layer.** The notebook-settings markdown documents each choice as the last spatial or token stage of that architecture.
- **InceptionV3 auxiliary head.** Adding `0.4 × aux` is the standard Inception training recipe. The notebook applies it without comment.
- **timm vs torchvision.** CoAtNet has no torchvision implementation. For Swin, the notebook comment only says `# For Swin Transformer`.

## Per-model results and training dynamics

Per-epoch validation accuracy and AUC, from `metrics/training_history.csv`. **Bold** marks the checkpointed (best) epoch.

| Epoch | EffNetB0 AUC | DenseNet AUC | ResNet50 AUC | Inception AUC | CoAtNet AUC | Swin AUC   | ViT AUC    | MobileNet acc / AUC | VGG16 acc / AUC    |
| ----- | ------------ | ------------ | ------------ | ------------- | ----------- | ---------- | ---------- | ------------------- | ------------------ |
| 1     | 0.9723       | 0.9997       | 0.9643       | 0.9965        | 0.8434      | 0.8100     | 0.7206     | 0.699 / 0.9518      | 0.541 / 0.7569     |
| 2     | 0.9987       | 0.99988      | 0.9948       | 0.99992       | 0.9881      | 0.8479     | 0.7974     | 0.656 / 0.9728      | 0.676 / 0.8633     |
| 3     | 0.99983      | 0.99993      | 0.9972       | **1.0000**    | 0.9972      | 0.8653     | 0.8814     | 0.703 / 0.9872      | 0.531 / 0.7319     |
| 4     | **0.99995**  | **1.0000**   | 0.9963       | 0.99990       | 0.9960      | 0.9576     | 0.9326     | 0.797 / 0.9929      | 0.666 / 0.8194     |
| 5     | 0.99994      | 0.99997      | 0.99982      | 0.99992       | 0.99983     | 0.9923     | 0.9520     | 0.814 / 0.9935      | 0.744 / 0.8572     |
| 6     | 0.99992      | 1.0000       | 0.99983      | 0.9998        | **0.99993** | **0.9973** | 0.9623     | 0.814 / 0.9965      | 0.836 / 0.9039     |
| 7     | 0.99993      | 1.0000       | **0.99992**  | 0.99988       | 0.9969      | 0.9961     | 0.9817     | 0.838 / 0.9982      | 0.855 / 0.9482     |
| 8     | 0.99993      | 1.0000       | 0.99990      | 0.99988       | 0.9991      | 0.9951     | 0.9720     | 0.850 / 0.9988      | 0.887 / 0.9580     |
| 9     | 0.99993      | 1.0000       | 0.99990      | 0.99990       | 0.9991      | 0.9960     | 0.9868     | 0.873 / 0.9991      | 0.893 / 0.9680     |
| 10    | 0.99993      | 0.99998      | 0.99990      | 0.99990       | 0.9993      | 0.9957     | **0.9899** | **0.881 / 0.9993**  | **0.898 / 0.9687** |

Test confusion matrices (rows are true, columns are predicted, as `[No Tumor, Tumor]`), from `metrics/confusion_matrix_test.csv`:

| Model             | No Tumor row | Tumor row | Val matrix      |
| ----------------- | ------------ | --------- | --------------- |
| CoAtNet-0         | 237, 3       | 0, 248    | 234,5 / 0,249   |
| DenseNet121       | 240, 0       | 3, 245    | 239,0 / 0,249   |
| EfficientNetB0    | 237, 3       | 0, 248    | 237,2 / 0,249   |
| InceptionV3       | 239, 1       | 5, 243    | 237,2 / 0,249   |
| MobileNetV3-Large | 240, 0       | 45, 203   | 239,0 / 58,191  |
| ResNet50          | 240, 0       | 3, 245    | 238,1 / 2,247   |
| SwinBase          | 236, 4       | 16, 232   | 238,1 / 14,235  |
| VGG16             | 217, 23      | 34, 214   | 222,17 / 33,216 |
| ViT-B16           | 224, 16      | 9, 239    | 216,23 / 6,243  |

Youden-optimal thresholds on test (`test_youden_optimal_threshold`): EfficientNetB0 0.845, CoAtNet-0 0.999, DenseNet121 0.177, ResNet50 0.070, InceptionV3 0.675, SwinBase 0.238, ViT-B16 0.633, VGG16 0.728, **MobileNetV3-Large 0.00044**. These thresholds are derived from the test set itself, so they describe the model and should not be used as tuned operating points.

**Per-model notes**

- **EfficientNetB0.** Converges by epoch 3–4. Validation accuracy peaks at 0.9959 (epochs 4–6), and validation loss keeps drifting down to about 0.012 while AUC is saturated. Test: 0 FN, 3 FP.
- **DenseNet121.** Fastest to converge. Validation AUC is 0.9997 after one epoch and reaches 1.0000 with 100% validation accuracy at epoch 4. Lowest test log loss (0.0189) and ECE (0.0044).
- **ResNet50.** Short 5-minute training. The best checkpoint (epoch 7) matches DenseNet121's test confusion matrix exactly.
- **InceptionV3.** Validation AUC reaches 1.0 at epoch 3. The test result (0.9877, 5 FN) is slightly worse than validation (0.9959). Training accuracy hits 1.0 by epoch 6, which suggests some overfitting to the 299-px inputs. Only one GPU was used for compute (see anomalies).
- **CoAtNet-0.** Slowest run (29 min). Validation loss is noisy (0.13 at epoch 4, then 0.021 at epoch 5). Training accuracy reaches 1.0 at epoch 10 while validation loss rises again from 0.021 to 0.040. Test: 0 FN, 3 FP.
- **SwinBase.** Slow start: validation AUC stays at or below 0.87 for 3 epochs, then climbs quickly at epochs 4–6. Most of its errors are missed tumours (16 FN).
- **ViT-B16.** Learns steadily but slowly. Validation AUC is still rising at epoch 10 (0.9899). Training accuracy only reaches 0.944. The model is clearly under-trained at 10 epochs with this recipe.
- **MobileNetV3-Large.** Training loss falls to about 0.001 and training accuracy reaches 0.999 by epoch 4. Validation loss **rises** to 1.06 at epoch 2 and only recovers to 0.40 by epoch 10. Validation accuracy creeps from 0.66 to 0.88, while validation AUC stays at or above 0.95 throughout. Every error is a missed tumour (45 FN on test, 58 on validation). The model ranks cases correctly but pushes tumour probabilities below 0.5. The notebook does not investigate why.
- **VGG16.** Epoch-1 training loss is 2.13, and accuracy stays near chance (0.53–0.63) for 4 epochs before improving. Best at the last epoch with validation AUC 0.969. Worst model overall. Batch 256 on this 134 M-parameter network used about 14.6 GB on GPU1.

## Visualisations

Each run folder contains the same 13 plots. Representative examples follow. Swap the folder name to see any other model.

**Training curves.** EfficientNetB0 (stable) vs MobileNetV3-Large (validation loss diverges from training loss):

![EfficientNetB0 training curves](binary-detection-efficientnetb0/plots/07_training_curves.png)
![MobileNetV3-Large training curves](binary-detection-mobilenetv3-large/plots/07_training_curves.png)

**Confusion matrices.** DenseNet121 (best) and VGG16 (worst):

![DenseNet121 confusion matrices](binary-detection-densenet121/plots/08_confusion_matrices.png)
![VGG16 confusion matrices](binary-detection-vgg16/plots/08_confusion_matrices.png)

**ROC / PR curves.** ResNet50:

![ResNet50 ROC and PR](binary-detection-resnet50/plots/09_roc_pr_curves.png)

**Calibration.** MobileNetV3-Large. Validation ECE is 0.0855 and test ECE 0.0685. All 45 test errors are distributed across confidence bins up to 1.0, so many of them are confidently wrong:

![MobileNetV3-Large calibration](binary-detection-mobilenetv3-large/plots/10_calibration.png)

**Per-class metrics and prediction samples.** EfficientNetB0:

![EfficientNetB0 per-class metrics](binary-detection-efficientnetb0/plots/11_per_class_metrics.png)
![EfficientNetB0 prediction samples](binary-detection-efficientnetb0/plots/12_prediction_samples.png)

**Augmentation.**

![Augmentation](binary-detection-efficientnetb0/plots/06_augmentation_before_after.png)

**Grad-CAM.** Correct and misclassified examples, EfficientNetB0 and SwinBase:

![EfficientNetB0 Grad-CAM correct](binary-detection-efficientnetb0/plots/13_gradcam_correct.png)
![EfficientNetB0 Grad-CAM misclassified](binary-detection-efficientnetb0/plots/14_gradcam_misclassified.png)
![SwinBase Grad-CAM misclassified](binary-detection-swinbase/plots/14_gradcam_misclassified.png)

Grad-CAM is used here for qualitative inspection only. The notebooks do not measure how well the heatmaps localise the tumour, and none of the datasets has lesion masks.

## Inconsistencies, anomalies and limitations

1. **InceptionV3 did not actually use DataParallel for compute.** The model is wrapped in `nn.DataParallel`, but `run_epoch` calls the unwrapped module (`m = model.module …; m(images)`) in both training and evaluation. `training_summary.json` confirms this: `peak_gpu_memory_allocated_mb_cuda1 = 0.0` against 13518.5 MB on cuda0. Yet `model_profile.json` reports `data_parallel_gpus: 2`, because that field only checks whether the model is wrapped. Its training time is therefore single-GPU at batch 128.
2. **MobileNetV3-Large threshold collapse.** AUC is 0.998 but sensitivity only 0.8185 at 0.5. The Youden threshold is 0.00044, and validation loss rose while training loss approached 0. The notebook does not test for a cause. It also differs from the other runs in pinning `IMAGENET1K_V1` weights. That difference is documented here, but it is not shown to be the cause.
3. **Unequal batch sizes (64, 128, 256)** at the same LR and epoch count mean each model got a different number of optimiser steps (80 to 350 over 10 epochs). The comparison is controlled for data and schedule, but not for optimisation budget.
4. **No deduplication or leakage check** across the two merged source datasets (see [Dataset](#dataset)).
5. **Validation is used for model selection.** The reported validation metrics are optimistic because the checkpoint was selected on validation AUC. The test set is the unbiased estimate.
6. **Checkpoint selection uses AUC, not accuracy or loss.** Several models saturate validation AUC early (DenseNet121 hits 1.0 at epoch 4), so later epochs cannot replace the checkpoint even when their validation loss is lower.
7. **"Test loss" differs slightly from `log_loss`.** For example, EfficientNetB0 prints `Test Loss: 0.0240` but has `log_loss 0.0233`. The first is a mean of per-batch fp16 BCE values from the DataParallel model. The second is the sklearn log loss on float64 probabilities from the single-GPU re-inference. Test accuracy agrees between the two paths for all nine binary runs.
8. **Only one seed.** There are no repeated runs, so there are no confidence intervals. With 488 test images, 1 error equals 0.2 percentage points of accuracy. The top four models are statistically indistinguishable.
9. **Evaluation does not use `torch.no_grad()`** in the original `run_epoch`. This wastes memory but does not affect results.
10. **Reproducibility is approximate.** `cudnn.benchmark=True`, TF32 and AMP allow non-deterministic kernels.
11. **`Binary Detection/EfficientNetB0.pth`** (16,342,545 bytes, dated 27 Feb 2026) has the same byte size as `binary-detection-efficientnetb0/weights/final_efficientnet_brain_tumor.pth` but a different MD5 (`f7dadc87…` vs `4709fe14…`). No notebook in this folder writes a file with this name. Its provenance and metrics are unknown. It is most likely an earlier training of the same architecture, but that is not verifiable from this folder.
12. **Hard-coded Kaggle paths** (`/kaggle/input/…`, `/kaggle/working/…`). The notebooks will not run elsewhere unless these paths are edited.
13. **Minor code drift** between notebooks: missing `try/except` in some datasets, unused `classification_report` imports, `GradScaler()` with and without `"cuda"`, `non_blocking` sometimes omitted. None of these affects results.

## Reproducing a run

1. Create a Kaggle notebook with **GPU T4 x2** and attach the two datasets `navoneel/brain-mri-images-for-brain-tumor-detection` and `ahmedhamada0/brain-tumor-detection`, so that the paths in the data collection cell resolve. Adjust the paths if your mount layout differs.
2. Environment used: Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8), torchvision, scikit-learn, pandas, matplotlib, Pillow, tqdm, tensorboard. CoAtNet-0 and SwinBase also need `timm` and internet access to download pretrained weights (Swin pulls from the Hugging Face Hub).
3. Run all cells in order. The tracker cell (30) must run before the training cell (32), or cell 34 raises `No epoch history was recorded`.
4. Download `/kaggle/working/<RUN_NAME>-outputs.zip`. Unzipping it reproduces the `<RUN_NAME>/` folder layout found here.

Loading a trained model for inference (EfficientNetB0 example; other models follow the same pattern with their own head):

```python
import torch, torch.nn as nn
from torchvision.models import efficientnet_b0
m = efficientnet_b0()
m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
m.load_state_dict(torch.load("binary-detection-efficientnetb0/weights/best_efficientnet.pth", map_location="cpu"))
m.eval()
# input: RGB, Resize((224,224)), ToTensor, Normalize(ImageNet); P(tumour) = sigmoid(m(x))
```

Checkpoints are stored without the `module.` prefix because they were saved from `model.module`. For InceptionV3, build the model with `aux_logits=True` and replace both `fc` and `AuxLogits.fc` before loading, and use a 299-px input.

## How Binary Detection relates to Classification

|                | Binary Detection (this folder)                 | [Classification](../Classification/README.md)                              |
| -------------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| Question       | Is there a tumour?                             | Which tumour type is it?                                                   |
| Classes        | `No Tumor`, `Tumor`                            | `Glioma`, `Meningioma`, `Pituitary` (no "no tumour" class)                 |
| Data           | navoneel + ahmedhamada0 (3253 images)          | sartajbhuvaji + Jun Cheng (5828 → 4705 after dedup)                        |
| Deduplication  | None                                           | pHash, Hamming distance ≤ 5                                                |
| Head / loss    | 1 logit, `BCEWithLogitsLoss`                   | 3 logits, `CrossEntropyLoss`                                               |
| Models         | 9 (adds DenseNet121, MobileNetV3-Large, VGG16) | 6                                                                          |
| Notebook cells | 65                                             | 75 (split config cell, separate path cell, dedup cells, split loader cell) |

The classification model has no "no tumour" output, so it is only meaningful on images that already contain a tumour. That implies a two-stage design: this detector first, then the type classifier on positive cases. **Neither folder implements or evaluates that cascade.** The two stages were trained on different source datasets, so combined end-to-end performance is unknown.
