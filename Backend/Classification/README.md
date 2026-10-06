# Classification: Glioma, Meningioma or Pituitary

This folder benchmarks pretrained image classifiers on a three-class task. Given a brain MRI slice that **already contains a tumour**, each model predicts whether it is a **Glioma**, **Meningioma** or **Pituitary** tumour. The legacy runs covered six backbones; all nine are now trained and evaluated by one Kaggle notebook, [`classification.ipynb`](classification.ipynb).

The rest of this README documents the **legacy per-model runs** behind the published results. Their notebooks were replaced by the unified notebook; they remain in Git history (for example `git show 9a8ac24:"Backend/Classification/classification-resnet50.ipynb"`), and their result folders are unchanged. Every number below is read from those saved outputs and run folders; the notebooks were not re-executed for this README. Where the evidence does not explain a choice, the README says so instead of guessing.

## Unified notebook

[`classification.ipynb`](classification.ipynb) trains and evaluates all nine backbones (ResNet50, VGG16, DenseNet121, EfficientNetB0, MobileNetV3-Large, InceptionV3, ViT-B/16, Swin-Base, CoAtNet-0) on subtype classification (the legacy runs covered only six) with one shared pipeline, the same code as the other 2D notebook. It has **no saved outputs yet**: it has not been run on Kaggle in this form, so no result in this README comes from it.

What it does differently from the legacy runs:

| Area                      | Legacy per-model notebooks                                                                   | Unified notebook                                                                                                                                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Deduplication             | Applied, but in a fixed class order, so cross-label pairs always kept the glioma copy        | pHash (Hamming 5) on this task's pool before splitting, label-aware; cross-label near-identical pairs (NCC >= 0.98) excluded as label noise                                            |
| Split                     | One stratified 70/15/15 split at image level                                                 | Group-aware: similar images (Hamming <= 8 or NCC >= 0.98) share a group; held-out test set plus 5 development folds; a leakage audit asserts no test image has a development neighbour |
| Optimisation budget       | Batch size 32 to 256, so step counts differed                                                | Same effective batch (64) and step count for every model through gradient accumulation; warm-up then per-step cosine decay                                                             |
| Model sources             | Swin-Base from different libraries per task; InceptionV3 auxiliary loss and head differences | One registry for both tasks; InceptionV3 auxiliary head removed everywhere                                                                                                             |
| Selection and calibration | Best validation AUC; fixed 0.5 threshold                                                     | Best validation log-loss; temperature scaling and (binary) Youden threshold fitted on each validation fold                                                                             |
| Statistics                | Single run, no intervals                                                                     | Fold mean and SD, fold-ensemble test predictions, bootstrap 95% CIs, exact McNemar tests with Holm correction                                                                          |
| Interpretability          | Grad-CAM galleries; outside-head share for EfficientNetB0 only                               | Outside-head Grad-CAM energy for every model on every test image, attention rollout for ViT, occlusion maps for all, inference masking test, and background-only retraining per model  |
| Preprocessing             | Plain resize                                                                                 | The web app's brain crop (`crop_to_brain` in `processor.py`), then resize; augmentation on the GPU                                                                                     |
| Efficiency                | Single-session latency                                                                       | Latency and throughput repeated three times with SD; training time, throughput and peak memory per run                                                                                 |

How to run it:

1. Create a Kaggle notebook from the file with the **GPU T4 x2** accelerator and **Internet on** (pretrained weights are downloaded).
2. Attach `sartajbhuvaji/brain-tumor-classification-mri` and `moundir8/jun-cheng-brain-tumor-mri-dataset-jpg`. The notebook finds them under both `/kaggle/input/<slug>` and `/kaggle/input/datasets/<owner>/<slug>`.
3. Run all cells. Training runs as two worker processes, one per GPU, with live progress bars, and stops claiming new jobs after `TRAINING_BUDGET_HOURS` (9 by default). To continue an interrupted run, attach the previous version's output and set `RESUME_DIR` in the configuration cell.
4. Download `outputs.zip` from the Output tab. It holds `weights/`, `metrics/`, `plots/`, `predictions/`, `logs/` and `src/`.
5. To update the web app, copy `weights/deploy/EfficientNetB0.pth` into this folder. The notebook checks that it loads into the backend's EfficientNet-B0 and reproduces the stored test logits; `weights/deploy/metadata.json` records its temperature and threshold.

Runtime, metrics and hardware use of the unified notebook: Not measured in the current repository.

## At a glance (legacy runs)

| Item                      | Value                                                                                                                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task                      | 3-class classification: Glioma (0), Meningioma (1), Pituitary (2). There is no "no tumour" class.                                                                                                              |
| Data                      | 5,828 JPG images from 2 Kaggle datasets; 4,705 remain after perceptual-hash (pHash) deduplication, a 19.27% reduction                                                                                          |
| Split                     | Stratified 70/15/15 with seed 42: train 3,293, validation 706, test 706                                                                                                                                        |
| Recipe                    | 10 epochs, AdamW (lr 3e-4), cosine LR schedule, `CrossEntropyLoss`, AMP fp16, 2x Tesla T4                                                                                                                      |
| Best test accuracy        | SwinBase, 0.9858 (10 of 706 wrong). CoAtNet-0 (0.9844) and InceptionV3 (0.9830) are within 2 errors.                                                                                                           |
| Lowest test accuracy      | ViT-B16, 0.9051                                                                                                                                                                                                |
| Efficiency                | EfficientNetB0 matches ResNet50's test accuracy (0.9759) with 0.769 vs 8.174 GFLOPs per image                                                                                                                  |
| Hardest class             | Meningioma, for every model                                                                                                                                                                                    |
| Model used by the web app | `EfficientNetB0.pth` in this folder, loaded by [`Backend/neuroscan/inference/processor.py`](../neuroscan/inference/processor.py). Its origin is not recorded (see [B6](#b-issues-that-affect-interpretation)). |

Main caveats, detailed in [Anomalies and limitations](#anomalies-and-limitations):

1. Deduplication compares images across classes and keeps whichever comes first, so Glioma always wins a cross-class tie.
2. The official Testing split of one source dataset is merged into the pool before re-splitting.
3. Every model scores higher on test than on validation. The notebooks do not explain why.
4. SwinBase's Grad-CAM highlights image borders for some classes.
5. Batch sizes range from 32 to 256, so optimiser steps are not equal.
6. Every result comes from a single seed.

## Contents

- [Unified notebook](#unified-notebook)
- [At a glance (legacy runs)](#at-a-glance-legacy-runs)
- [Folder layout](#folder-layout)
- [Pipeline overview](#pipeline-overview)
- [Dataset and deduplication](#dataset-and-deduplication)
- [Shared configuration](#shared-configuration)
- [Per-notebook differences](#per-notebook-differences)
- [Notebook walkthrough](#notebook-walkthrough)
- [Per-notebook run notes](#per-notebook-run-notes)
- [Results](#results)
- [What the results show](#what-the-results-show)
- [Plots for every run](#plots-for-every-run)
- [Output files reference](#output-files-reference)
- [Anomalies and limitations](#anomalies-and-limitations)
- [Reproducing a run](#reproducing-a-run)
- [Loading a trained model](#loading-a-trained-model)
- [Metric glossary](#metric-glossary)

## Folder layout

```text
Classification/
|-- README.md                              This file
|-- EfficientNetB0.pth                     Weights loaded by the backend (not tracked in Git)
|-- classification.ipynb                  Unified notebook for all nine models (no outputs yet)
`-- classification-<model>/                Unzipped outputs of the legacy per-model notebook
    |-- metrics/                           22 CSV, JSON and TXT files (tracked in Git)
    |-- plots/                             14 PNG files, numbered 01-14 (tracked in Git)
    |-- tensorboard/<YYYYMMDD-HHMMSS>/     One events.out.tfevents.* file (not tracked in Git)
    `-- weights/                           best_*_multi_class.pth and final_*_multi_class.pth (not tracked in Git)
```

`<model>` is one of `coatnet0`, `efficientnetb0`, `inceptionv3`, `resnet50`, `swinbase` and `vit-b16`.

**What a fresh clone contains.** The root [`.gitignore`](../../.gitignore) excludes `*.pth`, `tensorboard/` and `events.out.tfevents.*`. A clone therefore has the notebooks, `metrics/` and `plots/`, but no weights and no TensorBoard logs.

**`best_*` vs `final_*` weights.** The final-save cell runs after the best checkpoint has been reloaded, so both files hold the **best-validation-AUC weights**, not the last-epoch weights.

## Pipeline overview

```mermaid
flowchart LR
    A[Collect JPG paths] --> B[Explore data] --> C[pHash deduplication]
    C --> D[Stratified split 70/15/15] --> E[Train 10 epochs, keep best val AUC]
    E --> F[Test best checkpoint] --> G[Metrics, plots, Grad-CAM, benchmark, zip]
```

| Step                    | What happens                                                                                                                                   | Key numbers                       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| 1. Collect              | `os.walk` over three tumour folders in each of 2 Kaggle datasets, `.jpg` only. The Training and Testing folders of `sartajbhuvaji` are merged. | 5,828 images                      |
| 2. Explore              | Class counts, sample images, image sizes and intensity histograms, run **before** deduplication                                                | Plots `01`-`03`                   |
| 3. Deduplicate          | pHash every image; greedily drop any image within Hamming distance 5 of an already-kept image                                                  | 5,828 -> 4,705 (plot `04`)        |
| 4. Split                | Stratified 70/15/15, seed 42                                                                                                                   | 3,293 / 706 / 706                 |
| 5. Prepare              | `BrainDataset` (long labels) with resize, flip, rotation, colour jitter and ImageNet normalisation                                             | 224 px (299 px for InceptionV3)   |
| 6. Build                | Pretrained backbone, head replaced by 3 output logits, `nn.DataParallel` over 2 GPUs                                                           |                                   |
| 7. Train                | AdamW 3e-4, cosine LR, `CrossEntropyLoss`, AMP; save when macro one-vs-rest validation AUC improves                                            | `best_*_multi_class.pth`          |
| 8. Test                 | Reload the best checkpoint, evaluate the test set, save the same weights again                                                                 | `final_*_multi_class.pth`         |
| 9. Evaluate             | Single-GPU re-inference, metrics, confusion matrices, per-class and micro-average ROC/PR, calibration, samples                                 | Plots `08`-`12`                   |
| 10. Explain and package | Grad-CAM, latency and throughput benchmark, `benchmark_summary`, zip of every output                                                           | Plots `13`-`14`, 39 files per zip |

As in the Binary Detection notebooks, the first markdown cell states that the added cells "only observe the run", and their randomness runs inside `preserved_rng_state()`.

## Dataset and deduplication

### Sources

The paths are set in cell 10. All are prefixed with `/kaggle/input/datasets/`.

| Label | Class      | Folders                                                                                                                                                                                                                      |
| ----- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | Glioma     | `sartajbhuvaji/brain-tumor-classification-mri/Testing/glioma_tumor`, `sartajbhuvaji/brain-tumor-classification-mri/Training/glioma_tumor`, `moundir8/jun-cheng-brain-tumor-mri-dataset-jpg/brain tumor mri dataset/2_glioma` |
| 1     | Meningioma | `.../Testing/meningioma_tumor`, `.../Training/meningioma_tumor`, `.../1_meningioma`                                                                                                                                          |
| 2     | Pituitary  | `.../Testing/pituitary_tumor`, `.../Training/pituitary_tumor`, `.../3_pituitary_tumor`                                                                                                                                       |

- Only `.jpg` files are collected (`VALID_EXTENSIONS = (".jpg",)`).
- The `sartajbhuvaji` `no_tumor` folders are not used.
- The official `sartajbhuvaji` Training/Testing split is discarded: both halves go into one pool, which is then re-split.

### Class distribution before and after deduplication

Source: `metrics/deduplication_summary.csv`, identical in all six runs.

| Class      | Before    | Share | After     | Share | Removed   | Removed % |
| ---------- | --------- | ----- | --------- | ----- | --------- | --------- |
| Glioma     | 2,352     | 40.4% | 2,000     | 42.5% | 352       | 14.97     |
| Meningioma | 1,645     | 28.2% | 1,304     | 27.7% | 341       | 20.73     |
| Pituitary  | 1,831     | 31.4% | 1,401     | 29.8% | 430       | 23.48     |
| **Total**  | **5,828** |       | **4,705** |       | **1,123** | **19.27** |

### How deduplication works (cell 18)

1. Compute a 64-bit perceptual hash of every image with `imagehash.phash` on the RGB image. Unreadable files get `None` and are dropped.
2. Walk the table in collection order: all Glioma paths, then Meningioma, then Pituitary.
3. Keep an image only if its Hamming distance to **every** previously kept hash is greater than `HAMMING_TOLERANCE = 5`.
4. Keep the surviving rows. Output: `Total Images after deduplication: 4705`.

Implications:

- Deduplicating **before** splitting stops the same near-identical image from landing in both train and test.
- Comparison runs across classes and the first kept image wins. If a Meningioma or Pituitary image is within distance 5 of an already-kept Glioma image, the later one is silently removed. Label conflicts are not counted, so the number of cross-class removals is unknown.
- The loop compares each image against every kept hash in Python, which is quadratic in the number of images.
- The exploratory cells 14 and 16 (plots `01`-`03` and `image_dimension_stats.csv`) run on the 5,828-image set **before** deduplication. The split and all training use the 4,705 deduplicated images.

### Image properties (before deduplication)

Source: `metrics/image_dimension_stats.csv`.

| Class      | Width mean +/- std (min-max) | Width median | Height mean +/- std (min-max) | Height median |
| ---------- | ---------------------------- | ------------ | ----------------------------- | ------------- |
| Glioma     | 507.6 +/- 33.0 (175-640)     | 512          | 508.4 +/- 30.6 (167-665)      | 512           |
| Meningioma | 483.1 +/- 90.4 (174-1275)    | 512          | 485.9 +/- 85.1 (195-1427)     | 512           |
| Pituitary  | 512.5 +/- 84.4 (201-1365)    | 512          | 512.2 +/- 81.3 (202-1365)     | 512           |

### Split

Stratified 70/15/15 with `random_state=42`, applied to the deduplicated data. Source: `metrics/split_distribution.csv`.

| Class      | Train     | Validation | Test    |
| ---------- | --------- | ---------- | ------- |
| Glioma     | 1,400     | 300        | 300     |
| Meningioma | 913       | 195        | 196     |
| Pituitary  | 980       | 211        | 210     |
| **Total**  | **3,293** | **706**    | **706** |

The classes are moderately imbalanced (Glioma 42.5%, Meningioma 27.7%). No class weighting is used.

The dataset plots (`01` to `06`) are identical in every run folder and appear in the [plot gallery](#plots-for-every-run).

## Shared configuration

### Optimisation

| Setting                  | Value                                                                                                    |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| Seed                     | 42                                                                                                       |
| Epochs and learning rate | 10 epochs, 3e-4                                                                                          |
| Optimiser                | `AdamW` with the default weight decay (0.01). All parameters are trainable.                              |
| Scheduler                | `CosineAnnealingLR(T_max=10)`, stepped once per epoch. LR goes from 3.00e-04 to 7.34e-06.                |
| Loss                     | `CrossEntropyLoss` on 3 logits, `torch.long` labels                                                      |
| Precision                | AMP autocast, `GradScaler("cuda")`, TF32, `cudnn.benchmark=True`                                         |
| Checkpoint rule          | Save when `val_auc > best_auc`                                                                           |
| Epoch AUC                | `roc_auc_score(np.eye(3)[labels], probs, multi_class="ovr")`, which gives a macro-averaged per-class AUC |

### Data loading and augmentation

| Setting              | Value                                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Image size           | 224 px (299 px for InceptionV3)                                                                                             |
| Batch size           | 256 by default; some models override it                                                                                     |
| Loaders              | `num_workers=4`, `pin_memory`, `persistent_workers`, `prefetch_factor=4`. The train loader shuffles with `drop_last=True`.  |
| Train transform      | `Resize` -> `RandomHorizontalFlip` -> `RandomRotation(10)` -> `ColorJitter(0.2, 0.2)` -> `ToTensor` -> ImageNet `Normalize` |
| Evaluation transform | `Resize` -> `ToTensor` -> ImageNet `Normalize`                                                                              |

| Batch size | Training batches per epoch | Images used | Optimiser steps in 10 epochs |
| ---------- | -------------------------- | ----------- | ---------------------------- |
| 256        | 12                         | 3,072       | 120                          |
| 128        | 25                         | 3,200       | 250                          |
| 64         | 51                         | 3,264       | 510                          |
| 32         | 102                        | 3,264       | 1,020                        |

### Environment

Recorded in `benchmark_summary.json` for every run: Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, 2x Tesla T4 on Kaggle. All notebooks need `imagehash`; CoAtNet-0 also needs `timm`.

## Per-notebook differences

EfficientNetB0 is the reference template. Differences that are only comments, banners or print wording are not listed.

| Notebook       | Backbone and head                                                                             | Batch   | Input   | Grad-CAM layer (resolved type)                      |
| -------------- | --------------------------------------------------------------------------------------------- | ------- | ------- | --------------------------------------------------- |
| EfficientNetB0 | `torchvision.efficientnet_b0(DEFAULT)`, `classifier[1] -> Linear(1280, 3)`                    | 256     | 224     | `features[-1]` (Conv2dNormActivation)               |
| ResNet50       | `torchvision.resnet50(DEFAULT)`, `fc -> Linear(2048, 3)`                                      | 256     | 224     | `layer4[-1]` (Bottleneck)                           |
| InceptionV3    | `torchvision.inception_v3(DEFAULT)`, `fc -> Linear(2048, 3)`, then `model.aux_logits = False` | 128     | **299** | `Mixed_7c` (InceptionE)                             |
| CoAtNet-0      | `timm.create_model("coatnet_0_rw_224", pretrained=True)`, `reset_classifier(3)`               | **128** | 224     | `stages[-1]` (MaxxVitStage)                         |
| SwinBase       | **torchvision** `swin_b(Swin_B_Weights.DEFAULT)`, `head -> Linear(1024, 3)`                   | 64      | 224     | `features[-1][-1].norm1` (LayerNorm, channels-last) |
| ViT-B16        | `torchvision.vit_b_16(DEFAULT)`, `heads.head -> Linear(768, 3)`                               | **32**  | 224     | `encoder.layers[-1].ln_1` (LayerNorm, tokens)       |

Other code differences:

- **InceptionV3** uses no auxiliary loss. `AuxLogits` stays as an unused 1000-class head ([B5](#b-issues-that-affect-interpretation)). `import imagehash` sits in the imports cell.
- **SwinBase** uses the torchvision implementation (unlike the timm Swin in Binary Detection), prints two extra lines (`Model loaded successfully on cuda` and the parameter count) and imports `swin_b` twice.
- **ViT-B16** replaces only `heads.head`, the final linear layer inside `heads`.
- **CoAtNet-0** ran a day after the others (training started 2026-09-18 vs 2026-09-17 for the rest, from `training_summary.json`).

Why they differ, using only evidence from the notebooks:

- **Input 299 for InceptionV3** is the Inception default input size.
- **Head replacement** differs because each architecture names its classifier differently.
- **Batch sizes** have no stated reason. CoAtNet-0 at 128 peaked at 11,257.0 MB on GPU0. ViT-B16 at 32 peaked at 3,875.8 MB, far below the T4 limit, so memory does not explain its small batch.
- **No Inception auxiliary loss** has no stated reason. Setting `aux_logits=False` after construction makes torchvision return a plain tensor in train mode, so the shared `run_epoch` works unchanged.
- **timm for CoAtNet-0**, because CoAtNet has no torchvision implementation.

## Notebook walkthrough

All six notebooks have 75 cells in markdown and code pairs. Indices are 0-based. Code comments `# Cell 1` to `# Cell 14` mark the original training cells; unnumbered code cells are the added observation, evaluation and benchmark cells. Printed outputs quoted here are identical in all six runs unless a model is named.

| Section                        | Cells | Purpose                                             |
| ------------------------------ | ----- | --------------------------------------------------- |
| 1. Setup                       | 0-8   | Imports, configuration, notebook identity, helpers  |
| 2. Data                        | 9-32  | Collect, explore, deduplicate, split, load, augment |
| 3. Model and training          | 33-46 | Build, profile, train, summarise                    |
| 4. Test, save, evaluate        | 47-64 | Test, predictions, metrics and plots                |
| 5. Explain, benchmark, package | 65-74 | Grad-CAM, speed benchmark, summary, zip             |

### 1. Setup (cells 0-8)

- **Cell 0 (markdown).** Title, pipeline steps and the "added cells only observe the run" rule.
- **Cell 2, imports.** Data, model, metric and logging libraries.
- **Cell 4, configuration.** `SEED`, `BATCH_SIZE`, `EPOCHS`, `LR`, `IMG_SIZE`, loader settings, seeding, cuDNN and TF32 flags. Output: `CUDA available: True`, `GPU count: 2`.
- **Cell 6, notebook settings.** `MODEL_NAME`, `TASK="classification"`, `CLASS_NAMES=["Glioma","Meningioma","Pituitary"]`, `RUN_NAME` and `get_gradcam_target_layer(net)`.
- **Cell 8, output folders and helpers.** Same helpers as Binary Detection. `NUM_CLASSES=3` sets `IS_BINARY=False`, which selects the multi-class branch in every later cell.

### 2. Data (cells 9-32)

| Cell | What it does                                                                                                    | Output                                                                                   |
| ---- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 10   | Data paths for each tumour class                                                                                |                                                                                          |
| 12   | `collect(paths, label)` walks the folders and builds `df[path, label]`                                          | `Total Images before deduplication: 5828` (Glioma 2352, Meningioma 1645, Pituitary 1831) |
| 14   | Copies `df` to `df_raw`; class bar chart and random samples per class                                           | `01`, `02`                                                                               |
| 16   | Image sizes; grayscale intensity histograms from 100 sampled images per class (51 bins)                         | `03`, `image_dimension_stats.csv`                                                        |
| 18   | pHash deduplication ([details](#how-deduplication-works-cell-18))                                               | `Total Images after deduplication: 4705`                                                 |
| 20   | Before and after class counts on shared axes, and a summary table                                               | `04`, `deduplication_summary.csv`                                                        |
| 22   | Stratified 70/15/15 split                                                                                       | `Train: 3293 Val: 706 Test: 706`                                                         |
| 24   | Per-split class counts and proportions                                                                          | `05`, `split_distribution.csv`                                                           |
| 26   | `BrainDataset`: opens the image, converts it to RGB, applies the transform, returns `(image, torch.long label)` |                                                                                          |
| 28   | Training augmentation and evaluation transforms                                                                 |                                                                                          |
| 30   | Train, validation and test `DataLoader`s                                                                        |                                                                                          |
| 32   | For 2 training images per class, the evaluation transform (top) and one random augmentation (bottom)            | `06`                                                                                     |

### 3. Model and training (cells 33-46)

- **Cell 34, model build.** Pretrained backbone with a 3-output head, `DataParallel`, `CrossEntropyLoss`, AdamW, cosine schedule and `GradScaler`.
- **Cell 36, model profile.** Parameters, buffers, serialised size and FLOPs, as in Binary Detection. Saved as `model_profile.json`.
- **Cell 38, `run_epoch`.** Autocast forward pass, `CrossEntropyLoss(outputs, labels)`, scaled backward pass when training. Collects `softmax(dim=1)` probabilities and returns mean batch loss, argmax accuracy and macro one-vs-rest AUC. The evaluation pass does not use `torch.no_grad()`.
- **Cell 40, training tracker.** Same wrapper as Binary Detection: per-phase timing, per-epoch history, progress bar and peak-memory reset.
- **Cell 42, training loop.** 10 epochs of train, validate, `scheduler.step()`, with TensorBoard scalars and a checkpoint when validation AUC improves. Per-epoch results are in [Per-epoch training history](#per-epoch-training-history).
- **Cell 44, training summary.** Saved as `training_history.csv` and `training_summary.json`.
- **Cell 46, training curves.** Saves `07_training_curves.png`.

### 4. Test, save, evaluate (cells 47-64)

- **Cell 48, test.** Loads the best checkpoint into the `DataParallel` model and prints `FINAL UNSEEN TEST RESULTS`. Printed values are in [Original test cell output vs metrics files](#original-test-cell-output-vs-metrics-files).
- **Cell 50, save.** Saves the reloaded best weights again as `final_*_multi_class.pth`.
- **Cell 52, predictions.** Single-GPU re-inference of validation and test, softmax renormalised in float64. Saved as `predictions_{val,test}.csv`.
- **Cell 54, metrics.** The shared metric set plus `roc_auc_ovr_macro`, `roc_auc_ovr_weighted`, `roc_auc_ovo_macro`, `pr_auc_macro`, `pr_auc_micro`, `specificity_macro` and `top_2_accuracy`, and per-class one-vs-rest metrics.
- **Cell 56.** 3x3 confusion matrices, counts and row-normalised. `08`, `confusion_matrix_{val,test}.csv`.
- **Cell 58.** One-vs-rest ROC and PR curves per class plus the micro-average curve. `09`.
- **Cell 60.** Reliability diagram over 15 bins of maximum softmax confidence and confidence histograms. `10`, `calibration_bins_{val,test}.csv`.
- **Cell 62.** Per-class precision, recall, specificity, F1 and AUC on test. `11`.
- **Cell 64.** 6 random correct and 6 random misclassified test images. `12`.

### 5. Explain, benchmark, package (cells 65-74)

- **Cell 66, `GradCAM` class.** Same implementation as Binary Detection, except the score is the logit of the predicted class.
- **Cell 68, figures.** `13`: 2 most confident correct predictions per class. `14`: 6 most confident errors. `gradcam_status` is `ok` in all six runs.
- **Cell 70, inference benchmark.** Same protocol as Binary Detection. Saved as `inference_benchmark.json`.
- **Cell 72, benchmark summary.** `benchmark_summary.json` and `.csv`.
- **Cell 74, packaging.** Zips outputs, weights and the TensorBoard run into `<RUN_NAME>-outputs.zip`. Every run printed `Packaged 39 files`.

## Per-notebook run notes

### classification-swinbase.ipynb

- Strongest start: validation AUC 0.98525 after epoch 1. The checkpoint is epoch 7 (0.99743), although validation accuracy keeps rising to 0.9816 at epochs 9 and 10.
- Best test accuracy (0.9858), lowest test log loss (0.0410) and ECE (0.0076). Pituitary recall on test is 210/210.

### classification-coatnet0.ipynb

- Slow start: validation accuracy 0.6190 at epoch 1 and 0.6926 at epoch 2, then it converges. Checkpoint at epoch 9.
- Slowest run (0:28:01). The only model whose test top-2 accuracy is 1.0.

### classification-inceptionv3.ipynb

- Validation loss is noisy (0.2181 at epoch 4, 0.2528 at epoch 6) before settling near 0.10. Checkpoint at epoch 8.
- Best test ROC AUC (0.99938) and PR AUC (0.99872).

### classification-resnet50.ipynb

- Highest validation AUC of the six (0.99809, epoch 9), but mid-table on test accuracy (0.9759, 17 errors).

### classification-efficientnetb0.ipynb

- Lowest best-epoch validation accuracy among the five strong models (0.9547). Validation AUC plateaus near 0.996 from epoch 8. Checkpoint at epoch 9.
- The original test cell printed `Test Accuracy: 0.9773`, while the metrics files record 0.9759 ([A5](#a-issues-that-affect-results)).

### classification-vit-b16.ipynb

- Training accuracy only reaches 0.8808 and validation AUC is still rising at epoch 10 (0.96886), the checkpoint.
- Its batch size of 32 gives it the most optimiser steps (1,020), yet it still underfits. With this learning rate and 10 epochs, the recipe does not suit ViT-B16 on this data.

## Results

All tables below are generated from the files in each run folder. Models are sorted by test accuracy, then MCC.

### Test-set classification quality

| Model          | Accuracy | Errors | Balanced acc | Macro F1 | MCC    | Cohen kappa | Top-2 acc |
| -------------- | -------- | ------ | ------------ | -------- | ------ | ----------- | --------- |
| SwinBase       | 0.9858   | 10     | 0.9842       | 0.9847   | 0.9784 | 0.9783      | 0.9986    |
| CoAtNet-0      | 0.9844   | 11     | 0.9821       | 0.9835   | 0.9762 | 0.9761      | 1.0000    |
| InceptionV3    | 0.9830   | 12     | 0.9829       | 0.9822   | 0.9741 | 0.9740      | 0.9986    |
| ResNet50       | 0.9759   | 17     | 0.9757       | 0.9746   | 0.9633 | 0.9632      | 0.9986    |
| EfficientNetB0 | 0.9759   | 17     | 0.9750       | 0.9745   | 0.9632 | 0.9632      | 0.9972    |
| ViT-B16        | 0.9051   | 67     | 0.9037       | 0.9034   | 0.8550 | 0.8550      | 0.9943    |

Source: `metrics/metrics_test.json` in each run folder. Test set: 706 images.

### Test-set probability quality

| Model          | ROC AUC (OvR macro) | ROC AUC (OvO macro) | PR AUC (macro) | Log loss | Brier  | ECE    | Best epoch |
| -------------- | ------------------- | ------------------- | -------------- | -------- | ------ | ------ | ---------- |
| SwinBase       | 0.99929             | 0.99923             | 0.99848        | 0.0410   | 0.0211 | 0.0076 | 7          |
| CoAtNet-0      | 0.99892             | 0.99884             | 0.99773        | 0.0521   | 0.0253 | 0.0084 | 9          |
| InceptionV3    | 0.99938             | 0.99937             | 0.99872        | 0.0471   | 0.0271 | 0.0077 | 8          |
| ResNet50       | 0.99883             | 0.99886             | 0.99752        | 0.0593   | 0.0328 | 0.0116 | 9          |
| EfficientNetB0 | 0.99877             | 0.99880             | 0.99719        | 0.0588   | 0.0324 | 0.0112 | 9          |
| ViT-B16        | 0.97594             | 0.97608             | 0.95159        | 0.2637   | 0.1426 | 0.0266 | 10         |

### Model size and compute

| Model          | Parameters | GFLOPs / image | State dict (MB) | Input     | Batch size |
| -------------- | ---------- | -------------- | --------------- | --------- | ---------- |
| SwinBase       | 86,746,299 | 30.860         | 331.47          | 3x224x224 | 64         |
| CoAtNet-0      | 26,668,869 | 8.416          | 101.87          | 3x224x224 | 128        |
| InceptionV3    | 25,118,411 | 11.422         | 96.15           | 3x299x299 | 128        |
| ResNet50       | 23,514,179 | 8.174          | 90.00           | 3x224x224 | 256        |
| EfficientNetB0 | 4,011,391  | 0.769          | 15.59           | 3x224x224 | 256        |
| ViT-B16        | 85,800,963 | 22.541         | 327.36          | 3x224x224 | 32         |

Source: `metrics/model_profile.json` and `metrics/benchmark_summary.json`. All parameters are trainable in every run.

### Training time, memory and inference speed

| Model          | Train time | Mean epoch (s) | Train img/s | Peak GPU0 alloc (MB) | Peak GPU1 alloc (MB) | Latency b=1 mean (ms) | Latency b=1 p95 (ms) | Throughput b=32 (img/s) | Test set end-to-end (s) |
| -------------- | ---------- | -------------- | ----------- | -------------------- | -------------------- | --------------------- | -------------------- | ----------------------- | ----------------------- |
| SwinBase       | 0:09:28    | 56.42          | 67.43       | 10784.5              | 10048.3              | 35.13                 | 37.85                | 202.41                  | 5.970                   |
| CoAtNet-0      | 0:28:01    | 167.99         | 23.22       | 11257.0              | 11013.2              | 13.23                 | 14.69                | 498.58                  | 5.220                   |
| InceptionV3    | 0:12:14    | 73.26          | 52.05       | 7104.8               | 6838.7               | 14.15                 | 15.70                | 532.64                  | 5.144                   |
| ResNet50       | 0:08:07    | 48.52          | 77.01       | 11374.0              | 11121.4              | 7.82                  | 9.74                 | 791.58                  | 4.427                   |
| EfficientNetB0 | 0:26:53    | 161.20         | 23.51       | 11078.2              | 10974.2              | 13.85                 | 15.96                | 1470.84                 | 5.556                   |
| ViT-B16        | 0:07:23    | 43.79          | 85.78       | 3875.8               | 3220.6               | 8.39                  | 9.42                 | 289.50                  | 3.823                   |

Source: `metrics/training_summary.json` and `metrics/inference_benchmark.json`. Training used 2x Tesla T4; the inference benchmark used one Tesla T4 with fp16 autocast.

### Original test cell output vs metrics files

The original test cell evaluates the DataParallel model and prints loss, accuracy and AUC. The metrics files come from a separate single-GPU re-inference (see the walkthrough). Both are listed so they can be compared.

| Model          | Printed test loss | Printed test acc | Printed test AUC | File log loss | File accuracy | File AUC |
| -------------- | ----------------- | ---------------- | ---------------- | ------------- | ------------- | -------- |
| SwinBase       | 0.0377            | 0.9858           | 0.9993           | 0.0410        | 0.9858        | 0.9993   |
| CoAtNet-0      | 0.0499            | 0.9844           | 0.9989           | 0.0521        | 0.9844        | 0.9989   |
| InceptionV3    | 0.0442            | 0.9830           | 0.9994           | 0.0471        | 0.9830        | 0.9994   |
| ResNet50       | 0.0567            | 0.9759           | 0.9988           | 0.0593        | 0.9759        | 0.9988   |
| EfficientNetB0 | 0.0580            | 0.9773           | 0.9988           | 0.0588        | 0.9759        | 0.9988   |
| ViT-B16        | 0.2537            | 0.9051           | 0.9759           | 0.2637        | 0.9051        | 0.9759   |

### All test metrics

| Metric                       | SwinBase | CoAtNet-0 | InceptionV3 | ResNet50 | EfficientNetB0 | ViT-B16 |
| ---------------------------- | -------- | --------- | ----------- | -------- | -------------- | ------- |
| `samples`                    | 706      | 706       | 706         | 706      | 706            | 706     |
| `accuracy`                   | 0.9858   | 0.9844    | 0.9830      | 0.9759   | 0.9759         | 0.9051  |
| `balanced_accuracy`          | 0.9842   | 0.9821    | 0.9829      | 0.9757   | 0.9750         | 0.9037  |
| `error_rate`                 | 0.0142   | 0.0156    | 0.0170      | 0.0241   | 0.0241         | 0.0949  |
| `precision_macro`            | 0.9854   | 0.9850    | 0.9817      | 0.9736   | 0.9741         | 0.9031  |
| `recall_macro`               | 0.9842   | 0.9821    | 0.9829      | 0.9757   | 0.9750         | 0.9037  |
| `f1_macro`                   | 0.9847   | 0.9835    | 0.9822      | 0.9746   | 0.9745         | 0.9034  |
| `precision_weighted`         | 0.9859   | 0.9844    | 0.9832      | 0.9762   | 0.9759         | 0.9057  |
| `recall_weighted`            | 0.9858   | 0.9844    | 0.9830      | 0.9759   | 0.9759         | 0.9051  |
| `f1_weighted`                | 0.9858   | 0.9844    | 0.9831      | 0.9760   | 0.9759         | 0.9054  |
| `precision_micro`            | 0.9858   | 0.9844    | 0.9830      | 0.9759   | 0.9759         | 0.9051  |
| `recall_micro`               | 0.9858   | 0.9844    | 0.9830      | 0.9759   | 0.9759         | 0.9051  |
| `f1_micro`                   | 0.9858   | 0.9844    | 0.9830      | 0.9759   | 0.9759         | 0.9051  |
| `mcc`                        | 0.9784   | 0.9762    | 0.9741      | 0.9633   | 0.9632         | 0.8550  |
| `cohen_kappa`                | 0.9783   | 0.9761    | 0.9740      | 0.9632   | 0.9632         | 0.8550  |
| `jaccard_macro`              | 0.9699   | 0.9677    | 0.9653      | 0.9507   | 0.9505         | 0.8270  |
| `log_loss`                   | 0.0410   | 0.0521    | 0.0471      | 0.0593   | 0.0588         | 0.2637  |
| `brier_score`                | 0.0211   | 0.0253    | 0.0271      | 0.0328   | 0.0324         | 0.1426  |
| `expected_calibration_error` | 0.0076   | 0.0084    | 0.0077      | 0.0116   | 0.0112         | 0.0266  |
| `mean_confidence`            | 0.9881   | 0.9841    | 0.9879      | 0.9864   | 0.9740         | 0.8828  |
| `roc_auc_ovr_macro`          | 0.9993   | 0.9989    | 0.9994      | 0.9988   | 0.9988         | 0.9759  |
| `roc_auc_ovr_weighted`       | 0.9994   | 0.9990    | 0.9994      | 0.9988   | 0.9988         | 0.9766  |
| `roc_auc_ovo_macro`          | 0.9992   | 0.9988    | 0.9994      | 0.9989   | 0.9988         | 0.9761  |
| `pr_auc_macro`               | 0.9985   | 0.9977    | 0.9987      | 0.9975   | 0.9972         | 0.9516  |
| `pr_auc_micro`               | 0.9985   | 0.9982    | 0.9989      | 0.9979   | 0.9981         | 0.9604  |
| `specificity_macro`          | 0.9929   | 0.9918    | 0.9916      | 0.9883   | 0.9881         | 0.9517  |
| `top_2_accuracy`             | 0.9986   | 1.0000    | 0.9986      | 0.9986   | 0.9972         | 0.9943  |

Source: `metrics/metrics_test.json`. The validation split was also used to pick the checkpoint, so validation values are optimistic.

### All validation metrics

| Metric                       | SwinBase | CoAtNet-0 | InceptionV3 | ResNet50 | EfficientNetB0 | ViT-B16 |
| ---------------------------- | -------- | --------- | ----------- | -------- | -------------- | ------- |
| `samples`                    | 706      | 706       | 706         | 706      | 706            | 706     |
| `accuracy`                   | 0.9731   | 0.9660    | 0.9745      | 0.9731   | 0.9547         | 0.8810  |
| `balanced_accuracy`          | 0.9714   | 0.9675    | 0.9765      | 0.9737   | 0.9542         | 0.8791  |
| `error_rate`                 | 0.0269   | 0.0340    | 0.0255      | 0.0269   | 0.0453         | 0.1190  |
| `precision_macro`            | 0.9729   | 0.9637    | 0.9721      | 0.9710   | 0.9523         | 0.8749  |
| `recall_macro`               | 0.9714   | 0.9675    | 0.9765      | 0.9737   | 0.9542         | 0.8791  |
| `f1_macro`                   | 0.9720   | 0.9652    | 0.9738      | 0.9721   | 0.9532         | 0.8766  |
| `precision_weighted`         | 0.9730   | 0.9672    | 0.9757      | 0.9737   | 0.9550         | 0.8834  |
| `recall_weighted`            | 0.9731   | 0.9660    | 0.9745      | 0.9731   | 0.9547         | 0.8810  |
| `f1_weighted`                | 0.9730   | 0.9662    | 0.9747      | 0.9732   | 0.9548         | 0.8818  |
| `precision_micro`            | 0.9731   | 0.9660    | 0.9745      | 0.9731   | 0.9547         | 0.8810  |
| `recall_micro`               | 0.9731   | 0.9660    | 0.9745      | 0.9731   | 0.9547         | 0.8810  |
| `f1_micro`                   | 0.9731   | 0.9660    | 0.9745      | 0.9731   | 0.9547         | 0.8810  |
| `mcc`                        | 0.9589   | 0.9485    | 0.9615      | 0.9591   | 0.9308         | 0.8192  |
| `cohen_kappa`                | 0.9588   | 0.9482    | 0.9611      | 0.9589   | 0.9308         | 0.8188  |
| `jaccard_macro`              | 0.9459   | 0.9332    | 0.9492      | 0.9460   | 0.9117         | 0.7833  |
| `log_loss`                   | 0.0937   | 0.0987    | 0.1051      | 0.0980   | 0.1096         | 0.3202  |
| `brier_score`                | 0.0412   | 0.0478    | 0.0425      | 0.0453   | 0.0611         | 0.1840  |
| `expected_calibration_error` | 0.0191   | 0.0161    | 0.0149      | 0.0168   | 0.0173         | 0.0332  |
| `mean_confidence`            | 0.9902   | 0.9821    | 0.9871      | 0.9851   | 0.9702         | 0.8791  |
| `roc_auc_ovr_macro`          | 0.9974   | 0.9975    | 0.9963      | 0.9981   | 0.9959         | 0.9689  |
| `roc_auc_ovr_weighted`       | 0.9974   | 0.9974    | 0.9961      | 0.9981   | 0.9961         | 0.9696  |
| `roc_auc_ovo_macro`          | 0.9975   | 0.9976    | 0.9963      | 0.9981   | 0.9958         | 0.9687  |
| `pr_auc_macro`               | 0.9943   | 0.9947    | 0.9934      | 0.9961   | 0.9908         | 0.9388  |
| `pr_auc_micro`               | 0.9962   | 0.9954    | 0.9939      | 0.9963   | 0.9935         | 0.9478  |
| `specificity_macro`          | 0.9862   | 0.9835    | 0.9879      | 0.9869   | 0.9774         | 0.9413  |
| `top_2_accuracy`             | 0.9986   | 0.9929    | 0.9929      | 0.9972   | 0.9958         | 0.9887  |

Source: `metrics/metrics_val.json`. The validation split was also used to pick the checkpoint, so validation values are optimistic.

### Confusion matrices

Rows are the true class and columns the predicted class, both in the order `Glioma, Meningioma, Pituitary`. Rows are separated by `/`.

| Model          | Test                                | Validation                            |
| -------------- | ----------------------------------- | ------------------------------------- |
| SwinBase       | 298, 2, 0 / 3, 188, 5 / 0, 0, 210   | 294, 5, 1 / 7, 184, 4 / 1, 1, 209     |
| CoAtNet-0      | 299, 1, 0 / 6, 188, 2 / 0, 2, 208   | 287, 12, 1 / 4, 190, 1 / 1, 5, 205    |
| InceptionV3    | 295, 5, 0 / 3, 192, 1 / 0, 3, 207   | 289, 10, 1 / 2, 193, 0 / 0, 5, 206    |
| ResNet50       | 293, 7, 0 / 3, 190, 3 / 0, 4, 206   | 291, 8, 1 / 3, 191, 1 / 1, 5, 205     |
| EfficientNetB0 | 294, 5, 1 / 4, 188, 4 / 0, 3, 207   | 286, 14, 0 / 10, 181, 4 / 0, 4, 207   |
| ViT-B16        | 272, 27, 1 / 22, 167, 7 / 4, 6, 200 | 266, 29, 5 / 18, 163, 14 / 3, 15, 193 |

Source: `metrics/confusion_matrix_{test,val}.csv`.

### Per-class metrics

**Test split** (source: `metrics/per_class_metrics_test.csv`)

| Model          | Class      | Support | Precision | Recall | Specificity | F1     | ROC AUC | Average precision |
| -------------- | ---------- | ------- | --------- | ------ | ----------- | ------ | ------- | ----------------- |
| SwinBase       | Glioma     | 300     | 0.9900    | 0.9933 | 0.9926      | 0.9917 | 0.9998  | 0.9998            |
| SwinBase       | Meningioma | 196     | 0.9895    | 0.9592 | 0.9961      | 0.9741 | 0.9986  | 0.9972            |
| SwinBase       | Pituitary  | 210     | 0.9767    | 1.0000 | 0.9899      | 0.9882 | 0.9994  | 0.9985            |
| CoAtNet-0      | Glioma     | 300     | 0.9803    | 0.9967 | 0.9852      | 0.9884 | 0.9995  | 0.9994            |
| CoAtNet-0      | Meningioma | 196     | 0.9843    | 0.9592 | 0.9941      | 0.9716 | 0.9979  | 0.9954            |
| CoAtNet-0      | Pituitary  | 210     | 0.9905    | 0.9905 | 0.9960      | 0.9905 | 0.9994  | 0.9984            |
| InceptionV3    | Glioma     | 300     | 0.9899    | 0.9833 | 0.9926      | 0.9866 | 0.9993  | 0.9991            |
| InceptionV3    | Meningioma | 196     | 0.9600    | 0.9796 | 0.9843      | 0.9697 | 0.9989  | 0.9973            |
| InceptionV3    | Pituitary  | 210     | 0.9952    | 0.9857 | 0.9980      | 0.9904 | 0.9999  | 0.9997            |
| ResNet50       | Glioma     | 300     | 0.9899    | 0.9767 | 0.9926      | 0.9832 | 0.9987  | 0.9985            |
| ResNet50       | Meningioma | 196     | 0.9453    | 0.9694 | 0.9784      | 0.9572 | 0.9980  | 0.9945            |
| ResNet50       | Pituitary  | 210     | 0.9856    | 0.9810 | 0.9940      | 0.9833 | 0.9998  | 0.9996            |
| EfficientNetB0 | Glioma     | 300     | 0.9866    | 0.9800 | 0.9901      | 0.9833 | 0.9989  | 0.9986            |
| EfficientNetB0 | Meningioma | 196     | 0.9592    | 0.9592 | 0.9843      | 0.9592 | 0.9977  | 0.9936            |
| EfficientNetB0 | Pituitary  | 210     | 0.9764    | 0.9857 | 0.9899      | 0.9810 | 0.9997  | 0.9993            |
| ViT-B16        | Glioma     | 300     | 0.9128    | 0.9067 | 0.9360      | 0.9097 | 0.9783  | 0.9698            |
| ViT-B16        | Meningioma | 196     | 0.8350    | 0.8520 | 0.9353      | 0.8434 | 0.9580  | 0.9122            |
| ViT-B16        | Pituitary  | 210     | 0.9615    | 0.9524 | 0.9839      | 0.9569 | 0.9915  | 0.9728            |

**Validation split** (source: `metrics/per_class_metrics_val.csv`)

| Model          | Class      | Support | Precision | Recall | Specificity | F1     | ROC AUC | Average precision |
| -------------- | ---------- | ------- | --------- | ------ | ----------- | ------ | ------- | ----------------- |
| SwinBase       | Glioma     | 300     | 0.9735    | 0.9800 | 0.9803      | 0.9767 | 0.9967  | 0.9964            |
| SwinBase       | Meningioma | 195     | 0.9684    | 0.9436 | 0.9883      | 0.9558 | 0.9959  | 0.9875            |
| SwinBase       | Pituitary  | 211     | 0.9766    | 0.9905 | 0.9899      | 0.9835 | 0.9996  | 0.9991            |
| CoAtNet-0      | Glioma     | 300     | 0.9829    | 0.9567 | 0.9877      | 0.9696 | 0.9969  | 0.9964            |
| CoAtNet-0      | Meningioma | 195     | 0.9179    | 0.9744 | 0.9667      | 0.9453 | 0.9961  | 0.9889            |
| CoAtNet-0      | Pituitary  | 211     | 0.9903    | 0.9716 | 0.9960      | 0.9809 | 0.9994  | 0.9987            |
| InceptionV3    | Glioma     | 300     | 0.9931    | 0.9633 | 0.9951      | 0.9780 | 0.9944  | 0.9952            |
| InceptionV3    | Meningioma | 195     | 0.9279    | 0.9897 | 0.9706      | 0.9578 | 0.9959  | 0.9874            |
| InceptionV3    | Pituitary  | 211     | 0.9952    | 0.9763 | 0.9980      | 0.9856 | 0.9988  | 0.9977            |
| ResNet50       | Glioma     | 300     | 0.9864    | 0.9700 | 0.9901      | 0.9782 | 0.9978  | 0.9973            |
| ResNet50       | Meningioma | 195     | 0.9363    | 0.9795 | 0.9746      | 0.9574 | 0.9973  | 0.9931            |
| ResNet50       | Pituitary  | 211     | 0.9903    | 0.9716 | 0.9960      | 0.9809 | 0.9992  | 0.9980            |
| EfficientNetB0 | Glioma     | 300     | 0.9662    | 0.9533 | 0.9754      | 0.9597 | 0.9971  | 0.9962            |
| EfficientNetB0 | Meningioma | 195     | 0.9095    | 0.9282 | 0.9648      | 0.9188 | 0.9931  | 0.9796            |
| EfficientNetB0 | Pituitary  | 211     | 0.9810    | 0.9810 | 0.9919      | 0.9810 | 0.9976  | 0.9965            |
| ViT-B16        | Glioma     | 300     | 0.9268    | 0.8867 | 0.9483      | 0.9063 | 0.9713  | 0.9663            |
| ViT-B16        | Meningioma | 195     | 0.7874    | 0.8359 | 0.9139      | 0.8109 | 0.9488  | 0.8852            |
| ViT-B16        | Pituitary  | 211     | 0.9104    | 0.9147 | 0.9616      | 0.9125 | 0.9865  | 0.9649            |

### Per-epoch training history

Source: `metrics/training_history.csv`. Training accuracy and AUC are measured on augmented batches in train mode, so they are not directly comparable with the validation columns. **Bold** marks the checkpointed epoch (highest validation AUC, first occurrence wins).

**SwinBase**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.5009     | 0.7837    | 0.9273    | 0.3879   | 0.8697  | 0.98525 | 58.1           |
| 2     | 2.93e-04 | 0.2084     | 0.9259    | 0.9870    | 0.1560   | 0.9391  | 0.99367 | 55.4           |
| 3     | 2.71e-04 | 0.1317     | 0.9510    | 0.9943    | 0.1208   | 0.9603  | 0.99426 | 57.0           |
| 4     | 2.38e-04 | 0.0921     | 0.9700    | 0.9966    | 0.0832   | 0.9703  | 0.99701 | 56.3           |
| 5     | 1.96e-04 | 0.0693     | 0.9767    | 0.9982    | 0.0701   | 0.9773  | 0.99726 | 56.1           |
| 6     | 1.50e-04 | 0.0777     | 0.9749    | 0.9977    | 0.1145   | 0.9674  | 0.99662 | 56.5           |
| **7** | 1.04e-04 | 0.0333     | 0.9884    | 0.9996    | 0.0861   | 0.9731  | 0.99743 | 56.3           |
| 8     | 6.18e-05 | 0.0263     | 0.9905    | 0.9998    | 0.0874   | 0.9788  | 0.99728 | 56.4           |
| 9     | 2.86e-05 | 0.0170     | 0.9942    | 0.9999    | 0.0849   | 0.9816  | 0.99699 | 55.9           |
| 10    | 7.34e-06 | 0.0181     | 0.9942    | 0.9999    | 0.0805   | 0.9816  | 0.99717 | 56.2           |

**CoAtNet-0**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.9998     | 0.5003    | 0.6802    | 0.8346   | 0.6190  | 0.89421 | 187.3          |
| 2     | 2.93e-04 | 0.5000     | 0.7887    | 0.9240    | 0.7402   | 0.6926  | 0.93073 | 165.3          |
| 3     | 2.71e-04 | 0.2809     | 0.8859    | 0.9754    | 0.2812   | 0.8810  | 0.98413 | 165.6          |
| 4     | 2.38e-04 | 0.1686     | 0.9419    | 0.9911    | 0.1400   | 0.9589  | 0.99459 | 166.6          |
| 5     | 1.96e-04 | 0.1198     | 0.9534    | 0.9954    | 0.1497   | 0.9462  | 0.99453 | 165.8          |
| 6     | 1.50e-04 | 0.0875     | 0.9678    | 0.9974    | 0.1194   | 0.9618  | 0.99608 | 165.9          |
| 7     | 1.04e-04 | 0.0666     | 0.9769    | 0.9985    | 0.1154   | 0.9646  | 0.99689 | 165.4          |
| 8     | 6.18e-05 | 0.0382     | 0.9878    | 0.9994    | 0.1381   | 0.9603  | 0.99686 | 165.6          |
| **9** | 2.86e-05 | 0.0246     | 0.9928    | 0.9998    | 0.1019   | 0.9660  | 0.99752 | 167.1          |
| 10    | 7.34e-06 | 0.0216     | 0.9941    | 0.9998    | 0.1027   | 0.9688  | 0.99730 | 165.3          |

**InceptionV3**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.4232     | 0.8309    | 0.9500    | 0.6966   | 0.8031  | 0.97375 | 88.4           |
| 2     | 2.93e-04 | 0.1402     | 0.9456    | 0.9939    | 0.2708   | 0.8966  | 0.99141 | 71.1           |
| 3     | 2.71e-04 | 0.0829     | 0.9709    | 0.9978    | 0.1485   | 0.9575  | 0.99497 | 71.6           |
| 4     | 2.38e-04 | 0.0492     | 0.9847    | 0.9991    | 0.2181   | 0.9136  | 0.99119 | 71.4           |
| 5     | 1.96e-04 | 0.0313     | 0.9897    | 0.9997    | 0.1430   | 0.9490  | 0.99522 | 71.4           |
| 6     | 1.50e-04 | 0.0276     | 0.9922    | 0.9996    | 0.2528   | 0.9490  | 0.99058 | 71.7           |
| 7     | 1.04e-04 | 0.0197     | 0.9934    | 0.9999    | 0.1033   | 0.9717  | 0.99610 | 72.0           |
| **8** | 6.18e-05 | 0.0132     | 0.9966    | 1.0000    | 0.1061   | 0.9745  | 0.99633 | 71.8           |
| 9     | 2.86e-05 | 0.0059     | 0.9988    | 1.0000    | 0.1088   | 0.9759  | 0.99627 | 71.6           |
| 10    | 7.34e-06 | 0.0068     | 0.9988    | 1.0000    | 0.1067   | 0.9745  | 0.99627 | 71.7           |

**ResNet50**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.6624     | 0.7549    | 0.9015    | 0.4610   | 0.8031  | 0.95234 | 61.0           |
| 2     | 2.93e-04 | 0.2431     | 0.9059    | 0.9813    | 0.4870   | 0.8159  | 0.96865 | 45.4           |
| 3     | 2.71e-04 | 0.1267     | 0.9538    | 0.9948    | 0.2819   | 0.8909  | 0.99007 | 47.8           |
| 4     | 2.38e-04 | 0.0755     | 0.9704    | 0.9981    | 0.1739   | 0.9419  | 0.99279 | 47.0           |
| 5     | 1.96e-04 | 0.0547     | 0.9792    | 0.9989    | 0.1075   | 0.9660  | 0.99658 | 47.4           |
| 6     | 1.50e-04 | 0.0288     | 0.9886    | 0.9998    | 0.0925   | 0.9660  | 0.99753 | 47.5           |
| 7     | 1.04e-04 | 0.0203     | 0.9954    | 0.9999    | 0.1024   | 0.9674  | 0.99792 | 47.3           |
| 8     | 6.18e-05 | 0.0144     | 0.9961    | 0.9999    | 0.1030   | 0.9703  | 0.99804 | 47.2           |
| **9** | 2.86e-05 | 0.0121     | 0.9954    | 1.0000    | 0.0939   | 0.9731  | 0.99809 | 47.4           |
| 10    | 7.34e-06 | 0.0101     | 0.9980    | 1.0000    | 0.0930   | 0.9745  | 0.99807 | 47.2           |

**EfficientNetB0**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.6884     | 0.7262    | 0.8951    | 0.7265   | 0.7139  | 0.93822 | 196.4          |
| 2     | 2.93e-04 | 0.3004     | 0.8818    | 0.9707    | 0.4396   | 0.8272  | 0.96872 | 158.8          |
| 3     | 2.71e-04 | 0.1902     | 0.9310    | 0.9882    | 0.2777   | 0.8966  | 0.98013 | 155.6          |
| 4     | 2.38e-04 | 0.1454     | 0.9466    | 0.9929    | 0.1742   | 0.9334  | 0.99000 | 156.5          |
| 5     | 1.96e-04 | 0.0986     | 0.9622    | 0.9970    | 0.1539   | 0.9405  | 0.99291 | 155.7          |
| 6     | 1.50e-04 | 0.0675     | 0.9756    | 0.9986    | 0.1385   | 0.9433  | 0.99392 | 157.6          |
| 7     | 1.04e-04 | 0.0629     | 0.9785    | 0.9988    | 0.1174   | 0.9533  | 0.99522 | 155.6          |
| 8     | 6.18e-05 | 0.0526     | 0.9808    | 0.9993    | 0.1095   | 0.9547  | 0.99565 | 158.2          |
| **9** | 2.86e-05 | 0.0396     | 0.9893    | 0.9997    | 0.1069   | 0.9547  | 0.99590 | 159.3          |
| 10    | 7.34e-06 | 0.0458     | 0.9850    | 0.9995    | 0.1078   | 0.9561  | 0.99581 | 158.3          |

**ViT-B16**

| Epoch  | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ------ | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1      | 3.00e-04 | 1.0624     | 0.4881    | 0.6401    | 0.8701   | 0.5807  | 0.81536 | 44.7           |
| 2      | 2.93e-04 | 0.8584     | 0.6072    | 0.7804    | 0.8103   | 0.6671  | 0.82392 | 42.9           |
| 3      | 2.71e-04 | 0.8177     | 0.6403    | 0.8076    | 0.7734   | 0.6714  | 0.85561 | 42.9           |
| 4      | 2.38e-04 | 0.7361     | 0.6798    | 0.8445    | 0.7416   | 0.6827  | 0.85510 | 43.8           |
| 5      | 1.96e-04 | 0.6925     | 0.7108    | 0.8649    | 0.6893   | 0.7408  | 0.88847 | 43.8           |
| 6      | 1.50e-04 | 0.6125     | 0.7546    | 0.8934    | 0.5758   | 0.7805  | 0.92550 | 44.0           |
| 7      | 1.04e-04 | 0.5331     | 0.7800    | 0.9190    | 0.5075   | 0.8031  | 0.93321 | 44.2           |
| 8      | 6.18e-05 | 0.4136     | 0.8324    | 0.9498    | 0.4220   | 0.8499  | 0.95672 | 44.0           |
| 9      | 2.86e-05 | 0.3488     | 0.8609    | 0.9631    | 0.4047   | 0.8513  | 0.96435 | 43.9           |
| **10** | 7.34e-06 | 0.3040     | 0.8808    | 0.9712    | 0.3419   | 0.8810  | 0.96886 | 43.8           |

### Validation vs test accuracy

| Model          | Validation accuracy (best checkpoint) | Test accuracy |
| -------------- | ------------------------------------- | ------------- |
| SwinBase       | 0.9731                                | 0.9858        |
| CoAtNet-0      | 0.9660                                | 0.9844        |
| InceptionV3    | 0.9745                                | 0.9830        |
| ResNet50       | 0.9731                                | 0.9759        |
| EfficientNetB0 | 0.9547                                | 0.9759        |
| ViT-B16        | 0.8810                                | 0.9051        |

## What the results show

- **The top three are effectively tied.** SwinBase, CoAtNet-0 and InceptionV3 make 10, 11 and 12 test errors. With one seed and 706 test images (one error is 0.14 percentage points), that gap is not meaningful. Each leads on a different measure: SwinBase on accuracy and calibration, InceptionV3 on ROC and PR AUC, CoAtNet-0 on top-2 accuracy.
- **EfficientNetB0 is the efficiency pick among the models tested.** It makes the same 17 test errors as ResNet50 with 0.769 vs 8.174 GFLOPs and 15.59 vs 90.00 MB of weights, and has the highest batch-32 throughput (1,470.84 images/s).
- **ViT-B16 lags far behind** with 67 errors and is still improving at epoch 10.
- **Training time does not track model cost.** EfficientNetB0 (0.769 GFLOPs) trained at 23.51 images/s and took 0:26:53; SwinBase (30.860 GFLOPs) trained at 67.43 images/s and took 0:09:28. Runs used separate Kaggle sessions.
- **Meningioma is the hardest class for every model.** It is the smallest class, and nearly all errors involve it. On the test set, apart from ViT-B16, no model confuses Glioma and Pituitary more than once.
- **Every model scores higher on test than on validation** (table above). The notebooks give no explanation.

## Plots for every run

Each run folder has 14 plots. Plots `01` to `06` describe the shared data and are identical across runs; plots `07` to `14` are specific to each model.

What the SwinBase Grad-CAM "most confident correct" panel shows: for Meningioma the heatmaps sit on the visible lesion, but for Glioma and Pituitary they mostly highlight image corners and borders outside the brain. This is a visual observation from one figure, not a measured result. It may point to background or acquisition cues that differ between sources or classes; the notebooks do not test this.

### CoAtNet-0 plots

Run folder: [`classification-coatnet0/`](classification-coatnet0/). Notebook: `classification-coatnet0.ipynb` (removed; in Git history). Best epoch 9 of 10, test accuracy 0.9844.

![CoAtNet-0 01_class_distribution](classification-coatnet0/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![CoAtNet-0 02_sample_images](classification-coatnet0/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![CoAtNet-0 03_image_properties](classification-coatnet0/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![CoAtNet-0 04_deduplication_before_after](classification-coatnet0/plots/04_deduplication_before_after.png)

_04_deduplication_before_after: Class counts before and after pHash deduplication._

![CoAtNet-0 05_split_distribution](classification-coatnet0/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![CoAtNet-0 06_augmentation_before_after](classification-coatnet0/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![CoAtNet-0 07_training_curves](classification-coatnet0/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![CoAtNet-0 08_confusion_matrices](classification-coatnet0/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![CoAtNet-0 09_roc_pr_curves](classification-coatnet0/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![CoAtNet-0 10_calibration](classification-coatnet0/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![CoAtNet-0 11_per_class_metrics](classification-coatnet0/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![CoAtNet-0 12_prediction_samples](classification-coatnet0/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![CoAtNet-0 13_gradcam_correct](classification-coatnet0/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![CoAtNet-0 14_gradcam_misclassified](classification-coatnet0/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### EfficientNetB0 plots

Run folder: [`classification-efficientnetb0/`](classification-efficientnetb0/). Notebook: `classification-efficientnetb0.ipynb` (removed; in Git history). Best epoch 9 of 10, test accuracy 0.9759.

![EfficientNetB0 01_class_distribution](classification-efficientnetb0/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![EfficientNetB0 02_sample_images](classification-efficientnetb0/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![EfficientNetB0 03_image_properties](classification-efficientnetb0/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![EfficientNetB0 04_deduplication_before_after](classification-efficientnetb0/plots/04_deduplication_before_after.png)

_04_deduplication_before_after: Class counts before and after pHash deduplication._

![EfficientNetB0 05_split_distribution](classification-efficientnetb0/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![EfficientNetB0 06_augmentation_before_after](classification-efficientnetb0/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![EfficientNetB0 07_training_curves](classification-efficientnetb0/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![EfficientNetB0 08_confusion_matrices](classification-efficientnetb0/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![EfficientNetB0 09_roc_pr_curves](classification-efficientnetb0/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![EfficientNetB0 10_calibration](classification-efficientnetb0/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![EfficientNetB0 11_per_class_metrics](classification-efficientnetb0/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![EfficientNetB0 12_prediction_samples](classification-efficientnetb0/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![EfficientNetB0 13_gradcam_correct](classification-efficientnetb0/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![EfficientNetB0 14_gradcam_misclassified](classification-efficientnetb0/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### InceptionV3 plots

Run folder: [`classification-inceptionv3/`](classification-inceptionv3/). Notebook: `classification-inceptionv3.ipynb` (removed; in Git history). Best epoch 8 of 10, test accuracy 0.9830.

![InceptionV3 01_class_distribution](classification-inceptionv3/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![InceptionV3 02_sample_images](classification-inceptionv3/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![InceptionV3 03_image_properties](classification-inceptionv3/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![InceptionV3 04_deduplication_before_after](classification-inceptionv3/plots/04_deduplication_before_after.png)

_04_deduplication_before_after: Class counts before and after pHash deduplication._

![InceptionV3 05_split_distribution](classification-inceptionv3/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![InceptionV3 06_augmentation_before_after](classification-inceptionv3/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![InceptionV3 07_training_curves](classification-inceptionv3/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![InceptionV3 08_confusion_matrices](classification-inceptionv3/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![InceptionV3 09_roc_pr_curves](classification-inceptionv3/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![InceptionV3 10_calibration](classification-inceptionv3/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![InceptionV3 11_per_class_metrics](classification-inceptionv3/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![InceptionV3 12_prediction_samples](classification-inceptionv3/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![InceptionV3 13_gradcam_correct](classification-inceptionv3/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![InceptionV3 14_gradcam_misclassified](classification-inceptionv3/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### ResNet50 plots

Run folder: [`classification-resnet50/`](classification-resnet50/). Notebook: `classification-resnet50.ipynb` (removed; in Git history). Best epoch 9 of 10, test accuracy 0.9759.

![ResNet50 01_class_distribution](classification-resnet50/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![ResNet50 02_sample_images](classification-resnet50/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![ResNet50 03_image_properties](classification-resnet50/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![ResNet50 04_deduplication_before_after](classification-resnet50/plots/04_deduplication_before_after.png)

_04_deduplication_before_after: Class counts before and after pHash deduplication._

![ResNet50 05_split_distribution](classification-resnet50/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![ResNet50 06_augmentation_before_after](classification-resnet50/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![ResNet50 07_training_curves](classification-resnet50/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![ResNet50 08_confusion_matrices](classification-resnet50/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![ResNet50 09_roc_pr_curves](classification-resnet50/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![ResNet50 10_calibration](classification-resnet50/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![ResNet50 11_per_class_metrics](classification-resnet50/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![ResNet50 12_prediction_samples](classification-resnet50/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![ResNet50 13_gradcam_correct](classification-resnet50/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![ResNet50 14_gradcam_misclassified](classification-resnet50/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### SwinBase plots

Run folder: [`classification-swinbase/`](classification-swinbase/). Notebook: `classification-swinbase.ipynb` (removed; in Git history). Best epoch 7 of 10, test accuracy 0.9858.

![SwinBase 01_class_distribution](classification-swinbase/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![SwinBase 02_sample_images](classification-swinbase/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![SwinBase 03_image_properties](classification-swinbase/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![SwinBase 04_deduplication_before_after](classification-swinbase/plots/04_deduplication_before_after.png)

_04_deduplication_before_after: Class counts before and after pHash deduplication._

![SwinBase 05_split_distribution](classification-swinbase/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![SwinBase 06_augmentation_before_after](classification-swinbase/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![SwinBase 07_training_curves](classification-swinbase/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![SwinBase 08_confusion_matrices](classification-swinbase/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![SwinBase 09_roc_pr_curves](classification-swinbase/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![SwinBase 10_calibration](classification-swinbase/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![SwinBase 11_per_class_metrics](classification-swinbase/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![SwinBase 12_prediction_samples](classification-swinbase/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![SwinBase 13_gradcam_correct](classification-swinbase/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![SwinBase 14_gradcam_misclassified](classification-swinbase/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### ViT-B16 plots

Run folder: [`classification-vit-b16/`](classification-vit-b16/). Notebook: `classification-vit-b16.ipynb` (removed; in Git history). Best epoch 10 of 10, test accuracy 0.9051.

![ViT-B16 01_class_distribution](classification-vit-b16/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![ViT-B16 02_sample_images](classification-vit-b16/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![ViT-B16 03_image_properties](classification-vit-b16/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![ViT-B16 04_deduplication_before_after](classification-vit-b16/plots/04_deduplication_before_after.png)

_04_deduplication_before_after: Class counts before and after pHash deduplication._

![ViT-B16 05_split_distribution](classification-vit-b16/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![ViT-B16 06_augmentation_before_after](classification-vit-b16/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![ViT-B16 07_training_curves](classification-vit-b16/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![ViT-B16 08_confusion_matrices](classification-vit-b16/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![ViT-B16 09_roc_pr_curves](classification-vit-b16/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![ViT-B16 10_calibration](classification-vit-b16/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![ViT-B16 11_per_class_metrics](classification-vit-b16/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![ViT-B16 12_prediction_samples](classification-vit-b16/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![ViT-B16 13_gradcam_correct](classification-vit-b16/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![ViT-B16 14_gradcam_misclassified](classification-vit-b16/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

## Output files reference

Each run folder has the same 22 metric files: the 21 described in the [Binary Detection README](../Binary%20Detection/README.md#output-files-reference) plus `deduplication_summary.csv`. Differences for this task:

| File                              | Contents                                                                                                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `deduplication_summary.csv`       | Per class: `before`, `after`, `removed`, `removed_pct`                                                                                                              |
| `predictions_{val,test}.csv`      | 706 rows each. Columns: `path`, `true_label`, `true_class`, `pred_label`, `pred_class`, `confidence`, `correct`, `prob_glioma`, `prob_meningioma`, `prob_pituitary` |
| `confusion_matrix_{val,test}.csv` | 3x3 counts, rows true class, columns predicted class                                                                                                                |
| `metrics_{val,test}.json`         | The multi-class metric set (see [All test metrics](#all-test-metrics)); no TP/FP/FN/TN or Youden fields                                                             |

Weight files in the local working copy (not tracked in Git). Sizes are in bytes.

| Run                             | `best_*.pth`                                        | `final_*.pth`                                                    |
| ------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| `classification-coatnet0`       | `best_coatnet0_multi_class.pth` (106,821,835)       | `final_coatnet0_brain_tumor_multi_class.pth` (106,825,187)       |
| `classification-efficientnetb0` | `best_efficientnet_b0_multi_class.pth` (16,353,517) | `final_efficientnet_b0_brain_tumor_multi_class.pth` (16,359,363) |
| `classification-inceptionv3`    | `best_inceptionv3_multi_class.pth` (100,832,073)    | `final_inceptionv3_brain_tumor_multi_class.pth` (100,839,755)    |
| `classification-resnet50`       | `best_resnet50_multi_class.pth` (94,380,651)        | `final_resnet50_brain_tumor_multi_class.pth` (94,384,889)        |
| `classification-swinbase`       | `best_swinbase_multi_class.pth` (347,579,387)       | `final_swinbase_brain_tumor_multi_class.pth` (347,584,054)       |
| `classification-vit-b16`        | `best_vitb16_multi_class.pth` (343,267,183)         | `final_vitb16_brain_tumor_multi_class.pth` (343,269,301)         |

## Anomalies and limitations

### A. Issues that affect results

- **A1. Deduplication runs across classes and depends on order.** Glioma images always win ties because they are processed first, and cross-class removals are not reported.
- **A2. The official `sartajbhuvaji` Testing split is merged into the pool.** Results are not comparable with published numbers that use that split.
- **A3. Batch sizes differ (32 to 256)** with the same learning rate and epoch count, so optimisation budget is not controlled.
- **A4. SwinBase Grad-CAM focuses outside the brain for some classes** (see [Plots for every run](#plots-for-every-run)).
- **A5. EfficientNetB0 test accuracy differs between paths.** The original test cell prints 0.9773 (690/706); `metrics_test.json` records 0.9759 (689/706). The first comes from the DataParallel model, the second from the single-GPU copy in cell 52. A near-tied prediction probably flipped through numerical differences; the notebook does not show which image. The other five runs agree. This README uses the metrics files.

### B. Issues that affect interpretation

- **B1.** Every model scores higher on test than on validation. All runs use one seed with no confidence intervals.
- **B2.** Checkpoints are selected on validation macro AUC, which saturates near 0.997 and barely separates late epochs that differ in accuracy.
- **B3.** Plots `01`-`03` and `image_dimension_stats.csv` describe 5,828 images, not the 4,705 used for training.
- **B4. There is no "no tumour" class.** The model always outputs one of three tumour types, even for a healthy scan. The web app runs it on every upload but reports its class only when the binary detector flags a tumour.
- **B5. InceptionV3 keeps an unused auxiliary head.** Only `fc` is replaced; `AuxLogits.fc` still maps 768 inputs to 1000 ImageNet classes. Its 769,000 parameters are counted in the 25,118,411 total and stored in the checkpoint, so loading the weights needs the same unmodified auxiliary head.
- **B6. The folder-level `EfficientNetB0.pth` has unknown provenance.** It is the file the web app loads. It has the same size (16,359,363 bytes) as `classification-efficientnetb0/weights/final_efficientnet_b0_brain_tumor_multi_class.pth`, but a different MD5 (`15162bc973740e58d0a9e8dd2cd18821` vs `4ae7790efe1763feefca9b3ae9c112b6`). No notebook here writes this file, so its test metrics are unknown.

### C. Reproducibility (no effect on results)

- `cudnn.benchmark`, TF32 and AMP allow non-deterministic kernels.
- Kaggle paths are hard-coded.
- `run_epoch` evaluation does not use `torch.no_grad()`.

## Reproducing a run

To run the current pipeline, use the unified notebook (see [Unified notebook](#unified-notebook)). The steps below reproduce a legacy run from its notebook in Git history.

1. Create a Kaggle notebook with the **GPU T4 x2** accelerator and internet access enabled.
2. Attach `sartajbhuvaji/brain-tumor-classification-mri` and `moundir8/jun-cheng-brain-tumor-mri-dataset-jpg` so that the paths in cell 10 resolve.
3. Make sure `imagehash` is available, plus `timm` for CoAtNet-0. The runs used Python 3.12.13, torch 2.10.0+cu128 and CUDA 12.8.
4. Run all cells in order. The tracker (cell 40) must run before the training loop (cell 42).
5. Download `/kaggle/working/<RUN_NAME>-outputs.zip` and unzip it here.

## Loading a trained model

```python
import torch
import torch.nn as nn
from torchvision.models import swin_b

model = swin_b()
model.head = nn.Linear(model.head.in_features, 3)
state = torch.load("classification-swinbase/weights/best_swinbase_multi_class.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
# Input: RGB, Resize((224, 224)), ToTensor, ImageNet Normalize.
# Softmax output order: [Glioma, Meningioma, Pituitary]
```

For InceptionV3, build `inception_v3()` with the default 1000-class auxiliary head, replace only `fc` with `Linear(2048, 3)`, load the checkpoint and use 299 px input.

## Metric glossary

| Term                | Meaning                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| Balanced accuracy   | Mean of per-class recall                                                              |
| Macro F1            | Unweighted mean of per-class F1                                                       |
| MCC                 | Matthews correlation coefficient (multi-class form), from -1 to 1                     |
| Cohen's kappa       | Agreement with the true labels, corrected for chance                                  |
| Top-2 accuracy      | Share of images whose true class is among the two highest-scoring classes             |
| ROC AUC (OvR macro) | Each class scored against the rest, then averaged without weighting                   |
| ROC AUC (OvO macro) | Every pair of classes compared, then averaged                                         |
| PR AUC              | Area under the precision-recall curve (average precision)                             |
| Micro-average       | Pools all class decisions before computing a curve                                    |
| Log loss            | Penalises confident wrong probabilities. Lower is better.                             |
| Brier score         | Mean squared error of the predicted probability vector                                |
| ECE                 | Expected calibration error: weighted gap between confidence and accuracy over 15 bins |
