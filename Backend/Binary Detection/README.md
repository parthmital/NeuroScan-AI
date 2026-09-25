# Binary Detection: Tumour vs No Tumour

This folder benchmarks nine pretrained image classifiers on one question: does a brain MRI slice show a tumour (`Tumor`) or not (`No Tumor`)? There is one Kaggle notebook per model. All nine notebooks share the same data, split, training recipe and evaluation code, so the model is the main intended difference.

Every number in this README is read from the saved notebook outputs and the files in each run folder. Where the evidence does not explain a choice, the README says so instead of guessing. The notebooks were not re-executed for this README.

## At a glance

| Item                      | Value                                                                                                                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task                      | Binary classification: `No Tumor` (label 0) vs `Tumor` (label 1)                                                                                                                                               |
| Data                      | 3,253 JPG/PNG/TIF images from 2 Kaggle datasets, 1,598 No Tumor and 1,655 Tumor, **not deduplicated**                                                                                                          |
| Split                     | Stratified 70/15/15 with seed 42: train 2,277, validation 488, test 488                                                                                                                                        |
| Recipe                    | 10 epochs, AdamW (lr 3e-4), cosine LR schedule, `BCEWithLogitsLoss` on one logit, AMP fp16, 2x Tesla T4                                                                                                        |
| Best test accuracy        | 0.9939 (3 of 488 wrong), shared by DenseNet121, ResNet50, EfficientNetB0 and CoAtNet-0                                                                                                                         |
| Lowest test accuracy      | VGG16, 0.8832                                                                                                                                                                                                  |
| Only zero-miss models     | EfficientNetB0 and CoAtNet-0 (0 false negatives, 3 false positives each)                                                                                                                                       |
| Best probability quality  | DenseNet121: lowest test log loss (0.0189) and ECE (0.0044), ROC AUC 0.99998                                                                                                                                   |
| Model used by the web app | `EfficientNetB0.pth` in this folder, loaded by [`Backend/neuroscan/inference/processor.py`](../neuroscan/inference/processor.py). Its origin is not recorded (see [B4](#b-issues-that-affect-interpretation)). |

Main caveats, detailed in [Anomalies and limitations](#anomalies-and-limitations):

1. The two source datasets are merged without a duplicate check, so train/test leakage cannot be ruled out.
2. InceptionV3 is wrapped for two GPUs but ran its forward pass on one.
3. MobileNetV3-Large ranks images well (ROC AUC 0.998) but misses 45 tumours at the 0.5 threshold.
4. Batch sizes differ between models (64, 128, 256), so optimiser steps are not equal.
5. Every result comes from a single seed. The top four models cannot be separated statistically.

## Contents

- [Folder layout](#folder-layout)
- [Pipeline overview](#pipeline-overview)
- [Dataset](#dataset)
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
Binary Detection/
|-- README.md                              This file
|-- EfficientNetB0.pth                     Weights loaded by the backend (not tracked in Git)
|-- binary-detection-<model>.ipynb         9 notebooks, 65 cells each
`-- binary-detection-<model>/              Unzipped outputs of that notebook
    |-- metrics/                           21 CSV, JSON and TXT files (tracked in Git)
    |-- plots/                             13 PNG files, numbered 01-03 and 05-14 (tracked in Git)
    |-- tensorboard/<YYYYMMDD-HHMMSS>/     One events.out.tfevents.* file (not tracked in Git)
    `-- weights/                           best_*.pth and final_*.pth (not tracked in Git)
```

`<model>` is one of `coatnet0`, `densenet121`, `efficientnetb0`, `inceptionv3`, `mobilenetv3-large`, `resnet50`, `swinbase`, `vgg16` and `vit-b16`.

**What a fresh clone contains.** The root [`.gitignore`](../../.gitignore) excludes `*.pth`, `tensorboard/` and `events.out.tfevents.*`. A clone therefore has the notebooks, `metrics/` and `plots/`, but no weights and no TensorBoard logs. Those exist only in the working copy they were unzipped into.

**`best_*` vs `final_*` weights.** The final-save cell runs after the best checkpoint has been reloaded, so `final_*.pth` holds the **best-validation-AUC weights**, not the last-epoch weights. The two files differ slightly in size only because they are serialised separately.

## Pipeline overview

```mermaid
flowchart LR
    A[Collect image paths] --> B[Explore data] --> C[Stratified split 70/15/15]
    C --> D[Train 10 epochs, keep best val AUC] --> E[Test best checkpoint]
    E --> F[Metrics, plots, Grad-CAM] --> G[Benchmark and zip outputs]
```

| Step        | What happens                                                                                                | Key numbers                     |
| ----------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------- |
| 1. Collect  | `os.walk` over the `yes` and `no` folders of 2 Kaggle datasets builds a `(path, label)` table               | 3,253 images                    |
| 2. Explore  | Class counts, sample images, image sizes and intensity histograms. Nothing is changed.                      | Plots `01`-`03`                 |
| 3. Split    | Two chained `train_test_split` calls, stratified on the label, seed 42                                      | 2,277 / 488 / 488               |
| 4. Prepare  | `BrainDataset` converts each image to RGB; resize, flip, rotation, colour jitter and ImageNet normalisation | 224 px (299 px for InceptionV3) |
| 5. Build    | ImageNet-pretrained backbone, head replaced by one output logit, `nn.DataParallel` over 2 GPUs              |                                 |
| 6. Train    | AdamW 3e-4, cosine LR, `BCEWithLogitsLoss`, AMP; save when validation AUC improves                          | `best_*.pth`                    |
| 7. Test     | Reload the best checkpoint, evaluate the test set, save the same weights again                              | `final_*.pth`                   |
| 8. Evaluate | Single-GPU re-inference on validation and test, then metrics and plots                                      | Plots `07`-`12`                 |
| 9. Explain  | Grad-CAM on the most confident correct and misclassified test images                                        | Plots `13`-`14`                 |
| 10. Package | Latency and throughput benchmark, one-row summary, zip of every output                                      | 37 files per zip                |

The first markdown cell of every notebook states the design rule: the data, split, training and test cells are the original cells, and the added cells "only observe the run". Randomness in the added cells runs inside `preserved_rng_state()`, so adding them does not change the training outcome.

## Dataset

### Sources

The paths are hard-coded in the data collection cell (cell 8).

| Label        | Kaggle folders                                                                                                                                    |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 `No Tumor` | `/kaggle/input/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/no`, `/kaggle/input/datasets/ahmedhamada0/brain-tumor-detection/no`   |
| 1 `Tumor`    | `/kaggle/input/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/yes`, `/kaggle/input/datasets/ahmedhamada0/brain-tumor-detection/yes` |

Folders are walked recursively. Accepted extensions are `.jpg`, `.jpeg`, `.png` and `.tif`, case-insensitive.

### Class distribution

| Class     | Images    | Share |
| --------- | --------- | ----- |
| No Tumor  | 1,598     | 49.1% |
| Tumor     | 1,655     | 50.9% |
| **Total** | **3,253** |       |

The classes are close to balanced. No class weighting or resampling is used.

### Split

Source: `metrics/split_distribution.csv`, identical in all nine runs.

| Class     | Train     | Validation | Test    |
| --------- | --------- | ---------- | ------- |
| No Tumor  | 1,119     | 239        | 240     |
| Tumor     | 1,158     | 249        | 248     |
| **Total** | **2,277** | **488**    | **488** |

### Image properties

Source: `metrics/image_dimension_stats.csv`, identical in all nine runs.

| Class    | Width mean +/- std (min-max) | Width median | Height mean +/- std (min-max) | Height median |
| -------- | ---------------------------- | ------------ | ----------------------------- | ------------- |
| No Tumor | 302.6 +/- 154.9 (150-1920)   | 227          | 308.9 +/- 144.9 (168-1080)    | 240           |
| Tumor    | 351.5 +/- 195.2 (175-1275)   | 283          | 400.2 +/- 208.1 (167-1427)    | 342           |

Every image is resized to a fixed square (224 or 299 px) **without keeping the aspect ratio**. Because image sizes differ between the classes, the amount of resize distortion is partly class-dependent. The notebooks do not test whether the models use this as a shortcut.

The dataset plots (`01`, `02`, `03`, `05`, `06`) are identical in every run folder and appear in the [plot gallery](#plots-for-every-run).

### No deduplication

This pipeline does **not** remove duplicate or near-duplicate images before splitting. Two public datasets are merged with no overlap check between or within them, so a duplicate could land in both train and test. Read the near-perfect scores of the top models with that in mind. The [Classification](../Classification/README.md) benchmark, by contrast, removes near-duplicates with perceptual hashing.

## Shared configuration

These values are the same in every notebook unless [Per-notebook differences](#per-notebook-differences) says otherwise.

### Optimisation

| Setting         | Value                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Seed            | `42` for Python `random`, NumPy and `torch.manual_seed`                                                                   |
| Epochs          | `10`                                                                                                                      |
| Optimiser       | `AdamW(model.parameters(), lr=3e-4)` with the default weight decay (0.01). All layers are trainable.                      |
| Scheduler       | `CosineAnnealingLR(T_max=10)`, stepped once per epoch. The logged LR goes from 3.00e-04 (epoch 1) to 7.34e-06 (epoch 10). |
| Loss            | `BCEWithLogitsLoss` on a single logit, with `float32` labels                                                              |
| Precision       | `torch.amp.autocast("cuda")` with `GradScaler`, TF32 for matmul and cuDNN, `cudnn.benchmark = True`                       |
| Checkpoint rule | Save when `val_auc > best_auc`, so the first epoch to reach the maximum wins                                              |

### Data loading and augmentation

| Setting                     | Value                                                                                                                                                      |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image size                  | 224 px (299 px for InceptionV3)                                                                                                                            |
| Batch size                  | 256 by default; some models override it                                                                                                                    |
| DataLoader                  | `num_workers=4`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`                                                                         |
| Train loader                | `shuffle=True`, `drop_last=True`                                                                                                                           |
| Validation and test loaders | No shuffling, no dropping                                                                                                                                  |
| Train transform             | `Resize((S,S))` -> `RandomHorizontalFlip()` -> `RandomRotation(10)` -> `ColorJitter(brightness=0.2, contrast=0.2)` -> `ToTensor()` -> ImageNet `Normalize` |
| Evaluation transform        | `Resize((S,S))` -> `ToTensor()` -> ImageNet `Normalize`                                                                                                    |

`drop_last=True` skips part of the training set in every epoch. The skipped images change each epoch because the loader shuffles.

| Batch size | Batches per epoch | Images used per epoch | Optimiser steps in 10 epochs |
| ---------- | ----------------- | --------------------- | ---------------------------- |
| 256        | 8                 | 2,048 (229 dropped)   | 80                           |
| 128        | 17                | 2,176 (101 dropped)   | 170                          |
| 64         | 35                | 2,240 (37 dropped)    | 350                          |

### Environment

Recorded in `benchmark_summary.json` for every run: Python 3.12.13, torch 2.10.0+cu128, CUDA 12.8, 2x Tesla T4 on Kaggle. `nn.DataParallel` is used when more than one GPU is present, which was true in all nine runs.

## Per-notebook differences

EfficientNetB0 is the reference template. Differences that are only comments, cell titles or equivalent formatting are not listed.

| Notebook          | Backbone and head                                                                                               | Batch | Input   | Grad-CAM layer (resolved type)                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------- | ----- | ------- | -------------------------------------------------------- |
| EfficientNetB0    | `torchvision.efficientnet_b0(DEFAULT)`, `classifier[1] -> Linear(1280, 1)`                                      | 256   | 224     | `features[-1]` (Conv2dNormActivation)                    |
| MobileNetV3-Large | `torchvision.mobilenet_v3_large(IMAGENET1K_V1)`, `classifier[3] -> Linear(in, 1)`                               | 256   | 224     | `features[-1]` (Conv2dNormActivation)                    |
| ResNet50          | `torchvision.resnet50(DEFAULT)`, `fc -> Linear(2048, 1)`                                                        | 256   | 224     | `layer4[-1]` (Bottleneck)                                |
| VGG16             | `torchvision.vgg16(DEFAULT)`, `classifier[6] -> Linear(4096, 1)`                                                | 256   | 224     | `features[-1]` (**MaxPool2d**)                           |
| DenseNet121       | `torchvision.densenet121(DEFAULT)`, `classifier -> Linear(1024, 1)`                                             | 64    | 224     | `features[-1]` (BatchNorm2d, `norm5`)                    |
| InceptionV3       | `torchvision.inception_v3(DEFAULT, aux_logits=True)`, `fc -> Linear(2048, 1)`, `AuxLogits.fc -> Linear(768, 1)` | 128   | **299** | `Mixed_7c` (InceptionE)                                  |
| CoAtNet-0         | `timm.create_model("coatnet_0_rw_224", pretrained=True)`, `reset_classifier(1)`                                 | 64    | 224     | `stages[-1]` (MaxxVitStage)                              |
| SwinBase          | `timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=1)`                             | 64    | 224     | `layers[-1].blocks[-1].norm1` (LayerNorm, channels-last) |
| ViT-B16           | `torchvision.vit_b_16(DEFAULT)`, `heads -> Linear(768, 1)` (replaces the whole `heads` module)                  | 128   | 224     | `encoder.layers[-1].ln_1` (LayerNorm, tokens)            |

Other code differences:

- **MobileNetV3-Large** pins `IMAGENET1K_V1` weights. Every other torchvision model uses `.DEFAULT`.
- **InceptionV3** adds the auxiliary loss while training (`loss = main + 0.4 * aux`) and calls the unwrapped module (`m = model.module ...; m(images)`) in `run_epoch`. TF32 flags are removed and `GradScaler()` has no device argument.
- **CoAtNet-0** and **SwinBase** need `timm`. SwinBase downloads weights from the Hugging Face Hub, and the saved output shows an unauthenticated-request warning.
- **VGG16**, **DenseNet121**, **InceptionV3** and **CoAtNet-0** have no `try/except` around image loading in `BrainDataset`.
- **ResNet50** and **ViT-B16** import `classification_report` without using it.

Why they differ, using only evidence from the notebooks:

- **Batch size.** The only written reason is in InceptionV3: `# smaller batch due to inception memory`. No reason is given for 64 (DenseNet121, CoAtNet-0, SwinBase) or 128 (ViT-B16). Recorded peak memory shows VGG16 at batch 256 reaching 14,587.8 MB on GPU1, close to the 16 GB of a T4, so memory limits are plausible but unstated.
- **Head replacement.** Each architecture exposes its classifier under a different attribute, so this code has to differ.
- **Grad-CAM layer.** The notebook-settings markdown describes each choice as the last spatial or token stage of that architecture.
- **InceptionV3 auxiliary weight 0.4.** This is the standard Inception training recipe. The notebook applies it without comment.
- **timm.** CoAtNet has no torchvision implementation. For Swin, the only comment is `# For Swin Transformer`.

## Notebook walkthrough

All nine notebooks have 65 cells in markdown and code pairs. Indices below are 0-based. Code comments `# CELL 1` to `# CELL 10` mark the original training cells; unnumbered code cells are the added observation, evaluation and benchmark cells. Printed outputs quoted here are identical in all nine runs unless a model is named.

| Section                    | Cells | Purpose                                                 |
| -------------------------- | ----- | ------------------------------------------------------- |
| 1. Setup                   | 0-6   | Configuration, notebook identity, helpers               |
| 2. Data                    | 7-22  | Collect, explore, split, load, augment                  |
| 3. Model                   | 23-26 | Build and profile the model                             |
| 4. Training                | 27-36 | Epoch function, tracker, training loop, summary, curves |
| 5. Testing and saving      | 37-40 | Test the best checkpoint and save weights               |
| 6. Detailed evaluation     | 41-54 | Predictions, metrics and evaluation plots               |
| 7. Explainability          | 55-58 | Grad-CAM                                                |
| 8. Benchmark and packaging | 59-64 | Speed benchmark, summary and zip                        |

### 1. Setup (cells 0-6)

- **Cell 0 (markdown).** Title, the pipeline steps and the "added cells only observe the run" rule.
- **Cell 2, imports and configuration.** Sets `SEED`, `BATCH_SIZE`, `EPOCHS`, `LR`, `IMG_SIZE` and loader settings in one place, seeds the random generators and sets cuDNN and TF32 flags. Output: CUDA available `True`, GPU count `2`.
- **Cell 4, notebook settings.** The only per-notebook values the shared cells need: `MODEL_NAME`, `TASK="binary-detection"`, `CLASS_NAMES=["No Tumor","Tumor"]`, `RUN_NAME` and `get_gradcam_target_layer(net)`.
- **Cell 6, output folders and helpers.** Creates `/kaggle/working/outputs/<RUN_NAME>/{plots,metrics}` and defines `save_fig`, `save_json`, `save_table`, `save_text`, `unwrap_model` (removes the `DataParallel` wrapper), `denormalise`, `sync_cuda` and `preserved_rng_state()`, which snapshots and restores the Python, NumPy, torch and CUDA random states.

### 2. Data (cells 7-22)

| Cell | What it does                                                                                                                                   | Output                                                 |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 8    | `collect(paths, label)` walks the folders and builds `df[path, label]`                                                                         | `Total Images: 3253` (label 1: 1655, label 0: 1598)    |
| 10   | Copies `df` to `df_raw`, plots the class bar chart and 6 random samples per class                                                              | `01_class_distribution.png`, `02_sample_images.png`    |
| 12   | Reads every image size; grayscale intensity histograms from 100 sampled images per class (51 bins)                                             | `03_image_properties.png`, `image_dimension_stats.csv` |
| 14   | Stratified 70/15/15 split                                                                                                                      | `Train: 2277 Val: 488 Test: 488`                       |
| 16   | Per-split class counts and proportions                                                                                                         | `05_split_distribution.png`, `split_distribution.csv`  |
| 18   | `BrainDataset`: opens the image, converts it to RGB so grayscale scans get 3 channels, applies the transform, returns `(image, float32 label)` |                                                        |
| 20   | Transforms and the three `DataLoader`s                                                                                                         |                                                        |
| 22   | For 3 training images per class, the evaluation transform (top) and one random augmentation (bottom)                                           | `06_augmentation_before_after.png`                     |

No cell writes a plot numbered `04`. The numbering is shared with the Classification notebooks, where `04` is the deduplication plot.

### 3. Model (cells 23-26)

- **Cell 24, model build.** Loads the pretrained backbone, replaces the head with a 1-output `Linear`, wraps it in `DataParallel`, and creates the loss, optimiser, scheduler and `GradScaler`.
- **Cell 26, model profile.** Counts parameters and buffers, measures the serialised `state_dict` size with `BytesIO`, and counts FLOPs for one `1x3xSxS` input with `torch.utils.flop_counter.FlopCounterMode` on a CPU copy. Saved as `model_profile.json`.

### 4. Training (cells 27-36)

- **Cell 28, `run_epoch(loader, training)`.** Autocast forward pass, `BCEWithLogits(logits.squeeze(1), labels)`, scaled backward pass and optimiser step when training. Returns mean batch loss, accuracy at threshold 0.5 and ROC AUC. The evaluation pass does not use `torch.no_grad()`; it only skips the optimiser step.
- **Cell 30, training tracker.** Rebinds `run_epoch` to a wrapper that detects the phase from the loader, times it with CUDA syncs, records LR, loss, accuracy, AUC and timings per epoch, and drives a tqdm bar. This keeps the original training cell unchanged. A `_tracks_history` flag prevents double wrapping.
- **Cell 32, training loop.** 10 epochs of train, validate, `scheduler.step()`. Writes TensorBoard scalars `Loss/Train`, `Loss/Val`, `AUC/Train` and `AUC/Val` and saves a checkpoint when validation AUC improves. Per-epoch results are in [Per-epoch training history](#per-epoch-training-history).
- **Cell 34, training summary.** Best epoch (first maximum of validation AUC), total and mean times, time to best epoch, training images per second (`len(train_loader) * BATCH_SIZE / mean train time`) and peak memory per GPU. Saved as `training_history.csv` and `training_summary.json`.
- **Cell 36, training curves.** Saves `07_training_curves.png`.

### 5. Testing and saving (cells 37-40)

- **Cell 38, test.** Loads `best_*.pth` into the `DataParallel` model, runs the test pass and prints `===== FINAL UNSEEN TEST RESULTS =====` with loss, accuracy and AUC. The printed values for every model are in [Original test cell output vs metrics files](#original-test-cell-output-vs-metrics-files).
- **Cell 40, save.** Saves the reloaded best weights again as `final_*_brain_tumor.pth`.

### 6. Detailed evaluation (cells 41-54)

- **Cell 42, predictions.** Copies the unwrapped model onto one GPU and re-runs validation and test under autocast. The single logit becomes the probability pair `[1 - sigmoid, sigmoid]`, renormalised in float64. Saved as `predictions_val.csv` and `predictions_test.csv`.
- **Cell 44, metrics.** Accuracy, balanced accuracy, error rate, precision, recall and F1 (macro, weighted, micro), MCC, Cohen's kappa, macro Jaccard, log loss, Brier score, 15-bin ECE and mean confidence; for binary tasks also TP, FP, FN, TN, sensitivity, specificity, NPV, FPR, FNR, F1, F2, ROC AUC, average precision, positive-class Brier score and the Youden J threshold. Saved as `metrics_{val,test}.json`, `per_class_metrics_{val,test}.csv`, `classification_report_{val,test}.txt` and `metrics_summary.csv`.
- **Cell 46.** Confusion matrices as counts and row-normalised. `08_confusion_matrices.png`, `confusion_matrix_{val,test}.csv`.
- **Cell 48.** ROC and precision-recall curves for the positive class. `09_roc_pr_curves.png`.
- **Cell 50.** Reliability diagram over 15 confidence bins and confidence histograms for correct and incorrect predictions. `10_calibration.png`, `calibration_bins_{val,test}.csv`.
- **Cell 52.** Per-class precision, recall, specificity, F1 and AUC on the test set. `11_per_class_metrics.png`.
- **Cell 54.** 6 random correct and 6 random misclassified test images, seeded with `RandomState(42)`. `12_prediction_samples.png`.

### 7. Explainability (cells 55-58)

- **Cell 56, `GradCAM` class.** A forward hook captures activations and a tensor hook captures gradients at the target layer. Token layouts `(B, T, C)` and channels-last layouts `(B, H, W, C)` are converted to `(B, C, H, W)`. For a single-logit model the score is `+logit` for class 1 and `-logit` for class 0. The map is the ReLU of the gradient-weighted activation sum, upsampled and min-max normalised, computed on a float32 copy.
- **Cell 58, figures.** `13_gradcam_correct.png` shows the 2 most confident correct predictions per class; `14_gradcam_misclassified.png` shows the 6 most confident misclassifications. Each shows input, heatmap and overlay. `gradcam_status` is `ok` in all nine runs.

### 8. Benchmark and packaging (cells 59-64)

- **Cell 60, inference benchmark.** Synthetic `randn` input under fp16 autocast on one GPU. Batch 1: 10 warm-up and 100 timed iterations. Batch 32: 3 warm-up and 20 timed iterations. Also records the end-to-end test-set time from cell 42 and the original test-cell time. Saved as `inference_benchmark.json`.
- **Cell 62, benchmark summary.** Flattens configuration, profile, training, inference, validation and test metrics, Grad-CAM status and environment into `benchmark_summary.json` and `.csv`.
- **Cell 64, packaging.** Zips `outputs/<RUN_NAME>`, every `*.pth` in `/kaggle/working` and the TensorBoard run into `<RUN_NAME>-outputs.zip`. Every run printed `Packaged 37 files`. The run folders here are those zips unpacked.

## Per-notebook run notes

Each note covers what is specific to that notebook's run. Full numbers are in [Results](#results) and every plot is in the [gallery](#plots-for-every-run).

### binary-detection-efficientnetb0.ipynb

- Reference template. Validation AUC reaches 0.99995 at epoch 4 (the checkpoint) and stays saturated; validation loss keeps falling to 0.0142 at epoch 10.
- Test: 0 missed tumours, 3 false alarms, accuracy 0.9939.
- Training throughput was 25.84 images/s, against 77.52 for ResNet50, despite about one tenth of the FLOPs. This suggests that data loading, not model compute, limited training speed (see [What the results show](#what-the-results-show)).

### binary-detection-densenet121.ipynb

- Validation AUC is already 0.99971 after epoch 1 and reaches 1.0 with 100% validation accuracy at epoch 4, the checkpoint.
- Test: 3 missed tumours, 0 false alarms. Lowest test log loss (0.0189) and ECE (0.0044).

### binary-detection-resnet50.ipynb

- Checkpoint at epoch 7. The test confusion matrix is identical to DenseNet121's (3 missed tumours, 0 false alarms).

### binary-detection-coatnet0.ipynb

- Slowest run (0:29:08). Validation loss is noisy: 0.1304 at epoch 4, then 0.0211 at epoch 5.
- Training accuracy reaches 1.0 at epoch 10 while validation loss rises from 0.0211 (epoch 5) to 0.0404 (epoch 10).
- Test: 0 missed tumours, 3 false alarms.

### binary-detection-inceptionv3.ipynb

- Validation AUC reaches 1.0 at epoch 3, the checkpoint. Test accuracy 0.9877 with 5 missed tumours and 1 false alarm.
- Peak memory on GPU1 is 0.0 MB, which confirms the forward pass ran on one GPU ([A1](#a-issues-that-affect-results)).

### binary-detection-swinbase.ipynb

- Validation AUC stays at or below 0.87 for 3 epochs, then climbs to 0.99733 at epoch 6, the checkpoint.
- Most test errors are missed tumours (16 FN, 4 FP).

### binary-detection-vit-b16.ipynb

- Validation AUC is still rising at epoch 10 (0.98993), which is the checkpoint. Final training accuracy is 0.9439.
- Test: 9 missed tumours, 16 false alarms. Under-trained at 10 epochs with this recipe.

### binary-detection-mobilenetv3-large.ipynb

- Training loss falls to about 0.001, but validation loss **rises** to 1.0599 at epoch 2 and only recovers to 0.3995 by epoch 10.
- Validation AUC stays at or above 0.95 throughout, while validation accuracy starts at 0.6988, dips to 0.6557 at epoch 2 and reaches 0.8811 at epoch 10.
- Every test error is a missed tumour (45 FN, 0 FP). The test Youden threshold is 0.00044, far below 0.5. See [A2](#a-issues-that-affect-results).

### binary-detection-vgg16.ipynb

- Epoch-1 training loss is 2.1317 and training accuracy stays between 0.53 and 0.63 for 4 epochs before improving.
- Best at the last epoch (validation AUC 0.96872). Lowest test accuracy (0.8832) with 34 missed tumours and 23 false alarms.

## Results

All tables below are generated from the files in each run folder. Models are sorted by test accuracy, then MCC.

### Test-set classification quality

| Model             | Accuracy | Balanced acc | Sensitivity | Specificity | Missed tumours (FN) | False alarms (FP) | F1 (Tumor) | MCC    |
| ----------------- | -------- | ------------ | ----------- | ----------- | ------------------- | ----------------- | ---------- | ------ |
| DenseNet121       | 0.9939   | 0.9940       | 0.9879      | 1.0000      | 3                   | 0                 | 0.9939     | 0.9878 |
| ResNet50          | 0.9939   | 0.9940       | 0.9879      | 1.0000      | 3                   | 0                 | 0.9939     | 0.9878 |
| EfficientNetB0    | 0.9939   | 0.9938       | 1.0000      | 0.9875      | 0                   | 3                 | 0.9940     | 0.9878 |
| CoAtNet-0         | 0.9939   | 0.9938       | 1.0000      | 0.9875      | 0                   | 3                 | 0.9940     | 0.9878 |
| InceptionV3       | 0.9877   | 0.9878       | 0.9798      | 0.9958      | 5                   | 1                 | 0.9878     | 0.9755 |
| SwinBase          | 0.9590   | 0.9594       | 0.9355      | 0.9833      | 16                  | 4                 | 0.9587     | 0.9192 |
| ViT-B16           | 0.9488   | 0.9485       | 0.9637      | 0.9333      | 9                   | 16                | 0.9503     | 0.8978 |
| MobileNetV3-Large | 0.9078   | 0.9093       | 0.8185      | 1.0000      | 45                  | 0                 | 0.9002     | 0.8302 |
| VGG16             | 0.8832   | 0.8835       | 0.8629      | 0.9042      | 34                  | 23                | 0.8825     | 0.7673 |

Source: `metrics/metrics_test.json` in each run folder. Test set: 488 images.

### Test-set probability quality

| Model             | ROC AUC | PR AUC (AP) | Log loss | Brier (positive class) | ECE    | Youden threshold (test) | Best epoch |
| ----------------- | ------- | ----------- | -------- | ---------------------- | ------ | ----------------------- | ---------- |
| DenseNet121       | 0.99998 | 0.99998     | 0.0189   | 0.0052                 | 0.0044 | 0.17653                 | 4          |
| ResNet50          | 0.99997 | 0.99997     | 0.0217   | 0.0057                 | 0.0057 | 0.06979                 | 7          |
| EfficientNetB0    | 0.99983 | 0.99984     | 0.0233   | 0.0050                 | 0.0049 | 0.84518                 | 4          |
| CoAtNet-0         | 0.99987 | 0.99987     | 0.0253   | 0.0065                 | 0.0084 | 0.99864                 | 6          |
| InceptionV3       | 0.99861 | 0.99880     | 0.0419   | 0.0105                 | 0.0078 | 0.67523                 | 3          |
| SwinBase          | 0.99614 | 0.99658     | 0.1117   | 0.0276                 | 0.0235 | 0.23775                 | 6          |
| ViT-B16           | 0.98616 | 0.98888     | 0.1525   | 0.0407                 | 0.0323 | 0.63262                 | 10         |
| MobileNetV3-Large | 0.99808 | 0.99810     | 0.3698   | 0.0804                 | 0.0685 | 0.00044                 | 10         |
| VGG16             | 0.96760 | 0.96988     | 0.2448   | 0.0759                 | 0.0437 | 0.72836                 | 10         |

### Model size and compute

| Model             | Parameters  | GFLOPs / image | State dict (MB) | Input     | Batch size |
| ----------------- | ----------- | -------------- | --------------- | --------- | ---------- |
| DenseNet121       | 6,954,881   | 5.666          | 27.11           | 3x224x224 | 64         |
| ResNet50          | 23,510,081  | 8.174          | 89.99           | 3x224x224 | 256        |
| EfficientNetB0    | 4,008,829   | 0.769          | 15.58           | 3x224x224 | 256        |
| CoAtNet-0         | 26,667,331  | 8.416          | 101.86          | 3x224x224 | 64         |
| InceptionV3       | 24,346,082  | 11.422         | 93.20           | 3x299x299 | 128        |
| SwinBase          | 86,744,249  | 30.860         | 331.03          | 3x224x224 | 64         |
| ViT-B16           | 85,799,425  | 22.541         | 327.36          | 3x224x224 | 128        |
| MobileNetV3-Large | 4,203,313   | 0.431          | 16.23           | 3x224x224 | 256        |
| VGG16             | 134,264,641 | 30.932         | 512.19          | 3x224x224 | 256        |

Source: `metrics/model_profile.json` and `metrics/benchmark_summary.json`. All parameters are trainable in every run.

### Training time, memory and inference speed

| Model             | Train time | Mean epoch (s) | Train img/s | Peak GPU0 alloc (MB) | Peak GPU1 alloc (MB) | Latency b=1 mean (ms) | Latency b=1 p95 (ms) | Throughput b=32 (img/s) | Test set end-to-end (s) |
| ----------------- | ---------- | -------------- | ----------- | -------------------- | -------------------- | --------------------- | -------------------- | ----------------------- | ----------------------- |
| DenseNet121       | 0:08:28    | 50.71          | 52.81       | 4267.1               | 4197.7               | 19.43                 | 21.16                | 629.80                  | 2.131                   |
| ResNet50          | 0:05:24    | 32.27          | 77.52       | 10838.9              | 10591.7              | 7.35                  | 8.47                 | 771.21                  | 3.238                   |
| EfficientNetB0    | 0:16:17    | 97.69          | 25.84       | 10559.6              | 10463.1              | 10.76                 | 11.78                | 1364.24                 | 2.861                   |
| CoAtNet-0         | 0:29:08    | 174.68         | 15.59       | 5921.2               | 5693.8               | 13.91                 | 15.16                | 520.34                  | 2.313                   |
| InceptionV3       | 0:04:51    | 29.07          | 84.78       | 13518.5              | 0.0                  | 14.05                 | 15.84                | 514.15                  | 3.200                   |
| SwinBase          | 0:06:34    | 38.95          | 66.01       | 10801.8              | 10059.5              | 28.99                 | 31.39                | 195.63                  | 3.871                   |
| ViT-B16           | 0:04:41    | 27.62          | 90.93       | 10462.3              | 9768.1               | 8.80                  | 9.82                 | 286.88                  | 3.049                   |
| MobileNetV3-Large | 0:11:58    | 71.77          | 35.19       | 5415.8               | 5318.1               | 8.08                  | 9.26                 | 2408.52                 | 2.706                   |
| VGG16             | 0:12:28    | 74.06          | 33.73       | 13313.8              | 14587.8              | 8.19                  | 9.72                 | 338.15                  | 3.838                   |

Source: `metrics/training_summary.json` and `metrics/inference_benchmark.json`. Training used 2x Tesla T4; the inference benchmark used one Tesla T4 with fp16 autocast.

### Original test cell output vs metrics files

The original test cell evaluates the DataParallel model and prints loss, accuracy and AUC. The metrics files come from a separate single-GPU re-inference (see the walkthrough). Both are listed so they can be compared.

| Model             | Printed test loss | Printed test acc | Printed test AUC | File log loss | File accuracy | File AUC |
| ----------------- | ----------------- | ---------------- | ---------------- | ------------- | ------------- | -------- |
| DenseNet121       | 0.0180            | 0.9939           | 1.0000           | 0.0189        | 0.9939        | 1.0000   |
| ResNet50          | 0.0222            | 0.9939           | 1.0000           | 0.0217        | 0.9939        | 1.0000   |
| EfficientNetB0    | 0.0240            | 0.9939           | 0.9998           | 0.0233        | 0.9939        | 0.9998   |
| CoAtNet-0         | 0.0337            | 0.9939           | 0.9999           | 0.0253        | 0.9939        | 0.9999   |
| InceptionV3       | 0.0409            | 0.9877           | 0.9986           | 0.0419        | 0.9877        | 0.9986   |
| SwinBase          | 0.1102            | 0.9590           | 0.9961           | 0.1117        | 0.9590        | 0.9961   |
| ViT-B16           | 0.1545            | 0.9488           | 0.9862           | 0.1525        | 0.9488        | 0.9862   |
| MobileNetV3-Large | 0.3737            | 0.9078           | 0.9981           | 0.3698        | 0.9078        | 0.9981   |
| VGG16             | 0.2441            | 0.8832           | 0.9676           | 0.2448        | 0.8832        | 0.9676   |

### All test metrics

| Metric                            | DenseNet121 | ResNet50 | EfficientNetB0 | CoAtNet-0 | InceptionV3 | SwinBase | ViT-B16 | MobileNetV3-Large | VGG16  |
| --------------------------------- | ----------- | -------- | -------------- | --------- | ----------- | -------- | ------- | ----------------- | ------ |
| `samples`                         | 488         | 488      | 488            | 488       | 488         | 488      | 488     | 488               | 488    |
| `accuracy`                        | 0.9939      | 0.9939   | 0.9939         | 0.9939    | 0.9877      | 0.9590   | 0.9488  | 0.9078            | 0.8832 |
| `balanced_accuracy`               | 0.9940      | 0.9940   | 0.9938         | 0.9938    | 0.9878      | 0.9594   | 0.9485  | 0.9093            | 0.8835 |
| `error_rate`                      | 0.0061      | 0.0061   | 0.0061         | 0.0061    | 0.0123      | 0.0410   | 0.0512  | 0.0922            | 0.1168 |
| `precision_macro`                 | 0.9938      | 0.9938   | 0.9940         | 0.9940    | 0.9877      | 0.9598   | 0.9493  | 0.9211            | 0.8837 |
| `recall_macro`                    | 0.9940      | 0.9940   | 0.9938         | 0.9938    | 0.9878      | 0.9594   | 0.9485  | 0.9093            | 0.8835 |
| `f1_macro`                        | 0.9939      | 0.9939   | 0.9938         | 0.9938    | 0.9877      | 0.9590   | 0.9487  | 0.9073            | 0.8832 |
| `precision_weighted`              | 0.9939      | 0.9939   | 0.9939         | 0.9939    | 0.9878      | 0.9602   | 0.9491  | 0.9223            | 0.8841 |
| `recall_weighted`                 | 0.9939      | 0.9939   | 0.9939         | 0.9939    | 0.9877      | 0.9590   | 0.9488  | 0.9078            | 0.8832 |
| `f1_weighted`                     | 0.9939      | 0.9939   | 0.9939         | 0.9939    | 0.9877      | 0.9590   | 0.9487  | 0.9071            | 0.8832 |
| `precision_micro`                 | 0.9939      | 0.9939   | 0.9939         | 0.9939    | 0.9877      | 0.9590   | 0.9488  | 0.9078            | 0.8832 |
| `recall_micro`                    | 0.9939      | 0.9939   | 0.9939         | 0.9939    | 0.9877      | 0.9590   | 0.9488  | 0.9078            | 0.8832 |
| `f1_micro`                        | 0.9939      | 0.9939   | 0.9939         | 0.9939    | 0.9877      | 0.9590   | 0.9488  | 0.9078            | 0.8832 |
| `mcc`                             | 0.9878      | 0.9878   | 0.9878         | 0.9878    | 0.9755      | 0.9192   | 0.8978  | 0.8302            | 0.7673 |
| `cohen_kappa`                     | 0.9877      | 0.9877   | 0.9877         | 0.9877    | 0.9754      | 0.9181   | 0.8975  | 0.8161            | 0.7665 |
| `jaccard_macro`                   | 0.9878      | 0.9878   | 0.9878         | 0.9878    | 0.9757      | 0.9213   | 0.9025  | 0.8303            | 0.7908 |
| `log_loss`                        | 0.0189      | 0.0217   | 0.0233         | 0.0253    | 0.0419      | 0.1117   | 0.1525  | 0.3698            | 0.2448 |
| `brier_score`                     | 0.0105      | 0.0113   | 0.0100         | 0.0130    | 0.0211      | 0.0552   | 0.0813  | 0.1609            | 0.1518 |
| `expected_calibration_error`      | 0.0044      | 0.0057   | 0.0049         | 0.0084    | 0.0078      | 0.0235   | 0.0323  | 0.0685            | 0.0437 |
| `mean_confidence`                 | 0.9962      | 0.9987   | 0.9918         | 0.9948    | 0.9919      | 0.9784   | 0.9261  | 0.9611            | 0.8831 |
| `tp`                              | 245         | 245      | 248            | 248       | 243         | 232      | 239     | 203               | 214    |
| `fp`                              | 0           | 0        | 3              | 3         | 1           | 4        | 16      | 0                 | 23     |
| `fn`                              | 3           | 3        | 0              | 0         | 5           | 16       | 9       | 45                | 34     |
| `tn`                              | 240         | 240      | 237            | 237       | 239         | 236      | 224     | 240               | 217    |
| `precision`                       | 1.0000      | 1.0000   | 0.9880         | 0.9880    | 0.9959      | 0.9831   | 0.9373  | 1.0000            | 0.9030 |
| `recall_sensitivity`              | 0.9879      | 0.9879   | 1.0000         | 1.0000    | 0.9798      | 0.9355   | 0.9637  | 0.8185            | 0.8629 |
| `specificity`                     | 1.0000      | 1.0000   | 0.9875         | 0.9875    | 0.9958      | 0.9833   | 0.9333  | 1.0000            | 0.9042 |
| `npv`                             | 0.9877      | 0.9877   | 1.0000         | 1.0000    | 0.9795      | 0.9365   | 0.9614  | 0.8421            | 0.8645 |
| `fpr`                             | 0.0000      | 0.0000   | 0.0125         | 0.0125    | 0.0042      | 0.0167   | 0.0667  | 0.0000            | 0.0958 |
| `fnr`                             | 0.0121      | 0.0121   | 0.0000         | 0.0000    | 0.0202      | 0.0645   | 0.0363  | 0.1815            | 0.1371 |
| `f1`                              | 0.9939      | 0.9939   | 0.9940         | 0.9940    | 0.9878      | 0.9587   | 0.9503  | 0.9002            | 0.8825 |
| `f2`                              | 0.9903      | 0.9903   | 0.9976         | 0.9976    | 0.9830      | 0.9446   | 0.9583  | 0.8494            | 0.8706 |
| `roc_auc`                         | 1.0000      | 1.0000   | 0.9998         | 0.9999    | 0.9986      | 0.9961   | 0.9862  | 0.9981            | 0.9676 |
| `pr_auc_average_precision`        | 1.0000      | 1.0000   | 0.9998         | 0.9999    | 0.9988      | 0.9966   | 0.9889  | 0.9981            | 0.9699 |
| `brier_score_positive_class`      | 0.0052      | 0.0057   | 0.0050         | 0.0065    | 0.0105      | 0.0276   | 0.0407  | 0.0804            | 0.0759 |
| `youden_j`                        | 0.9960      | 0.9960   | 0.9958         | 0.9879    | 0.9758      | 0.9426   | 0.9137  | 0.9626            | 0.7813 |
| `youden_optimal_threshold`        | 0.1765      | 0.0698   | 0.8452         | 0.9986    | 0.6752      | 0.2378   | 0.6326  | 0.0004            | 0.7284 |
| `sensitivity_at_youden_threshold` | 0.9960      | 0.9960   | 1.0000         | 0.9879    | 0.9758      | 0.9718   | 0.9637  | 0.9960            | 0.8105 |
| `specificity_at_youden_threshold` | 1.0000      | 1.0000   | 0.9958         | 1.0000    | 1.0000      | 0.9708   | 0.9500  | 0.9667            | 0.9708 |

Source: `metrics/metrics_test.json`. The validation split was also used to pick the checkpoint, so validation values are optimistic.

### All validation metrics

| Metric                            | DenseNet121 | ResNet50 | EfficientNetB0 | CoAtNet-0 | InceptionV3 | SwinBase | ViT-B16 | MobileNetV3-Large | VGG16  |
| --------------------------------- | ----------- | -------- | -------------- | --------- | ----------- | -------- | ------- | ----------------- | ------ |
| `samples`                         | 488         | 488      | 488            | 488       | 488         | 488      | 488     | 488               | 488    |
| `accuracy`                        | 1.0000      | 0.9939   | 0.9959         | 0.9898    | 0.9959      | 0.9693   | 0.9406  | 0.8811            | 0.8975 |
| `balanced_accuracy`               | 1.0000      | 0.9939   | 0.9958         | 0.9895    | 0.9958      | 0.9698   | 0.9398  | 0.8835            | 0.8982 |
| `error_rate`                      | 0.0000      | 0.0061   | 0.0041         | 0.0102    | 0.0041      | 0.0307   | 0.0594  | 0.1189            | 0.1025 |
| `precision_macro`                 | 1.0000      | 0.9938   | 0.9960         | 0.9902    | 0.9960      | 0.9701   | 0.9433  | 0.9024            | 0.8988 |
| `recall_macro`                    | 1.0000      | 0.9939   | 0.9958         | 0.9895    | 0.9958      | 0.9698   | 0.9398  | 0.8835            | 0.8982 |
| `f1_macro`                        | 1.0000      | 0.9939   | 0.9959         | 0.9897    | 0.9959      | 0.9693   | 0.9404  | 0.8800            | 0.8975 |
| `precision_weighted`              | 1.0000      | 0.9939   | 0.9959         | 0.9900    | 0.9959      | 0.9706   | 0.9426  | 0.9044            | 0.8994 |
| `recall_weighted`                 | 1.0000      | 0.9939   | 0.9959         | 0.9898    | 0.9959      | 0.9693   | 0.9406  | 0.8811            | 0.8975 |
| `f1_weighted`                     | 1.0000      | 0.9939   | 0.9959         | 0.9898    | 0.9959      | 0.9693   | 0.9405  | 0.8797            | 0.8975 |
| `precision_micro`                 | 1.0000      | 0.9939   | 0.9959         | 0.9898    | 0.9959      | 0.9693   | 0.9406  | 0.8811            | 0.8975 |
| `recall_micro`                    | 1.0000      | 0.9939   | 0.9959         | 0.9898    | 0.9959      | 0.9693   | 0.9406  | 0.8811            | 0.8975 |
| `f1_micro`                        | 1.0000      | 0.9939   | 0.9959         | 0.9898    | 0.9959      | 0.9693   | 0.9406  | 0.8811            | 0.8975 |
| `mcc`                             | 1.0000      | 0.9877   | 0.9918         | 0.9797    | 0.9918      | 0.9399   | 0.8831  | 0.7857            | 0.7970 |
| `cohen_kappa`                     | 1.0000      | 0.9877   | 0.9918         | 0.9795    | 0.9918      | 0.9386   | 0.8809  | 0.7633            | 0.7953 |
| `jaccard_macro`                   | 1.0000      | 0.9878   | 0.9918         | 0.9797    | 0.9918      | 0.9404   | 0.8875  | 0.7859            | 0.8141 |
| `log_loss`                        | 0.0043      | 0.0193   | 0.0180         | 0.0337    | 0.0109      | 0.0943   | 0.1455  | 0.4001            | 0.2419 |
| `brier_score`                     | 0.0019      | 0.0099   | 0.0094         | 0.0155    | 0.0049      | 0.0479   | 0.0816  | 0.1888            | 0.1511 |
| `expected_calibration_error`      | 0.0036      | 0.0058   | 0.0089         | 0.0098    | 0.0092      | 0.0198   | 0.0294  | 0.0855            | 0.0503 |
| `mean_confidence`                 | 0.9964      | 0.9979   | 0.9916         | 0.9946    | 0.9910      | 0.9891   | 0.9261  | 0.9586            | 0.8886 |
| `tp`                              | 249         | 247      | 249            | 249       | 249         | 235      | 243     | 191               | 216    |
| `fp`                              | 0           | 1        | 2              | 5         | 2           | 1        | 23      | 0                 | 17     |
| `fn`                              | 0           | 2        | 0              | 0         | 0           | 14       | 6       | 58                | 33     |
| `tn`                              | 239         | 238      | 237            | 234       | 237         | 238      | 216     | 239               | 222    |
| `precision`                       | 1.0000      | 0.9960   | 0.9920         | 0.9803    | 0.9920      | 0.9958   | 0.9135  | 1.0000            | 0.9270 |
| `recall_sensitivity`              | 1.0000      | 0.9920   | 1.0000         | 1.0000    | 1.0000      | 0.9438   | 0.9759  | 0.7671            | 0.8675 |
| `specificity`                     | 1.0000      | 0.9958   | 0.9916         | 0.9791    | 0.9916      | 0.9958   | 0.9038  | 1.0000            | 0.9289 |
| `npv`                             | 1.0000      | 0.9917   | 1.0000         | 1.0000    | 1.0000      | 0.9444   | 0.9730  | 0.8047            | 0.8706 |
| `fpr`                             | 0.0000      | 0.0042   | 0.0084         | 0.0209    | 0.0084      | 0.0042   | 0.0962  | 0.0000            | 0.0711 |
| `fnr`                             | 0.0000      | 0.0080   | 0.0000         | 0.0000    | 0.0000      | 0.0562   | 0.0241  | 0.2329            | 0.1325 |
| `f1`                              | 1.0000      | 0.9940   | 0.9960         | 0.9901    | 0.9960      | 0.9691   | 0.9437  | 0.8682            | 0.8963 |
| `f2`                              | 1.0000      | 0.9928   | 0.9984         | 0.9960    | 0.9984      | 0.9537   | 0.9628  | 0.8045            | 0.8788 |
| `roc_auc`                         | 1.0000      | 0.9999   | 0.9999         | 0.9999    | 1.0000      | 0.9973   | 0.9900  | 0.9993            | 0.9687 |
| `pr_auc_average_precision`        | 1.0000      | 0.9999   | 1.0000         | 0.9999    | 1.0000      | 0.9979   | 0.9915  | 0.9993            | 0.9705 |
| `brier_score_positive_class`      | 0.0010      | 0.0049   | 0.0047         | 0.0077    | 0.0024      | 0.0240   | 0.0408  | 0.0944            | 0.0755 |
| `youden_j`                        | 1.0000      | 0.9920   | 0.9920         | 0.9918    | 1.0000      | 0.9798   | 0.9180  | 0.9754            | 0.8039 |
| `youden_optimal_threshold`        | 0.5251      | 0.8426   | 0.9607         | 0.9921    | 0.6439      | 0.0952   | 0.6529  | 0.0023            | 0.4215 |
| `sensitivity_at_youden_threshold` | 1.0000      | 0.9920   | 0.9920         | 0.9960    | 1.0000      | 0.9839   | 0.9598  | 0.9880            | 0.8876 |
| `specificity_at_youden_threshold` | 1.0000      | 1.0000   | 1.0000         | 0.9958    | 1.0000      | 0.9958   | 0.9582  | 0.9874            | 0.9163 |

Source: `metrics/metrics_val.json`. The validation split was also used to pick the checkpoint, so validation values are optimistic.

### Confusion matrices

Rows are the true class and columns the predicted class, both in the order `No Tumor, Tumor`. Rows are separated by `/`.

| Model             | Test              | Validation        |
| ----------------- | ----------------- | ----------------- |
| DenseNet121       | 240, 0 / 3, 245   | 239, 0 / 0, 249   |
| ResNet50          | 240, 0 / 3, 245   | 238, 1 / 2, 247   |
| EfficientNetB0    | 237, 3 / 0, 248   | 237, 2 / 0, 249   |
| CoAtNet-0         | 237, 3 / 0, 248   | 234, 5 / 0, 249   |
| InceptionV3       | 239, 1 / 5, 243   | 237, 2 / 0, 249   |
| SwinBase          | 236, 4 / 16, 232  | 238, 1 / 14, 235  |
| ViT-B16           | 224, 16 / 9, 239  | 216, 23 / 6, 243  |
| MobileNetV3-Large | 240, 0 / 45, 203  | 239, 0 / 58, 191  |
| VGG16             | 217, 23 / 34, 214 | 222, 17 / 33, 216 |

Source: `metrics/confusion_matrix_{test,val}.csv`.

### Per-class metrics

**Test split** (source: `metrics/per_class_metrics_test.csv`)

| Model             | Class    | Support | Precision | Recall | Specificity | F1     | ROC AUC | Average precision |
| ----------------- | -------- | ------- | --------- | ------ | ----------- | ------ | ------- | ----------------- |
| DenseNet121       | No Tumor | 240     | 0.9877    | 1.0000 | 0.9879      | 0.9938 | 1.0000  | 1.0000            |
| DenseNet121       | Tumor    | 248     | 1.0000    | 0.9879 | 1.0000      | 0.9939 | 1.0000  | 1.0000            |
| ResNet50          | No Tumor | 240     | 0.9877    | 1.0000 | 0.9879      | 0.9938 | 1.0000  | 1.0000            |
| ResNet50          | Tumor    | 248     | 1.0000    | 0.9879 | 1.0000      | 0.9939 | 1.0000  | 1.0000            |
| EfficientNetB0    | No Tumor | 240     | 1.0000    | 0.9875 | 1.0000      | 0.9937 | 0.9998  | 0.9998            |
| EfficientNetB0    | Tumor    | 248     | 0.9880    | 1.0000 | 0.9875      | 0.9940 | 0.9998  | 0.9998            |
| CoAtNet-0         | No Tumor | 240     | 1.0000    | 0.9875 | 1.0000      | 0.9937 | 0.9999  | 0.9999            |
| CoAtNet-0         | Tumor    | 248     | 0.9880    | 1.0000 | 0.9875      | 0.9940 | 0.9999  | 0.9999            |
| InceptionV3       | No Tumor | 240     | 0.9795    | 0.9958 | 0.9798      | 0.9876 | 0.9986  | 0.9985            |
| InceptionV3       | Tumor    | 248     | 0.9959    | 0.9798 | 0.9958      | 0.9878 | 0.9986  | 0.9988            |
| SwinBase          | No Tumor | 240     | 0.9365    | 0.9833 | 0.9355      | 0.9593 | 0.9961  | 0.9959            |
| SwinBase          | Tumor    | 248     | 0.9831    | 0.9355 | 0.9833      | 0.9587 | 0.9961  | 0.9966            |
| ViT-B16           | No Tumor | 240     | 0.9614    | 0.9333 | 0.9637      | 0.9471 | 0.9862  | 0.9822            |
| ViT-B16           | Tumor    | 248     | 0.9373    | 0.9637 | 0.9333      | 0.9503 | 0.9862  | 0.9889            |
| MobileNetV3-Large | No Tumor | 240     | 0.8421    | 1.0000 | 0.8185      | 0.9143 | 0.9981  | 0.9982            |
| MobileNetV3-Large | Tumor    | 248     | 1.0000    | 0.8185 | 1.0000      | 0.9002 | 0.9981  | 0.9981            |
| VGG16             | No Tumor | 240     | 0.8645    | 0.9042 | 0.8629      | 0.8839 | 0.9676  | 0.9687            |
| VGG16             | Tumor    | 248     | 0.9030    | 0.8629 | 0.9042      | 0.8825 | 0.9676  | 0.9699            |

**Validation split** (source: `metrics/per_class_metrics_val.csv`)

| Model             | Class    | Support | Precision | Recall | Specificity | F1     | ROC AUC | Average precision |
| ----------------- | -------- | ------- | --------- | ------ | ----------- | ------ | ------- | ----------------- |
| DenseNet121       | No Tumor | 239     | 1.0000    | 1.0000 | 1.0000      | 1.0000 | 1.0000  | 1.0000            |
| DenseNet121       | Tumor    | 249     | 1.0000    | 1.0000 | 1.0000      | 1.0000 | 1.0000  | 1.0000            |
| ResNet50          | No Tumor | 239     | 0.9917    | 0.9958 | 0.9920      | 0.9937 | 0.9999  | 0.9999            |
| ResNet50          | Tumor    | 249     | 0.9960    | 0.9920 | 0.9958      | 0.9940 | 0.9999  | 0.9999            |
| EfficientNetB0    | No Tumor | 239     | 1.0000    | 0.9916 | 1.0000      | 0.9958 | 0.9999  | 0.9999            |
| EfficientNetB0    | Tumor    | 249     | 0.9920    | 1.0000 | 0.9916      | 0.9960 | 0.9999  | 1.0000            |
| CoAtNet-0         | No Tumor | 239     | 1.0000    | 0.9791 | 1.0000      | 0.9894 | 0.9999  | 0.9999            |
| CoAtNet-0         | Tumor    | 249     | 0.9803    | 1.0000 | 0.9791      | 0.9901 | 0.9999  | 0.9999            |
| InceptionV3       | No Tumor | 239     | 1.0000    | 0.9916 | 1.0000      | 0.9958 | 1.0000  | 1.0000            |
| InceptionV3       | Tumor    | 249     | 0.9920    | 1.0000 | 0.9916      | 0.9960 | 1.0000  | 1.0000            |
| SwinBase          | No Tumor | 239     | 0.9444    | 0.9958 | 0.9438      | 0.9695 | 0.9973  | 0.9967            |
| SwinBase          | Tumor    | 249     | 0.9958    | 0.9438 | 0.9958      | 0.9691 | 0.9973  | 0.9979            |
| ViT-B16           | No Tumor | 239     | 0.9730    | 0.9038 | 0.9759      | 0.9371 | 0.9900  | 0.9882            |
| ViT-B16           | Tumor    | 249     | 0.9135    | 0.9759 | 0.9038      | 0.9437 | 0.9900  | 0.9915            |
| MobileNetV3-Large | No Tumor | 239     | 0.8047    | 1.0000 | 0.7671      | 0.8918 | 0.9993  | 0.9993            |
| MobileNetV3-Large | Tumor    | 249     | 1.0000    | 0.7671 | 1.0000      | 0.8682 | 0.9993  | 0.9993            |
| VGG16             | No Tumor | 239     | 0.8706    | 0.9289 | 0.8675      | 0.8988 | 0.9687  | 0.9702            |
| VGG16             | Tumor    | 249     | 0.9270    | 0.8675 | 0.9289      | 0.8963 | 0.9687  | 0.9705            |

### Per-epoch training history

Source: `metrics/training_history.csv`. Training accuracy and AUC are measured on augmented batches in train mode, so they are not directly comparable with the validation columns. **Bold** marks the checkpointed epoch (highest validation AUC, first occurrence wins).

**DenseNet121**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.1666     | 0.9344    | 0.9852    | 0.0900   | 0.9631  | 0.99971 | 60.8           |
| 2     | 2.93e-04 | 0.0324     | 0.9902    | 0.9993    | 0.0162   | 0.9939  | 0.99988 | 49.3           |
| 3     | 2.71e-04 | 0.0265     | 0.9915    | 0.9997    | 0.0239   | 0.9898  | 0.99993 | 49.9           |
| **4** | 2.38e-04 | 0.0099     | 0.9969    | 1.0000    | 0.0042   | 1.0000  | 1.00000 | 49.3           |
| 5     | 1.96e-04 | 0.0106     | 0.9982    | 0.9995    | 0.0070   | 0.9980  | 0.99997 | 49.6           |
| 6     | 1.50e-04 | 0.0070     | 0.9978    | 1.0000    | 0.0043   | 0.9980  | 1.00000 | 49.5           |
| 7     | 1.04e-04 | 0.0041     | 0.9991    | 1.0000    | 0.0049   | 0.9980  | 1.00000 | 49.7           |
| 8     | 6.18e-05 | 0.0018     | 1.0000    | 1.0000    | 0.0044   | 0.9980  | 1.00000 | 49.5           |
| 9     | 2.86e-05 | 0.0021     | 0.9996    | 1.0000    | 0.0054   | 0.9959  | 1.00000 | 49.8           |
| 10    | 7.34e-06 | 0.0028     | 0.9996    | 1.0000    | 0.0049   | 0.9980  | 0.99998 | 49.5           |

**ResNet50**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.4627     | 0.8276    | 0.9194    | 0.4154   | 0.8340  | 0.96431 | 47.6           |
| 2     | 2.93e-04 | 0.0762     | 0.9756    | 0.9972    | 0.0961   | 0.9836  | 0.99484 | 29.4           |
| 3     | 2.71e-04 | 0.0209     | 0.9946    | 0.9998    | 0.0665   | 0.9898  | 0.99715 | 29.7           |
| 4     | 2.38e-04 | 0.0192     | 0.9932    | 0.9998    | 0.0766   | 0.9877  | 0.99634 | 29.8           |
| 5     | 1.96e-04 | 0.0138     | 0.9956    | 0.9999    | 0.0326   | 0.9898  | 0.99982 | 30.1           |
| 6     | 1.50e-04 | 0.0079     | 0.9980    | 1.0000    | 0.0180   | 0.9939  | 0.99983 | 31.0           |
| **7** | 1.04e-04 | 0.0042     | 0.9990    | 1.0000    | 0.0184   | 0.9939  | 0.99992 | 31.6           |
| 8     | 6.18e-05 | 0.0046     | 0.9976    | 1.0000    | 0.0179   | 0.9959  | 0.99990 | 30.8           |
| 9     | 2.86e-05 | 0.0021     | 0.9995    | 1.0000    | 0.0161   | 0.9959  | 0.99990 | 31.6           |
| 10    | 7.34e-06 | 0.0025     | 0.9995    | 1.0000    | 0.0156   | 0.9959  | 0.99990 | 31.0           |

**EfficientNetB0**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.5223     | 0.7847    | 0.8772    | 0.4066   | 0.8033  | 0.97229 | 127.5          |
| 2     | 2.93e-04 | 0.2061     | 0.9434    | 0.9859    | 0.0535   | 0.9836  | 0.99872 | 93.6           |
| 3     | 2.71e-04 | 0.0692     | 0.9795    | 0.9984    | 0.0186   | 0.9918  | 0.99983 | 94.7           |
| **4** | 2.38e-04 | 0.0307     | 0.9937    | 0.9996    | 0.0178   | 0.9959  | 0.99995 | 94.4           |
| 5     | 1.96e-04 | 0.0162     | 0.9971    | 0.9999    | 0.0153   | 0.9959  | 0.99994 | 94.9           |
| 6     | 1.50e-04 | 0.0103     | 0.9980    | 1.0000    | 0.0139   | 0.9959  | 0.99992 | 95.1           |
| 7     | 1.04e-04 | 0.0076     | 0.9985    | 1.0000    | 0.0116   | 0.9939  | 0.99993 | 94.3           |
| 8     | 6.18e-05 | 0.0058     | 0.9995    | 1.0000    | 0.0128   | 0.9939  | 0.99993 | 94.2           |
| 9     | 2.86e-05 | 0.0071     | 0.9990    | 1.0000    | 0.0135   | 0.9939  | 0.99993 | 94.2           |
| 10    | 7.34e-06 | 0.0065     | 0.9990    | 1.0000    | 0.0142   | 0.9939  | 0.99993 | 93.9           |

**CoAtNet-0**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.6636     | 0.6558    | 0.7110    | 0.5917   | 0.7111  | 0.84336 | 201.5          |
| 2     | 2.93e-04 | 0.3094     | 0.8763    | 0.9404    | 0.2289   | 0.9570  | 0.98805 | 176.7          |
| 3     | 2.71e-04 | 0.1125     | 0.9567    | 0.9922    | 0.0497   | 0.9836  | 0.99723 | 177.1          |
| 4     | 2.38e-04 | 0.0436     | 0.9839    | 0.9982    | 0.1304   | 0.9672  | 0.99599 | 177.9          |
| 5     | 1.96e-04 | 0.0586     | 0.9839    | 0.9962    | 0.0211   | 0.9898  | 0.99983 | 177.4          |
| **6** | 1.50e-04 | 0.0114     | 0.9969    | 0.9999    | 0.0322   | 0.9898  | 0.99993 | 172.9          |
| 7     | 1.04e-04 | 0.0134     | 0.9955    | 0.9999    | 0.0481   | 0.9918  | 0.99692 | 167.8          |
| 8     | 6.18e-05 | 0.0118     | 0.9960    | 0.9999    | 0.0383   | 0.9898  | 0.99913 | 165.6          |
| 9     | 2.86e-05 | 0.0027     | 0.9996    | 1.0000    | 0.0394   | 0.9898  | 0.99912 | 166.4          |
| 10    | 7.34e-06 | 0.0010     | 1.0000    | 1.0000    | 0.0404   | 0.9898  | 0.99927 | 163.5          |

**InceptionV3**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.2562     | 0.9182    | 0.9806    | 0.0511   | 0.9898  | 0.99647 | 64.0           |
| 2     | 2.93e-04 | 0.0355     | 0.9908    | 0.9995    | 0.0117   | 0.9959  | 0.99992 | 24.6           |
| **3** | 2.71e-04 | 0.0314     | 0.9922    | 0.9997    | 0.0106   | 0.9959  | 1.00000 | 24.7           |
| 4     | 2.38e-04 | 0.0137     | 0.9972    | 1.0000    | 0.0187   | 0.9939  | 0.99990 | 24.9           |
| 5     | 1.96e-04 | 0.0086     | 0.9986    | 1.0000    | 0.0159   | 0.9918  | 0.99992 | 25.0           |
| 6     | 1.50e-04 | 0.0018     | 1.0000    | 1.0000    | 0.0262   | 0.9939  | 0.99978 | 25.3           |
| 7     | 1.04e-04 | 0.0017     | 1.0000    | 1.0000    | 0.0229   | 0.9939  | 0.99988 | 25.2           |
| 8     | 6.18e-05 | 0.0011     | 1.0000    | 1.0000    | 0.0220   | 0.9939  | 0.99988 | 25.5           |
| 9     | 2.86e-05 | 0.0025     | 0.9991    | 1.0000    | 0.0209   | 0.9939  | 0.99990 | 25.7           |
| 10    | 7.34e-06 | 0.0014     | 0.9995    | 1.0000    | 0.0202   | 0.9939  | 0.99990 | 25.7           |

**SwinBase**

| Epoch | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ----- | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1     | 3.00e-04 | 0.7003     | 0.5813    | 0.6079    | 0.5639   | 0.7275  | 0.81000 | 40.6           |
| 2     | 2.93e-04 | 0.5554     | 0.7205    | 0.7884    | 0.4889   | 0.7664  | 0.84788 | 39.5           |
| 3     | 2.71e-04 | 0.5452     | 0.7379    | 0.7976    | 0.4862   | 0.7848  | 0.86529 | 38.7           |
| 4     | 2.38e-04 | 0.4249     | 0.8063    | 0.8855    | 0.3235   | 0.8750  | 0.95756 | 38.7           |
| 5     | 1.96e-04 | 0.2490     | 0.9000    | 0.9613    | 0.1590   | 0.9426  | 0.99235 | 38.3           |
| **6** | 1.50e-04 | 0.0953     | 0.9665    | 0.9938    | 0.0928   | 0.9693  | 0.99733 | 39.3           |
| 7     | 1.04e-04 | 0.0484     | 0.9857    | 0.9975    | 0.0584   | 0.9836  | 0.99605 | 39.6           |
| 8     | 6.18e-05 | 0.0177     | 0.9951    | 0.9998    | 0.0535   | 0.9877  | 0.99505 | 38.6           |
| 9     | 2.86e-05 | 0.0146     | 0.9969    | 0.9997    | 0.0516   | 0.9877  | 0.99597 | 38.1           |
| 10    | 7.34e-06 | 0.0077     | 0.9978    | 1.0000    | 0.0539   | 0.9877  | 0.99572 | 38.1           |

**ViT-B16**

| Epoch  | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ------ | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1      | 3.00e-04 | 0.8011     | 0.6011    | 0.6233    | 0.6407   | 0.6189  | 0.72058 | 30.1           |
| 2      | 2.93e-04 | 0.6209     | 0.6604    | 0.7126    | 0.6200   | 0.6270  | 0.79739 | 26.4           |
| 3      | 2.71e-04 | 0.5596     | 0.7091    | 0.7835    | 0.4561   | 0.7807  | 0.88137 | 26.6           |
| 4      | 2.38e-04 | 0.4623     | 0.7904    | 0.8635    | 0.4351   | 0.7869  | 0.93260 | 26.9           |
| 5      | 1.96e-04 | 0.4216     | 0.8042    | 0.8890    | 0.3015   | 0.8811  | 0.95198 | 27.2           |
| 6      | 1.50e-04 | 0.2905     | 0.8879    | 0.9482    | 0.2481   | 0.8934  | 0.96233 | 28.3           |
| 7      | 1.04e-04 | 0.2607     | 0.8938    | 0.9582    | 0.1872   | 0.9180  | 0.98172 | 27.7           |
| 8      | 6.18e-05 | 0.2329     | 0.9076    | 0.9666    | 0.2880   | 0.8791  | 0.97197 | 27.6           |
| 9      | 2.86e-05 | 0.2264     | 0.9113    | 0.9684    | 0.1630   | 0.9406  | 0.98679 | 27.7           |
| **10** | 7.34e-06 | 0.1649     | 0.9439    | 0.9837    | 0.1449   | 0.9406  | 0.98993 | 27.8           |

**MobileNetV3-Large**

| Epoch  | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ------ | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1      | 3.00e-04 | 0.3997     | 0.8164    | 0.9172    | 0.5823   | 0.6988  | 0.95181 | 103.8          |
| 2      | 2.93e-04 | 0.0992     | 0.9707    | 0.9947    | 1.0599   | 0.6557  | 0.97283 | 67.7           |
| 3      | 2.71e-04 | 0.0294     | 0.9897    | 0.9996    | 0.9699   | 0.7029  | 0.98723 | 68.2           |
| 4      | 2.38e-04 | 0.0080     | 0.9985    | 1.0000    | 0.6483   | 0.7971  | 0.99288 | 67.8           |
| 5      | 1.96e-04 | 0.0054     | 0.9985    | 1.0000    | 0.6628   | 0.8135  | 0.99347 | 68.2           |
| 6      | 1.50e-04 | 0.0037     | 0.9990    | 1.0000    | 0.6682   | 0.8135  | 0.99649 | 67.8           |
| 7      | 1.04e-04 | 0.0027     | 0.9990    | 1.0000    | 0.5784   | 0.8381  | 0.99819 | 68.3           |
| 8      | 6.18e-05 | 0.0017     | 0.9995    | 1.0000    | 0.5277   | 0.8504  | 0.99879 | 68.2           |
| 9      | 2.86e-05 | 0.0010     | 1.0000    | 1.0000    | 0.4601   | 0.8730  | 0.99909 | 68.4           |
| **10** | 7.34e-06 | 0.0023     | 0.9990    | 1.0000    | 0.3995   | 0.8811  | 0.99931 | 69.4           |

**VGG16**

| Epoch  | LR       | Train loss | Train acc | Train AUC | Val loss | Val acc | Val AUC | Epoch time (s) |
| ------ | -------- | ---------- | --------- | --------- | -------- | ------- | ------- | -------------- |
| 1      | 3.00e-04 | 2.1317     | 0.5278    | 0.5792    | 0.6825   | 0.5410  | 0.75687 | 90.0           |
| 2      | 2.93e-04 | 0.6821     | 0.5654    | 0.6156    | 0.6208   | 0.6762  | 0.86334 | 71.0           |
| 3      | 2.71e-04 | 0.6727     | 0.5845    | 0.6041    | 0.6839   | 0.5307  | 0.73189 | 72.0           |
| 4      | 2.38e-04 | 0.6578     | 0.6338    | 0.6875    | 0.5813   | 0.6660  | 0.81937 | 72.4           |
| 5      | 1.96e-04 | 0.5878     | 0.7041    | 0.7717    | 0.5090   | 0.7439  | 0.85716 | 72.6           |
| 6      | 1.50e-04 | 0.4776     | 0.7725    | 0.8557    | 0.3911   | 0.8361  | 0.90387 | 72.1           |
| 7      | 1.04e-04 | 0.3603     | 0.8413    | 0.9238    | 0.3229   | 0.8545  | 0.94824 | 72.1           |
| 8      | 6.18e-05 | 0.3251     | 0.8530    | 0.9368    | 0.2648   | 0.8873  | 0.95797 | 72.5           |
| 9      | 2.86e-05 | 0.2768     | 0.8838    | 0.9540    | 0.2531   | 0.8934  | 0.96799 | 73.0           |
| **10** | 7.34e-06 | 0.2590     | 0.8872    | 0.9603    | 0.2421   | 0.8975  | 0.96872 | 72.9           |

## What the results show

- **The top four fail in opposite directions.** DenseNet121, ResNet50, EfficientNetB0 and CoAtNet-0 each make 3 errors out of 488. DenseNet121 and ResNet50 miss 3 tumours and raise no false alarms; EfficientNetB0 and CoAtNet-0 catch every tumour and raise 3 false alarms. With a single seed and 488 test images, one error is 0.2 percentage points, so these four are not statistically separable.
- **Training time does not track model cost.** EfficientNetB0 (0.769 GFLOPs) took 0:16:17, while ResNet50 (8.174 GFLOPs) took 0:05:24. Each run used a separate Kaggle session (TensorBoard host ids differ), so I/O caching was not controlled. Compare training times with caution.
- **MobileNetV3-Large ranks well but its 0.5 threshold is badly placed.** Test ROC AUC is 0.99808, yet accuracy is 0.9078.
- **Three models were still improving at epoch 10.** VGG16, ViT-B16 and MobileNetV3-Large have their best validation AUC at the last epoch, so their scores reflect this 10-epoch budget rather than each architecture's ceiling.
- **Youden thresholds are descriptive only.** They are computed on the test set itself and must not be used as tuned operating points.

## Plots for every run

Each run folder has 13 plots. Plots `01`, `02`, `03`, `05` and `06` describe the shared data and are identical across runs; plots `07` to `14` are specific to each model. Grad-CAM heatmaps are for qualitative inspection only: no dataset here has lesion masks, and the notebooks do not measure how well the heatmaps localise tumours.

### CoAtNet-0 plots

Run folder: [`binary-detection-coatnet0/`](binary-detection-coatnet0/). Notebook: [`binary-detection-coatnet0.ipynb`](binary-detection-coatnet0.ipynb). Best epoch 6 of 10, test accuracy 0.9939.

![CoAtNet-0 01_class_distribution](binary-detection-coatnet0/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![CoAtNet-0 02_sample_images](binary-detection-coatnet0/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![CoAtNet-0 03_image_properties](binary-detection-coatnet0/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![CoAtNet-0 05_split_distribution](binary-detection-coatnet0/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![CoAtNet-0 06_augmentation_before_after](binary-detection-coatnet0/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![CoAtNet-0 07_training_curves](binary-detection-coatnet0/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![CoAtNet-0 08_confusion_matrices](binary-detection-coatnet0/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![CoAtNet-0 09_roc_pr_curves](binary-detection-coatnet0/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![CoAtNet-0 10_calibration](binary-detection-coatnet0/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![CoAtNet-0 11_per_class_metrics](binary-detection-coatnet0/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![CoAtNet-0 12_prediction_samples](binary-detection-coatnet0/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![CoAtNet-0 13_gradcam_correct](binary-detection-coatnet0/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![CoAtNet-0 14_gradcam_misclassified](binary-detection-coatnet0/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### DenseNet121 plots

Run folder: [`binary-detection-densenet121/`](binary-detection-densenet121/). Notebook: [`binary-detection-densenet121.ipynb`](binary-detection-densenet121.ipynb). Best epoch 4 of 10, test accuracy 0.9939.

![DenseNet121 01_class_distribution](binary-detection-densenet121/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![DenseNet121 02_sample_images](binary-detection-densenet121/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![DenseNet121 03_image_properties](binary-detection-densenet121/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![DenseNet121 05_split_distribution](binary-detection-densenet121/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![DenseNet121 06_augmentation_before_after](binary-detection-densenet121/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![DenseNet121 07_training_curves](binary-detection-densenet121/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![DenseNet121 08_confusion_matrices](binary-detection-densenet121/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![DenseNet121 09_roc_pr_curves](binary-detection-densenet121/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![DenseNet121 10_calibration](binary-detection-densenet121/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![DenseNet121 11_per_class_metrics](binary-detection-densenet121/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![DenseNet121 12_prediction_samples](binary-detection-densenet121/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![DenseNet121 13_gradcam_correct](binary-detection-densenet121/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![DenseNet121 14_gradcam_misclassified](binary-detection-densenet121/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### EfficientNetB0 plots

Run folder: [`binary-detection-efficientnetb0/`](binary-detection-efficientnetb0/). Notebook: [`binary-detection-efficientnetb0.ipynb`](binary-detection-efficientnetb0.ipynb). Best epoch 4 of 10, test accuracy 0.9939.

![EfficientNetB0 01_class_distribution](binary-detection-efficientnetb0/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![EfficientNetB0 02_sample_images](binary-detection-efficientnetb0/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![EfficientNetB0 03_image_properties](binary-detection-efficientnetb0/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![EfficientNetB0 05_split_distribution](binary-detection-efficientnetb0/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![EfficientNetB0 06_augmentation_before_after](binary-detection-efficientnetb0/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![EfficientNetB0 07_training_curves](binary-detection-efficientnetb0/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![EfficientNetB0 08_confusion_matrices](binary-detection-efficientnetb0/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![EfficientNetB0 09_roc_pr_curves](binary-detection-efficientnetb0/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![EfficientNetB0 10_calibration](binary-detection-efficientnetb0/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![EfficientNetB0 11_per_class_metrics](binary-detection-efficientnetb0/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![EfficientNetB0 12_prediction_samples](binary-detection-efficientnetb0/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![EfficientNetB0 13_gradcam_correct](binary-detection-efficientnetb0/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![EfficientNetB0 14_gradcam_misclassified](binary-detection-efficientnetb0/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### InceptionV3 plots

Run folder: [`binary-detection-inceptionv3/`](binary-detection-inceptionv3/). Notebook: [`binary-detection-inceptionv3.ipynb`](binary-detection-inceptionv3.ipynb). Best epoch 3 of 10, test accuracy 0.9877.

![InceptionV3 01_class_distribution](binary-detection-inceptionv3/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![InceptionV3 02_sample_images](binary-detection-inceptionv3/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![InceptionV3 03_image_properties](binary-detection-inceptionv3/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![InceptionV3 05_split_distribution](binary-detection-inceptionv3/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![InceptionV3 06_augmentation_before_after](binary-detection-inceptionv3/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![InceptionV3 07_training_curves](binary-detection-inceptionv3/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![InceptionV3 08_confusion_matrices](binary-detection-inceptionv3/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![InceptionV3 09_roc_pr_curves](binary-detection-inceptionv3/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![InceptionV3 10_calibration](binary-detection-inceptionv3/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![InceptionV3 11_per_class_metrics](binary-detection-inceptionv3/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![InceptionV3 12_prediction_samples](binary-detection-inceptionv3/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![InceptionV3 13_gradcam_correct](binary-detection-inceptionv3/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![InceptionV3 14_gradcam_misclassified](binary-detection-inceptionv3/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### MobileNetV3-Large plots

Run folder: [`binary-detection-mobilenetv3-large/`](binary-detection-mobilenetv3-large/). Notebook: [`binary-detection-mobilenetv3-large.ipynb`](binary-detection-mobilenetv3-large.ipynb). Best epoch 10 of 10, test accuracy 0.9078.

![MobileNetV3-Large 01_class_distribution](binary-detection-mobilenetv3-large/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![MobileNetV3-Large 02_sample_images](binary-detection-mobilenetv3-large/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![MobileNetV3-Large 03_image_properties](binary-detection-mobilenetv3-large/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![MobileNetV3-Large 05_split_distribution](binary-detection-mobilenetv3-large/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![MobileNetV3-Large 06_augmentation_before_after](binary-detection-mobilenetv3-large/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![MobileNetV3-Large 07_training_curves](binary-detection-mobilenetv3-large/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![MobileNetV3-Large 08_confusion_matrices](binary-detection-mobilenetv3-large/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![MobileNetV3-Large 09_roc_pr_curves](binary-detection-mobilenetv3-large/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![MobileNetV3-Large 10_calibration](binary-detection-mobilenetv3-large/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![MobileNetV3-Large 11_per_class_metrics](binary-detection-mobilenetv3-large/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![MobileNetV3-Large 12_prediction_samples](binary-detection-mobilenetv3-large/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![MobileNetV3-Large 13_gradcam_correct](binary-detection-mobilenetv3-large/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![MobileNetV3-Large 14_gradcam_misclassified](binary-detection-mobilenetv3-large/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### ResNet50 plots

Run folder: [`binary-detection-resnet50/`](binary-detection-resnet50/). Notebook: [`binary-detection-resnet50.ipynb`](binary-detection-resnet50.ipynb). Best epoch 7 of 10, test accuracy 0.9939.

![ResNet50 01_class_distribution](binary-detection-resnet50/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![ResNet50 02_sample_images](binary-detection-resnet50/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![ResNet50 03_image_properties](binary-detection-resnet50/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![ResNet50 05_split_distribution](binary-detection-resnet50/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![ResNet50 06_augmentation_before_after](binary-detection-resnet50/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![ResNet50 07_training_curves](binary-detection-resnet50/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![ResNet50 08_confusion_matrices](binary-detection-resnet50/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![ResNet50 09_roc_pr_curves](binary-detection-resnet50/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![ResNet50 10_calibration](binary-detection-resnet50/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![ResNet50 11_per_class_metrics](binary-detection-resnet50/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![ResNet50 12_prediction_samples](binary-detection-resnet50/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![ResNet50 13_gradcam_correct](binary-detection-resnet50/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![ResNet50 14_gradcam_misclassified](binary-detection-resnet50/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### SwinBase plots

Run folder: [`binary-detection-swinbase/`](binary-detection-swinbase/). Notebook: [`binary-detection-swinbase.ipynb`](binary-detection-swinbase.ipynb). Best epoch 6 of 10, test accuracy 0.9590.

![SwinBase 01_class_distribution](binary-detection-swinbase/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![SwinBase 02_sample_images](binary-detection-swinbase/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![SwinBase 03_image_properties](binary-detection-swinbase/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![SwinBase 05_split_distribution](binary-detection-swinbase/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![SwinBase 06_augmentation_before_after](binary-detection-swinbase/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![SwinBase 07_training_curves](binary-detection-swinbase/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![SwinBase 08_confusion_matrices](binary-detection-swinbase/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![SwinBase 09_roc_pr_curves](binary-detection-swinbase/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![SwinBase 10_calibration](binary-detection-swinbase/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![SwinBase 11_per_class_metrics](binary-detection-swinbase/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![SwinBase 12_prediction_samples](binary-detection-swinbase/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![SwinBase 13_gradcam_correct](binary-detection-swinbase/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![SwinBase 14_gradcam_misclassified](binary-detection-swinbase/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### VGG16 plots

Run folder: [`binary-detection-vgg16/`](binary-detection-vgg16/). Notebook: [`binary-detection-vgg16.ipynb`](binary-detection-vgg16.ipynb). Best epoch 10 of 10, test accuracy 0.8832.

![VGG16 01_class_distribution](binary-detection-vgg16/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![VGG16 02_sample_images](binary-detection-vgg16/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![VGG16 03_image_properties](binary-detection-vgg16/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![VGG16 05_split_distribution](binary-detection-vgg16/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![VGG16 06_augmentation_before_after](binary-detection-vgg16/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![VGG16 07_training_curves](binary-detection-vgg16/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![VGG16 08_confusion_matrices](binary-detection-vgg16/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![VGG16 09_roc_pr_curves](binary-detection-vgg16/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![VGG16 10_calibration](binary-detection-vgg16/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![VGG16 11_per_class_metrics](binary-detection-vgg16/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![VGG16 12_prediction_samples](binary-detection-vgg16/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![VGG16 13_gradcam_correct](binary-detection-vgg16/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![VGG16 14_gradcam_misclassified](binary-detection-vgg16/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

### ViT-B16 plots

Run folder: [`binary-detection-vit-b16/`](binary-detection-vit-b16/). Notebook: [`binary-detection-vit-b16.ipynb`](binary-detection-vit-b16.ipynb). Best epoch 10 of 10, test accuracy 0.9488.

![ViT-B16 01_class_distribution](binary-detection-vit-b16/plots/01_class_distribution.png)

_01_class_distribution: Class counts of all collected images. In Classification this is before deduplication._

![ViT-B16 02_sample_images](binary-detection-vit-b16/plots/02_sample_images.png)

_02_sample_images: Random sample images per class, seeded with SEED=42._

![ViT-B16 03_image_properties](binary-detection-vit-b16/plots/03_image_properties.png)

_03_image_properties: Image width and height distributions and grayscale intensity histograms per class._

![ViT-B16 05_split_distribution](binary-detection-vit-b16/plots/05_split_distribution.png)

_05_split_distribution: Per-split class counts and class proportions after the stratified 70/15/15 split._

![ViT-B16 06_augmentation_before_after](binary-detection-vit-b16/plots/06_augmentation_before_after.png)

_06_augmentation_before_after: Evaluation transform (top row) against one random training augmentation (bottom row)._

![ViT-B16 07_training_curves](binary-detection-vit-b16/plots/07_training_curves.png)

_07_training_curves: Loss, accuracy and AUC per epoch (train and validation), generalisation gap, learning rate and time per epoch._

![ViT-B16 08_confusion_matrices](binary-detection-vit-b16/plots/08_confusion_matrices.png)

_08_confusion_matrices: Confusion matrices as counts and row-normalised, validation (top) and test (bottom)._

![ViT-B16 09_roc_pr_curves](binary-detection-vit-b16/plots/09_roc_pr_curves.png)

_09_roc_pr_curves: ROC and precision-recall curves, validation (top) and test (bottom)._

![ViT-B16 10_calibration](binary-detection-vit-b16/plots/10_calibration.png)

_10_calibration: Reliability diagram (15 confidence bins) and confidence histograms for correct and incorrect predictions._

![ViT-B16 11_per_class_metrics](binary-detection-vit-b16/plots/11_per_class_metrics.png)

_11_per_class_metrics: Per-class precision, recall, specificity, F1 and AUC on the test set._

![ViT-B16 12_prediction_samples](binary-detection-vit-b16/plots/12_prediction_samples.png)

_12_prediction_samples: Random correct (top) and misclassified (bottom) test images._

![ViT-B16 13_gradcam_correct](binary-detection-vit-b16/plots/13_gradcam_correct.png)

_13_gradcam_correct: Grad-CAM for the most confident correct test predictions: input, heatmap and overlay._

![ViT-B16 14_gradcam_misclassified](binary-detection-vit-b16/plots/14_gradcam_misclassified.png)

_14_gradcam_misclassified: Grad-CAM for the most confident misclassified test predictions._

## Output files reference

Each run folder has the same 21 metric files.

| File                                                  | Contents                                                                                                                                                                |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark_summary.json` / `.csv`                     | One flat record: configuration, model profile, training summary, inference benchmark, all validation and test metrics, Grad-CAM status and environment                  |
| `metrics_val.json`, `metrics_test.json`               | Every scalar metric for that split (see [All test metrics](#all-test-metrics))                                                                                          |
| `metrics_summary.csv`                                 | The same metrics with one column per split                                                                                                                              |
| `per_class_metrics_{val,test}.csv`                    | One row per class: support, TP, FP, FN, TN, precision, recall, specificity, NPV, FPR, FNR, F1, ROC AUC, average precision                                               |
| `classification_report_{val,test}.txt`                | scikit-learn classification report with 4 digits                                                                                                                        |
| `confusion_matrix_{val,test}.csv`                     | 2x2 counts, rows true class, columns predicted class                                                                                                                    |
| `calibration_bins_{val,test}.csv`                     | 15 rows: `bin_low`, `bin_high`, `count`, `mean_confidence`, `accuracy`; empty bins have blank values                                                                    |
| `predictions_{val,test}.csv`                          | 488 rows each. Columns: `path`, `true_label`, `true_class`, `pred_label`, `pred_class`, `confidence`, `correct`, `prob_no_tumor`, `prob_tumor`. Paths are Kaggle paths. |
| `training_history.csv`                                | 10 rows: `epoch`, `lr`, `train_loss`, `train_acc`, `train_auc`, `train_time_s`, `val_loss`, `val_acc`, `val_auc`, `val_time_s`, `epoch_time_s`, `cumulative_time_s`     |
| `training_summary.json`                               | Best epoch, times, throughput and peak GPU memory                                                                                                                       |
| `model_profile.json`                                  | Parameter counts, buffer elements, state-dict size, input size, FLOPs, DataParallel GPU count                                                                           |
| `inference_benchmark.json`                            | Latency and throughput on one Tesla T4                                                                                                                                  |
| `split_distribution.csv`, `image_dimension_stats.csv` | Dataset tables shown in [Dataset](#dataset)                                                                                                                             |

Weight files in the local working copy (not tracked in Git). Sizes are in bytes.

| Run                                  | `best_*.pth`                              | `final_*.pth`                                          |
| ------------------------------------ | ----------------------------------------- | ------------------------------------------------------ |
| `binary-detection-coatnet0`          | `best_coatnet.pth` (106,811,955)          | `final_coatnet_brain_tumor.pth` (106,815,691)          |
| `binary-detection-densenet121`       | `best_densenet121.pth` (28,431,126)       | `final_densenet121_brain_tumor.pth` (28,440,719)       |
| `binary-detection-efficientnetb0`    | `best_efficientnet.pth` (16,337,403)      | `final_efficientnet_brain_tumor.pth` (16,342,545)      |
| `binary-detection-inceptionv3`       | `best_inception.pth` (97,734,525)         | `final_inception_brain_tumor.pth` (97,742,207)         |
| `binary-detection-mobilenetv3-large` | `best_mobilenetv3_large.pth` (17,023,537) | `final_mobilenetv3_large_brain_tumor.pth` (17,027,735) |
| `binary-detection-resnet50`          | `best_resnet50.pth` (94,360,291)          | `final_resnet50_brain_tumor.pth` (94,364,593)          |
| `binary-detection-swinbase`          | `best_swin.pth` (347,107,083)             | `final_swin_brain_tumor.pth` (347,111,502)             |
| `binary-detection-vgg16`             | `best_vgg16.pth` (537,070,645)            | `final_vgg16_brain_tumor.pth` (537,071,267)            |
| `binary-detection-vit-b16`           | `best_vit.pth` (343,258,605)              | `final_vit_brain_tumor.pth` (343,260,659)              |

## Anomalies and limitations

### A. Issues that affect results

- **A1. InceptionV3 did not use DataParallel for compute.** The model is wrapped in `nn.DataParallel`, but `run_epoch` calls the unwrapped module in training and evaluation. `training_summary.json` records `peak_gpu_memory_allocated_mb_cuda1 = 0.0`. `model_profile.json` still reports `data_parallel_gpus: 2` because it only checks the wrapper. Its training time is therefore single-GPU at batch 128.
- **A2. MobileNetV3-Large threshold collapse.** ROC AUC is 0.998 but sensitivity at 0.5 is 0.8185, the test Youden threshold is 0.00044, and validation loss rose while training loss approached zero. The notebook does not test for a cause. It is also the only run that pins `IMAGENET1K_V1` weights; nothing shows that this is the cause.
- **A3. Unequal batch sizes.** With the same learning rate and epoch count, the models got between 80 and 350 optimiser steps. The comparison is controlled for data and schedule, not for optimisation budget.
- **A4. No deduplication or leakage check** across the two merged datasets.
- **A5. Single seed.** No repeated runs, so no confidence intervals.

### B. Issues that affect interpretation

- **B1.** The checkpoint is chosen on validation AUC, so validation metrics are optimistic. The test set is the unbiased estimate.
- **B2.** Validation AUC saturates early for several models (DenseNet121 reaches 1.0 at epoch 4), so later epochs cannot replace the checkpoint even when their validation loss is lower.
- **B3.** The printed test loss and the file `log_loss` differ slightly (for example 0.0240 vs 0.0233 for EfficientNetB0). The first is a mean of per-batch fp16 losses from the DataParallel model; the second is scikit-learn's log loss on float64 probabilities from the single-GPU re-inference. Test accuracy agrees between the two paths in all nine runs.
- **B4. The folder-level `EfficientNetB0.pth` has unknown provenance.** It is the file the web app loads. It has the same size (16,342,545 bytes) as `binary-detection-efficientnetb0/weights/final_efficientnet_brain_tumor.pth`, but a different MD5 (`f7dadc871213b16293337109ecdfe537` vs `4709fe14863376fef2fa64b10f2135b4`). No notebook here writes a file with this name, so its test metrics are unknown.

### C. Reproducibility (no effect on results)

- `cudnn.benchmark=True`, TF32 and AMP allow non-deterministic kernels, so reruns are only approximately reproducible.
- Kaggle paths (`/kaggle/input/...`, `/kaggle/working/...`) are hard-coded.
- Minor code drift between notebooks, listed in [Per-notebook differences](#per-notebook-differences).

## Reproducing a run

The notebooks are designed for Kaggle and have not been run elsewhere.

1. Create a Kaggle notebook with the **GPU T4 x2** accelerator and internet access enabled.
2. Attach the Kaggle datasets `navoneel/brain-mri-images-for-brain-tumor-detection` and `ahmedhamada0/brain-tumor-detection`. If your mount paths differ from `/kaggle/input/datasets/...`, edit the data collection cell.
3. The runs used Python 3.12.13 and torch 2.10.0+cu128 with torchvision, scikit-learn, pandas, matplotlib, Pillow, tqdm and TensorBoard. CoAtNet-0 and SwinBase also need `timm`.
4. Run all cells in order. The tracker (cell 30) must run before the training loop (cell 32), otherwise cell 34 raises `No epoch history was recorded`.
5. Download `/kaggle/working/<RUN_NAME>-outputs.zip` and unzip it here. It recreates the `<RUN_NAME>/` layout above.

## Loading a trained model

EfficientNetB0 example. The other torchvision models follow the same pattern with their own head (see [Per-notebook differences](#per-notebook-differences)).

```python
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

model = efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
state = torch.load("binary-detection-efficientnetb0/weights/best_efficientnet.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
# Input: RGB image, Resize((224, 224)), ToTensor, ImageNet Normalize.
# P(tumour) = torch.sigmoid(model(x))
```

- Checkpoints have no `module.` prefix because they were saved from `model.module`.
- For InceptionV3, build the model with `aux_logits=True`, replace both `fc` and `AuxLogits.fc` with 1-output layers, and use 299 px input.

## Metric glossary

| Term                 | Meaning                                                                               |
| -------------------- | ------------------------------------------------------------------------------------- |
| Sensitivity (recall) | Share of tumour images detected: TP / (TP + FN)                                       |
| Specificity          | Share of tumour-free images correctly cleared: TN / (TN + FP)                         |
| FN / FP              | Missed tumours / false alarms                                                         |
| NPV                  | Share of "No Tumor" predictions that are correct                                      |
| Balanced accuracy    | Mean of per-class recall                                                              |
| MCC                  | Matthews correlation coefficient, from -1 to 1                                        |
| Cohen's kappa        | Agreement with the true labels, corrected for chance                                  |
| ROC AUC              | Probability that a random tumour image scores higher than a random tumour-free image  |
| PR AUC (AP)          | Area under the precision-recall curve (average precision)                             |
| Log loss             | Penalises confident wrong probabilities. Lower is better.                             |
| Brier score          | Mean squared error of predicted probabilities                                         |
| ECE                  | Expected calibration error: weighted gap between confidence and accuracy over 15 bins |
| Youden J             | Sensitivity + specificity - 1. The optimal threshold maximises it.                    |
