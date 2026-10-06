# Segmentation: 3D Brain Tumour Sub-Region Segmentation on BraTS 2020

This folder benchmarks six 3D segmentation networks on the BraTS 2020 training data. Each model takes the four MRI modalities of one patient (FLAIR, T1, T1CE, T2) as a 4-channel volume and predicts three overlapping tumour masks: Whole Tumour (WT), Tumour Core (TC) and Enhancing Tumour (ET). All six are now trained and evaluated by one Kaggle notebook, [`segmentation.ipynb`](segmentation.ipynb), written from scratch in plain PyTorch (no MONAI or timm).

The rest of this README documents the **legacy per-model runs** behind the published results. Their notebooks were replaced by the unified notebook; they remain in Git history (for example `git show 9a8ac24:"Backend/Segmentation/UNet.ipynb"`), and every number below is copied from their saved cell outputs; the notebooks were not re-executed for this README.

## Unified notebook

[`segmentation.ipynb`](segmentation.ipynb) replaces the seven per-model notebooks. It trains and evaluates UNet, nnU-Net, VNet, UNETR, Swin UNETR and TransBTS with one shared data pipeline, training loop and evaluation. It has **no saved outputs yet**: it has not been run on Kaggle in this form, so no result in this README comes from it.

What it does differently from the legacy runs:

| Area            | Legacy per-model notebooks                      | Unified notebook                                                                                                                                                                          |
| --------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cases           | 368 (the misnamed mask of case 355 was skipped) | Every case with a mask, found by name pattern                                                                                                                                             |
| Split           | Random 70/15/15, not stratified                 | 70/15/15 stratified by tumour grade (HGG/LGG); results also reported per grade                                                                                                            |
| Training length | 10 epochs, no warm-up                           | `EPOCHS` (50 by default) for every model; warm-up then cosine decay for the transformer and hybrid models; per-epoch checkpoints so runs resume across Kaggle sessions                    |
| Batch           | 4 (2 for Swin-UNETR)                            | Effective batch 4 for every model through gradient accumulation                                                                                                                           |
| Evaluation      | Dice averaged over batches on the 128^3 grid    | Per-case Dice, HD95 (mm), sensitivity and specificity at the native resolution, BraTS conventions for empty regions, 8-flip test-time augmentation, ET size threshold tuned on validation |
| Swin-UNETR      | No test result (checkpoint loading failed)      | Evaluated like every other model; adds the reference design's full-resolution skip                                                                                                        |
| Statistics      | None computed                                   | Bootstrap 95% CIs, Wilcoxon signed-rank tests with Holm correction, seed spread when `SEEDS` has several entries                                                                          |
| Extras          | One VNet re-run with three figures              | Seed, all-model and CNN-only ensembles; TTA uncertainty against per-case Dice; latency per volume and peak memory per model                                                               |

How to run it:

1. Create a Kaggle notebook from the file with the **GPU T4 x2** accelerator; Internet is needed only if `nibabel` is missing.
2. Attach `awsaf49/brats20-dataset-training-validation`.
3. Run all cells. Two worker processes train one job each per GPU with live progress. The full six-model run at 50 epochs does not fit one 12-hour session: training stops at `TRAINING_BUDGET_HOURS` (9.5 by default) at an epoch boundary and the notebook analyses whatever finished. To continue, attach the previous version's output and set `RESUME_DIR`; or reduce `EPOCHS` or `MODELS` to fit one session.
4. Download `outputs.zip` from the Output tab.
5. To update the web app, copy `weights/deploy/VNet.pth` into this folder. Its module names match `VNet3D` in [`networks.py`](../neuroscan/inference/networks.py), and the notebook checks the reload against the stored test probabilities.

Runtime, metrics and hardware use of the unified notebook: Not measured in the current repository.

## At a glance (legacy runs)

| Item                      | Value                                                                                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Task                      | Multi-label 3D segmentation: WT, TC and ET masks per voxel                                                                                                                     |
| Data                      | Kaggle dataset `awsaf49/brats20-dataset-training-validation`, 368 cases that have a `_seg.nii` file                                                                            |
| Input                     | 4 x 128 x 128 x 128 (FLAIR, T1, T1CE, T2), resampled from the original volumes                                                                                                 |
| Split                     | Random 70/15/15 with seed 42: train 257, validation 55, test 56 cases                                                                                                          |
| Training                  | 10 epochs, batch 4 (2 for Swin-UNETR), AMP, gradient clipping at 1.0, 2x Tesla T4                                                                                              |
| Best test mean Dice       | 3D UNet, 0.7999                                                                                                                                                                |
| Model used by the web app | VNet: `VNet.pth` in this folder, loaded by [`Backend/neuroscan/inference/processor.py`](../neuroscan/inference/processor.py). Test mean Dice of the VNet notebook run: 0.7655. |
| Missing result            | Swin-UNETR has no test result: its test cell failed with a `state_dict` loading error                                                                                          |

## Contents

- [Unified notebook](#unified-notebook)
- [At a glance (legacy runs)](#at-a-glance-legacy-runs)
- [Folder layout](#folder-layout)
- [Dataset and split](#dataset-and-split)
- [Shared pipeline](#shared-pipeline)
- [Models](#models)
- [Notebook walkthroughs](#notebook-walkthroughs)
- [Results](#results)
- [What the results show](#what-the-results-show)
- [Anomalies and limitations](#anomalies-and-limitations)
- [Reproducing a run](#reproducing-a-run)
- [Loading the deployed VNet](#loading-the-deployed-vnet)
- [Metric glossary](#metric-glossary)

## Folder layout

```text
Segmentation/
|-- README.md                  This file
|-- segmentation.ipynb         Unified notebook for all six models (no outputs yet)
`-- VNet.pth                   Weights loaded by the backend (not tracked in Git)
```

Unlike Binary Detection and Classification, there are **no output folders** here. Metrics exist only as printed cell outputs, and the only figures are the three images saved inside `VNet-Visualisation.ipynb`. The notebooks save checkpoints (`best_model.pth`, `best_model_nnunet.pth`, `best_model_swinunetr.pth`, `best_model_transbts.pth`) to the Kaggle working directory; none of those files is in this repository.

`VNet.pth` (5,617,589 bytes) is excluded by the root [`.gitignore`](../../.gitignore) rule `*.pth`, so a fresh clone does not contain it. It loads into the `VNet3D` class in `neuroscan/inference/networks.py` without errors and has 1,398,761 parameters, the same count the VNet notebook prints. No notebook writes a file named `VNet.pth`, so which training run produced it is not recorded.

## Dataset and split

- **Source.** `DATA_ROOT = "/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation"`, using `BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData`.
- **Case selection.** Every `BraTS20_Training_*` folder is listed and kept only if `<case>/<case>_seg.nii` exists. Output: `Total cases: 368`.
- **Split.** Two chained `train_test_split` calls with `random_state=42`: 70% train, then the remaining 30% halved. Output: `Train: 257  Val: 55  Test: 56`. The split is **not stratified**; tumour grade is not used anywhere in the notebooks.
- **Labels.** The BraTS label map is turned into three binary channels: WT = label > 0, TC = label 1 or 4, ET = label 4.

## Shared pipeline

These cells are identical in all seven notebooks apart from cell numbering.

### Preprocessing and RAM cache

`load_and_preprocess(path)` runs once per case before training:

1. Load each modality with `nibabel` as float32.
2. Z-score normalise over non-zero (brain) voxels only: `(v - mean) / (std + 1e-8)`.
3. Resize to 128 x 128 x 128 with trilinear `F.interpolate` (`align_corners=False`) and store as float16.
4. Resize the segmentation with nearest-neighbour interpolation and build the WT, TC and ET channels.
5. Stack into an image of shape `(4, 128, 128, 128)` (order FLAIR, T1, T1CE, T2) and a label of shape `(3, 128, 128, 128)`.

All 368 cases are cached in RAM. Every notebook prints `Total RAM used: 10.80 GB`. Cache build time was 637.5 s to 768.3 s depending on the run (see each walkthrough).

### Augmentation

`BraTSDataset` applies augmentation to the training set only:

- a random flip along each spatial axis with probability 0.5 (image and label together);
- a per-channel intensity scale drawn from U(0.9, 1.1);
- a per-channel intensity shift drawn from U(-0.1, 0.1).

### Loaders

`batch_size=BATCH_SIZE`, `num_workers=4`, `pin_memory=True`, `persistent_workers=True`. The training loader shuffles and uses `drop_last=True`. With batch 4 that gives 64 training batches and 14 validation batches (128 and 28 with batch 2).

### Loss

`region_loss(logits, target)` = mean soft Dice loss over the three channels + binary cross-entropy with logits. The soft Dice sums over the whole batch and all voxels for each channel. Models with extra output heads use a weighted deep-supervision sum, with targets downsampled by nearest-neighbour interpolation:

| Model                | Loss                                                |
| -------------------- | --------------------------------------------------- |
| UNet, VNet, TransBTS | `1.0 * main + 0.5 * ds2 + 0.25 * ds3`               |
| nnU-Net              | `1.0 * main + 0.5 * ds2 + 0.25 * ds3 + 0.125 * ds4` |
| UNETR, Swin-UNETR    | `region_loss` on the single output                  |

Because the loss definitions differ, loss values cannot be compared across these two groups.

### Evaluation metric

`brats_metrics` thresholds `sigmoid(logits) > 0.5` and computes a hard Dice score `(2 * |P and G| + 1e-6) / (|P| + |G| + 1e-6)` for each channel over the **whole batch** of 4 volumes. The per-batch values are averaged over batches, and the mean Dice is the average of the WT, TC and ET means. This is not the same as the per-case mean Dice used by the official BraTS evaluation.

### Training loop

- Each epoch: train with AMP autocast, `GradScaler`, gradient clipping at max-norm 1.0, then `scheduler.step()`, then validate under `torch.no_grad()`.
- The checkpoint is saved whenever the validation mean Dice improves (`mean_dice > best_mean_dice`).
- Early stopping with `PATIENCE = 6` epochs is implemented but never triggered in any run.
- After training, the test cell reloads the best checkpoint and prints `FINAL TEST RESULTS`.
- Seeds: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed` and `random.seed` all set to 42, with `cudnn.benchmark = True`.

### Hardware and environment

Every notebook prints `CUDA: True`, `GPUs: 2` and two `Tesla T4` devices, and wraps the model in `nn.DataParallel`. The notebook metadata records Python 3.12.12 on a Kaggle T4 accelerator with internet enabled. PyTorch and CUDA versions are not printed.

## Models

Parameter counts are the values each notebook prints.

| Model         | Notebook          | Parameters  | Optimiser                                               | LR schedule                                 | Batch | Key design                                                                                                                                                                          |
| ------------- | ----------------- | ----------- | ------------------------------------------------------- | ------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3D UNet       | `UNet.ipynb`      | 5,604,361   | SGD, Nesterov momentum 0.99, weight decay 3e-5, lr 0.01 | Polynomial `(1 - epoch/10) ** 0.9`          | 4     | 3 max-pool levels, channels 32-64-128-256, double conv + InstanceNorm + LeakyReLU(0.01), deep-supervision heads at 64 and 128 channels                                              |
| nnU-Net style | `nnUNet.ipynb`    | 16,548,108  | Same as UNet                                            | Same as UNet                                | 4     | 4 stride-2 conv levels, channels 32-64-128-256-320, InstanceNorm + LeakyReLU, deep-supervision heads at 3 decoder levels                                                            |
| VNet          | `VNet.ipynb`      | 1,398,761   | Same as UNet                                            | Same as UNet                                | 4     | Residual blocks with PReLU and 1x1x1 projection shortcuts, stride-2 conv downsampling, channels 16-32-64-128, gradient checkpointing on every block, 2 deep-supervision heads       |
| UNETR         | `UNETR.ipynb`     | 137,596,579 | AdamW, weight decay 1e-5, lr 1e-4                       | `CosineAnnealingLR(T_max=10, eta_min=1e-6)` | 4     | 16^3 patches (512 tokens), 12 transformer layers, hidden 768, 12 heads, MLP 3072, skips from layers 3, 6, 9 and 12 into a CNN decoder, gradient checkpointing on transformer layers |
| Swin-UNETR    | `SwinUNETR.ipynb` | 10,658,581  | AdamW, weight decay 1e-5, lr 1e-4                       | `CosineAnnealingLR(T_max=10)`               | 2     | 2^3 patches, embed dim 48, 4 stages (48-96-192-384), depths (2, 2, 2, 2), heads (3, 6, 12, 24), window 7 with shifted windows and relative position bias, CNN decoder               |
| TransBTS      | `TransBTS.ipynb`  | 35,805,449  | AdamW, weight decay 1e-5, lr 1e-4                       | `CosineAnnealingLR(T_max=10)`               | 4     | Residual CNN encoder 32-64-128-256-512, 4 transformer layers (dim 512, 8 heads, MLP ratio 4) on the 8^3 bottleneck, CNN decoder, 2 deep-supervision heads                           |

The model code comments name the source papers: UNETR (Hatamizadeh et al., 2021), Swin-UNETR (Hatamizadeh et al., MICCAI 2022) and TransBTS (W. Wang et al., MICCAI 2021). These are re-implementations written in the notebooks, not the official code. The nnU-Net notebook implements a fixed "3D full-resolution" style network; it does not run the nnU-Net framework's automatic configuration.

## Notebook walkthroughs

Each walkthrough lists the cells in order. Cells that match the [shared pipeline](#shared-pipeline) are summarised in one line. The epoch tables are copied from the training cell output.

### UNet.ipynb

| Cell | What it does                                                                                                                     | Output                                                            |
| ---- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 0    | `pip install nibabel tqdm`                                                                                                       |                                                                   |
| 1    | Imports                                                                                                                          |                                                                   |
| 2    | Configuration: `SEED=42`, `IMG_SIZE=128`, `EPOCHS=10`, `BATCH_SIZE=4`, `LR=0.01`, `NUM_WORKERS=4`, `PATIENCE=6`; seeding; device | `CUDA: True`, `GPUs: 2` (Tesla T4 x2)                             |
| 3    | Find cases with a segmentation file                                                                                              | `Total cases: 368`                                                |
| 4    | Preprocess and cache all cases                                                                                                   | `Cache built in 637.5s (10.6min)`, `Total RAM used: 10.80 GB`     |
| 5    | `BraTSDataset` with augmentation                                                                                                 |                                                                   |
| 6    | Split and loaders                                                                                                                | `Train: 257 Val: 55 Test: 56`, 64 train and 14 validation batches |
| 7    | `DoubleConv` and `UNet3D`                                                                                                        |                                                                   |
| 8    | `region_loss` and 3-head `deep_supervision_loss`                                                                                 |                                                                   |
| 9    | Model, `DataParallel`, SGD, polynomial schedule, `GradScaler`                                                                    | `Parameters: 5,604,361`                                           |
| 10   | `dice_score` and `brats_metrics`                                                                                                 |                                                                   |
| 11   | Training loop                                                                                                                    | `Training complete. Total time: 32.4min`                          |
| 12   | Test with the best checkpoint (`best_model.pth`)                                                                                 | Test loss 0.4109, WT 0.8647, TC 0.7844, ET 0.7505, mean 0.7999    |

| Epoch | LR      | Train loss | Val loss | Val WT | Val TC | Val ET | Val mean Dice | Epoch time (s) | Checkpoint |
| ----- | ------- | ---------- | -------- | ------ | ------ | ------ | ------------- | -------------- | ---------- |
| 1     | 0.01000 | 1.8430     | 0.8222   | 0.7889 | 0.6631 | 0.6422 | 0.6980        | 184.7          | saved      |
| 2     | 0.00910 | 0.6593     | 0.5985   | 0.8035 | 0.6806 | 0.6667 | 0.7169        | 192.9          | saved      |
| 3     | 0.00818 | 0.5495     | 0.5418   | 0.8183 | 0.7242 | 0.6572 | 0.7332        | 196.7          | saved      |
| 4     | 0.00725 | 0.5129     | 0.5226   | 0.8398 | 0.7344 | 0.6924 | 0.7555        | 195.6          | saved      |
| 5     | 0.00631 | 0.4832     | 0.4840   | 0.8541 | 0.7259 | 0.7105 | 0.7635        | 194.1          | saved      |
| 6     | 0.00536 | 0.4654     | 0.4445   | 0.8629 | 0.7676 | 0.7285 | 0.7863        | 193.8          | saved      |
| 7     | 0.00438 | 0.4319     | 0.4384   | 0.8639 | 0.7693 | 0.7380 | 0.7904        | 194.4          | saved      |
| 8     | 0.00338 | 0.4153     | 0.4307   | 0.8570 | 0.7839 | 0.7340 | 0.7916        | 194.1          | saved      |
| 9     | 0.00235 | 0.3962     | 0.4119   | 0.8697 | 0.7904 | 0.7397 | 0.8000        | 197.0          | saved      |
| 10    | 0.00126 | 0.3852     | 0.4061   | 0.8729 | 0.7844 | 0.7488 | 0.8021        | 198.4          | saved      |

Validation mean Dice improved in every epoch, so the checkpoint is epoch 10.

### nnUNet.ipynb

| Cell | What it does                                                                                          | Output                                                         |
| ---- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 0    | Markdown title: self-configuring 3D U-Net                                                             |                                                                |
| 1-2  | Install and imports                                                                                   |                                                                |
| 3    | Configuration, same values as UNet                                                                    | `CUDA: True`, `GPUs: 2`                                        |
| 4    | Find cases                                                                                            | `Total cases: 368`                                             |
| 5    | Preprocess and cache                                                                                  | `Cache built in 712.4s (11.9min)`, `Total RAM used: 10.80 GB`  |
| 6-7  | Dataset, split and loaders                                                                            | `Train: 257 Val: 55 Test: 56`                                  |
| 8    | Markdown: encoder 32-64-128-256-320, deep supervision at every decoder stage, InstanceNorm, LeakyReLU |                                                                |
| 9    | `nnUNetConvBlock` and `nnUNet3D`                                                                      |                                                                |
| 10   | `region_loss` and 4-head `deep_supervision_loss`                                                      |                                                                |
| 11   | Model, SGD, polynomial schedule                                                                       | `Parameters: 16,548,108`                                       |
| 12   | Metrics                                                                                               |                                                                |
| 13   | Training loop, checkpoint `best_model_nnunet.pth`                                                     | `Training complete. Total time: 55.1min`                       |
| 14   | Test                                                                                                  | Test loss 0.4553, WT 0.8569, TC 0.7813, ET 0.7452, mean 0.7945 |

| Epoch | LR      | Train loss | Val loss | Val WT | Val TC | Val ET | Val mean Dice | Epoch time (s) | Checkpoint           |
| ----- | ------- | ---------- | -------- | ------ | ------ | ------ | ------------- | -------------- | -------------------- |
| 1     | 0.01000 | 2.0607     | 0.9917   | 0.7364 | 0.6566 | 0.5888 | 0.6606        | 337.1          | saved                |
| 2     | 0.00910 | 0.8282     | 0.7676   | 0.7851 | 0.6606 | 0.6280 | 0.6912        | 330.6          | saved                |
| 3     | 0.00818 | 0.6635     | 0.6168   | 0.8095 | 0.7057 | 0.6844 | 0.7332        | 331.1          | saved                |
| 4     | 0.00725 | 0.5980     | 0.5970   | 0.8150 | 0.6909 | 0.6701 | 0.7253        | 329.9          | no improvement (1/6) |
| 5     | 0.00631 | 0.5588     | 0.5906   | 0.8432 | 0.7147 | 0.6321 | 0.7300        | 329.2          | no improvement (2/6) |
| 6     | 0.00536 | 0.5280     | 0.5261   | 0.8474 | 0.7443 | 0.7060 | 0.7659        | 329.7          | saved                |
| 7     | 0.00438 | 0.5073     | 0.5065   | 0.8522 | 0.7376 | 0.7108 | 0.7668        | 329.4          | saved                |
| 8     | 0.00338 | 0.4776     | 0.4853   | 0.8711 | 0.7522 | 0.7231 | 0.7822        | 330.4          | saved                |
| 9     | 0.00235 | 0.4667     | 0.4650   | 0.8739 | 0.7633 | 0.7354 | 0.7909        | 329.2          | saved                |
| 10    | 0.00126 | 0.4431     | 0.4514   | 0.8733 | 0.7758 | 0.7378 | 0.7956        | 328.9          | saved                |

### VNet.ipynb

| Cell | What it does                                                                                                                 | Output                                                         |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 0-1  | Install and imports                                                                                                          |                                                                |
| 2    | Configuration, same values as UNet                                                                                           | `CUDA: True`, `GPUs: 2`                                        |
| 3    | Find cases                                                                                                                   | `Total cases: 368`                                             |
| 4    | Preprocess and cache                                                                                                         | `Cache built in 768.3s (12.8min)`, `Total RAM used: 10.80 GB`  |
| 5-6  | Dataset, split and loaders                                                                                                   | `Train: 257 Val: 55 Test: 56`                                  |
| 7    | `ResBlock` and `VNet3D` with gradient checkpointing. A code comment says channels are halved "to save memory at full 128^3". |                                                                |
| 8    | `region_loss` and 3-head `deep_supervision_loss`                                                                             |                                                                |
| 9    | Model, SGD, polynomial schedule                                                                                              | `Parameters: 1,398,761`                                        |
| 10   | Metrics                                                                                                                      |                                                                |
| 11   | Training loop (also calls `torch.cuda.empty_cache()` after every step)                                                       | `Training complete. Total time: 16.9min`                       |
| 12   | Test with `best_model.pth`                                                                                                   | Test loss 0.4880, WT 0.8541, TC 0.7108, ET 0.7316, mean 0.7655 |

| Epoch | LR      | Train loss | Val loss | Val WT | Val TC | Val ET | Val mean Dice | Epoch time (s) | Checkpoint           |
| ----- | ------- | ---------- | -------- | ------ | ------ | ------ | ------------- | -------------- | -------------------- |
| 1     | 0.01000 | 1.9324     | 0.7990   | 0.6041 | 0.5583 | 0.6345 | 0.5990        | 105.9          | saved                |
| 2     | 0.00910 | 0.7572     | 0.6744   | 0.7566 | 0.6101 | 0.6828 | 0.6832        | 102.3          | saved                |
| 3     | 0.00818 | 0.6859     | 0.6477   | 0.7703 | 0.6091 | 0.7033 | 0.6942        | 102.0          | saved                |
| 4     | 0.00725 | 0.6333     | 0.6417   | 0.7919 | 0.5712 | 0.7131 | 0.6921        | 103.2          | no improvement (1/6) |
| 5     | 0.00631 | 0.5763     | 0.6343   | 0.7872 | 0.6187 | 0.6665 | 0.6908        | 100.5          | no improvement (2/6) |
| 6     | 0.00536 | 0.5584     | 0.5351   | 0.8128 | 0.6818 | 0.7071 | 0.7339        | 100.1          | saved                |
| 7     | 0.00438 | 0.5132     | 0.5432   | 0.8333 | 0.6445 | 0.7047 | 0.7275        | 101.5          | no improvement (1/6) |
| 8     | 0.00338 | 0.4854     | 0.5040   | 0.8505 | 0.6879 | 0.7182 | 0.7522        | 101.1          | saved                |
| 9     | 0.00235 | 0.4817     | 0.5188   | 0.8404 | 0.6631 | 0.7145 | 0.7393        | 100.0          | no improvement (1/6) |
| 10    | 0.00126 | 0.4812     | 0.4719   | 0.8567 | 0.7059 | 0.7356 | 0.7661        | 100.1          | saved                |

### VNet-Visualisation.ipynb

This notebook repeats the VNet experiment with the same configuration and adds three figures. It merges the dataset, split and loader code into one cell, drops the per-epoch WT, TC and ET printout and records a `history` dictionary instead.

| Cell | What it does                                                                                         | Output                                   |
| ---- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 0-1  | Install (adds `matplotlib` and `seaborn`) and imports                                                |                                          |
| 2    | Configuration, same values as VNet                                                                   | `CUDA: True`, `GPUs: 2`                  |
| 3    | Find cases                                                                                           | `Total cases: 368`                       |
| 4    | Preprocess and cache                                                                                 | `Cache built. RAM used: 10.80 GB`        |
| 5    | `visualize_sample`: middle axial slice (z = 64) of the first case, 4 modalities and 3 masks          | Figure 1                                 |
| 6    | Dataset, split and loaders                                                                           |                                          |
| 7    | `ResBlock` and `VNet3D`, same as VNet                                                                |                                          |
| 8    | Loss and metric functions                                                                            |                                          |
| 9    | Model, SGD, polynomial schedule, empty `history`                                                     |                                          |
| 10   | Training loop                                                                                        | Per-epoch line below                     |
| 11   | `plot_training_history`: loss curves and Dice curves (mean, WT, TC, ET)                              | Figure 2                                 |
| 12   | Test with `best_model.pth`                                                                           | `Test Loss: 0.4847`, `Mean Dice: 0.7658` |
| 13   | `visualize_predictions`: ground truth and prediction overlays for the first test batch, slice z = 64 | Figure 3                                 |

| Epoch | Train loss | Val loss | Val mean Dice |
| ----- | ---------- | -------- | ------------- |
| 1     | 1.9324     | 0.7989   | 0.5990        |
| 2     | 0.7573     | 0.6747   | 0.6830        |
| 3     | 0.6844     | 0.6557   | 0.6888        |
| 4     | 0.6318     | 0.6334   | 0.6953        |
| 5     | 0.5740     | 0.6276   | 0.6941        |
| 6     | 0.5585     | 0.5357   | 0.7342        |
| 7     | 0.5129     | 0.5458   | 0.7252        |
| 8     | 0.4845     | 0.5024   | 0.7518        |
| 9     | 0.4817     | 0.5143   | 0.7407        |
| 10    | 0.4790     | 0.4698   | 0.7664        |

The three figures are stored only as embedded images inside the notebook; they were not exported as image files, so they cannot be embedded here. Open the notebook to view them. The paper folder [`Backend/Materials/A_Comprehensive_Benchmark_of_3D_Segmentation_Architectures_for_Brain_Tumour_Segmentation/figures/`](../Materials/A_Comprehensive_Benchmark_of_3D_Segmentation_Architectures_for_Brain_Tumour_Segmentation/figures/) holds three figures with matching subjects, but their bytes differ from the notebook images, so they are not confirmed to come from this run.

This re-run ends slightly differently from `VNet.ipynb` (test mean Dice 0.7658 vs 0.7655, test loss 0.4847 vs 0.4880) with the same seed, which shows run-to-run variation from non-deterministic GPU kernels (`cudnn.benchmark = True`).

### UNETR.ipynb

| Cell | What it does                                                      | Output                                                         |
| ---- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| 0-1  | Install and imports                                               |                                                                |
| 2    | Configuration with `LR=1e-4`                                      | `CUDA: True`, `GPUs: 2`                                        |
| 3    | Find cases                                                        | `Total cases: 368`                                             |
| 4    | Preprocess and cache                                              | `Cache built in 662.5s (11.0min)`, `Total RAM used: 10.80 GB`  |
| 5-6  | Dataset, split and loaders                                        | `Train: 257 Val: 55 Test: 56`                                  |
| 7    | `PatchEmbedding3D`, `TransformerBlock`, `ConvBlock3D` and `UNETR` |                                                                |
| 8    | `region_loss` only (no deep supervision)                          |                                                                |
| 9    | Model, AdamW, cosine schedule with `eta_min=1e-6`                 | `Parameters: 137,596,579`                                      |
| 10   | Metrics                                                           |                                                                |
| 11   | Training loop                                                     | `Training complete. Total time: 117.2min`                      |
| 12   | Test                                                              | Test loss 1.3411, WT 0.7128, TC 0.6261, ET 0.5314, mean 0.6234 |

| Epoch | LR       | Train loss | Val loss | Val WT | Val TC | Val ET | Val mean Dice | Epoch time (s) | Checkpoint           |
| ----- | -------- | ---------- | -------- | ------ | ------ | ------ | ------------- | -------------- | -------------------- |
| 1     | 0.000100 | 1.5717     | 1.5129   | 0.5497 | 0.1266 | 0.0388 | 0.2383        | 714.4          | saved                |
| 2     | 0.000098 | 1.4838     | 1.4628   | 0.6650 | 0.4638 | 0.2496 | 0.4595        | 700.7          | saved                |
| 3     | 0.000091 | 1.4454     | 1.4295   | 0.6857 | 0.5791 | 0.3797 | 0.5482        | 701.5          | saved                |
| 4     | 0.000080 | 1.4142     | 1.4004   | 0.6852 | 0.6194 | 0.4427 | 0.5824        | 702.1          | saved                |
| 5     | 0.000066 | 1.3885     | 1.3786   | 0.6815 | 0.6123 | 0.4480 | 0.5806        | 700.1          | no improvement (1/6) |
| 6     | 0.000051 | 1.3690     | 1.3623   | 0.7042 | 0.6386 | 0.5032 | 0.6154        | 700.0          | saved                |
| 7     | 0.000035 | 1.3552     | 1.3511   | 0.7147 | 0.6583 | 0.5168 | 0.6299        | 700.6          | saved                |
| 8     | 0.000021 | 1.3458     | 1.3445   | 0.6827 | 0.6433 | 0.5010 | 0.6090        | 702.7          | no improvement (1/6) |
| 9     | 0.000010 | 1.3408     | 1.3409   | 0.7217 | 0.6657 | 0.5230 | 0.6368        | 700.1          | saved                |
| 10    | 0.000003 | 1.3384     | 1.3397   | 0.7263 | 0.6691 | 0.5149 | 0.6368        | 701.2          | no improvement (1/6) |

Epochs 9 and 10 both print a mean of 0.6368; only epoch 9 was saved because the rule is a strict `>`.

### SwinUNETR.ipynb

| Cell | What it does                                                                                                                                                                | Output                                                                                     |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 0    | Markdown title: Swin Transformer backbone with shifted-window attention                                                                                                     |                                                                                            |
| 1-2  | Install and imports                                                                                                                                                         |                                                                                            |
| 3    | Configuration with `BATCH_SIZE=2` (comment: "Swin-UNETR is memory-heavy; reduce if OOM"), `LR=1e-4`, patch 2, window 7, embed 48, depths (2, 2, 2, 2), heads (3, 6, 12, 24) | `CUDA: True`, `GPUs: 2`                                                                    |
| 4    | Find cases                                                                                                                                                                  | `Total cases: 368`                                                                         |
| 5    | Preprocess and cache                                                                                                                                                        | `Cache built in 715.9s (11.9min)`, `Total RAM used: 10.80 GB`                              |
| 6-7  | Dataset, split and loaders                                                                                                                                                  | `Train: 257 Val: 55 Test: 56`, 128 train and 28 validation batches                         |
| 8    | Markdown: architecture summary                                                                                                                                              |                                                                                            |
| 9    | Window partition helpers, `WindowAttention3D`, `SwinTransformerBlock3D`, `PatchMerging3D`, `SwinStage3D`, `PatchEmbed3D`, `DoubleConv3D`, `SwinUNETR`                       |                                                                                            |
| 10   | `region_loss` only                                                                                                                                                          |                                                                                            |
| 11   | Model, AdamW, cosine schedule, `torch.cuda.amp.GradScaler()`                                                                                                                | `Parameters: 10,658,581`, plus a `FutureWarning` that this `GradScaler` form is deprecated |
| 12   | Metrics                                                                                                                                                                     |                                                                                            |
| 13   | Training loop, checkpoint `best_model_swinunetr.pth`                                                                                                                        | `Training complete. Total time: 49.5min`                                                   |
| 14   | Test                                                                                                                                                                        | **`RuntimeError: Error(s) in loading state_dict for DataParallel`**                        |

| Epoch | LR       | Train loss | Val loss | Val WT | Val TC | Val ET | Val mean Dice | Epoch time (s) | Checkpoint           |
| ----- | -------- | ---------- | -------- | ------ | ------ | ------ | ------------- | -------------- | -------------------- |
| 1     | 0.000100 | 1.5204     | 1.4600   | 0.5982 | 0.4328 | 0.2473 | 0.4261        | 296.5          | saved                |
| 2     | 0.000098 | 1.4257     | 1.3922   | 0.6976 | 0.5225 | 0.3505 | 0.5235        | 296.4          | saved                |
| 3     | 0.000090 | 1.3630     | 1.3356   | 0.7692 | 0.6146 | 0.4189 | 0.6009        | 298.8          | saved                |
| 4     | 0.000079 | 1.3122     | 1.2918   | 0.6865 | 0.5407 | 0.4221 | 0.5498        | 297.2          | no improvement (1/6) |
| 5     | 0.000065 | 1.2728     | 1.2560   | 0.7680 | 0.6153 | 0.4848 | 0.6227        | 293.5          | saved                |
| 6     | 0.000050 | 1.2417     | 1.2304   | 0.8090 | 0.6529 | 0.4507 | 0.6375        | 295.8          | saved                |
| 7     | 0.000035 | 1.2200     | 1.2125   | 0.8025 | 0.6622 | 0.4797 | 0.6481        | 296.2          | saved                |
| 8     | 0.000021 | 1.2059     | 1.2024   | 0.8174 | 0.6808 | 0.4926 | 0.6636        | 297.8          | saved                |
| 9     | 0.000010 | 1.1983     | 1.1977   | 0.8066 | 0.6630 | 0.4655 | 0.6451        | 295.6          | no improvement (1/6) |
| 10    | 0.000002 | 1.1954     | 1.1964   | 0.8213 | 0.6817 | 0.4835 | 0.6622        | 295.6          | no improvement (2/6) |

Why the test cell failed: the checkpoint was saved from the `DataParallel` model, so its keys start with `module.`. The test cell strips that prefix and then loads the stripped keys back into the same `DataParallel` model, which expects the prefix. Every key is reported as both missing and unexpected, and no test metrics were produced. The best validation mean Dice was 0.6636 at epoch 8.

The `proj0` to `proj3` linear layers are defined in `SwinUNETR.__init__` but never used in `forward`; their weights are still counted in the parameter total.

### TransBTS.ipynb

| Cell | What it does                                                                                   | Output                                                         |
| ---- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 0-1  | Install and imports                                                                            |                                                                |
| 2    | Configuration with `LR=1e-4`, `TRANS_DIM=512`, `TRANS_DEPTH=4`, `TRANS_HEADS=8`, `MLP_RATIO=4` | `CUDA: True`, `GPUs: 2`                                        |
| 3    | Find cases                                                                                     | `Total cases: 368`                                             |
| 4    | Preprocess and cache                                                                           | `Cache built in 737.7s (12.3min)`, `Total RAM used: 10.80 GB`  |
| 5-6  | Dataset, split and loaders                                                                     | `Train: 257 Val: 55 Test: 56`                                  |
| 7    | `ResConvBlock`, `TransformerLayer`, `UpBlock` and `TransBTS`                                   |                                                                |
| 8    | `region_loss` and 3-head `deep_supervision_loss`                                               |                                                                |
| 9    | Model, AdamW, cosine schedule                                                                  | `Parameters: 35,805,449`                                       |
| 10   | Metrics                                                                                        |                                                                |
| 11   | Training loop, checkpoint `best_model_transbts.pth`                                            | `Training complete. Total time: 70.1min`                       |
| 12   | Test                                                                                           | Test loss 2.1940, WT 0.6383, TC 0.4618, ET 0.4325, mean 0.5109 |

| Epoch | LR       | Train loss | Val loss | Val WT | Val TC | Val ET | Val mean Dice | Epoch time (s) | Checkpoint           |
| ----- | -------- | ---------- | -------- | ------ | ------ | ------ | ------------- | -------------- | -------------------- |
| 1     | 0.000100 | 2.7437     | 2.6248   | 0.1952 | 0.0646 | 0.1089 | 0.1229        | 427.0          | saved                |
| 2     | 0.000098 | 2.5548     | 2.4975   | 0.3157 | 0.1240 | 0.2096 | 0.2164        | 420.6          | saved                |
| 3     | 0.000090 | 2.4405     | 2.3992   | 0.4785 | 0.3209 | 0.3448 | 0.3814        | 420.1          | saved                |
| 4     | 0.000079 | 2.3612     | 2.3384   | 0.4569 | 0.3152 | 0.3052 | 0.3591        | 423.0          | no improvement (1/6) |
| 5     | 0.000065 | 2.3062     | 2.2819   | 0.5722 | 0.4618 | 0.3774 | 0.4705        | 417.0          | saved                |
| 6     | 0.000050 | 2.2615     | 2.2440   | 0.5882 | 0.4665 | 0.4247 | 0.4931        | 415.6          | saved                |
| 7     | 0.000035 | 2.2266     | 2.2164   | 0.6117 | 0.4829 | 0.4035 | 0.4994        | 419.7          | saved                |
| 8     | 0.000021 | 2.2036     | 2.1997   | 0.6155 | 0.5016 | 0.4266 | 0.5146        | 420.1          | saved                |
| 9     | 0.000010 | 2.1915     | 2.1937   | 0.6308 | 0.5239 | 0.4435 | 0.5327        | 417.3          | saved                |
| 10    | 0.000002 | 2.1882     | 2.1914   | 0.6480 | 0.5390 | 0.4428 | 0.5433        | 419.9          | saved                |

## Results

### Test set (56 cases)

| Model         | WT Dice       | TC Dice       | ET Dice       | Mean Dice     | Test loss     | Parameters  | Training time | Best epoch |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ----------- | ------------- | ---------- |
| 3D UNet       | 0.8647        | 0.7844        | 0.7505        | **0.7999**    | 0.4109        | 5,604,361   | 32.4 min      | 10         |
| nnU-Net style | 0.8569        | 0.7813        | 0.7452        | 0.7945        | 0.4553        | 16,548,108  | 55.1 min      | 10         |
| VNet          | 0.8541        | 0.7108        | 0.7316        | 0.7655        | 0.4880        | 1,398,761   | 16.9 min      | 10         |
| UNETR         | 0.7128        | 0.6261        | 0.5314        | 0.6234        | 1.3411        | 137,596,579 | 117.2 min     | 9          |
| TransBTS      | 0.6383        | 0.4618        | 0.4325        | 0.5109        | 2.1940        | 35,805,449  | 70.1 min      | 10         |
| Swin-UNETR    | Not available | Not available | Not available | Not available | Not available | 10,658,581  | 49.5 min      | 8          |

Sources: the `FINAL TEST RESULTS` output of each notebook, the `Parameters:` line and the `Training complete` line. Training time excludes the 10 to 13 minutes of preprocessing. Test losses are not comparable between models that use deep supervision and those that do not (see [Loss](#loss)).

### Best validation scores (55 cases)

| Model         | Best epoch | Val WT | Val TC | Val ET | Val mean Dice |
| ------------- | ---------- | ------ | ------ | ------ | ------------- |
| 3D UNet       | 10         | 0.8729 | 0.7844 | 0.7488 | 0.8021        |
| nnU-Net style | 10         | 0.8733 | 0.7758 | 0.7378 | 0.7956        |
| VNet          | 10         | 0.8567 | 0.7059 | 0.7356 | 0.7661        |
| Swin-UNETR    | 8          | 0.8174 | 0.6808 | 0.4926 | 0.6636        |
| UNETR         | 9          | 0.7217 | 0.6657 | 0.5230 | 0.6368        |
| TransBTS      | 10         | 0.6480 | 0.5390 | 0.4428 | 0.5433        |

The validation set also selects the checkpoint, so these values are optimistic.

## What the results show

- **The three CNNs lead.** 3D UNet (0.7999), the nnU-Net style network (0.7945) and VNet (0.7655) have the highest test mean Dice. The two transformer models with test results score 0.6234 (UNETR) and 0.5109 (TransBTS).
- **Parameter count does not predict the ranking.** UNETR has the most parameters (137.6 million) and ranks fourth; VNet has the fewest (1.4 million) and ranks third.
- **ET is usually the hardest region.** For every model with test results, WT has the highest Dice. ET is lower than TC for UNet, nnU-Net, UNETR and TransBTS, but VNet scores ET (0.7316) above TC (0.7108).
- **Most models were still improving at epoch 10.** UNet, nnU-Net, VNet and TransBTS saved their best checkpoint at the last epoch, so these results reflect the 10-epoch budget, not each architecture's ceiling.
- **Only one run per model was recorded (VNet has two).** The two VNet runs, with the same seed, differ by 0.0003 test mean Dice. How much results vary across different seeds is not measured, so small gaps such as UNet vs nnU-Net (0.0054) should not be read as a firm ranking.

## Anomalies and limitations

- **Swin-UNETR has no test result** because the checkpoint loading code in the test cell is wrong (see [SwinUNETR.ipynb](#swinunetripynb)). Its best validation score is the only evaluation available.
- **Optimisers and learning rates differ by model family.** The three CNNs use SGD at lr 0.01 with a polynomial schedule; the three transformer models use AdamW at lr 1e-4 with cosine annealing. Swin-UNETR also uses batch 2. The comparison is therefore not a single controlled recipe.
- **Dice is computed per batch of 4 volumes, not per case,** and averaged over batches. Scores are not directly comparable with official BraTS leaderboard numbers.
- **Evaluation is at 128^3,** not at the original resolution. Masks are compared after resampling.
- **The split is random, not stratified,** and there is no cross-validation or repeated seed.
- **Only 10 epochs** were run, and most models peaked at the last epoch.
- **No checkpoints or output files** are stored with the notebooks, so the results cannot be re-evaluated without re-training.
- **Provenance of `VNet.pth` is not recorded.** It matches the VNet architecture and parameter count, but no notebook writes that filename.

## Reproducing a run

To run the current pipeline, use the unified notebook (see [Unified notebook](#unified-notebook)). The steps below reproduce a legacy run from its notebook in Git history. The notebooks are designed for Kaggle and have not been run elsewhere.

1. Create a Kaggle notebook with the **GPU T4 x2** accelerator and internet access (the first cell runs `pip install`).
2. Attach the Kaggle dataset `awsaf49/brats20-dataset-training-validation`. If it mounts somewhere other than `/kaggle/input/datasets/awsaf49/brats20-dataset-training-validation`, edit `DATA_ROOT`.
3. Make sure the session has more than 10.80 GB of free RAM for the volume cache.
4. Run all cells in order. Preprocessing takes about 11 to 13 minutes before training starts.
5. Download the checkpoint from `/kaggle/working/` (for VNet, `best_model.pth`) if you want to keep it.

For Swin-UNETR, the test cell needs to load the checkpoint without stripping the `module.` prefix (or load it into `model.module`); as saved, it fails.

## Loading the deployed VNet

The backend defines the same `VNet3D` in [`networks.py`](../neuroscan/inference/networks.py), without gradient checkpointing, and loads `VNet.pth` from this folder at start-up. To use it directly:

```python
import torch
from neuroscan.inference.networks import VNet3D  # run from the Backend folder

model = VNet3D()
state = torch.load("Segmentation/VNet.pth", map_location="cpu", weights_only=True)
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()
# Input: tensor of shape (1, 4, 128, 128, 128) in the order FLAIR, T1, T1CE, T2,
# each modality z-scored over non-zero voxels and resized with trilinear interpolation.
# Output: tuple (main, ds2, ds3); sigmoid(main) > 0.5 gives the WT, TC and ET masks.
```

## Metric glossary

| Term             | Meaning                                                                                                                     |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- |
| WT               | Whole Tumour: every labelled tumour voxel (BraTS labels 1, 2 and 4)                                                         |
| TC               | Tumour Core: labels 1 and 4                                                                                                 |
| ET               | Enhancing Tumour: label 4                                                                                                   |
| Dice             | 2 x (voxels in both P and G) / (voxels in P + voxels in G), for predicted mask P and ground truth G; 1 is a perfect overlap |
| Mean Dice        | Average of the WT, TC and ET Dice scores                                                                                    |
| Deep supervision | Extra output heads at lower decoder resolutions whose losses are added to the main loss during training                     |
