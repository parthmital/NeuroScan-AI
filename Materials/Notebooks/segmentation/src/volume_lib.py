"""Shared code for the BraTS 2020 hybrid CNN-Transformer ensemble segmentation notebook.

The notebook writes this file to /kaggle/working/src so the notebook and the training workers
(one process per GPU) import exactly the same data handling, models, inference, metrics, fusion
and explainability code.
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODALITIES = ("flair", "t1", "t1ce", "t2")  # input channel order, as in the web app
# File-name suffixes per sequence: BraTS 2020 (`_t1ce`) and BraTS 2023 (`-t1c`) conventions.
MODALITY_ALIASES = {
    "flair": ("flair", "t2f"),
    "t1": ("t1", "t1n"),
    "t1ce": ("t1ce", "t1c"),
    "t2": ("t2", "t2w"),
}
REGIONS = ("WT", "TC", "ET")
HD95_EMPTY = 373.13  # BraTS value when exactly one of prediction and reference is empty

# Family, optimiser recipe, memory settings, deep-supervision weights and the decoder layer used
# for Seg-Grad-CAM (a stage at 1/4 of the input resolution in every network).
RECIPES = {
    "nnU-Net": {
        "family": "CNN",
        "optimizer": "sgd",
        "lr": 1e-2,
        "weight_decay": 3e-5,
        "warmup_epochs": 0,
        "grad_clip": 12.0,
        "micro_batch": 2,
        "ds_weights": (1.0, 0.5, 0.25),
        "cam_layer": "upsamples.2",
    },
    "SegResNet": {
        "family": "CNN",
        "optimizer": "adamw",
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 2,
        "grad_clip": 1.0,
        "micro_batch": 2,
        "ds_weights": (1.0, 0.5, 0.25),
        "cam_layer": "up_layers.0.blocks",
    },
    "Swin UNETR": {
        "family": "Transformer",
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 2,
        "grad_clip": 1.0,
        "micro_batch": 1,  # 2 ran out of memory on a T4 (a 2.6 GB fp32 attention-mask sum in stage 1)
        "ds_weights": (1.0,),
        "cam_layer": "decoder3",
    },
}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def run_id(model, seed):
    return f"{slug(model)}_seed{seed}"


# ---------------------------------------------------------------------------
# Models (MONAI reference implementations; 4 input sequences, 3 sigmoid region outputs)
# ---------------------------------------------------------------------------


def build_model(name):
    """nnU-Net: six-stage residual DynUNet, 32-320 channels, deep supervision at two decoder stages.
    SegResNet: the deep-supervision SegResNet of MONAI Auto3DSeg (Myronenko's BraTS 2018 design).
    Swin UNETR: Swin UNETR-v2, feature size 48, gradient checkpointing."""
    from monai.networks.nets import DynUNet, SegResNetDS, SwinUNETR

    if name == "nnU-Net":
        strides = [1, 2, 2, 2, 2, 2]
        return DynUNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            kernel_size=[3] * 6,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=[32, 64, 128, 256, 320, 320],
            norm_name=("INSTANCE", {"affine": True}),
            deep_supervision=True,
            deep_supr_num=2,
            res_block=True,
        )
    if name == "SegResNet":
        return SegResNetDS(
            init_filters=32,
            in_channels=4,
            out_channels=3,
            norm="instance",
            blocks_down=(1, 2, 2, 4),
            dsdepth=3,
        )
    if name == "Swin UNETR":
        return SwinUNETR(
            in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True, use_v2=True
        )
    raise ValueError(f"Unknown model {name!r}")


def output_list(output):
    """Main output first, then deep-supervision outputs, whatever the network returns."""
    if isinstance(output, (list, tuple)):
        return list(output)
    if output.ndim == 6:  # DynUNet stacks its deep-supervision outputs on dim 1 while training
        return list(output.unbind(1))
    return [output]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def case_files(case_dir):
    """Sequence and segmentation files of one case.

    The segmentation is any NIfTI whose name contains "seg", which also finds the misnamed mask of
    BraTS20_Training_355; sequences are recognised by the last token of the file name. A folder
    named like a NIfTI file (Kaggle can unpack an uploaded volume that way) stands for the file in it.
    """
    files = {}
    for path in sorted(Path(case_dir).iterdir()):
        name = path.name.lower()
        if not name.endswith((".nii", ".nii.gz")):
            continue
        if path.is_dir():
            path = next(iter(sorted(path.glob("*.nii*"))), None)
        if path is None or not path.is_file():
            continue
        stem = name.split(".nii")[0]
        if "seg" in stem:
            files["seg"] = path
            continue
        token = re.split(r"[_-]", stem)[-1]
        for modality, aliases in MODALITY_ALIASES.items():
            if token in aliases:
                files[modality] = path
    return files


def zscore_foreground(volume):
    brain = volume != 0
    if brain.any():
        volume[brain] = (volume[brain] - volume[brain].mean()) / (volume[brain].std() + 1e-8)
    return volume


def label_bits(seg, et_label=4):
    """BraTS labels -> bit field: 1 = WT (any tumour), 2 = TC (necrosis and ET), 4 = ET.

    ET is label 4 in BraTS 2020 and label 3 in BraTS 2023 (including BraTS-Africa).
    """
    seg = np.asarray(seg)
    return ((seg > 0) * 1 + np.isin(seg, (1, et_label)) * 2 + (seg == et_label) * 4).astype(
        np.uint8
    )


def bits_to_regions(bits):
    return np.stack([(bits & 1) > 0, (bits & 2) > 0, (bits & 4) > 0])


def preprocess_case(args):
    """Load one case, crop it to the bounding box of the brain (any non-zero sequence voxel), z-score
    each sequence over its non-zero voxels and save float16 images and the 3-bit region mask to the
    cache at native 1 mm resolution (run in a worker process)."""
    import time

    import nibabel as nib

    files, et_label, prefix, fg_samples = args
    start = time.perf_counter()
    volumes = [np.asarray(nib.load(str(files[m])).dataobj, dtype=np.float32) for m in MODALITIES]
    seg_image = nib.load(str(files["seg"]))
    seg = np.rint(np.asarray(seg_image.dataobj, dtype=np.float32)).astype(np.uint8)
    read_seconds = time.perf_counter() - start
    brain = np.any(np.stack(volumes) != 0, axis=0)
    coords = np.argwhere(brain) if brain.any() else np.argwhere(np.ones_like(brain))
    low, high = coords.min(0), coords.max(0) + 1
    box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    image = np.stack([zscore_foreground(v[box].copy()) for v in volumes]).astype(np.float16)
    bits = label_bits(seg, et_label)
    crop_bits = np.ascontiguousarray(bits[box])
    np.save(f"{prefix}_img.npy", image)
    np.save(f"{prefix}_lbl.npy", crop_bits)
    spacing = tuple(float(z) for z in seg_image.header.get_zooms()[:3])
    voxel_cm3 = float(np.prod(spacing)) / 1000
    rng = np.random.default_rng(0)
    fg = {}
    crop_regions = bits_to_regions(crop_bits)
    for index, region in enumerate(REGIONS):
        points = np.argwhere(crop_regions[index])
        if len(points) > fg_samples:
            points = points[rng.choice(len(points), fg_samples, replace=False)]
        fg[region] = points.astype(np.int16)
    return {
        "shape": seg.shape,
        "spacing": spacing,
        "box_low": [int(v) for v in low],
        "crop_shape": image.shape[1:],
        "volumes_cm3": [float(r.sum()) * voxel_cm3 for r in bits_to_regions(bits)],
        "label_outside_crop": int((bits > 0).sum() - (crop_bits > 0).sum()),
        "labels_present": sorted(int(v) for v in np.unique(seg)),
        "fg": fg,
        "read_seconds": read_seconds,
        "process_seconds": time.perf_counter() - start - read_seconds,
    }


def load_image(cache_dir, case):
    return np.load(Path(cache_dir) / f"{case}_img.npy", mmap_mode="r")


def load_regions(cache_dir, case):
    return bits_to_regions(np.load(Path(cache_dir) / f"{case}_lbl.npy"))


def crop_pad(array, start, size):
    """Cube of `size` voxels starting at `start` over the last three axes; parts outside are zero."""
    out = np.zeros(array.shape[:-3] + (size,) * 3, dtype=array.dtype)
    src, dst = [], []
    for begin, length in zip(start, array.shape[-3:]):
        a, b = max(int(begin), 0), min(int(begin) + size, length)
        if b <= a:
            return out
        src.append(slice(a, b))
        dst.append(slice(a - int(begin), b - int(begin)))
    out[(..., *dst)] = array[(..., *src)]
    return out


class PatchDataset(torch.utils.data.Dataset):
    """Random training cubes of `source` voxels from the native-resolution cache.

    Item `index` is drawn from a generator seeded by (seed, index), so a run that resumes at an
    epoch boundary sees the same patches as an uninterrupted run. With probability `fg_probability`
    the cube is centred on a random voxel of a random region present in the case (foreground
    oversampling, as in nnU-Net); otherwise the centre is uniform inside the brain box.
    """

    def __init__(self, cache_dir, cases, fg_path, source, patch, fg_probability, seed):
        self.cache_dir, self.cases, self.fg_path = Path(cache_dir), list(cases), fg_path
        self.source, self.patch, self.fg_probability, self.seed = (
            source,
            patch,
            fg_probability,
            seed,
        )
        self.fg = None

    def __len__(self):
        return 2**62

    def __getitem__(self, index):
        if self.fg is None:
            with np.load(self.fg_path) as archive:
                self.fg = {key: archive[key] for key in archive.files}
        rng = np.random.default_rng([self.seed, int(index)])
        case = self.cases[rng.integers(len(self.cases))]
        image = load_image(self.cache_dir, case)
        bits = np.load(self.cache_dir / f"{case}_lbl.npy", mmap_mode="r")
        centre = None
        if rng.random() < self.fg_probability:
            present = [key for key in (f"{case}/{r}" for r in REGIONS) if len(self.fg.get(key, ()))]
            if present:
                points = self.fg[present[rng.integers(len(present))]]
                centre = points[rng.integers(len(points))].astype(np.int64)
        if centre is None:
            low = [min(self.patch // 2, s // 2) for s in bits.shape]
            centre = np.array(
                [
                    rng.integers(lo, max(lo, s - self.patch // 2) + 1)
                    for lo, s in zip(low, bits.shape)
                ]
            )
        start = centre - self.source // 2
        return torch.from_numpy(
            crop_pad(image, start, self.source).astype(np.float32)
        ), torch.from_numpy(crop_pad(bits, start, self.source))


def _rotation(angles):
    """(B, 3) angles in radians -> (B, 3, 3) rotation matrices Rz @ Ry @ Rx."""
    c, s = angles.cos(), angles.sin()
    one, zero = torch.ones_like(c[:, 0]), torch.zeros_like(c[:, 0])

    def matrix(*entries):
        return torch.stack(entries, -1).view(-1, 3, 3)

    rx = matrix(one, zero, zero, zero, c[:, 0], -s[:, 0], zero, s[:, 0], c[:, 0])
    ry = matrix(c[:, 1], zero, s[:, 1], zero, one, zero, -s[:, 1], zero, c[:, 1])
    rz = matrix(c[:, 2], -s[:, 2], zero, s[:, 2], c[:, 2], zero, zero, zero, one)
    return rz @ ry @ rx


def augment(image, bits, patch, max_angle=30.0, zoom_range=(0.7, 1.4), p_rotate=0.2, p_zoom=0.2):
    """GPU augmentation of a batch of source cubes (B, 4, S, S, S) and region bits (B, S, S, S).

    nnU-Net-style recipe: random rotation (up to `max_angle` degrees per axis) and zoom, resampled
    to the central `patch`^3 (bilinear for images, nearest for masks); mirroring on every axis;
    Gaussian noise, brightness, contrast and gamma per sequence. Returns images (B, 4, P, P, P)
    and region targets (B, 3, P, P, P).
    """
    b, device = image.shape[0], image.device
    label = torch.stack([(bits & 1) > 0, (bits & 2) > 0, (bits & 4) > 0], 1).float()
    angles = (torch.rand(b, 3, device=device) * 2 - 1) * math.radians(max_angle)
    angles = angles * (torch.rand(b, 1, device=device) < p_rotate)
    zoom = torch.empty(b, device=device).uniform_(*zoom_range)
    zoom = torch.where(torch.rand(b, device=device) < p_zoom, zoom, torch.ones_like(zoom))
    theta = torch.zeros(b, 3, 4, device=device)
    theta[:, :, :3] = _rotation(angles) / zoom[:, None, None] * (patch / image.shape[-1])
    grid = F.affine_grid(theta, [b, 1, patch, patch, patch], align_corners=False)
    image = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    label = F.grid_sample(label, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    for axis in (2, 3, 4):
        flip = (torch.rand(b, device=device) < 0.5)[:, None, None, None, None]
        image = torch.where(flip, image.flip(axis), image)
        label = torch.where(flip, label.flip(axis), label)

    sample, channel = (b, 1, 1, 1, 1), (b, image.shape[1], 1, 1, 1)

    def chance(p, shape):
        return torch.rand(shape, device=device) < p

    def uniform(low, high, shape):
        return torch.empty(shape, device=device).uniform_(low, high)

    image = image + torch.randn_like(image) * uniform(0.0, 0.1, sample) * chance(0.15, sample)
    image = image * torch.where(chance(0.15, channel), uniform(0.75, 1.25, channel), 1.0)
    mean = image.mean((2, 3, 4), keepdim=True)
    image = torch.where(
        chance(0.15, channel), (image - mean) * uniform(0.75, 1.25, channel) + mean, image
    )
    low = image.amin((2, 3, 4), keepdim=True)
    span = image.amax((2, 3, 4), keepdim=True) - low + 1e-6
    gamma = torch.where(chance(0.15, channel), uniform(0.7, 1.5, channel), 1.0)
    image = ((image - low) / span).clamp(0, 1) ** gamma * span + low
    return image, label


# ---------------------------------------------------------------------------
# Loss, inference, post-processing, metrics
# ---------------------------------------------------------------------------


def region_loss(logits, target):
    """Soft Dice over the batch plus binary cross-entropy, per region."""
    logits = logits.float()
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3, 4)
    dice = 1 - (2 * (prob * target).sum(dims) + 1e-6) / (prob.sum(dims) + target.sum(dims) + 1e-6)
    return dice.mean() + F.binary_cross_entropy_with_logits(logits, target)


def supervised_loss(outputs, target, weights):
    """Deep-supervision loss: weights normalised to sum to one, targets downsampled (nearest) to
    each lower-resolution output."""
    outputs = output_list(outputs)
    total = sum(weights[: len(outputs)])
    loss = 0.0
    for output, weight in zip(outputs, weights):
        scaled = (
            target
            if output.shape[2:] == target.shape[2:]
            else F.interpolate(target, size=output.shape[2:], mode="nearest")
        )
        loss = loss + weight / total * region_loss(output, scaled)
    return loss


FLIP_SETS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


@torch.no_grad()
def sliding_probs(model, image, roi, overlap, tta, sw_batch=2):
    """Sigmoid probabilities of a whole cropped case (1, 4, D, H, W): Gaussian-weighted sliding
    windows of roi^3 voxels, averaged over the 8 axis flips when tta is on.

    Returns the mean (1, 3, D, H, W) and the standard deviation across flips (zero without TTA).
    """
    from monai.inferers import sliding_window_inference

    model.eval()

    def predictor(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            return output_list(model(x))[0].float()

    flips = FLIP_SETS if tta else [()]
    total = total_sq = None
    for dims in flips:
        inputs = image.flip(dims) if dims else image
        prob = torch.sigmoid(
            sliding_window_inference(
                inputs, (roi,) * 3, sw_batch, predictor, overlap=overlap, mode="gaussian"
            )
        )
        prob = prob.flip(dims) if dims else prob
        total = prob if total is None else total + prob
        total_sq = prob**2 if total_sq is None else total_sq + prob**2
    mean = total / len(flips)
    std = (total_sq / len(flips) - mean**2).clamp_min(0).sqrt()
    return mean, std


def save_probs(path, probs, tta_std=0.0):
    """Probabilities (3, D, H, W) as compressed uint8 (resolution 1/255) with the case's TTA spread."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probs=(probs.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy(),
        tta_std=np.float32(tta_std),
    )


def load_probs(path):
    with np.load(path) as archive:
        return archive["probs"], float(archive["tta_std"])


def regions_from_probs(probs, et_min_voxels=0, thresholds=0.5):
    """Nested regions with the BraTS label hierarchy: ET inside TC inside WT.

    `thresholds` is one value or one per region. An ET prediction smaller than et_min_voxels is
    removed, a standard BraTS post-processing step that helps cases without enhancing tumour.
    """
    limit = torch.as_tensor(thresholds, dtype=probs.dtype, device=probs.device).reshape(
        -1, *[1] * (probs.ndim - 1)
    )
    wt, tc, et = (probs > limit).unbind(0)
    if int(et.sum()) < et_min_voxels:
        et = torch.zeros_like(et)
    tc = tc | et
    wt = wt | tc
    return torch.stack([wt, tc, et])


def dice(pred, ref):
    pred_sum, ref_sum = int(pred.sum()), int(ref.sum())
    if pred_sum + ref_sum == 0:
        return 1.0
    return 2 * int((pred & ref).sum()) / (pred_sum + ref_sum)


def hd95(pred, ref, spacing):
    """95th-percentile symmetric Hausdorff distance in mm (MedPy definition), on the bounding box
    of both masks for speed."""
    from scipy import ndimage

    if not pred.any() and not ref.any():
        return 0.0
    if not pred.any() or not ref.any():
        return HD95_EMPTY
    coords = np.argwhere(pred | ref)
    low = np.maximum(coords.min(0) - 2, 0)
    high = coords.max(0) + 3
    box = tuple(slice(a, b) for a, b in zip(low, high))
    pred = np.pad(pred[box], 2)
    ref = np.pad(ref[box], 2)
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure)
    ref_surface = ref ^ ndimage.binary_erosion(ref, structure)
    to_ref = ndimage.distance_transform_edt(~ref_surface, sampling=spacing)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[ref_surface]
    return float(max(np.percentile(to_ref, 95), np.percentile(to_pred, 95)))


def slice_stats(pred, ref):
    """Sums for the slice-wise protocol of 2D studies, per region: the Dice of every axial slice
    whose reference contains the region (slices without it are skipped, as when only tumour slices
    are kept) and the pooled overlap over those slices."""
    row = {}
    for index, region in enumerate(REGIONS):
        p, r = pred[index], ref[index]
        inter, p_sum, r_sum = (p & r).sum((0, 1)), p.sum((0, 1)), r.sum((0, 1))
        keep = r_sum > 0
        row[f"slice_dice_sum_{region}"] = float(
            (2 * inter[keep] / (p_sum[keep] + r_sum[keep])).sum()
        )
        row[f"slices_{region}"] = int(keep.sum())
        row[f"slice_overlap_{region}"] = int(inter[keep].sum())
        row[f"slice_total_{region}"] = int(p_sum[keep].sum() + r_sum[keep].sum())
    return row


def case_metrics(pred, ref, spacing, total_voxels=None):
    """Dice, HD95, sensitivity, specificity and volumes for the three regions of one case.

    Masks may be cropped to the brain box; `total_voxels` is the native volume size so that
    specificity counts the true negatives outside the box.
    """
    voxel_cm3 = float(np.prod(spacing)) / 1000
    total_voxels = total_voxels or pred[0].size
    row = {}
    for index, region in enumerate(REGIONS):
        p, r = pred[index], ref[index]
        tp = int((p & r).sum())
        fn = int(r.sum()) - tp
        fp = int(p.sum()) - tp
        tn = total_voxels - tp - fn - fp
        row[f"dice_{region}"] = dice(p, r)
        row[f"hd95_{region}"] = hd95(p, r, spacing)
        row[f"sensitivity_{region}"] = tp / (tp + fn) if tp + fn else 1.0
        row[f"specificity_{region}"] = tn / (tn + fp) if tn + fp else 1.0
        row[f"volume_pred_cm3_{region}"] = float(p.sum()) * voxel_cm3
        row[f"volume_ref_cm3_{region}"] = float(r.sum()) * voxel_cm3
    row["dice_mean"] = float(np.mean([row[f"dice_{r}"] for r in REGIONS]))
    row["hd95_mean"] = float(np.mean([row[f"hd95_{r}"] for r in REGIONS]))
    row.update(slice_stats(pred, ref))
    return row


def score_case(args):
    """case_metrics of one prediction against the cached reference (run in a worker process)."""
    pred, case, cache_dir, spacing, voxels = args
    return case_metrics(pred, load_regions(cache_dir, case), spacing, voxels)


def bootstrap_mean_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Region-adaptive fusion of the ensemble members
# ---------------------------------------------------------------------------


def simplex_grid(members, step):
    """Every weight vector over `members` with entries in multiples of `step` that sum to one."""
    n = round(1 / step)
    combos = [c for c in itertools.product(range(n + 1), repeat=members) if sum(c) == n]
    return np.array(combos, dtype=np.float32) / n


def fuse(probs, weights):
    """Region-wise weighted mean of member probabilities: probs (M, 3, ...), weights (3, M)."""
    return torch.einsum("rm,mr...->r...", weights.to(probs.dtype), probs)


@torch.no_grad()
def fusion_dice_table(probs, ref, combos, thresholds):
    """Dice of every (weight vector, threshold) pair for every region of one case.

    probs (M, 3, ...) member probabilities, ref (3, ...) bool reference, combos (K, M).
    Returns an array (3, K, T).
    """
    out = np.zeros((3, len(combos), len(thresholds)))
    weights = torch.as_tensor(combos, device=probs.device, dtype=probs.dtype)
    for region in range(3):
        fused = torch.einsum("km,m...->k...", weights, probs[:, region])
        target = ref[region]
        ref_sum = int(target.sum())
        for t_index, threshold in enumerate(thresholds):
            pred = fused > threshold
            inter = (pred & target).flatten(1).sum(1).double()
            pred_sum = pred.flatten(1).sum(1).double()
            denominator = pred_sum + ref_sum
            scores = torch.where(
                denominator > 0, 2 * inter / denominator.clamp_min(1), torch.ones_like(denominator)
            )
            out[region, :, t_index] = scores.cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def seg_grad_cam(model, x, layer_name, region):
    """Seg-Grad-CAM (Vinogradova et al., AAAI 2020) for one output region of a 3D network.

    The target is the sum of the region's logits over the voxels the model assigns to the region
    (all voxels if none), the class-activation map is the ReLU of the gradient-weighted sum of the
    layer's channels, upsampled to the input and scaled to [0, 1].
    Returns the map (D, H, W) and the region probabilities (3, D, H, W).

    Memory: the weights are frozen during the call and the layer's output is replaced by a detached
    leaf, so the encoder keeps no autograd graph, no weight gradients are allocated and the backward
    pass stops at the layer. A full backward of three networks on one T4 ran out of memory.
    """
    layer = model.get_submodule(layer_name)
    store = {}

    def hook(module, inputs, output):
        store["activation"] = output.detach().requires_grad_()
        return store["activation"]

    trainable = [p for p in model.parameters() if p.requires_grad]
    handle = layer.register_forward_hook(hook)
    try:
        model.eval()
        for parameter in trainable:
            parameter.requires_grad_(False)
        with torch.enable_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            logits = output_list(model(x))[0].float()
            mask = (logits[:, region] > 0).float()
            if mask.sum() == 0:
                mask = torch.ones_like(mask)
            (gradient,) = torch.autograd.grad((logits[:, region] * mask).sum(), store["activation"])
        activation, gradient = store.pop("activation").detach().float(), gradient.float()
        cam = F.relu((gradient.mean(dim=(2, 3, 4), keepdim=True) * activation).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
        cam = cam / cam.max().clamp_min(1e-8)
        return cam, torch.sigmoid(logits[0]).detach()
    finally:
        handle.remove()
        for parameter in trainable:
            parameter.requires_grad_(True)


def attribution_in_region(cam, region_mask, margin=5):
    """Share of the map's mass inside the reference region dilated by `margin` voxels, and the
    share expected by chance (the dilated region's volume fraction)."""
    mask = region_mask.float()[None, None]
    dilated = F.max_pool3d(mask, 2 * margin + 1, stride=1, padding=margin)[0, 0] > 0
    total = float(cam.sum())
    inside = float(cam[dilated].sum()) / total if total > 0 else float("nan")
    return inside, float(dilated.float().mean())
