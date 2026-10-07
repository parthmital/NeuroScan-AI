"""Shared code for the unified BraTS 2020 segmentation notebook.

The notebook writes this file to /kaggle/working/src so the notebook and the training workers
(one process per GPU) import exactly the same data handling, models and metrics.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

MODALITIES = ("flair", "t1", "t1ce", "t2")  # input channel order, as in the web app
REGIONS = ("WT", "TC", "ET")
HD95_EMPTY = 373.13  # BraTS value when exactly one of prediction and reference is empty

# Family, optimiser recipe and memory settings per model. CNNs keep the paper's SGD recipe;
# transformers keep AdamW and now get a linear warm-up before cosine decay.
RECIPES = {
    "UNet": {
        "family": "CNN",
        "optimizer": "sgd",
        "lr": 1e-2,
        "weight_decay": 3e-5,
        "warmup_epochs": 0,
        "micro_batch": 2,
        "checkpointing": False,
        "ds_weights": (1.0, 0.5, 0.25),
    },
    "nnU-Net": {
        "family": "CNN",
        "optimizer": "sgd",
        "lr": 1e-2,
        "weight_decay": 3e-5,
        "warmup_epochs": 0,
        "micro_batch": 2,
        "checkpointing": False,
        "ds_weights": (1.0, 0.5, 0.25, 0.125),
    },
    "VNet": {
        "family": "CNN",
        "optimizer": "sgd",
        "lr": 1e-2,
        "weight_decay": 3e-5,
        "warmup_epochs": 0,
        "micro_batch": 2,
        "checkpointing": False,
        "ds_weights": (1.0, 0.5, 0.25),
    },
    "UNETR": {
        "family": "Transformer",
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "micro_batch": 2,
        "checkpointing": True,
        "ds_weights": (1.0,),
    },
    "Swin UNETR": {
        "family": "Transformer",
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "micro_batch": 1,
        "checkpointing": True,
        "ds_weights": (1.0,),
    },
    "TransBTS": {
        "family": "Hybrid",
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "warmup_epochs": 3,
        "micro_batch": 2,
        "checkpointing": False,
        "ds_weights": (1.0, 0.5, 0.25),
    },
}


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def run_id(model, seed):
    return f"{slug(model)}_seed{seed}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def case_files(case_dir):
    """Modality and segmentation files of one BraTS case.

    The segmentation is any NIfTI whose name contains "seg", which also finds the misnamed
    mask of BraTS20_Training_355 that the earlier notebooks skipped.
    """
    case_dir = Path(case_dir)
    files = {}
    for path in sorted(case_dir.iterdir()):
        name = path.name.lower()
        if not name.endswith((".nii", ".nii.gz")):
            continue
        stem = name.split(".nii")[0]
        if "seg" in stem:
            files["seg"] = path
        else:
            for modality in MODALITIES:
                if stem.endswith("_" + modality):
                    files[modality] = path
    return files


def zscore_foreground(volume):
    brain = volume != 0
    if brain.any():
        volume[brain] = (volume[brain] - volume[brain].mean()) / (volume[brain].std() + 1e-8)
    return volume


def label_bits(seg):
    """BraTS labels -> bit field: 1 = WT (any tumour), 2 = TC (labels 1 and 4), 4 = ET (label 4)."""
    seg = np.asarray(seg)
    return ((seg > 0) * 1 + np.isin(seg, (1, 4)) * 2 + (seg == 4) * 4).astype(np.uint8)


def bits_to_regions(bits):
    return np.stack([(bits & 1) > 0, (bits & 2) > 0, (bits & 4) > 0])


def preprocess_case(args):
    """Load, normalise and resample one case (run in a worker process)."""
    import nibabel as nib

    torch.set_num_threads(1)
    files, size = args
    channels = []
    for modality in MODALITIES:
        volume = np.asarray(nib.load(str(files[modality])).dataobj, dtype=np.float32)
        volume = zscore_foreground(volume)
        resized = F.interpolate(
            torch.from_numpy(volume)[None, None],
            size=(size,) * 3,
            mode="trilinear",
            align_corners=False,
        )
        channels.append(resized[0, 0].numpy().astype(np.float16))
    seg_image = nib.load(str(files["seg"]))
    seg = np.asarray(seg_image.dataobj, dtype=np.float32)
    resized_seg = F.interpolate(
        torch.from_numpy(seg)[None, None], size=(size,) * 3, mode="nearest"
    )[0, 0].numpy()
    spacing = tuple(float(z) for z in seg_image.header.get_zooms()[:3])
    regions = bits_to_regions(label_bits(seg.astype(np.uint8)))
    voxel_cm3 = float(np.prod(spacing)) / 1000
    return {
        "image": np.stack(channels),
        "label": label_bits(resized_seg.astype(np.uint8)),
        "shape": seg.shape,
        "spacing": spacing,
        "volumes_cm3": [float(r.sum()) * voxel_cm3 for r in regions],
    }


def load_reference(seg_path):
    import nibabel as nib

    image = nib.load(str(seg_path))
    seg = np.asarray(image.dataobj).astype(np.uint8)
    return bits_to_regions(label_bits(seg)), tuple(float(z) for z in image.header.get_zooms()[:3])


class CachedCases(torch.utils.data.Dataset):
    """Cases from the memory-mapped cache; opened lazily so each loader worker maps the file."""

    def __init__(self, images_path, labels_path, indices):
        self.images_path, self.labels_path = images_path, labels_path
        self.indices = np.asarray(indices)
        self.images = self.labels = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self.images is None:
            self.images = np.load(self.images_path, mmap_mode="r")
            self.labels = np.load(self.labels_path, mmap_mode="r")
        index = self.indices[item]
        image = torch.from_numpy(np.asarray(self.images[index], dtype=np.float32))
        bits = torch.from_numpy(np.asarray(self.labels[index]))
        label = torch.stack([(bits & 1) > 0, (bits & 2) > 0, (bits & 4) > 0]).float()
        return image, label


def augment(image, label):
    """GPU augmentation of the earlier notebooks: random flips on each axis, per-channel
    intensity scale (0.9 to 1.1) and shift (-0.1 to 0.1)."""
    for axis in (2, 3, 4):
        flip = torch.rand(image.shape[0], device=image.device) < 0.5
        image = torch.where(flip[:, None, None, None, None], image.flip(axis), image)
        label = torch.where(flip[:, None, None, None, None], label.flip(axis), label)
    scale = torch.empty(image.shape[0], image.shape[1], 1, 1, 1, device=image.device).uniform_(
        0.9, 1.1
    )
    shift = torch.empty_like(scale).uniform_(-0.1, 0.1)
    return image * scale + shift, label


# ---------------------------------------------------------------------------
# Models (all from scratch in PyTorch; every model returns a tuple, main output first)
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """3D UNet, channels 32-64-128-256, max pooling, deep supervision at decoder stages 2 and 3."""

    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3, self.b = (
            DoubleConv(4, 32),
            DoubleConv(32, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
        )
        self.pool = nn.MaxPool3d(2)
        self.u3, self.d3 = nn.ConvTranspose3d(256, 128, 2, 2), DoubleConv(256, 128)
        self.u2, self.d2 = nn.ConvTranspose3d(128, 64, 2, 2), DoubleConv(128, 64)
        self.u1, self.d1 = nn.ConvTranspose3d(64, 32, 2, 2), DoubleConv(64, 32)
        self.out, self.ds2, self.ds3 = (
            nn.Conv3d(32, 3, 1),
            nn.Conv3d(64, 3, 1),
            nn.Conv3d(128, 3, 1),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1), self.ds2(d2), self.ds3(d3)


class NNUNet3D(nn.Module):
    """nnU-Net style full-resolution network: five stages 32-64-128-256-320, stride-2 convolutions,
    deep supervision at four decoder stages."""

    def __init__(self, f=(32, 64, 128, 256, 320)):
        super().__init__()
        self.e1, self.e2 = DoubleConv(4, f[0]), DoubleConv(f[0], f[1], 2)
        self.e3, self.e4, self.b = (
            DoubleConv(f[1], f[2], 2),
            DoubleConv(f[2], f[3], 2),
            DoubleConv(f[3], f[4], 2),
        )
        self.u4, self.d4 = nn.ConvTranspose3d(f[4], f[3], 2, 2), DoubleConv(2 * f[3], f[3])
        self.u3, self.d3 = nn.ConvTranspose3d(f[3], f[2], 2, 2), DoubleConv(2 * f[2], f[2])
        self.u2, self.d2 = nn.ConvTranspose3d(f[2], f[1], 2, 2), DoubleConv(2 * f[1], f[1])
        self.u1, self.d1 = nn.ConvTranspose3d(f[1], f[0], 2, 2), DoubleConv(2 * f[0], f[0])
        self.out = nn.Conv3d(f[0], 3, 1)
        self.ds2, self.ds3, self.ds4 = (
            nn.Conv3d(f[1], 3, 1),
            nn.Conv3d(f[2], 3, 1),
            nn.Conv3d(f[3], 3, 1),
        )

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b = self.b(s4)
        d4 = self.d4(torch.cat([self.u4(b), s4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.out(d1), self.ds2(d2), self.ds3(d3), self.ds4(d4)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_convs):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(n_convs):
            layers += [
                nn.Conv3d(ch, out_ch, 3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.PReLU(out_ch),
            ]
            ch = out_ch
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.convs(x) + self.proj(x)


class VNet3D(nn.Module):
    """Residual VNet with PReLU, channels 16-32-64-128.

    Module names match Backend/neuroscan/inference/networks.py, so the exported state dict loads
    into the web app unchanged.
    """

    def __init__(self):
        super().__init__()
        self.e1 = ResBlock(4, 16, 1)
        self.d1 = nn.Conv3d(16, 16, 2, stride=2, bias=False)
        self.e2 = ResBlock(16, 32, 1)
        self.d2 = nn.Conv3d(32, 32, 2, stride=2, bias=False)
        self.e3 = ResBlock(32, 64, 2)
        self.d3 = nn.Conv3d(64, 64, 2, stride=2, bias=False)
        self.b = ResBlock(64, 128, 2)
        self.u3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.r3 = ResBlock(128, 64, 2)
        self.u2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.r2 = ResBlock(64, 32, 1)
        self.u1 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.r1 = ResBlock(32, 16, 1)
        self.out = nn.Conv3d(16, 3, 1)
        self.ds2 = nn.Conv3d(32, 3, 1)
        self.ds3 = nn.Conv3d(64, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        b = self.b(self.d3(e3))
        d3 = self.r3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.r2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.r1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1), self.ds2(d2), self.ds3(d3)


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer."""

    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ConvBlock(nn.Module):
    """Two Conv-InstanceNorm-ReLU layers (UNETR decoder)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNETR(nn.Module):
    """ViT-Base encoder (16^3 patches, 768 dim, 12 layers, 12 heads), skips from layers 3, 6, 9, 12."""

    def __init__(
        self, img_size=128, patch=16, dim=768, heads=12, layers=12, mlp_dim=3072, dropout=0.1
    ):
        super().__init__()
        self.grid = img_size // patch
        self.patch_embed = nn.Conv3d(4, dim, patch, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid**3, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.transformer = nn.ModuleList(
            [TransformerLayer(dim, heads, mlp_dim, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.use_checkpoint = False
        self.l4_main_up = nn.ConvTranspose3d(dim, 512, 2, 2)
        self.l4_skip_proj = ConvBlock(dim, 512)
        self.l4_skip_up = nn.ConvTranspose3d(512, 512, 2, 2)
        self.l4_conv = ConvBlock(1024, 256)
        self.l3_up = nn.ConvTranspose3d(256, 256, 2, 2)
        self.l3_skip_up = nn.Sequential(
            nn.ConvTranspose3d(dim, 256, 2, 2), nn.ConvTranspose3d(256, 256, 2, 2)
        )
        self.l3_conv = ConvBlock(512, 128)
        self.l2_up = nn.ConvTranspose3d(128, 128, 2, 2)
        self.l2_skip_up = nn.Sequential(
            nn.ConvTranspose3d(dim, 256, 2, 2),
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.ConvTranspose3d(128, 128, 2, 2),
        )
        self.l2_conv = ConvBlock(256, 64)
        self.l1_up = nn.ConvTranspose3d(64, 64, 2, 2)
        self.l1_skip_proj = ConvBlock(4, 64)
        self.l1_conv = ConvBlock(128, 32)
        self.out_conv = nn.Conv3d(32, 3, 1)

    def _volume(self, z):
        return z.transpose(1, 2).reshape(z.shape[0], z.shape[2], self.grid, self.grid, self.grid)

    def forward(self, x):
        tokens = self.pos_drop(self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed)
        skips = {}
        for index, layer in enumerate(self.transformer, start=1):
            if self.use_checkpoint and self.training:
                tokens = checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
            if index in (3, 6, 9, 12):
                skips[index] = tokens
        z3, z6, z9 = (self._volume(skips[i]) for i in (3, 6, 9))
        z12 = self._volume(self.norm(skips[12]))
        d4 = self.l4_conv(
            torch.cat([self.l4_main_up(z12), self.l4_skip_up(self.l4_skip_proj(z9))], 1)
        )
        d3 = self.l3_conv(torch.cat([self.l3_up(d4), self.l3_skip_up(z6)], 1))
        d2 = self.l2_conv(torch.cat([self.l2_up(d3), self.l2_skip_up(z3)], 1))
        d1 = self.l1_conv(torch.cat([self.l1_up(d2), self.l1_skip_proj(x)], 1))
        return (self.out_conv(d1),)


def _window_partition(x, w):
    b, d, h, wd, c = x.shape
    x = x.view(b, d // w, w, h // w, w, wd // w, w, c).permute(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(-1, w * w * w, c)


def _window_reverse(windows, w, b, d, h, wd):
    x = windows.view(b, d // w, h // w, wd // w, w, w, w, -1).permute(0, 1, 4, 2, 5, 3, 6, 7)
    return x.reshape(b, d, h, wd, -1)


class WindowAttention3D(nn.Module):
    """Window self-attention with relative position bias, computed with scaled_dot_product_attention."""

    def __init__(self, dim, window, heads):
        super().__init__()
        self.heads, self.window = heads, window
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window - 1) ** 3, heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid(*(torch.arange(window),) * 3, indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window - 1)
        index = rel[..., 0] * (2 * window - 1) ** 2 + rel[..., 1] * (2 * window - 1) + rel[..., 2]
        self.register_buffer("relative_position_index", index, persistent=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        windows, n, c = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(windows, n, 3, self.heads, c // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(n, n, -1)
            .permute(2, 0, 1)
        )
        bias = bias.unsqueeze(0)  # (1, heads, n, n)
        if mask is not None:  # (nW, n, n): windows are ordered batch-major
            n_windows = mask.shape[0]
            bias = (bias.unsqueeze(0) + mask[None, :, None]).expand(
                windows // n_windows, -1, -1, -1, -1
            )
            bias = bias.reshape(windows, self.heads, n, n)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.proj(out.transpose(1, 2).reshape(windows, n, c))


class SwinBlock3D(nn.Module):
    def __init__(self, dim, resolution, heads, window, shift):
        super().__init__()
        self.resolution, self.window, self.shift = resolution, window, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.padded = tuple(int(math.ceil(r / window)) * window for r in resolution)
        mask = None
        if shift:
            regions = torch.zeros(1, *self.padded, 1)
            spans = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            label = 0
            for sd in spans:
                for sh in spans:
                    for sw in spans:
                        regions[:, sd, sh, sw, :] = label
                        label += 1
            ids = _window_partition(regions, window)[..., 0]
            mask = (ids[:, None, :] - ids[:, :, None]).ne(0).float() * -100.0
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x):
        b, length, c = x.shape
        d, h, w = self.resolution
        dp, hp, wp = self.padded
        shortcut = x
        x = F.pad(self.norm1(x).view(b, d, h, w, c), (0, 0, 0, wp - w, 0, hp - h, 0, dp - d))
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift,) * 3, dims=(1, 2, 3))
        x = _window_reverse(
            self.attn(_window_partition(x, self.window), self.attn_mask), self.window, b, dp, hp, wp
        )
        if self.shift:
            x = torch.roll(x, shifts=(self.shift,) * 3, dims=(1, 2, 3))
        x = shortcut + x[:, :d, :h, :w, :].reshape(b, length, c)
        return x + self.mlp(self.norm2(x))


class PatchMerging3D(nn.Module):
    def __init__(self, resolution, dim):
        super().__init__()
        self.resolution = resolution
        self.norm = nn.LayerNorm(8 * dim)
        self.reduce = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x):
        d, h, w = self.resolution
        x = x.view(x.shape[0], d, h, w, -1)
        x = torch.cat(
            [x[:, i::2, j::2, k::2] for i in range(2) for j in range(2) for k in range(2)], -1
        )
        return self.reduce(self.norm(x.reshape(x.shape[0], -1, x.shape[-1])))


class SwinUNETR(nn.Module):
    """Swin UNETR: 2^3 patches, embed 48 doubling per stage, window 7, depths 2, heads 3-6-12-24.

    Compared with the earlier notebook it adds the full-resolution convolutional skip of the
    reference design, so the last decoder stage no longer upsamples without any skip.
    """

    def __init__(self, img_size=128, embed=48, window=7, depths=(2, 2, 2, 2), heads=(3, 6, 12, 24)):
        super().__init__()
        tokens = img_size // 2
        dims = [embed * 2**i for i in range(4)]
        self.resolutions = [(tokens // 2**i,) * 3 for i in range(4)]
        self.patch_embed = nn.Conv3d(4, embed, 2, 2)
        self.embed_norm = nn.LayerNorm(embed)
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        self.skip_norms = nn.ModuleList()
        for i in range(4):
            self.stages.append(
                nn.ModuleList(
                    [
                        SwinBlock3D(
                            dims[i],
                            self.resolutions[i],
                            heads[i],
                            window,
                            0 if j % 2 == 0 else window // 2,
                        )
                        for j in range(depths[i])
                    ]
                )
            )
            self.merges.append(
                PatchMerging3D(self.resolutions[i], dims[i]) if i < 3 else nn.Identity()
            )
            self.skip_norms.append(nn.LayerNorm(dims[i]))
        self.use_checkpoint = False
        self.e0 = DoubleConv(4, 32)
        self.up3, self.dc3 = nn.ConvTranspose3d(dims[3], dims[2], 2, 2), DoubleConv(
            2 * dims[2], dims[2]
        )
        self.up2, self.dc2 = nn.ConvTranspose3d(dims[2], dims[1], 2, 2), DoubleConv(
            2 * dims[1], dims[1]
        )
        self.up1, self.dc1 = nn.ConvTranspose3d(dims[1], dims[0], 2, 2), DoubleConv(
            2 * dims[0], dims[0]
        )
        self.up0, self.dc0 = nn.ConvTranspose3d(dims[0], 32, 2, 2), DoubleConv(64, 32)
        self.out = nn.Conv3d(32, 3, 1)

    def forward(self, x):
        t = self.embed_norm(self.patch_embed(x).flatten(2).transpose(1, 2))
        skips = []
        for i, (blocks, merge) in enumerate(zip(self.stages, self.merges)):
            for block in blocks:
                t = (
                    checkpoint(block, t, use_reentrant=False)
                    if self.use_checkpoint and self.training
                    else block(t)
                )
            d = self.resolutions[i][0]
            skips.append(self.skip_norms[i](t).transpose(1, 2).reshape(t.shape[0], -1, d, d, d))
            t = merge(t)
        d3 = self.dc3(torch.cat([self.up3(skips[3]), skips[2]], 1))
        d2 = self.dc2(torch.cat([self.up2(d3), skips[1]], 1))
        d1 = self.dc1(torch.cat([self.up1(d2), skips[0]], 1))
        d0 = self.dc0(torch.cat([self.up0(d1), self.e0(x)], 1))
        return (self.out(d0),)


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, 2)
        self.conv = ResConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], 1))


class TransBTS(nn.Module):
    """Residual CNN encoder 32-64-128-256-512, four-layer transformer bottleneck (8 heads, 512 dim)."""

    def __init__(self, img_size=128, dim=512, depth=4, heads=8):
        super().__init__()
        self.e1, self.e2 = ResConvBlock(4, 32), ResConvBlock(32, 64, 2)
        self.e3, self.e4 = ResConvBlock(64, 128, 2), ResConvBlock(128, 256, 2)
        self.b = ResConvBlock(256, dim, 2)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // 16) ** 3, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = nn.Sequential(
            *[TransformerLayer(dim, heads, 4 * dim) for _ in range(depth)]
        )
        self.trans_norm = nn.LayerNorm(dim)
        self.d4, self.d3 = UpBlock(dim, 256, 256), UpBlock(256, 128, 128)
        self.d2, self.d1 = UpBlock(128, 64, 64), UpBlock(64, 32, 32)
        self.out, self.ds2, self.ds3 = (
            nn.Conv3d(32, 3, 1),
            nn.Conv3d(64, 3, 1),
            nn.Conv3d(128, 3, 1),
        )

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b = self.b(s4)
        n, c, d, h, w = b.shape
        tokens = self.trans_norm(self.transformer(b.flatten(2).transpose(1, 2) + self.pos_embed))
        b = tokens.transpose(1, 2).reshape(n, c, d, h, w)
        d3 = self.d3(self.d4(b, s4), s3)
        d2 = self.d2(d3, s2)
        d1 = self.d1(d2, s1)
        return self.out(d1), self.ds2(d2), self.ds3(d3)


def build_model(name, img_size):
    builders = {
        "UNet": UNet3D,
        "nnU-Net": NNUNet3D,
        "VNet": VNet3D,
        "UNETR": lambda: UNETR(img_size),
        "Swin UNETR": lambda: SwinUNETR(img_size),
        "TransBTS": lambda: TransBTS(img_size),
    }
    model = builders[name]()
    if RECIPES[name]["checkpointing"]:
        model.use_checkpoint = True
    return model


# ---------------------------------------------------------------------------
# Loss, inference, post-processing, metrics
# ---------------------------------------------------------------------------


def region_loss(logits, target):
    """Soft Dice over the batch plus binary cross-entropy, per region, as in the paper."""
    logits = logits.float()
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3, 4)
    dice = 1 - (2 * (prob * target).sum(dims) + 1e-6) / (prob.sum(dims) + target.sum(dims) + 1e-6)
    return dice.mean() + F.binary_cross_entropy_with_logits(logits, target)


def supervised_loss(outputs, target, weights):
    loss = 0.0
    for output, weight in zip(outputs, weights):
        scaled = (
            target
            if output.shape[2:] == target.shape[2:]
            else F.interpolate(target, size=output.shape[2:], mode="nearest")
        )
        loss = loss + weight * region_loss(output, scaled)
    return loss


FLIP_SETS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


@torch.no_grad()
def predict_probs(model, x, tta):
    """Sigmoid probabilities of the main output, averaged over the 8 axis flips when tta is on.

    Returns the mean and the standard deviation across flips (zero without TTA).
    """
    model.eval()
    flips = FLIP_SETS if tta else [()]
    total = total_sq = None
    for dims in flips:
        inputs = x.flip(dims) if dims else x
        with torch.autocast("cuda", dtype=torch.float16):
            prob = torch.sigmoid(model(inputs)[0].float())
        prob = prob.flip(dims) if dims else prob
        total = prob if total is None else total + prob
        total_sq = prob**2 if total_sq is None else total_sq + prob**2
    mean = total / len(flips)
    std = (total_sq / len(flips) - mean**2).clamp_min(0).sqrt()
    return mean, std


def to_native(probs, shape):
    """Resample (3, S, S, S) probabilities to the original grid (trilinear on probabilities)."""
    return F.interpolate(
        probs[None].float(), size=tuple(shape), mode="trilinear", align_corners=False
    )[0]


def regions_from_probs(probs, et_min_voxels=0, threshold=0.5):
    """Nested regions with the BraTS label hierarchy: ET inside TC inside WT.

    An ET prediction smaller than et_min_voxels is removed, a standard BraTS post-processing
    step that helps cases without enhancing tumour (mostly lower-grade gliomas).
    """
    wt, tc, et = (probs > threshold).unbind(0)
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


def case_metrics(pred, ref, spacing):
    """Dice, HD95, sensitivity, specificity and volumes for the three regions of one case."""
    voxel_cm3 = float(np.prod(spacing)) / 1000
    row = {}
    for index, region in enumerate(REGIONS):
        p, r = pred[index], ref[index]
        tp = int((p & r).sum())
        fn = int(r.sum()) - tp
        fp = int(p.sum()) - tp
        tn = p.size - tp - fn - fp
        row[f"dice_{region}"] = dice(p, r)
        row[f"hd95_{region}"] = hd95(p, r, spacing)
        row[f"sensitivity_{region}"] = tp / (tp + fn) if tp + fn else 1.0
        row[f"specificity_{region}"] = tn / (tn + fp) if tn + fp else 1.0
        row[f"volume_pred_cm3_{region}"] = float(p.sum()) * voxel_cm3
        row[f"volume_ref_cm3_{region}"] = float(r.sum()) * voxel_cm3
    row["dice_mean"] = float(np.mean([row[f"dice_{r}"] for r in REGIONS]))
    row["hd95_mean"] = float(np.mean([row[f"hd95_{r}"] for r in REGIONS]))
    return row


def bootstrap_mean_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
