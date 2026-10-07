"""Shared code for the unified brain tumour detection and classification notebooks.

The notebook writes this file to /kaggle/working/src so that the notebook and the
training workers (one process per GPU) import exactly the same definitions.
"""

from __future__ import annotations

import io
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
THUMB_SIZE = 64  # thumbnail side used by the normalised cross-correlation duplicate detector

# Every backbone is built from one registry for both tasks, so a model name always means the
# same implementation and the same pretrained checkpoint.
MODEL_SPECS = {
    "ResNet50": {"family": "Classical CNN", "input_size": 224, "micro_batch": 64},
    "VGG16": {"family": "Classical CNN", "input_size": 224, "micro_batch": 64},
    "DenseNet121": {"family": "Classical CNN", "input_size": 224, "micro_batch": 64},
    "EfficientNetB0": {"family": "Efficient CNN", "input_size": 224, "micro_batch": 64},
    "MobileNetV3-Large": {"family": "Efficient CNN", "input_size": 224, "micro_batch": 64},
    "InceptionV3": {"family": "Efficient CNN", "input_size": 299, "micro_batch": 64},
    "ViT-B/16": {"family": "Transformer", "input_size": 224, "micro_batch": 64},
    "Swin-Base": {"family": "Transformer", "input_size": 224, "micro_batch": 32},
    "CoAtNet-0": {"family": "Hybrid", "input_size": 224, "micro_batch": 64},
}
TOKEN_MODELS = {"ViT-B/16", "Swin-Base"}  # no channels_last benefit for these


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def run_id(model, fold, mode):
    return f"{slug(model)}_fold{fold}" + ("_background" if mode == "background" else "")


# ---------------------------------------------------------------------------
# Image loading, cropping, hashing
# ---------------------------------------------------------------------------


def crop_to_brain(bgr):
    """The crop the web app applies before inference (Backend/neuroscan/inference/processor.py).

    Training on the same crop removes the train/serve mismatch of the earlier notebooks and
    removes most of the border and framing that differs between source collections.
    """
    grey = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, thresh = cv2.threshold(grey, 45, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        bgr = bgr[y : y + h, x : x + w]
    return bgr


def head_mask(gray):
    """Otsu threshold, closing, largest component, hole filling (as in the revision audit)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    try:
        foreground = gray > threshold_otsu(gray)
    except ValueError:
        return np.ones_like(gray, dtype=bool)
    foreground = ndimage.binary_closing(foreground, iterations=2)
    labels, count = ndimage.label(foreground)
    if count == 0:
        return np.ones_like(gray, dtype=bool)
    sizes = ndimage.sum(foreground, labels, range(1, count + 1))
    return ndimage.binary_fill_holes(labels == int(np.argmax(sizes)) + 1)


def describe_image(args):
    """Decode one file once and return everything later stages need.

    The perceptual hash and thumbnail are computed on the original image (as in the paper);
    the cached training image is the brain crop resized to cache_size x cache_size RGB.
    """
    import imagehash
    from PIL import Image

    path, cache_size = args
    empty = np.zeros((cache_size, cache_size, 3), np.uint8)
    try:
        data = Path(path).read_bytes()
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            rgb = img.convert("RGB")
        bits = np.packbits(imagehash.phash(rgb).hash.flatten().astype(np.uint8))
        hash_value = int.from_bytes(bits.tobytes(), "big")
        thumb = np.asarray(
            rgb.convert("L").resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR), dtype=np.uint8
        )
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:  # formats OpenCV cannot decode: fall back to the PIL decode
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        cropped = crop_to_brain(bgr)
        crop_fraction = cropped.shape[0] * cropped.shape[1] / (bgr.shape[0] * bgr.shape[1])
        cached = cv2.cvtColor(
            cv2.resize(cropped, (cache_size, cache_size), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        mask = head_mask(cv2.cvtColor(cached, cv2.COLOR_RGB2GRAY))
        return {
            "ok": True,
            "width": width,
            "height": height,
            "hash": hash_value,
            "thumb": thumb,
            "image": cached,
            "mask": mask.astype(np.uint8),
            "crop_fraction": crop_fraction,
        }
    except Exception as exc:  # unreadable files are reported and excluded, never silently kept
        print(f"Unreadable image {path}: {exc}", flush=True)
        return {
            "ok": False,
            "width": 0,
            "height": 0,
            "hash": 0,
            "thumb": np.zeros((THUMB_SIZE, THUMB_SIZE), np.uint8),
            "image": empty,
            "mask": np.zeros((cache_size, cache_size), np.uint8),
            "crop_fraction": 0.0,
        }


_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def popcount64(values):
    values = np.ascontiguousarray(values, dtype=np.uint64)
    return _POPCOUNT[values.view(np.uint8)].reshape(*values.shape, 8).sum(axis=-1, dtype=np.uint8)


def hamming_matrix(hashes_a, hashes_b, chunk=512):
    """Pairwise Hamming distances between two lists of 64-bit hashes."""
    hashes_a = np.asarray(hashes_a, dtype=np.uint64)
    hashes_b = np.asarray(hashes_b, dtype=np.uint64)
    out = np.empty((len(hashes_a), len(hashes_b)), dtype=np.uint8)
    for start in range(0, len(hashes_a), chunk):
        out[start : start + chunk] = popcount64(
            hashes_a[start : start + chunk, None] ^ hashes_b[None, :]
        )
    return out


def ncc_vectors(thumbs):
    vectors = thumbs.reshape(len(thumbs), -1).astype(np.float32)
    vectors -= vectors.mean(axis=1, keepdims=True)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors


def greedy_dedup(distance, valid, tolerance, labels):
    """Label-aware greedy deduplication in a fixed order.

    An image is dropped when a previously kept image of the same label lies within the
    tolerance, so the outcome no longer depends on the order in which classes are read.
    """
    kept_by_label = {}
    keep = np.zeros(len(distance), dtype=bool)
    nearest = np.full(len(distance), -1)
    for index in range(len(distance)):
        if not valid[index]:
            continue
        pool = kept_by_label.setdefault(labels[index], [])
        if pool:
            row = distance[index, pool]
            best = int(row.argmin())
            if row[best] <= tolerance:
                nearest[index] = pool[best]
                continue
        pool.append(index)
        keep[index] = True
    return keep, nearest


def connected_groups(adjacency_pairs, count):
    """Union-find over index pairs; returns a group id per index."""
    parent = np.arange(count)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in adjacency_pairs:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    return np.array([find(i) for i in range(count)])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_model(name, num_outputs, pretrained=True):
    tv = torchvision.models
    if name == "ResNet50":
        model = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
    elif name == "VGG16":
        model = tv.vgg16(weights=tv.VGG16_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_outputs)
    elif name == "DenseNet121":
        model = tv.densenet121(weights=tv.DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_outputs)
    elif name == "EfficientNetB0":
        model = tv.efficientnet_b0(
            weights=tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_outputs)
    elif name == "MobileNetV3-Large":
        model = tv.mobilenet_v3_large(
            weights=tv.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        )
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_outputs)
    elif name == "InceptionV3":
        # transform_input is fixed so a skeleton built without weights matches the trained model.
        model = tv.inception_v3(
            weights=tv.Inception_V3_Weights.DEFAULT if pretrained else None,
            transform_input=True,
            init_weights=False,
        )
        # The auxiliary classifier is removed for every task, so no model receives an extra
        # loss term and no unused parameters are counted.
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
    elif name == "ViT-B/16":
        model = tv.vit_b_16(weights=tv.ViT_B_16_Weights.DEFAULT if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_outputs)
    elif name == "Swin-Base":
        model = tv.swin_b(weights=tv.Swin_B_Weights.DEFAULT if pretrained else None)
        model.head = nn.Linear(model.head.in_features, num_outputs)
    elif name == "CoAtNet-0":
        import timm  # CoAtNet has no torchvision implementation

        model = timm.create_model(
            "coatnet_0_rw_224", pretrained=pretrained, num_classes=num_outputs
        )
    else:
        raise KeyError(f"Unknown model {name!r}; choose from {list(MODEL_SPECS)}")
    return model


def gradcam_layer(model, name):
    """Last spatial or token stage of each backbone."""
    if name == "ResNet50":
        return model.layer4[-1]
    if name == "VGG16":
        return model.features[28]  # last convolution, not the final max-pooling layer
    if name == "DenseNet121":
        return model.features.norm5
    if name in ("EfficientNetB0", "MobileNetV3-Large"):
        return model.features[-1]
    if name == "InceptionV3":
        return model.Mixed_7c
    if name == "ViT-B/16":
        return model.encoder.layers[-1].ln_1
    if name == "Swin-Base":
        return model.norm  # final norm, (B, 7, 7, C) channels-last grid
    if name == "CoAtNet-0":
        return model.stages[-1]
    raise KeyError(name)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_weights(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model


# ---------------------------------------------------------------------------
# GPU preprocessing and augmentation
# ---------------------------------------------------------------------------


class GpuPreprocessor:
    """Resize, augment and normalise uint8 batches on the GPU.

    Augmentation matches the earlier notebooks (horizontal flip p=0.5, rotation up to 10
    degrees, brightness and contrast jitter of 20 percent) but runs on the GPU, so the four
    Kaggle CPU cores are not the bottleneck. ``mode`` selects the input variant:
    "full", "background" (head region set to zero) or "head" (background set to zero).
    """

    def __init__(self, input_size, device, channels_last=False):
        self.size = input_size
        self.device = device
        self.channels_last = channels_last
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, images, masks=None, mode="full", train=False):
        x = images.to(self.device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        if mode != "full":
            head = masks.to(self.device, non_blocking=True).float().unsqueeze(1)
            x = x * (1 - head) if mode == "background" else x * head
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", antialias=True)
        if train:
            batch = x.shape[0]
            flip = torch.rand(batch, device=self.device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(-1), x)
            angle = (torch.rand(batch, device=self.device) * 2 - 1) * math.radians(10)
            cos, sin = torch.cos(angle), torch.sin(angle)
            zeros = torch.zeros_like(cos)
            theta = torch.stack(
                [torch.stack([cos, -sin, zeros], 1), torch.stack([sin, cos, zeros], 1)], 1
            )
            grid = F.affine_grid(theta, list(x.shape), align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            brightness = torch.empty(batch, 1, 1, 1, device=self.device).uniform_(0.8, 1.2)
            x = (x * brightness).clamp_(0, 1)
            contrast = torch.empty(batch, 1, 1, 1, device=self.device).uniform_(0.8, 1.2)
            grey_mean = (
                (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])
                .mean(dim=(1, 2))
                .view(-1, 1, 1, 1)
            )
            x = ((x - grey_mean) * contrast + grey_mean).clamp_(0, 1)
        x = (x - self.mean) / self.std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x


def denormalise(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.detach().float().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_logits(model, prep, images, masks=None, mode="full", batch_size=128):
    """Logits for a uint8 array (N, S, S, 3); masks are needed for the masked modes."""
    model.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(images[start : start + batch_size]))
        batch_masks = (
            None
            if masks is None
            else torch.from_numpy(np.ascontiguousarray(masks[start : start + batch_size]))
        )
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(prep(batch, batch_masks, mode=mode))
        outputs.append(logits.float().cpu())
    return torch.cat(outputs).numpy()


def logits_to_probs(logits, temperature=1.0):
    """Class probabilities (N, K); a single-logit binary model gives two columns."""
    logits = np.asarray(logits, dtype=np.float64) / temperature
    if logits.ndim == 1 or logits.shape[1] == 1:
        positive = 1 / (1 + np.exp(-logits.reshape(-1)))
        return np.stack([1 - positive, positive], axis=1)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels):
    """Temperature that minimises validation negative log-likelihood (Guo et al., 2017)."""
    from scipy.optimize import minimize_scalar

    labels = np.asarray(labels, dtype=int)

    def nll(log_t):
        probs = logits_to_probs(logits, math.exp(log_t))
        return -np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)))

    result = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
    return float(math.exp(result.x))


# ---------------------------------------------------------------------------
# Metrics and statistics
# ---------------------------------------------------------------------------


def calibration_table(y_true, probs, n_bins=15):
    import pandas as pd

    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        rows.append(
            {
                "bin_low": low,
                "bin_high": high,
                "count": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
                "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, probs, n_bins=15):
    table = calibration_table(y_true, probs, n_bins).dropna()
    weights = table["count"] / len(y_true)
    return float((weights * (table["accuracy"] - table["mean_confidence"]).abs()).sum())


def predictions_from_probs(probs, threshold=None):
    """Arg-max, or for binary tasks the positive-class threshold when one is given."""
    if probs.shape[1] == 2 and threshold is not None:
        return (probs[:, 1] >= threshold).astype(int)
    return probs.argmax(axis=1)


def per_class_metrics(y_true, y_pred, probs, class_names):
    import pandas as pd
    from sklearn import metrics as skm

    k = len(class_names)
    cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(k)))
    rows = []
    for index, name in enumerate(class_names):
        tp = cm[index, index]
        fn = cm[index].sum() - tp
        fp = cm[:, index].sum() - tp
        tn = cm.sum() - tp - fn - fp
        positive = (y_true == index).astype(int)
        rows.append(
            {
                "class": name,
                "support": int(cm[index].sum()),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                "roc_auc": safe_score(skm.roc_auc_score, positive, probs[:, index]),
                "average_precision": safe_score(
                    skm.average_precision_score, positive, probs[:, index]
                ),
            }
        )
    return pd.DataFrame(rows)


def safe_score(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def classification_metrics(y_true, probs, threshold=None):
    """Headline metrics of one prediction set; ``threshold`` applies to binary tasks only."""
    from sklearn import metrics as skm

    y_true = np.asarray(y_true, dtype=int)
    k = probs.shape[1]
    y_pred = predictions_from_probs(probs, threshold)
    labels = list(range(k))
    result = {
        "n": int(len(y_true)),
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
        "precision_macro": skm.precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": skm.recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_macro": skm.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "mcc": skm.matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": skm.cohen_kappa_score(y_true, y_pred),
        "log_loss": skm.log_loss(y_true, np.clip(probs, 1e-12, 1), labels=labels),
        "brier": float(np.mean(np.sum((probs - np.eye(k)[y_true]) ** 2, axis=1))),
        "ece": expected_calibration_error(y_true, probs),
    }
    if k == 2:
        cm = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        result.update(
            {
                "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
                "specificity": tn / (tn + fp) if tn + fp else 0.0,
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "roc_auc": safe_score(skm.roc_auc_score, y_true, probs[:, 1]),
                "pr_auc": safe_score(skm.average_precision_score, y_true, probs[:, 1]),
            }
        )
    else:
        result.update(
            {
                "roc_auc": safe_score(
                    skm.roc_auc_score, y_true, probs, multi_class="ovr", average="macro"
                ),
                "pr_auc": safe_score(
                    skm.average_precision_score, np.eye(k)[y_true], probs, average="macro"
                ),
            }
        )
    return {key: float(value) for key, value in result.items()}


def youden_threshold(y_true, positive_probs):
    from sklearn import metrics as skm

    fpr, tpr, thresholds = skm.roc_curve(y_true, positive_probs)
    best = int(np.argmax(tpr - fpr))
    return float(min(thresholds[best], 1.0))


def bootstrap_ci(y_true, probs, metric_fn, n_boot=2000, seed=0, threshold=None):
    """Percentile 95 percent interval of a metric over test-set resamples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        if len(np.unique(y_true[index])) < probs.shape[1]:
            continue  # a resample missing a class has no defined AUC
        values.append(metric_fn(y_true[index], probs[index], threshold))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def mcnemar_p(correct_a, correct_b):
    """Exact McNemar test on paired correctness vectors."""
    from scipy.stats import binomtest

    only_a = int(np.sum(correct_a & ~correct_b))
    only_b = int(np.sum(~correct_a & correct_b))
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


# ---------------------------------------------------------------------------
# Attention maps
# ---------------------------------------------------------------------------


def to_nchw(tensor):
    if tensor.dim() == 3:  # transformer tokens (batch, tokens, channels)
        tokens = tensor.shape[1]
        side = int(round(math.sqrt(tokens)))
        if side * side != tokens:  # drop the class token
            tensor = tensor[:, 1:, :]
            side = int(round(math.sqrt(tensor.shape[1])))
        return tensor.reshape(tensor.shape[0], side, side, tensor.shape[2]).permute(0, 3, 1, 2)
    if (
        tensor.dim() == 4
        and tensor.shape[1] == tensor.shape[2]
        and tensor.shape[3] != tensor.shape[1]
    ):
        return tensor.permute(0, 3, 1, 2)  # channels-last grid
    return tensor


def normalise_maps(maps):
    maps = maps - maps.flatten(1).min(dim=1).values[:, None, None]
    peak = maps.flatten(1).max(dim=1).values[:, None, None]
    return torch.where(peak > 0, maps / peak.clamp_min(1e-12), maps)


def gradcam_batch(model, layer, inputs, binary):
    """Grad-CAM of the predicted class for a batch (samples are independent in eval mode)."""
    store = {}

    def hook(module, args, output):
        output = output[0] if isinstance(output, (tuple, list)) else output
        store["activation"] = output
        output.register_hook(lambda grad: store.__setitem__("gradient", grad))

    handle = layer.register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(inputs).float()
            if binary:
                predicted = (logits[:, 0] > 0).long()
                score = torch.where(predicted == 1, logits[:, 0], -logits[:, 0])
            else:
                predicted = logits.argmax(dim=1)
                score = logits.gather(1, predicted[:, None])[:, 0]
            score.sum().backward()
    finally:
        handle.remove()
    activation = to_nchw(store["activation"].detach().float())
    gradient = to_nchw(store["gradient"].detach().float())
    cam = F.relu((gradient.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    model.zero_grad(set_to_none=True)
    return normalise_maps(cam), predicted


def vit_rollout(model, inputs):
    """Attention rollout (Abnar and Zuidema, 2020) for the torchvision ViT."""
    attentions, patched = [], []
    for block in model.encoder.layers:
        module, original = block.self_attention, block.self_attention.forward

        def forward(query, key, value, _original=original, **kwargs):
            kwargs["need_weights"], kwargs["average_attn_weights"] = True, False
            output, weights = _original(query, key, value, **kwargs)
            attentions.append(weights.detach().float())
            return output, weights

        module.forward = forward
        patched.append(module)
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for module in patched:
            del module.forward
    tokens = attentions[0].shape[-1]
    identity = torch.eye(tokens, device=inputs.device)
    rollout = identity.expand(inputs.shape[0], tokens, tokens).clone()
    for attention in attentions:
        attention = 0.5 * attention.mean(dim=1) + 0.5 * identity
        rollout = (attention / attention.sum(dim=-1, keepdim=True)) @ rollout
    side = int(math.sqrt(tokens - 1))
    maps = rollout[:, 0, 1:].reshape(-1, 1, side, side)
    maps = F.interpolate(maps, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return normalise_maps(maps)


@torch.no_grad()
def occlusion_maps(model, inputs, binary, patch=32, stride=32):
    """Drop in predicted-class probability when each patch is replaced by the image mean.

    Model-agnostic, so unlike Grad-CAM it is directly comparable between convolutional and
    attention backbones.
    """
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        base = model(inputs).float()
    if binary:
        predicted = (base[:, 0] > 0).long()
        base_score = torch.sigmoid(torch.where(predicted == 1, base[:, 0], -base[:, 0]))
    else:
        predicted = base.argmax(dim=1)
        base_score = base.softmax(dim=1).gather(1, predicted[:, None])[:, 0]
    size = inputs.shape[-1]
    positions = [
        (y, x)
        for y in range(0, size - patch + 1, stride)
        for x in range(0, size - patch + 1, stride)
    ]
    maps = torch.zeros(inputs.shape[0], size, size, device=inputs.device)
    counts = torch.zeros(size, size, device=inputs.device)
    fill = inputs.mean(dim=(2, 3), keepdim=True)
    for y, x in positions:
        occluded = inputs.clone()
        occluded[:, :, y : y + patch, x : x + patch] = fill
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(occluded).float()
        if binary:
            score = torch.sigmoid(torch.where(predicted == 1, out[:, 0], -out[:, 0]))
        else:
            score = out.softmax(dim=1).gather(1, predicted[:, None])[:, 0]
        maps[:, y : y + patch, x : x + patch] += (base_score - score).clamp_min(0)[:, None, None]
        counts[y : y + patch, x : x + patch] += 1
    return normalise_maps(maps / counts.clamp_min(1)), predicted


def outside_fraction(maps, masks):
    """Share of map energy outside the head mask, per image."""
    total = maps.flatten(1).sum(dim=1)
    outside = (maps * (~masks)).flatten(1).sum(dim=1)
    return torch.where(total > 0, outside / total, torch.full_like(total, float("nan")))


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------


def gflops(model, input_size):
    from torch.utils.flop_counter import FlopCounterMode

    model = model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, 3, input_size, input_size))
    return counter.get_total_flops() / 1e9


@torch.no_grad()
def time_passes(model, inputs, warmup, iterations):
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.float16):
            model(inputs)
    torch.cuda.synchronize(inputs.device)
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            model(inputs)
        torch.cuda.synchronize(inputs.device)
        timings.append(time.perf_counter() - start)
    return np.array(timings)
