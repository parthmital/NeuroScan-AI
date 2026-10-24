"""Loads the trained models and runs detection, classification, and segmentation."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from monai.inferers.utils import sliding_window_inference
from PIL import Image
from torchvision import transforms

from neuroscan.inference import volumes
from neuroscan.inference.networks import efficientnet_b0_head, segresnet

logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)
CLASSES: list[str] = ["glioma", "meningioma", "pituitary"]
# Order of input channels the segmentation network was trained with.
SEGMENTATION_CHANNELS = ("flair", "t1", "t1ce", "t2")
# Sliding-window inference as in the segmentation notebook: 128^3 windows, half overlap.
SEG_WINDOW = 128
SEG_OVERLAP = 0.5
# Smallest ET prediction kept, tuned on the validation cases for SegResNet
# (Materials/Notebooks/segmentation/metrics/runs/segresnet_seed42/summary.json).
ET_MIN_VOXELS = 800

_NORMALISE = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@dataclass(frozen=True)
class Classification:
    label: str
    confidence: float
    # Probability of every class, keyed by class name.
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Volumes:
    """Tumour sub-region volumes in cm^3 (voxel count x FLAIR voxel volume in mm^3 / 1000)."""

    wt: float
    tc: float
    et: float


def _load_state_dict(model: nn.Module, path: Path, device: torch.device) -> None:
    # weights_only=True refuses pickled code, so a tampered weights file cannot execute on load.
    state_dict = torch.load(path, map_location=device, weights_only=True)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()


class BrainProcessor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_model: nn.Module | None = None
        self.classification_model: nn.Module | None = None
        self.segmentation_model: nn.Module | None = None

    @property
    def ready(self) -> bool:
        return self.detection_model is not None and self.classification_model is not None

    def load_models(self, detection: Path, classification: Path, segmentation: Path) -> None:
        """Load models independently; a missing segmentation model only skips segmentation."""
        specs: tuple[tuple[str, Callable[[], nn.Module], Path], ...] = (
            ("detection_model", lambda: efficientnet_b0_head(1), detection),
            (
                "classification_model",
                lambda: efficientnet_b0_head(len(CLASSES)),
                classification,
            ),
            ("segmentation_model", segresnet, segmentation),
        )
        for attribute, factory, path in specs:
            try:
                model = factory().to(self.device)
                _load_state_dict(model, path, self.device)
                setattr(self, attribute, model)
                logger.info("Model loaded", extra={"fields": {"model": attribute}})
            except (OSError, RuntimeError, KeyError, ValueError):
                logger.exception("Model failed to load", extra={"fields": {"model": attribute}})

    def _preprocess_2d(self, image: np.ndarray) -> torch.Tensor:
        grey = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, thresh = cv2.threshold(grey, 45, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            image = image[y : y + h, x : x + w]
        cropped = cv2.resize(image, IMG_SIZE)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        tensor: torch.Tensor = _NORMALISE(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        return tensor

    def detect(self, image: np.ndarray) -> float:
        """Probability that the slice contains a tumour."""
        if self.detection_model is None:
            raise RuntimeError("Detection model not loaded")
        with torch.no_grad():
            return float(torch.sigmoid(self.detection_model(self._preprocess_2d(image))).item())

    def classify(self, image: np.ndarray) -> Classification:
        if self.classification_model is None:
            raise RuntimeError("Classification model not loaded")
        with torch.no_grad():
            logits = self.classification_model(self._preprocess_2d(image))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        index = int(np.argmax(probs))
        return Classification(
            label=CLASSES[index],
            confidence=float(probs[index]),
            probabilities={name: float(p) for name, p in zip(CLASSES, probs, strict=True)},
        )

    @staticmethod
    def _preprocess_3d(modality_paths: Mapping[str, Path]) -> tuple[np.ndarray, tuple[slice, ...]]:
        """Crop the co-registered sequences to the brain and z-score each over its brain voxels.

        Same as the notebook's `preprocess_case`: no resampling, so the model sees the native grid
        (1 mm in BraTS). Returns the (4, D, H, W) image and the crop box in the native grid.
        """
        data = [volumes.load_first_volume(modality_paths[m])[0] for m in SEGMENTATION_CHANNELS]
        if len({d.shape for d in data}) != 1:
            raise ValueError("The four sequences must share one voxel grid (co-registered)")
        brain = np.any(np.stack(data) != 0, axis=0)
        coords = np.argwhere(brain) if brain.any() else np.argwhere(np.ones_like(brain))
        low, high = coords.min(0), coords.max(0) + 1
        box = tuple(slice(int(a), int(b)) for a, b in zip(low, high, strict=True))
        cropped = []
        for volume in data:
            volume = volume[box].copy()
            inside = volume != 0
            if inside.any():
                volume[inside] = (volume[inside] - float(volume[inside].mean())) / (
                    float(volume[inside].std()) + 1e-8
                )
            cropped.append(volume)
        return np.stack(cropped).astype(np.float32), box

    @property
    def can_segment(self) -> bool:
        return self.segmentation_model is not None

    def segment(self, modality_paths: Mapping[str, Path], mask_path: Path) -> Volumes:
        """Segment WT/TC/ET, save a label mask (1=WT, 2=TC, 3=ET) in FLAIR space, return volumes."""
        model = self.segmentation_model
        if model is None:
            raise RuntimeError("Segmentation model not loaded")
        image, box = self._preprocess_3d(modality_paths)

        def predictor(x: torch.Tensor) -> torch.Tensor:
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                return model(x).float()  # type: ignore[no-any-return]

        with torch.no_grad():
            logits = sliding_window_inference(
                torch.from_numpy(image)[None].to(self.device),
                (SEG_WINDOW,) * 3,
                2,
                predictor,
                overlap=SEG_OVERLAP,
                mode="gaussian",
            )
            wt, tc, et = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy()
        # BraTS label hierarchy (ET inside TC inside WT) after removing a small ET prediction.
        if int(et.sum()) < ET_MIN_VOXELS:
            et = np.zeros_like(et)
        tc |= et
        wt |= tc

        reference, affine = volumes.load_first_volume(modality_paths["flair"])
        labels = np.zeros(reference.shape, dtype=np.uint8)
        for label, region in enumerate((wt, tc, et), start=1):
            labels[box][region] = label
        volumes.save_mask(labels, affine, mask_path)
        voxel_cm3 = float(np.prod(volumes.geometry(modality_paths["flair"])[1])) / 1e3
        return Volumes(*(float(region.sum()) * voxel_cm3 for region in (wt, tc, et)))
