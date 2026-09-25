"""Loads the trained models and runs detection, classification, and segmentation."""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from neuroscan.inference import volumes
from neuroscan.inference.networks import VNet3D, efficientnet_b0_head

logger = logging.getLogger(__name__)

IMG_SIZE = (224, 224)
SEG_SIZE = (128, 128, 128)
CLASSES: list[str] = ["glioma", "meningioma", "pituitary"]
# Order of input channels the segmentation network was trained with.
SEGMENTATION_CHANNELS = ("flair", "t1", "t1ce", "t2")

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


@dataclass(frozen=True)
class Volumes:
    """Tumour sub-region volumes in cm^3 (voxel count / 1000 on the resampled 128^3 grid)."""

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
        self.segmentation_model: VNet3D | None = None

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
            ("segmentation_model", VNet3D, segmentation),
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
        return Classification(label=CLASSES[index], confidence=float(probs[index]))

    @staticmethod
    def _preprocess_3d(path: Path) -> np.ndarray:
        data, _ = volumes.load_first_volume(path)
        brain = data != 0
        if brain.any():
            data[brain] = (data[brain] - float(data[brain].mean())) / (
                float(data[brain].std()) + 1e-8
            )
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(data)[None, None],
            size=SEG_SIZE,
            mode="trilinear",
            align_corners=False,
        )
        return resized.squeeze().numpy().astype(np.float32)

    @property
    def can_segment(self) -> bool:
        return self.segmentation_model is not None

    def segment(self, modality_paths: Mapping[str, Path], mask_path: Path) -> Volumes:
        """Segment WT/TC/ET, save a label mask (1=WT, 2=TC, 3=ET) in FLAIR space, return volumes."""
        if self.segmentation_model is None:
            raise RuntimeError("Segmentation model not loaded")
        stacked = np.stack([self._preprocess_3d(modality_paths[m]) for m in SEGMENTATION_CHANNELS])
        with torch.no_grad():
            output = self.segmentation_model(torch.from_numpy(stacked)[None].to(self.device))[0]
            regions = torch.sigmoid(output).cpu().numpy()[0] > 0.5
        wt, tc, et = (float(regions[i].sum() / 1e3) for i in range(3))

        labels = np.zeros(SEG_SIZE, dtype=np.uint8)
        for label, region in enumerate(regions, start=1):
            labels[region] = label
        reference, affine = volumes.load_first_volume(modality_paths["flair"])
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(labels.astype(np.float32))[None, None],
            size=reference.shape,
            mode="nearest",
        )
        volumes.save_mask(resized.squeeze().numpy().astype(np.uint8), affine, mask_path)
        return Volumes(wt=wt, tc=tc, et=et)
