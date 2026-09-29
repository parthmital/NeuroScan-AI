"""NIfTI validation, loading, and 2D slice rendering."""

from pathlib import Path
from typing import Any

import cv2
import nibabel
import numpy as np

# 32 Mi voxels (for example 320^3). BraTS volumes are 240x240x155 (about 9 Mi).
MAX_VOXELS = 32 * 1024 * 1024
MAX_DIM = 1024

# Segmentation label -> RGB mask. Nested regions: WT covers TC covers ET,
# so channel R marks WT, G marks TC, and B marks ET.
_LABEL_COLOURS = {1: (255, 0, 0), 2: (255, 255, 0), 3: (255, 255, 255)}


class InvalidVolumeError(ValueError):
    """The file is not a usable 3D or 4D NIfTI volume."""


def _load(path: Path) -> Any:
    return nibabel.load(str(path))


def validate_volume(path: Path) -> tuple[int, ...]:
    """Read only the header and reject anything that is not a bounded 3D/4D NIfTI image."""
    try:
        image = _load(path)
    except Exception as exc:  # nibabel raises many exception types for malformed files
        raise InvalidVolumeError("File is not a readable NIfTI image") from exc
    if not isinstance(image, (nibabel.Nifti1Image, nibabel.Nifti2Image)):
        raise InvalidVolumeError("File is not a NIfTI image")
    shape = tuple(int(d) for d in image.shape)
    if len(shape) not in (3, 4) or any(d < 1 or d > MAX_DIM for d in shape):
        raise InvalidVolumeError("Volume must be 3D or 4D with sides between 1 and 1024")
    if int(np.prod(shape, dtype=np.int64)) > MAX_VOXELS:
        raise InvalidVolumeError("Volume is too large")
    return shape


def load_first_volume(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (float32 3D data, affine), taking the first frame of a 4D series."""
    image = _load(path)
    data = image.dataobj[..., 0] if len(image.shape) == 4 else image.dataobj
    return np.asarray(data, dtype=np.float32), np.asarray(image.affine)


def geometry(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Spatial matrix size and voxel spacing in mm, read from the header."""
    image = _load(path)
    return (
        tuple(int(d) for d in image.shape[:3]),
        tuple(float(z) for z in image.header.get_zooms()[:3]),
    )


def slice_count(path: Path) -> int:
    return int(_load(path).shape[2])


def _axial_slice(path: Path, index: int) -> np.ndarray:
    image = _load(path)
    shape = image.shape
    if not 0 <= index < shape[2]:
        raise IndexError(index)
    if len(shape) == 4:
        return np.asarray(image.dataobj[:, :, index, 0])
    return np.asarray(image.dataobj[:, :, index])


def _to_uint8(data: np.ndarray, upper: float) -> np.ndarray:
    lower = float(np.min(data))
    if upper <= lower:
        return np.zeros(data.shape, dtype=np.uint8)
    scaled: np.ndarray = (np.clip(data, lower, upper) - lower) / (upper - lower) * 255
    return scaled.astype(np.uint8)


def render_slice(path: Path, index: int) -> tuple[np.ndarray, int]:
    """Axial slice as a BGR image (99.5th percentile windowing) plus the total slice count."""
    data = _axial_slice(path, index)
    grey = _to_uint8(data, float(np.percentile(data, 99.5)))
    grey = cv2.rotate(grey, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), slice_count(path)


def render_mask_slice(path: Path, index: int) -> np.ndarray:
    """Axial segmentation slice as an RGB image using the nested WT/TC/ET channel encoding."""
    labels = np.rot90(_axial_slice(path, index), 1)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, colour in _LABEL_COLOURS.items():
        rgb[labels == label] = colour
    return rgb


def representative_slice(path: Path) -> np.ndarray:
    """Middle axial slice as min-max scaled BGR: the input for 2D detection and classification."""
    data, _ = load_first_volume(path)
    middle = data[:, :, data.shape[2] // 2]
    grey = _to_uint8(middle, float(np.max(middle)))
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def save_mask(mask: np.ndarray, affine: np.ndarray, path: Path) -> None:
    nibabel.save(nibabel.Nifti1Image(mask, affine), str(path))  # type: ignore[no-untyped-call]
