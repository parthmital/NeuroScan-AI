"""Segmentation inference: network parity with the notebook and mask post-processing."""

from pathlib import Path

import nibabel
import numpy as np
import pytest
import torch
import torch.nn as nn

from neuroscan.inference import volumes
from neuroscan.inference.networks import segresnet
from neuroscan.inference.processor import ET_MIN_VOXELS, SEGMENTATION_CHANNELS, BrainProcessor

SHAPE = (24, 24, 12)
SPACING = 2.0  # mm, so one voxel is 8 mm^3
TUMOUR = (slice(8, 12), slice(8, 12), slice(4, 8))


class BrightFlairNet(nn.Module):
    """Stand-in network: every region is predicted wherever the z-scored FLAIR is above 1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x[:, :1] > 1).float() * 20 - 10).repeat(1, 3, 1, 1, 1)


def _write_case(folder: Path) -> dict[str, Path]:
    data = np.zeros(SHAPE, dtype=np.float32)
    data[2:22, 2:22, 1:11] = 1.0  # brain; the zero border is cropped away
    data[TUMOUR] = 10.0
    affine = np.diag([SPACING, SPACING, SPACING, 1.0])
    paths = {}
    for modality in SEGMENTATION_CHANNELS:
        paths[modality] = folder / f"case_{modality}.nii"
        nibabel.save(nibabel.Nifti1Image(data, affine), str(paths[modality]))  # type: ignore[no-untyped-call]
    return paths


def test_segresnet_matches_the_notebook_checkpoint_layout() -> None:
    # Parameter and tensor counts of segresnet_seed42.pth (metrics/model_profiles.csv).
    state = segresnet().state_dict()
    assert len(state) == 85
    assert sum(t.numel() for t in state.values()) == 19_917_353


def test_segment_places_the_mask_in_flair_space_and_filters_small_et(tmp_path: Path) -> None:
    processor = BrainProcessor()
    processor.segmentation_model = BrightFlairNet().eval()
    mask_path = tmp_path / "mask.nii"

    result = processor.segment(_write_case(tmp_path), mask_path)

    mask, _ = volumes.load_first_volume(mask_path)
    tumour_voxels = 4 * 4 * 4
    assert tumour_voxels < ET_MIN_VOXELS  # so the predicted ET is removed as too small
    assert mask.shape == SHAPE
    assert set(np.unique(mask[TUMOUR])) == {2}  # TC, the deepest region left
    assert int((mask > 0).sum()) == tumour_voxels
    voxel_cm3 = SPACING**3 / 1000
    assert result.wt == pytest.approx(tumour_voxels * voxel_cm3)
    assert result.tc == pytest.approx(tumour_voxels * voxel_cm3)
    assert result.et == 0.0


def test_segment_rejects_sequences_on_different_grids(tmp_path: Path) -> None:
    processor = BrainProcessor()
    processor.segmentation_model = BrightFlairNet().eval()
    paths = _write_case(tmp_path)
    nibabel.save(  # type: ignore[no-untyped-call]
        nibabel.Nifti1Image(np.ones((10, 10, 10), dtype=np.float32), np.eye(4)), str(paths["t2"])
    )
    with pytest.raises(ValueError, match="co-registered"):
        processor.segment(paths, tmp_path / "mask.nii")
