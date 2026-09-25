"""Scan file storage. Client file names never become paths; every path stays inside the root."""

import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

from neuroscan.inference import volumes
from neuroscan.models import Modality

logger = logging.getLogger(__name__)

SCAN_ID_PATTERN = r"^scan-[0-9a-f]{6,32}$"
_SCAN_ID_RE = re.compile(SCAN_ID_PATTERN)
_CHUNK = 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"
MASK_FILE = "segmentation.nii"

# Checked in order: "T1CE" must win over "T1", and "FLAIR" over everything.
_MODALITY_MARKERS = (
    ("FLAIR", Modality.FLAIR),
    ("T1CE", Modality.T1CE),
    ("T1C", Modality.T1CE),
    ("T2", Modality.T2),
    ("T1", Modality.T1),
)


class UploadRejected(ValueError):
    """The upload cannot be accepted; the message is safe to show to the user."""


class UploadTooLarge(UploadRejected):
    pass


def detect_modality(filename: str) -> Optional[Modality]:
    upper = filename.upper()
    return next((modality for marker, modality in _MODALITY_MARKERS if marker in upper), None)


class ScanStorage:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()

    @staticmethod
    def new_scan_id() -> str:
        return f"scan-{uuid.uuid4().hex}"

    def scan_dir(self, scan_id: str) -> Path:
        if not _SCAN_ID_RE.fullmatch(scan_id):
            raise ValueError("Invalid scan id")
        return self.root / scan_id

    def mask_path(self, scan_id: str) -> Path:
        return self.scan_dir(scan_id) / MASK_FILE

    def has_mask(self, scan_id: str) -> bool:
        return self.mask_path(scan_id).is_file()

    def resolve(self, relative: str) -> Path:
        """Resolve a stored relative path (legacy rows use Windows separators) inside the root."""
        path = (self.root / relative.replace("\\", "/")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Stored path escapes the upload directory")
        return path

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def save_modality(
        self, scan_id: str, modality: Modality, source: BinaryIO, max_bytes: int
    ) -> Path:
        """Stream an upload to <scan>/<modality>.nii[.gz], enforcing size and NIfTI validity."""
        directory = self.scan_dir(scan_id)
        directory.mkdir(parents=True, exist_ok=True)
        partial = directory / f"{modality.value}.upload"
        written = 0
        with partial.open("wb") as target:
            while chunk := source.read(_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    target.close()
                    partial.unlink(missing_ok=True)
                    raise UploadTooLarge(
                        f"Each file must be at most {max_bytes // (1024 * 1024)} MB"
                    )
                target.write(chunk)
        with partial.open("rb") as check:
            compressed = check.read(2) == _GZIP_MAGIC
        final = directory / f"{modality.value}{'.nii.gz' if compressed else '.nii'}"
        partial.replace(final)
        try:
            volumes.validate_volume(final)
        except volumes.InvalidVolumeError as exc:
            final.unlink(missing_ok=True)
            raise UploadRejected(f"{modality.value.upper()}: {exc}") from exc
        return final

    def modality_paths(self, file_paths: dict[str, str]) -> dict[str, Path]:
        return {modality: self.resolve(relative) for modality, relative in file_paths.items()}

    def delete_scan(self, scan_id: str) -> None:
        directory = self.scan_dir(scan_id)
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Could not delete scan files", extra={"fields": {"scanId": scan_id}})
