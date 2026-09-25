"""Background processing of uploaded scans.

Jobs run on a single worker thread so only one model inference holds GPU/CPU memory at a time.
Jobs are idempotent (each run overwrites the scan's results and mask), so scans interrupted by a
restart are simply queued again at startup.
"""

import logging
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, col, select

from neuroscan.inference import volumes
from neuroscan.inference.processor import SEGMENTATION_CHANNELS, Classification, Volumes
from neuroscan.models import PipelineStep, Scan, ScanStatus
from neuroscan.scans.storage import ScanStorage

logger = logging.getLogger(__name__)

DETECTION_THRESHOLD = 0.5
NO_TUMOUR = "No Tumour Detected"
FAILURE_MESSAGE = "Processing failed. Check that the files are valid MRI volumes and retry."
# Modality used for the 2D detection and classification slice, in order of preference.
REPRESENTATIVE_ORDER = ("flair", "t1ce", "t2", "t1")


class InferenceEngine(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def can_segment(self) -> bool: ...

    def detect(self, image: np.ndarray) -> float: ...

    def classify(self, image: np.ndarray) -> Classification: ...

    def segment(self, modality_paths: Mapping[str, Path], mask_path: Path) -> Volumes: ...


class QueueFull(RuntimeError):
    pass


def _update(session: Session, scan: Scan, step: PipelineStep, progress: int) -> None:
    scan.pipelineStep = step.value
    scan.progress = progress
    session.add(scan)
    session.commit()


def _segment(
    engine: InferenceEngine, storage: ScanStorage, scan: Scan, paths: dict[str, Path]
) -> Volumes | None:
    if not engine.can_segment or not all(m in paths for m in SEGMENTATION_CHANNELS):
        return None
    try:
        return engine.segment(paths, storage.mask_path(scan.id))
    except Exception:  # segmentation is optional; detection and classification still stand
        logger.exception("Segmentation failed", extra={"fields": {"scanId": scan.id}})
        return None


def process_scan(
    session: Session, engine: InferenceEngine, storage: ScanStorage, scan: Scan
) -> None:
    scan.status = ScanStatus.PROCESSING.value
    scan.errorMessage = None
    _update(session, scan, PipelineStep.DETECTION, 10)

    paths = storage.modality_paths(scan.filePaths or {})
    representative = next(paths[m] for m in REPRESENTATIVE_ORDER if m in paths)
    image = volumes.representative_slice(representative)
    probability = engine.detect(image)
    detected = probability > DETECTION_THRESHOLD
    _update(session, scan, PipelineStep.CLASSIFICATION, 35)

    classification = engine.classify(image)
    _update(session, scan, PipelineStep.SEGMENTATION, 60)

    storage.mask_path(scan.id).unlink(missing_ok=True)
    segmented = _segment(engine, storage, scan, paths)
    wt, tc, et = (segmented.wt, segmented.tc, segmented.et) if segmented else (0.0, 0.0, 0.0)

    scan.results = {
        "detected": detected,
        "classification": classification.label if detected else NO_TUMOUR,
        "confidence": round(classification.confidence if detected else 1 - probability, 4),
        "tumorVolume": round(wt, 2),
        "wtVolume": round(wt, 2),
        "tcVolume": round(tc, 2),
        "etVolume": round(et, 2),
    }
    scan.status = ScanStatus.COMPLETED.value
    _update(session, scan, PipelineStep.COMPLETE, 100)


class PipelineRunner:
    def __init__(
        self,
        db: Engine,
        engine: InferenceEngine,
        storage: ScanStorage,
        max_pending: int,
    ) -> None:
        self._db = db
        self._engine = engine
        self._storage = storage
        self._max_pending = max_pending
        self._pending = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-pipeline")

    def submit(self, scan_id: str, enforce_limit: bool = True) -> None:
        with self._lock:
            if enforce_limit and self._pending >= self._max_pending:
                raise QueueFull("Too many scans are waiting to be processed")
            self._pending += 1
        self._executor.submit(self._run, scan_id)

    def recover(self) -> None:
        """Queue scans left unfinished by a previous process."""
        unfinished = (ScanStatus.QUEUED.value, ScanStatus.PROCESSING.value)
        with Session(self._db) as session:
            scans = session.exec(select(Scan).where(col(Scan.status).in_(unfinished))).all()
            for scan in scans:
                scan.status = ScanStatus.QUEUED.value
                scan.pipelineStep = PipelineStep.QUEUED.value
                scan.progress = 0
                session.add(scan)
            session.commit()
            ids = [scan.id for scan in scans]
        for scan_id in ids:
            self.submit(scan_id, enforce_limit=False)
        if ids:
            logger.info("Re-queued unfinished scans", extra={"fields": {"count": len(ids)}})

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, scan_id: str) -> None:
        try:
            with Session(self._db) as session:
                scan = session.get(Scan, scan_id)
                if scan is None:
                    return
                try:
                    process_scan(session, self._engine, self._storage, scan)
                    logger.info("Scan processed", extra={"fields": {"scanId": scan_id}})
                except Exception:
                    logger.exception(
                        "Scan processing failed", extra={"fields": {"scanId": scan_id}}
                    )
                    self._mark_failed(session, scan_id)
        finally:
            with self._lock:
                self._pending -= 1

    @staticmethod
    def _mark_failed(session: Session, scan_id: str) -> None:
        session.rollback()
        scan = session.get(Scan, scan_id)
        if scan is None:  # deleted while processing
            return
        scan.status = ScanStatus.FAILED.value
        scan.errorMessage = FAILURE_MESSAGE
        session.add(scan)
        try:
            session.commit()
        except SQLAlchemyError:
            logger.exception("Could not record scan failure", extra={"fields": {"scanId": scan_id}})
