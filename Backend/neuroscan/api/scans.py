"""Scan upload, listing, viewing, editing, deletion, retry, and exports.

Every route is owner-scoped."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import cv2
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from sqlalchemy import func
from sqlmodel import Session, col, select

from neuroscan.api.deps import (
    get_current_user,
    get_pipeline,
    get_processor,
    get_session,
    get_settings,
    get_storage,
)
from neuroscan.config import Settings
from neuroscan.inference import volumes
from neuroscan.models import Modality, PipelineStep, Scan, ScanStatus, User
from neuroscan.scans import exports
from neuroscan.scans.pipeline import InferenceEngine, PipelineRunner, QueueFull
from neuroscan.scans.storage import (
    SCAN_ID_PATTERN,
    ScanStorage,
    UploadRejected,
    UploadTooLarge,
    detect_modality,
)
from neuroscan.schemas import Message, ScanPublic, ScanResults, ScanUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["scans"])

ScanId = Annotated[str, PathParam(pattern=SCAN_ID_PATTERN)]
SliceIndex = Annotated[int, PathParam(ge=0, le=4096)]
MAX_PAGE_SIZE = 500
MASK_KEY = "segmentation"


def owned_scan(
    scan_id: ScanId,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
) -> Scan:
    scan = session.get(Scan, scan_id)
    # Another user's scan is reported as missing so scan IDs cannot be probed.
    if scan is None or scan.userId != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scan not found")
    return scan


def to_public(scan: Scan, storage: ScanStorage) -> ScanPublic:
    return ScanPublic(
        id=scan.id,
        patientId=scan.patientId,
        patientName=scan.patientName,
        scanDate=scan.scanDate,
        modalities=scan.modalities or [],
        status=scan.status,
        progress=scan.progress,
        pipelineStep=scan.pipelineStep,
        results=ScanResults.model_validate(scan.results) if scan.results else None,
        createdAt=scan.createdAt,
        hasSegmentation=storage.has_mask(scan.id),
        errorMessage=scan.errorMessage,
    )


def _owner_scans(user: User):  # type: ignore[no-untyped-def]
    return select(Scan).where(Scan.userId == user.id)


def _file_for(scan: Scan, key: str, storage: ScanStorage) -> Path:
    if key == MASK_KEY:
        path = storage.mask_path(scan.id)
    elif scan.filePaths and key in scan.filePaths:
        path = storage.resolve(scan.filePaths[key])
    else:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found")
    return path


def _attachment(
    content: bytes | str, media_type: str, filename: str, inline: bool = False
) -> Response:
    disposition = "inline" if inline else "attachment"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


@router.get("/scans", response_model=list[ScanPublic])
def list_scans(
    response: Response,
    limit: int = Query(100, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    storage: ScanStorage = Depends(get_storage),
) -> list[ScanPublic]:
    total = session.exec(select(func.count()).select_from(Scan).where(Scan.userId == user.id)).one()
    scans = session.exec(
        _owner_scans(user).order_by(col(Scan.createdAt).desc()).offset(offset).limit(limit)
    ).all()
    response.headers["X-Total-Count"] = str(total)
    return [to_public(scan, storage) for scan in scans]


@router.get("/scans/export.csv")
def export_all_csv(
    user: User = Depends(get_current_user), session: Session = Depends(get_session)
) -> Response:
    scans = session.exec(
        _owner_scans(user)
        .where(Scan.status == ScanStatus.COMPLETED.value)
        .order_by(col(Scan.createdAt).desc())
    ).all()
    return _attachment(
        exports.scans_to_csv(scans), "text/csv; charset=utf-8", "neuroscan-reports.csv"
    )


@router.get("/scans/{scan_id}", response_model=ScanPublic)
def get_scan(
    scan: Scan = Depends(owned_scan), storage: ScanStorage = Depends(get_storage)
) -> ScanPublic:
    return to_public(scan, storage)


@router.put("/scans/{scan_id}", response_model=ScanPublic)
def update_scan(
    data: ScanUpdate,
    scan: Scan = Depends(owned_scan),
    session: Session = Depends(get_session),
    storage: ScanStorage = Depends(get_storage),
) -> ScanPublic:
    for field, value in data.model_dump(exclude_unset=True, exclude_none=True).items():
        setattr(scan, field, value)
    session.add(scan)
    session.commit()
    session.refresh(scan)
    return to_public(scan, storage)


@router.delete("/scans/{scan_id}", response_model=Message)
def delete_scan(
    scan: Scan = Depends(owned_scan),
    session: Session = Depends(get_session),
    storage: ScanStorage = Depends(get_storage),
) -> Message:
    scan_id = scan.id
    session.delete(scan)
    session.commit()
    # Files go after the row, so a failure never leaves a row pointing at missing files.
    storage.delete_scan(scan_id)
    return Message(message="Scan deleted successfully")


@router.post(
    "/scans/{scan_id}/retry",
    response_model=ScanPublic,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_scan(
    scan: Scan = Depends(owned_scan),
    session: Session = Depends(get_session),
    storage: ScanStorage = Depends(get_storage),
    processor: InferenceEngine = Depends(get_processor),
    pipeline: PipelineRunner = Depends(get_pipeline),
) -> ScanPublic:
    if scan.status != ScanStatus.FAILED.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Only failed scans can be retried")
    if not processor.ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "AI models are not loaded")
    _queue(scan, session, pipeline)
    return to_public(scan, storage)


def _queue(scan: Scan, session: Session, pipeline: PipelineRunner) -> None:
    scan.status = ScanStatus.QUEUED.value
    scan.pipelineStep = PipelineStep.QUEUED.value
    scan.progress = 0
    scan.errorMessage = None
    session.add(scan)
    session.commit()
    session.refresh(scan)
    try:
        pipeline.submit(scan.id)
    except QueueFull as exc:
        scan.status = ScanStatus.FAILED.value
        scan.errorMessage = "The server is busy. Retry in a few minutes."
        session.add(scan)
        session.commit()
        session.refresh(scan)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, str(exc), headers={"Retry-After": "60"}
        ) from exc


@router.get("/scans/{scan_id}/slice/{slice_idx}")
def get_slice(
    slice_idx: SliceIndex,
    modality: Modality = Modality.FLAIR,
    scan: Scan = Depends(owned_scan),
    storage: ScanStorage = Depends(get_storage),
) -> Response:
    if not scan.filePaths:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Scan files not found")
    key = modality.value if modality.value in scan.filePaths else next(iter(scan.filePaths))
    try:
        image, total = volumes.render_slice(_file_for(scan, key, storage), slice_idx)
    except IndexError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Slice out of range") from exc
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"X-Total-Slices": str(total)},
    )


@router.get("/scans/{scan_id}/segmentation/{slice_idx}")
def get_segmentation_slice(
    slice_idx: SliceIndex,
    scan: Scan = Depends(owned_scan),
    storage: ScanStorage = Depends(get_storage),
) -> Response:
    try:
        mask = volumes.render_mask_slice(_file_for(scan, MASK_KEY, storage), slice_idx)
    except IndexError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Slice out of range") from exc
    _, buffer = cv2.imencode(".png", cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
    return Response(content=buffer.tobytes(), media_type="image/png")


@router.get("/scans/{scan_id}/download/{key}")
def download_file(
    key: Annotated[str, PathParam(pattern=r"^(flair|t1|t1ce|t2|segmentation)$")],
    scan: Scan = Depends(owned_scan),
    storage: ScanStorage = Depends(get_storage),
) -> FileResponse:
    path = _file_for(scan, key, storage)
    return FileResponse(
        path, media_type="application/octet-stream", filename=f"{scan.id}-{path.name}"
    )


@router.get("/scans/{scan_id}/report.pdf")
def report_pdf(
    scan: Scan = Depends(owned_scan),
    user: User = Depends(get_current_user),
    storage: ScanStorage = Depends(get_storage),
) -> Response:
    if scan.status != ScanStatus.COMPLETED.value:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Report is available once processing completes"
        )
    # Inline, so the browser shows it in a tab; its viewer can still save it under this name.
    return _attachment(
        exports.scan_to_pdf(scan, user, storage),
        "application/pdf",
        f"{scan.id}-report.pdf",
        inline=True,
    )


@router.get("/scans/{scan_id}/report.csv")
def report_csv(scan: Scan = Depends(owned_scan)) -> Response:
    if scan.status != ScanStatus.COMPLETED.value:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Report is available once processing completes"
        )
    return _attachment(
        exports.scans_to_csv([scan]), "text/csv; charset=utf-8", f"{scan.id}-report.csv"
    )


def _default_patient_id() -> str:
    return f"PT-{datetime.now().year}-{ScanStorage.new_scan_id()[5:9].upper()}"


@router.post("/process-mri", response_model=ScanPublic, status_code=status.HTTP_202_ACCEPTED)
def process_mri(
    files: list[UploadFile] = File(...),
    patientName: Annotated[str, Form(min_length=1, max_length=120)] = "Uploaded Scan",
    patientId: Annotated[Optional[str], Form(pattern=r"^[A-Za-z0-9 _.\-/]{0,64}$")] = None,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    storage: ScanStorage = Depends(get_storage),
    processor: InferenceEngine = Depends(get_processor),
    pipeline: PipelineRunner = Depends(get_pipeline),
) -> ScanPublic:
    """Store the uploaded modalities and queue them; poll GET /api/scans/{id} for progress."""
    if not processor.ready:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "AI models are not loaded")
    if len(files) > len(Modality):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Upload at most one file per modality")

    by_modality: dict[Modality, UploadFile] = {}
    for upload in files:
        name = (upload.filename or "")[:100]
        modality = detect_modality(name)
        if modality is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Cannot tell the modality of '{name}'. "
                "Include FLAIR, T1, T1CE, or T2 in the file name.",
            )
        if modality in by_modality:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"More than one {modality.value.upper()} file",
            )
        by_modality[modality] = upload

    scan_id = storage.new_scan_id()
    try:
        saved = {
            modality.value: storage.save_modality(
                scan_id, modality, upload.file, settings.max_upload_file_bytes
            )
            for modality, upload in by_modality.items()
        }
    except UploadRejected as exc:
        storage.delete_scan(scan_id)
        too_large = isinstance(exc, UploadTooLarge)
        code = status.HTTP_413_CONTENT_TOO_LARGE if too_large else status.HTTP_400_BAD_REQUEST
        raise HTTPException(code, str(exc)) from exc

    scan = Scan(
        id=scan_id,
        patientId=(patientId or "").strip() or _default_patient_id(),
        patientName=patientName.strip() or "Uploaded Scan",
        scanDate=datetime.now().strftime("%Y-%m-%d"),
        modalities=[m.upper() for m in saved],
        filePaths={m: storage.relative(p) for m, p in saved.items()},
        status=ScanStatus.QUEUED.value,
        userId=user.id,
    )
    # If the queue is full the scan is kept as failed, so the user can retry without re-uploading.
    _queue(scan, session, pipeline)
    logger.info("Scan uploaded", extra={"fields": {"scanId": scan_id, "userId": user.id}})
    return to_public(scan, storage)
