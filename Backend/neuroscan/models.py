"""Database tables. Column names match the original schema so existing databases keep working."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Optional

from sqlmodel import JSON, Column, Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class ScanStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStep(StrEnum):
    QUEUED = "queued"
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    COMPLETE = "complete"


class Modality(StrEnum):
    FLAIR = "flair"
    T1 = "t1"
    T1CE = "t1ce"
    T2 = "t2"


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str
    fullName: str
    title: Optional[str] = None
    department: Optional[str] = None
    institution: Optional[str] = None
    createdAt: datetime = Field(default_factory=utcnow)
    # Incremented to revoke every session token issued before the change.
    tokenVersion: int = Field(default=0)
    failedLoginCount: int = Field(default=0)
    lockedUntil: Optional[datetime] = None
    # Fernet-encrypted base32 TOTP secret.
    totpSecret: Optional[str] = None
    totpEnabled: bool = Field(default=False)
    # Last accepted TOTP time step, so a code cannot be replayed.
    totpLastStep: Optional[int] = None


class Scan(SQLModel, table=True):
    id: str = Field(primary_key=True)
    patientId: str
    patientName: str
    scanDate: str
    modalities: list[str] = Field(sa_column=Column(JSON))
    filePaths: Optional[dict[str, str]] = Field(default=None, sa_column=Column(JSON))
    status: str
    progress: int = Field(default=0)
    pipelineStep: str = Field(default="queued")
    results: Optional[dict[str, object]] = Field(default=None, sa_column=Column(JSON))
    createdAt: datetime = Field(default_factory=utcnow)
    userId: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    errorMessage: Optional[str] = None
