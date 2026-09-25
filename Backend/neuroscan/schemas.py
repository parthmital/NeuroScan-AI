"""Request and response shapes. Responses never expose password hashes, secrets, or file paths."""

from datetime import datetime
from typing import Annotated, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    field_validator,
)

Username = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9_.-]{3,32}$")
]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
OptionalText = Optional[Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)]]
PatientName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
PatientId = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9 _.\-/]{1,64}$")
]
OtpCode = Annotated[str, StringConstraints(strip_whitespace=True, pattern=r"^\d{6}$")]

PASSWORD_MIN_LENGTH = 12
# bcrypt only reads the first 72 bytes; longer passwords are rejected rather than truncated.
PASSWORD_MAX_BYTES = 72


def _check_password(value: str) -> str:
    if len(value) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters")
    if len(value.encode("utf-8")) > PASSWORD_MAX_BYTES:
        raise ValueError(f"Password must be at most {PASSWORD_MAX_BYTES} bytes")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UserCreate(StrictModel):
    username: Username
    email: EmailStr
    password: str
    fullName: Name
    title: OptionalText = None
    department: OptionalText = None
    institution: OptionalText = None

    _password = field_validator("password")(_check_password)


class UserLogin(StrictModel):
    username: Annotated[str, StringConstraints(max_length=64)]
    password: Annotated[str, StringConstraints(max_length=256)]
    otp: Optional[OtpCode] = None


class UserUpdate(StrictModel):
    fullName: Optional[Name] = None
    email: Optional[EmailStr] = None
    title: OptionalText = None
    department: OptionalText = None
    institution: OptionalText = None


class PasswordChange(StrictModel):
    currentPassword: Annotated[str, StringConstraints(max_length=256)]
    newPassword: str

    _password = field_validator("newPassword")(_check_password)


class PasswordConfirm(StrictModel):
    password: Annotated[str, StringConstraints(max_length=256)]


class OtpConfirm(StrictModel):
    code: OtpCode


class TwoFactorDisable(StrictModel):
    password: Annotated[str, StringConstraints(max_length=256)]
    code: OtpCode


class TwoFactorSetup(BaseModel):
    secret: str
    otpauthUri: str
    qrSvg: str


class UserPublic(BaseModel):
    id: int
    username: str
    email: str
    fullName: str
    title: Optional[str] = None
    department: Optional[str] = None
    institution: Optional[str] = None
    createdAt: datetime
    totpEnabled: bool


class AuthResponse(BaseModel):
    user: UserPublic


class ScanUpdate(StrictModel):
    patientName: Optional[PatientName] = None
    patientId: Optional[PatientId] = None


class ScanResults(BaseModel):
    detected: bool
    classification: str
    confidence: float
    tumorVolume: float
    wtVolume: float
    tcVolume: float
    etVolume: float


class ScanPublic(BaseModel):
    id: str
    patientId: str
    patientName: str
    scanDate: str
    modalities: list[str]
    status: str
    progress: int
    pipelineStep: str
    results: Optional[ScanResults] = None
    createdAt: datetime
    hasSegmentation: bool
    errorMessage: Optional[str] = None


class Message(BaseModel):
    message: str


class HealthStatus(BaseModel):
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
