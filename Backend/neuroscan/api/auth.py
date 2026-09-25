"""Registration, login, session management, password change, and TOTP two-factor authentication."""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlmodel import Session, select

from neuroscan.api.deps import get_current_user, get_session, get_settings, rate_limited
from neuroscan.config import Settings
from neuroscan.models import User
from neuroscan.schemas import (
    AuthResponse,
    OtpConfirm,
    PasswordChange,
    PasswordConfirm,
    TwoFactorDisable,
    TwoFactorSetup,
    UserCreate,
    UserLogin,
    UserPublic,
    UserUpdate,
)
from neuroscan.security import totp
from neuroscan.security.passwords import hash_password, verify_password
from neuroscan.security.tokens import SESSION_COOKIE, create_token, decode_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

MAX_FAILED_LOGINS = 5
LOCKOUT = timedelta(minutes=15)
INVALID_CREDENTIALS = "Incorrect username or password"


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    # SQLite returns naive datetimes; they are always stored in UTC.
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _audit(event: str, user: User | None = None, **fields: object) -> None:
    logger.info(
        event,
        extra={"fields": {"event": event, "userId": user.id if user else None, **fields}},
    )


def _public(user: User) -> UserPublic:
    return UserPublic.model_validate(user, from_attributes=True)


def _start_session(response: Response, user: User, settings: Settings) -> AuthResponse:
    if user.id is None:
        raise RuntimeError("Cannot start a session for an unsaved user")
    lifetime = timedelta(minutes=settings.access_token_expire_minutes)
    response.set_cookie(
        SESSION_COOKIE,
        create_token(user.id, user.tokenVersion, settings.jwt_secret, lifetime),
        max_age=int(lifetime.total_seconds()),
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/api",
    )
    return AuthResponse(user=_public(user))


def _end_session(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        SESSION_COOKIE,
        path="/api",
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
    )


def _revoke_sessions(user: User) -> None:
    user.tokenVersion += 1


def _record_failure(session: Session, user: User) -> None:
    user.failedLoginCount += 1
    if user.failedLoginCount >= MAX_FAILED_LOGINS:
        user.lockedUntil = _now() + LOCKOUT
        user.failedLoginCount = 0
        _audit("account_locked", user)
    session.add(user)
    session.commit()


def _check_totp(session: Session, user: User, code: str, settings: Settings) -> bool:
    secret = totp.decrypt_secret(user.totpSecret or "", settings.data_encryption_key)
    if secret is None:
        return False
    step = totp.verify_code(secret, code, user.totpLastStep)
    if step is None:
        return False
    user.totpLastStep = step
    session.add(user)
    return True


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limited("register"))],
)
def register(
    data: UserCreate,
    response: Response,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    if not settings.allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled")
    email = data.email.lower()
    if session.exec(select(User).where(func.lower(User.username) == data.username.lower())).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Username already registered")
    if session.exec(select(User).where(func.lower(User.email) == email)).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = User(
        username=data.username,
        email=email,
        hashed_password=hash_password(data.password),
        fullName=data.fullName,
        title=data.title,
        department=data.department,
        institution=data.institution,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    _audit("user_registered", user)
    return _start_session(response, user, settings)


@router.post(
    "/login",
    response_model=AuthResponse,
    responses={401: {"description": "Invalid credentials or two-factor code required"}},
    dependencies=[Depends(rate_limited("login"))],
)
def login(
    data: UserLogin,
    response: Response,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AuthResponse | JSONResponse:
    user = session.exec(select(User).where(User.username == data.username)).first()
    password_ok = verify_password(data.password, user.hashed_password if user else None)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIALS)

    locked = user.lockedUntil is not None and _as_utc(user.lockedUntil) > _now()
    if locked:
        # Only someone who knows the password learns that the account is locked.
        detail = (
            "Account temporarily locked. Try again later." if password_ok else INVALID_CREDENTIALS
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail)
    if not password_ok:
        _record_failure(session, user)
        _audit("login_failed", user)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIALS)

    if user.totpEnabled:
        if data.otp is None:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Two-factor code required", "otpRequired": True},
            )
        if not _check_totp(session, user, data.otp, settings):
            _record_failure(session, user)
            _audit("login_otp_failed", user)
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid two-factor code")

    user.failedLoginCount = 0
    user.lockedUntil = None
    session.add(user)
    session.commit()
    _audit("login_succeeded", user)
    return _start_session(response, user, settings)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> Response:
    """End the session and revoke every token issued to this user."""
    token = request.cookies.get(SESSION_COOKIE)
    claims = decode_token(token, settings.jwt_secret) if token else None
    user = session.get(User, claims.user_id) if claims else None
    if user is not None and claims is not None and user.tokenVersion == claims.token_version:
        _revoke_sessions(user)
        session.add(user)
        session.commit()
        _audit("logout", user)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _end_session(response, settings)
    return response


@router.get("/me", response_model=UserPublic)
def get_me(user: User = Depends(get_current_user)) -> UserPublic:
    return _public(user)


@router.put("/me", response_model=UserPublic)
def update_me(
    data: UserUpdate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> UserPublic:
    changes = data.model_dump(exclude_unset=True, exclude_none=True)
    if "email" in changes:
        changes["email"] = changes["email"].lower()
        taken = session.exec(
            select(User).where(func.lower(User.email) == changes["email"]).where(User.id != user.id)
        ).first()
        if taken:
            raise HTTPException(status.HTTP_409_CONFLICT, "Email already in use")
    for field, value in changes.items():
        setattr(user, field, value)
    session.add(user)
    session.commit()
    session.refresh(user)
    return _public(user)


@router.post(
    "/password",
    response_model=AuthResponse,
    dependencies=[Depends(rate_limited("password"))],
)
def change_password(
    data: PasswordChange,
    response: Response,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    if not verify_password(data.currentPassword, user.hashed_password):
        _audit("password_change_failed", user)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current password is incorrect")
    user.hashed_password = hash_password(data.newPassword)
    _revoke_sessions(user)
    session.add(user)
    session.commit()
    session.refresh(user)
    _audit("password_changed", user)
    return _start_session(response, user, settings)


@router.post(
    "/2fa/setup",
    response_model=TwoFactorSetup,
    dependencies=[Depends(rate_limited("2fa"))],
)
def setup_two_factor(
    data: PasswordConfirm,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> TwoFactorSetup:
    if user.totpEnabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "Two-factor authentication is already on")
    if not verify_password(data.password, user.hashed_password):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Password is incorrect")
    secret = totp.new_secret()
    user.totpSecret = totp.encrypt_secret(secret, settings.data_encryption_key)
    user.totpLastStep = None
    session.add(user)
    session.commit()
    uri = totp.provisioning_uri(secret, user.username)
    return TwoFactorSetup(secret=secret, otpauthUri=uri, qrSvg=totp.qr_svg(uri))


@router.post(
    "/2fa/enable",
    response_model=AuthResponse,
    dependencies=[Depends(rate_limited("2fa"))],
)
def enable_two_factor(
    data: OtpConfirm,
    response: Response,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    if user.totpEnabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "Two-factor authentication is already on")
    if user.totpSecret is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Start two-factor setup first")
    if not _check_totp(session, user, data.code, settings):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid two-factor code")
    user.totpEnabled = True
    _revoke_sessions(user)
    session.add(user)
    session.commit()
    session.refresh(user)
    _audit("two_factor_enabled", user)
    return _start_session(response, user, settings)


@router.post(
    "/2fa/disable",
    response_model=AuthResponse,
    dependencies=[Depends(rate_limited("2fa"))],
)
def disable_two_factor(
    data: TwoFactorDisable,
    response: Response,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    if not user.totpEnabled:
        raise HTTPException(status.HTTP_409_CONFLICT, "Two-factor authentication is already off")
    if not verify_password(data.password, user.hashed_password) or not _check_totp(
        session, user, data.code, settings
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Password or two-factor code is incorrect")
    user.totpEnabled = False
    user.totpSecret = None
    user.totpLastStep = None
    _revoke_sessions(user)
    session.add(user)
    session.commit()
    session.refresh(user)
    _audit("two_factor_disabled", user)
    return _start_session(response, user, settings)
