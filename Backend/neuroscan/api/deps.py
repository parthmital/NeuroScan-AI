"""Shared request dependencies. Application services live on app.state (see neuroscan.app)."""

from collections.abc import Callable, Generator

from fastapi import Depends, HTTPException, Request, status
from sqlmodel import Session

from neuroscan.config import Settings
from neuroscan.models import User
from neuroscan.scans.pipeline import InferenceEngine, PipelineRunner
from neuroscan.scans.storage import ScanStorage
from neuroscan.security.rate_limit import RateLimiter
from neuroscan.security.tokens import SESSION_COOKIE, decode_token


def get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_storage(request: Request) -> ScanStorage:
    return request.app.state.storage  # type: ignore[no-any-return]


def get_processor(request: Request) -> InferenceEngine:
    return request.app.state.processor  # type: ignore[no-any-return]


def get_pipeline(request: Request) -> PipelineRunner:
    return request.app.state.pipeline  # type: ignore[no-any-return]


def get_session(request: Request) -> Generator[Session, None, None]:
    with Session(request.app.state.engine) as session:
        yield session


def _unauthorised() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


def get_current_user(
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> User:
    token = request.cookies.get(SESSION_COOKIE)
    claims = decode_token(token, settings.jwt_secret) if token else None
    if claims is None:
        raise _unauthorised()
    user = session.get(User, claims.user_id)
    if user is None or user.tokenVersion != claims.token_version:
        raise _unauthorised()
    return user


def rate_limited(name: str) -> Callable[[Request], None]:
    def dependency(request: Request) -> None:
        limiter: RateLimiter = request.app.state.auth_limiter
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(f"{name}:{client}"):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many attempts. Wait a minute and try again.",
                headers={"Retry-After": "60"},
            )

    return dependency
