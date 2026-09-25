"""Liveness and readiness probes."""

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from neuroscan.schemas import HealthStatus

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    return HealthStatus(status="ok")


@router.get("/ready", response_model=HealthStatus)
def ready(request: Request, response: Response) -> HealthStatus:
    try:
        with request.app.state.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database = True
    except SQLAlchemyError:
        database = False
    checks = {"database": database, "models": bool(request.app.state.processor.ready)}
    if not all(checks.values()):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthStatus(status="ok" if all(checks.values()) else "unavailable", checks=checks)
