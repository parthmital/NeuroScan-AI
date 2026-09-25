"""Application factory: wires settings, storage, models, pipeline, middleware, and routers."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from neuroscan.api import auth, health, scans
from neuroscan.api.middleware import (
    CSRF_HEADER,
    BodySizeLimitMiddleware,
    CsrfHeaderMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    UnhandledErrorMiddleware,
)
from neuroscan.config import Settings, get_settings
from neuroscan.db import build_engine, migrate
from neuroscan.inference.processor import BrainProcessor
from neuroscan.models import Modality
from neuroscan.observability import configure_logging
from neuroscan.scans.pipeline import InferenceEngine, PipelineRunner
from neuroscan.scans.storage import ScanStorage
from neuroscan.security.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

DEFAULT_BODY_LIMIT = 1024 * 1024
UPLOAD_PATH = "/api/process-mri"
DOCS_PATHS = frozenset({"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})


def _load_default_models(settings: Settings) -> BrainProcessor:
    processor = BrainProcessor()
    processor.load_models(
        settings.detection_weights,
        settings.classification_weights,
        settings.segmentation_weights,
    )
    return processor


def create_app(
    settings: Settings | None = None, processor: InferenceEngine | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        migrate(app.state.engine)
        if app.state.processor is None:
            app.state.processor = _load_default_models(settings)
        if not app.state.processor.ready:
            logger.error(
                "AI models are not loaded; uploads are disabled until weights are available"
            )
        app.state.pipeline = PipelineRunner(
            app.state.engine,
            app.state.processor,
            app.state.storage,
            settings.max_pending_jobs,
        )
        app.state.pipeline.recover()
        yield
        app.state.pipeline.shutdown()
        app.state.engine.dispose()

    docs_enabled = not settings.is_production
    app = FastAPI(
        title="NeuroScan AI API",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    app.state.settings = settings
    app.state.engine = build_engine(settings.database_url)
    app.state.storage = ScanStorage(settings.upload_dir)
    app.state.processor = processor
    app.state.auth_limiter = RateLimiter(settings.auth_rate_limit_per_minute)

    upload_limit = settings.max_upload_file_bytes * len(Modality) + DEFAULT_BODY_LIMIT

    # Added innermost first. Request order: context, headers, CORS, errors, CSRF, size limit.
    app.add_middleware(
        BodySizeLimitMiddleware,
        limit_for=lambda path: upload_limit if path == UPLOAD_PATH else DEFAULT_BODY_LIMIT,
    )
    app.add_middleware(CsrfHeaderMiddleware)
    app.add_middleware(UnhandledErrorMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", CSRF_HEADER, "X-Request-ID"],
            expose_headers=[
                "X-Total-Count",
                "X-Total-Slices",
                "X-Request-ID",
                "Content-Disposition",
            ],
        )
    app.add_middleware(
        SecurityHeadersMiddleware, hsts=settings.is_production, docs_paths=DOCS_PATHS
    )
    app.add_middleware(RequestContextMiddleware)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Drop the echoed "input" values so passwords never appear in responses or proxy logs.
        errors = [
            {
                "loc": list(error.get("loc", ())),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": errors},
        )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(scans.router)
    return app
