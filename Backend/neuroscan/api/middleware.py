"""Pure ASGI middleware: request IDs, security headers, CSRF check, and body size limits."""

import json
import logging
import re
import time
import uuid
from collections.abc import Callable

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from neuroscan.observability import request_id_var

logger = logging.getLogger("neuroscan.access")

CSRF_HEADER = "x-requested-with"
CSRF_VALUE = "NeuroScan"
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")


async def _send_json(
    send: Send,
    status: int,
    body: dict[str, object],
    headers: dict[str, str] | None = None,
) -> None:
    payload = json.dumps(body).encode()
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode()),
    ]
    raw_headers += [(k.encode(), v.encode()) for k, v in (headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": payload})


class HttpMiddleware:
    """Pure ASGI middleware base: non-HTTP scopes (lifespan, websockets) pass straight through."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        await self.handle(scope, receive, send)

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        raise NotImplementedError


class RequestContextMiddleware(HttpMiddleware):
    """Assigns a request ID (or keeps a well-formed X-Request-ID) and logs each request."""

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            logger.info(
                "request",
                extra={
                    "fields": {
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status_code,
                        "durationMs": round((time.perf_counter() - started) * 1000, 1),
                    }
                },
            )
            request_id_var.reset(token)


class SecurityHeadersMiddleware(HttpMiddleware):
    def __init__(self, app: ASGIApp, hsts: bool, docs_paths: frozenset[str]) -> None:
        super().__init__(app)
        self.hsts = hsts
        self.docs_paths = docs_paths

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        is_docs = scope["path"] in self.docs_paths

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["Referrer-Policy"] = "no-referrer"
                headers["Cross-Origin-Resource-Policy"] = "same-origin"
                headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
                if not is_docs:
                    # API responses carry patient data: never render as a document, never cache.
                    headers["Content-Security-Policy"] = (
                        "default-src 'none'; frame-ancestors 'none'"
                    )
                    headers["Cache-Control"] = "no-store"
                if self.hsts:
                    headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            await send(message)

        await self.app(scope, receive, send_with_headers)


class UnhandledErrorMiddleware(HttpMiddleware):
    """Turns unexpected exceptions into a generic 500 that carries the request ID, never internals.

    Sits inside the header and CORS middleware so error responses still get those headers.
    """

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            logger.exception("Unhandled error", extra={"fields": {"path": scope["path"]}})
            if response_started:
                raise
            await _send_json(
                send,
                500,
                {"detail": "Internal server error", "requestId": request_id_var.get()},
            )


class CsrfHeaderMiddleware(HttpMiddleware):
    """Unsafe requests must carry X-Requested-With: NeuroScan.

    Browsers cannot add custom headers to cross-site form posts, and a cross-origin fetch with a
    custom header triggers a CORS preflight that only allowed origins pass. Together with the
    SameSite=Strict session cookie this blocks cross-site request forgery.
    """

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        unsafe = scope["method"] not in _SAFE_METHODS
        if unsafe and Headers(scope=scope).get(CSRF_HEADER) != CSRF_VALUE:
            await _send_json(send, 403, {"detail": "Missing or invalid X-Requested-With header"})
            return
        await self.app(scope, receive, send)


class BodySizeLimitMiddleware(HttpMiddleware):
    """Rejects bodies above a per-path limit, whether or not Content-Length is declared."""

    def __init__(self, app: ASGIApp, limit_for: Callable[[str], int]) -> None:
        super().__init__(app)
        self.limit_for = limit_for

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self.limit_for(scope["path"])
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            await _send_json(send, 413, {"detail": "Request body too large"})
            return

        received = 0
        exceeded = False
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            # Frameworks may turn the aborted read into their own error; replace it with 413.
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                if exceeded:
                    await _send_json(send, 413, {"detail": "Request body too large"})
                    return
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            if not response_started:
                await _send_json(send, 413, {"detail": "Request body too large"})


class _BodyTooLarge(Exception):
    pass
