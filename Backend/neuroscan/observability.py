"""Logging with a per-request correlation ID: JSON in production, readable lines in development."""

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import TextIO

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_LEVEL_COLOURS = {
    "DEBUG": "2",
    "INFO": "32",
    "WARNING": "33",
    "ERROR": "31",
    "CRITICAL": "1;31",
}
_STATUS_COLOURS = {2: "32", 3: "36", 4: "33", 5: "31"}
_LOGGER_WIDTH = 20


def _fields(record: logging.LogRecord) -> dict[str, object]:
    extra = getattr(record, "fields", None)
    return extra if isinstance(extra, dict) else {}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "requestId": request_id_var.get(),
            "message": record.getMessage(),
        }
        entry.update(_fields(record))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def _enable_windows_ansi(stream: TextIO) -> bool:
    """Turn on ANSI escape handling for a Windows console; False if it is not a console."""
    if sys.platform != "win32":
        return True
    import ctypes
    import msvcrt

    kernel32 = ctypes.windll.kernel32
    handle = msvcrt.get_osfhandle(stream.fileno())
    mode = ctypes.c_uint32()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    enable_virtual_terminal_processing = 0x0004
    return bool(kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing))


def _supports_colour(stream: TextIO) -> bool:
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return stream.isatty() and _enable_windows_ansi(stream)
    except (AttributeError, OSError, ValueError):
        return False


class PrettyFormatter(logging.Formatter):
    """One aligned line per record: time, level, source, message, then key=value fields.

    Access records read as "GET /api/scans 200 6.0 ms". The request ID is shortened to 8
    characters, enough to match the lines of one request.
    """

    def __init__(self, colour: bool | None = None) -> None:
        super().__init__()
        self.colour = _supports_colour(sys.stdout) if colour is None else colour

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.colour else text

    @staticmethod
    def _source(name: str) -> str:
        if name.startswith("uvicorn"):
            return "server"
        return name.removeprefix("neuroscan.").removeprefix("alembic.runtime.")

    def _message(self, record: logging.LogRecord, fields: dict[str, object]) -> str:
        if record.name == "neuroscan.access" and {"method", "path", "status"} <= fields.keys():
            status = fields.pop("status")
            method, path = fields.pop("method"), fields.pop("path")
            duration = fields.pop("durationMs", None)
            status_code = status if isinstance(status, int) else 0
            painted = self._paint(str(status), _STATUS_COLOURS.get(status_code // 100, "0"))
            timing = self._paint(f"{duration} ms", "2") if duration is not None else ""
            return f"{method} {path} {painted} {timing}".rstrip()
        message = record.getMessage()
        # Audit events repeat their name as an "event" field.
        if fields.get("event") == message:
            fields.pop("event")
        return message

    def format(self, record: logging.LogRecord) -> str:
        fields = dict(_fields(record))
        time = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = record.levelname
        parts = [
            self._paint(time, "2"),
            self._paint(f"{level:<8}", _LEVEL_COLOURS.get(level, "0")),
            self._paint(f"{self._source(record.name):<{_LOGGER_WIDTH}}", "36"),
            self._message(record, fields),
        ]
        if fields:
            parts.append(
                "  ".join(self._paint(f"{key}=", "2") + str(value) for key, value in fields.items())
            )
        request_id = request_id_var.get()
        if request_id != "-":
            parts.append(self._paint(f"[{request_id[:8]}]", "2"))
        line = "  ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str, pretty: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(PrettyFormatter() if pretty else JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Route uvicorn's server messages through the same format. Its access log is dropped because
    # RequestContextMiddleware logs every request with its status, duration, and request ID.
    server = logging.getLogger("uvicorn")
    server.handlers[:] = []
    server.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
