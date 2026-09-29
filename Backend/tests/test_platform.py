"""Security headers, error handling, health probes, configuration validation, migrations, logs."""

import json
import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import inspect

from neuroscan.config import Settings
from neuroscan.db import build_engine, migrate
from neuroscan.inference import volumes
from neuroscan.observability import JsonFormatter, PrettyFormatter, request_id_var


def test_security_headers_on_api_responses(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    headers = response.headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert headers["cache-control"] == "no-store"
    assert len(headers["x-request-id"]) == 32


def test_well_formed_request_id_is_propagated(client: TestClient) -> None:
    assert (
        client.get("/api/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"]
        == "abc-123"
    )
    injected = client.get("/api/health", headers={"X-Request-ID": "a\tb"}).headers["x-request-id"]
    assert injected != "a\tb"


def test_production_hides_docs_and_sets_hsts(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(environment="production")
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    assert "max-age" in client.get("/api/health").headers["strict-transport-security"]


def test_unhandled_errors_return_generic_message(client: TestClient) -> None:
    @client.app.get("/api/boom")  # type: ignore[attr-defined]
    def boom() -> None:
        raise RuntimeError("database password is hunter2")

    response = client.get("/api/boom")
    assert response.status_code == 500
    assert "hunter2" not in response.text
    assert response.json()["requestId"] == response.headers["x-request-id"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_cors_allows_only_configured_origins(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(cors_origins="https://app.example.org")
    allowed = client.options(
        "/api/scans",
        headers={
            "Origin": "https://app.example.org",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "https://app.example.org"
    denied = client.options(
        "/api/scans",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in denied.headers


def test_readiness_reports_models(client: TestClient, engine) -> None:  # type: ignore[no-untyped-def]
    assert client.get("/api/ready").json() == {
        "status": "ok",
        "checks": {"database": True, "models": True},
    }
    engine.ready = False
    assert client.get("/api/ready").status_code == 503


@pytest.mark.parametrize(
    "overrides",
    [
        {"jwt_secret": "change-me"},
        {"jwt_secret": "too-short"},
        {"data_encryption_key": "not-a-fernet-key"},
        {"cors_origins": "*"},
        {"environment": "staging"},
    ],
)
def test_insecure_configuration_is_refused(
    make_settings: Callable[..., Settings], overrides: dict[str, str]
) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_relative_paths_resolve_against_backend_folder(
    make_settings: Callable[..., Settings],
) -> None:
    from neuroscan.config import BACKEND_DIR

    settings = make_settings(segmentation_weights="Segmentation/VNet.pth")
    assert settings.segmentation_weights == BACKEND_DIR / "Segmentation" / "VNet.pth"


def test_production_is_the_default_environment(
    make_settings: Callable[..., Settings],
) -> None:
    settings = make_settings()
    assert Settings.model_fields["environment"].default == "production"
    assert Settings.model_fields["cookie_secure"].default is True
    assert settings.max_upload_file_bytes == 300 * 1024 * 1024


LEGACY_SCHEMA = """
CREATE TABLE user (id INTEGER NOT NULL, username VARCHAR NOT NULL, email VARCHAR NOT NULL,
  hashed_password VARCHAR NOT NULL, "fullName" VARCHAR NOT NULL, title VARCHAR, department VARCHAR,
  institution VARCHAR, "createdAt" DATETIME NOT NULL, PRIMARY KEY (id));
CREATE UNIQUE INDEX ix_user_email ON user (email);
CREATE UNIQUE INDEX ix_user_username ON user (username);
CREATE TABLE scan (id VARCHAR NOT NULL, "patientId" VARCHAR NOT NULL,
  "patientName" VARCHAR NOT NULL,
  "scanDate" VARCHAR NOT NULL, modalities JSON, "filePaths" JSON, status VARCHAR NOT NULL,
  progress INTEGER NOT NULL, "pipelineStep" VARCHAR NOT NULL, results JSON,
  "createdAt" DATETIME NOT NULL,
  "userId" INTEGER, PRIMARY KEY (id), FOREIGN KEY("userId") REFERENCES user (id));
INSERT INTO user VALUES
  (1, 'legacy', 'l@x.org', 'hash', 'Legacy User', NULL, NULL, NULL, '2026-01-01');
INSERT INTO scan VALUES ('scan-abc123', 'PT-1', 'Old', '2026-01-01', '["FLAIR"]',
  '{"flair": "scan-abc123\\\\flair.nii"}', 'completed', 100, 'complete', NULL, '2026-01-01', 1);
"""


def test_pre_migration_database_is_upgraded_in_place(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(LEGACY_SCHEMA)
    engine = build_engine(f"sqlite:///{path}")
    migrate(engine)
    migrate(engine)  # idempotent

    columns = {c["name"] for c in inspect(engine).get_columns("user")}
    assert {
        "tokenVersion",
        "failedLoginCount",
        "lockedUntil",
        "totpSecret",
        "totpEnabled",
    } <= columns
    with engine.connect() as connection:
        user = connection.exec_driver_sql(
            'SELECT username, "tokenVersion", "totpEnabled" FROM user'
        ).one()
        scan = connection.exec_driver_sql('SELECT id, "errorMessage" FROM scan').one()
    assert tuple(user) == ("legacy", 0, 0)
    assert tuple(scan) == ("scan-abc123", None)


def test_legacy_windows_file_paths_resolve_inside_upload_dir(tmp_path: Path) -> None:
    from neuroscan.scans.storage import ScanStorage

    storage = ScanStorage(tmp_path)
    assert (
        storage.resolve("scan-abc123\\flair.nii")
        == (tmp_path / "scan-abc123" / "flair.nii").resolve()
    )
    with pytest.raises(ValueError):
        storage.resolve("..\\..\\secrets.txt")
    with pytest.raises(ValueError):
        storage.scan_dir("../scan-abc123")


def test_oversized_volume_header_rejected_without_reading_data(tmp_path: Path) -> None:
    import nibabel

    header = nibabel.Nifti1Header()
    header.set_data_shape((1024, 1024, 1024))
    header.set_data_dtype(np.uint8)
    path = tmp_path / "bomb.nii"
    path.write_bytes(header.binaryblock + b"\0" * 4)
    with pytest.raises(volumes.InvalidVolumeError):
        volumes.validate_volume(path)


def test_four_dimensional_series_uses_first_frame(tmp_path: Path) -> None:
    import nibabel

    data = np.zeros((8, 8, 4, 2), dtype=np.float32)
    data[..., 0] = 1.0
    data[2:6, 2:6, 2, 0] = 5.0
    path = tmp_path / "series.nii"
    nibabel.save(nibabel.Nifti1Image(data, np.eye(4)), str(path))
    assert volumes.validate_volume(path) == (8, 8, 4, 2)
    image = volumes.representative_slice(path)
    assert image.shape == (8, 8, 3) and image.max() == 255


def _access_record() -> logging.LogRecord:
    record = logging.LogRecord("neuroscan.access", logging.INFO, "", 0, "request", None, None)
    record.fields = {"method": "GET", "path": "/api/scans", "status": 200, "durationMs": 6.0}
    return record


def test_log_formats() -> None:
    token = request_id_var.set("853e59a8cbe74bf2ac7074a23bc43c9f")
    try:
        pretty = PrettyFormatter(colour=False).format(_access_record())
        entry = json.loads(JsonFormatter().format(_access_record()))
    finally:
        request_id_var.reset(token)
    assert pretty.endswith("access                GET /api/scans 200 6.0 ms  [853e59a8]")
    assert "[" not in pretty
    assert entry["status"] == 200 and entry["requestId"] == "853e59a8cbe74bf2ac7074a23bc43c9f"
