"""Shared fixtures: an isolated app per test with a fake inference engine and temporary storage."""

import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import nibabel
import numpy as np
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from neuroscan.app import create_app
from neuroscan.config import Settings
from neuroscan.inference import volumes
from neuroscan.inference.processor import Classification, Volumes

CSRF = {"X-Requested-With": "NeuroScan"}
PASSWORD = "correct-horse-battery"
SHAPE = (16, 16, 8)


class FakeEngine:
    """Deterministic stand-in for BrainProcessor, so tests need no model weights."""

    def __init__(self) -> None:
        self.ready = True
        self.can_segment = True
        self.fail = False

    def detect(self, image: np.ndarray) -> float:
        if self.fail:
            raise RuntimeError("inference exploded")
        return 0.9

    def classify(self, image: np.ndarray) -> Classification:
        return Classification(label="glioma", confidence=0.8)

    def segment(self, modality_paths: Mapping[str, Path], mask_path: Path) -> Volumes:
        mask = np.zeros(SHAPE, dtype=np.uint8)
        mask[4:8, 4:8, 2:4] = 3
        volumes.save_mask(mask, np.eye(4), mask_path)
        return Volumes(wt=1.5, tc=1.0, et=0.5)


def nifti_bytes(shape: tuple[int, ...] = SHAPE) -> bytes:
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    return bytes(nibabel.Nifti1Image(data, np.eye(4)).to_bytes())


def all_modalities(
    prefix: str = "BraTS_001",
) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("files", (f"{prefix}_{m}.nii", nifti_bytes(), "application/octet-stream"))
        for m in ("flair", "t1", "t1ce", "t2")
    ]


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    def factory(**overrides: object) -> Settings:
        values: dict[str, object] = {
            "environment": "development",
            "jwt_secret": "test-secret-" + "x" * 40,
            "data_encryption_key": Fernet.generate_key().decode(),
            "database_url": f"sqlite:///{tmp_path / 'test.db'}",
            "upload_dir": tmp_path / "uploads",
            "cookie_secure": False,
            "auth_rate_limit_per_minute": 1000,
            "log_level": "WARNING",
        }
        values.update(overrides)
        return Settings(**values)  # type: ignore[arg-type]

    return factory


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def make_client(
    make_settings: Callable[..., Settings], engine: FakeEngine
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def factory(**overrides: object) -> TestClient:
        client = TestClient(create_app(make_settings(**overrides), processor=engine), headers=CSRF)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def register(client: TestClient, username: str = "drrivera", **extra: object) -> dict[str, object]:
    body = {
        "username": username,
        "email": f"{username}@hospital.org",
        "password": PASSWORD,
        "fullName": "Alex Rivera",
        **extra,
    }
    response = client.post("/api/auth/register", json=body)
    assert response.status_code == 201, response.text
    return response.json()["user"]  # type: ignore[no-any-return]


@pytest.fixture
def user_client(client: TestClient) -> TestClient:
    register(client)
    return client


def wait_for(client: TestClient, scan_id: str, timeout: float = 20.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        scan: dict[str, object] = client.get(f"/api/scans/{scan_id}").json()
        if scan["status"] in ("completed", "failed"):
            return scan
        time.sleep(0.05)
    raise AssertionError(f"scan {scan_id} did not finish")


def upload(client: TestClient, files=None, **form: str):  # type: ignore[no-untyped-def]
    return client.post("/api/process-mri", files=files or all_modalities(), data=form)
