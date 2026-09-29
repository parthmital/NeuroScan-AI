"""Scan upload, processing, access control, file handling, and export regression tests."""

import csv
import io
from collections.abc import Callable
from pathlib import Path

from conftest import FakeEngine, nifti_bytes, register, upload, wait_for
from fastapi.testclient import TestClient


def test_upload_is_queued_then_processed(user_client: TestClient) -> None:
    response = upload(user_client, patientName="Jane Doe", patientId="PT-100")
    assert response.status_code == 202, response.text
    queued = response.json()
    assert queued["status"] == "queued"
    assert not {"filePaths", "userId"} & queued.keys()

    scan = wait_for(user_client, queued["id"])
    assert scan["status"] == "completed" and scan["pipelineStep"] == "complete"
    assert scan["hasSegmentation"] is True
    assert scan["modalities"] == ["FLAIR", "T1", "T1CE", "T2"]
    assert scan["results"] == {
        "detected": True,
        "classification": "glioma",
        "confidence": 0.8,
        "tumorVolume": 1.5,
        "wtVolume": 1.5,
        "tcVolume": 1.0,
        "etVolume": 0.5,
    }


def test_client_file_name_never_becomes_a_path(user_client: TestClient, tmp_path: Path) -> None:
    files = [
        (
            "files",
            ("../../../evil_flair.nii", nifti_bytes(), "application/octet-stream"),
        )
    ]
    response = upload(user_client, files=files)
    assert response.status_code == 202
    scan_dir = tmp_path / "uploads" / response.json()["id"]
    assert (scan_dir / "flair.nii").is_file()
    assert not list(tmp_path.rglob("evil_flair.nii"))


def test_gzip_upload_keeps_compressed_extension(user_client: TestClient, tmp_path: Path) -> None:
    import gzip

    files = [
        (
            "files",
            ("case_flair.nii.gz", gzip.compress(nifti_bytes()), "application/gzip"),
        )
    ]
    response = upload(user_client, files=files)
    assert response.status_code == 202
    assert (tmp_path / "uploads" / response.json()["id"] / "flair.nii.gz").is_file()


def test_non_nifti_upload_rejected_and_cleaned_up(user_client: TestClient, tmp_path: Path) -> None:
    files = [("files", ("scan_flair.nii", b"<?php system($_GET['c']); ?>", "text/plain"))]
    response = upload(user_client, files=files)
    assert response.status_code == 400
    assert "FLAIR" in response.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_unknown_and_duplicate_modalities_rejected(user_client: TestClient) -> None:
    unknown = [("files", ("brain.nii", nifti_bytes(), "application/octet-stream"))]
    assert upload(user_client, files=unknown).status_code == 400
    twice = [
        ("files", ("a_flair.nii", nifti_bytes(), "application/octet-stream")),
        ("files", ("b_flair.nii", nifti_bytes(), "application/octet-stream")),
    ]
    assert upload(user_client, files=twice).status_code == 400


def test_oversized_file_rejected(make_client: Callable[..., TestClient], tmp_path: Path) -> None:
    client = make_client(max_upload_file_mb=1)
    register(client)
    big = [
        (
            "files",
            ("x_flair.nii", b"\0" * (1024 * 1024 + 1), "application/octet-stream"),
        )
    ]
    assert upload(client, files=big).status_code == 413
    assert list((tmp_path / "uploads").iterdir()) == []


def test_request_body_limit_applies_to_undeclared_length(
    user_client: TestClient,
) -> None:
    response = user_client.put(
        "/api/auth/me",
        content=(b'{"title": "' + b"a" * (2 * 1024 * 1024) + b'"}' for _ in range(1)),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413


def test_invalid_patient_id_rejected(user_client: TestClient) -> None:
    assert upload(user_client, patientId="<script>alert(1)</script>").status_code == 422


def test_models_not_loaded_returns_503(user_client: TestClient, engine: FakeEngine) -> None:
    engine.ready = False
    assert upload(user_client).status_code == 503


def test_scans_are_isolated_between_users(
    make_client: Callable[..., TestClient],
) -> None:
    owner = make_client()
    register(owner, "owner")
    scan_id = wait_for(owner, upload(owner).json()["id"])["id"]

    attacker = TestClient(owner.app, headers=owner.headers)
    register(attacker, "attacker")
    for method, path in [
        ("GET", f"/api/scans/{scan_id}"),
        ("PUT", f"/api/scans/{scan_id}"),
        ("DELETE", f"/api/scans/{scan_id}"),
        ("POST", f"/api/scans/{scan_id}/retry"),
        ("GET", f"/api/scans/{scan_id}/slice/0"),
        ("GET", f"/api/scans/{scan_id}/segmentation/0"),
        ("GET", f"/api/scans/{scan_id}/download/flair"),
        ("GET", f"/api/scans/{scan_id}/report.pdf"),
        ("GET", f"/api/scans/{scan_id}/report.csv"),
    ]:
        response = attacker.request(method, path, json={} if method == "PUT" else None)
        assert response.status_code == 404, (method, path)
    assert attacker.get("/api/scans").json() == []
    assert owner.get(f"/api/scans/{scan_id}").status_code == 200


def test_unauthenticated_requests_rejected(client: TestClient) -> None:
    assert client.get("/api/scans").status_code == 401
    assert upload(client).status_code == 401


def test_scan_id_and_download_key_are_allowlisted(user_client: TestClient) -> None:
    scan_id = wait_for(user_client, upload(user_client).json()["id"])["id"]
    assert user_client.get("/api/scans/..%2F..%2Fmain.py").status_code in (404, 422)
    assert user_client.get("/api/scans/not-a-scan").status_code == 422
    assert user_client.get(f"/api/scans/{scan_id}/download/flair.nii").status_code == 422
    assert user_client.get(f"/api/scans/{scan_id}/download/..%2F..%2Fmain.py").status_code in (
        404,
        422,
    )
    download = user_client.get(f"/api/scans/{scan_id}/download/segmentation")
    assert download.status_code == 200 and download.content[:4] == b"\x5c\x01\x00\x00"


def test_slices_render_and_bounds_are_checked(user_client: TestClient) -> None:
    scan_id = wait_for(user_client, upload(user_client).json()["id"])["id"]
    image = user_client.get(f"/api/scans/{scan_id}/slice/3?modality=t2")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    assert image.headers["x-total-slices"] == "8"
    assert user_client.get(f"/api/scans/{scan_id}/slice/8").status_code == 404
    assert user_client.get(f"/api/scans/{scan_id}/slice/-1").status_code == 422
    assert user_client.get(f"/api/scans/{scan_id}/slice/0?modality=evil").status_code == 422
    mask = user_client.get(f"/api/scans/{scan_id}/segmentation/2")
    assert mask.status_code == 200 and mask.headers["content-type"] == "image/png"


def test_failed_scan_reports_generic_error_and_can_be_retried(
    user_client: TestClient, engine: FakeEngine
) -> None:
    engine.fail = True
    scan = wait_for(user_client, upload(user_client).json()["id"])
    assert scan["status"] == "failed"
    assert "exploded" not in str(scan["errorMessage"])
    assert user_client.get(f"/api/scans/{scan['id']}/report.pdf").status_code == 409

    engine.fail = False
    assert user_client.post(f"/api/scans/{scan['id']}/retry").status_code == 202
    assert wait_for(user_client, str(scan["id"]))["status"] == "completed"
    assert user_client.post(f"/api/scans/{scan['id']}/retry").status_code == 409


def test_segmentation_skipped_without_all_modalities(user_client: TestClient) -> None:
    files = [("files", ("only_flair.nii", nifti_bytes(), "application/octet-stream"))]
    scan = wait_for(user_client, upload(user_client, files=files).json()["id"])
    assert scan["status"] == "completed" and scan["hasSegmentation"] is False
    assert scan["results"]["wtVolume"] == 0.0  # type: ignore[index]


def test_update_and_delete(user_client: TestClient, tmp_path: Path) -> None:
    scan_id = wait_for(user_client, upload(user_client).json()["id"])["id"]
    updated = user_client.put(f"/api/scans/{scan_id}", json={"patientName": "Renamed"})
    assert updated.json()["patientName"] == "Renamed"
    assert user_client.put(f"/api/scans/{scan_id}", json={"userId": 99}).status_code == 422
    assert user_client.delete(f"/api/scans/{scan_id}").status_code == 200
    assert not (tmp_path / "uploads" / str(scan_id)).exists()
    assert user_client.get(f"/api/scans/{scan_id}").status_code == 404


def test_list_is_paginated(user_client: TestClient) -> None:
    for _ in range(3):
        wait_for(user_client, upload(user_client).json()["id"])
    page = user_client.get("/api/scans?limit=2&offset=0")
    assert len(page.json()) == 2 and page.headers["x-total-count"] == "3"
    assert len(user_client.get("/api/scans?limit=2&offset=2").json()) == 1
    assert user_client.get("/api/scans?limit=501").status_code == 422


def test_csv_export_neutralises_spreadsheet_formulas(user_client: TestClient) -> None:
    payload = '=HYPERLINK("http://evil.example","x")'
    scan_id = wait_for(user_client, upload(user_client, patientName=payload).json()["id"])["id"]
    for path in ("/api/scans/export.csv", f"/api/scans/{scan_id}/report.csv"):
        response = user_client.get(path)
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        rows = list(csv.reader(io.StringIO(response.text)))
        assert rows[1][2] == "'" + payload


def test_pdf_report(user_client: TestClient) -> None:
    scan_id = wait_for(user_client, upload(user_client, patientName="<b>Jane</b>").json()["id"])[
        "id"
    ]
    response = user_client.get(f"/api/scans/{scan_id}/report.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.content.startswith(b"%PDF")
