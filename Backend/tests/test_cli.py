"""Operator recovery commands."""

import pyotp
import pytest
from conftest import PASSWORD
from fastapi.testclient import TestClient

from neuroscan import cli


@pytest.fixture
def operator(user_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The CLI pointed at the same database as the test app."""
    settings = user_client.app.state.settings  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return user_client


def login(client: TestClient, password: str = PASSWORD):  # type: ignore[no-untyped-def]
    return client.post("/api/auth/login", json={"username": "drrivera", "password": password})


def test_reset_password_signs_user_out_and_sets_new_password(
    operator: TestClient,
) -> None:
    message = cli.run(["reset-password", "drrivera"], password_prompt=lambda: "operator-set-pass-1")
    assert "signed out" in message
    assert operator.get("/api/auth/me").status_code == 401
    assert login(operator).status_code == 401
    assert login(operator, "operator-set-pass-1").status_code == 200


def test_unlock_clears_lockout(operator: TestClient) -> None:
    operator.cookies.clear()
    for _ in range(5):
        login(operator, "wrong-password-123")
    assert "locked" in login(operator).json()["detail"]
    cli.run(["unlock", "drrivera"])
    assert login(operator).status_code == 200


def test_disable_2fa_for_lost_device(operator: TestClient) -> None:
    secret = operator.post("/api/auth/2fa/setup", json={"password": PASSWORD}).json()["secret"]
    operator.post("/api/auth/2fa/enable", json={"code": pyotp.TOTP(secret).now()})
    operator.cookies.clear()
    assert login(operator).json().get("otpRequired") is True
    cli.run(["disable-2fa", "drrivera"])
    response = login(operator)
    assert response.status_code == 200 and response.json()["user"]["totpEnabled"] is False


def test_unknown_user_is_reported(operator: TestClient) -> None:
    with pytest.raises(SystemExit):
        cli.run(["unlock", "nobody"])
