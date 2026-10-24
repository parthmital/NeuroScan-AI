"""Authentication, session, lockout, and two-factor regression tests."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import jwt
import pyotp
import pytest
from conftest import PASSWORD, register
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from neuroscan.models import User
from neuroscan.security.tokens import AUDIENCE, ISSUER, SESSION_COOKIE


def login(client: TestClient, username: str = "drrivera", password: str = PASSWORD, **extra: str):  # type: ignore[no-untyped-def]
    return client.post(
        "/api/auth/login", json={"username": username, "password": password, **extra}
    )


def db_user(client: TestClient, username: str = "drrivera") -> User:
    with Session(client.app.state.engine) as session:  # type: ignore[attr-defined]
        user = session.exec(select(User).where(User.username == username)).one()
        session.expunge(user)
        return user


def test_register_sets_httponly_strict_cookie_and_hides_secrets(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "username": "drrivera",
            "email": "Alex@Hospital.org",
            "password": PASSWORD,
            "fullName": "Alex Rivera",
        },
    )
    assert response.status_code == 201
    user = response.json()["user"]
    assert "hashed_password" not in user and "totpSecret" not in user
    assert "access_token" not in response.json()
    assert user["email"] == "alex@hospital.org"
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "path=/api" in cookie


def test_me_never_returns_password_hash(user_client: TestClient) -> None:
    body = user_client.get("/api/auth/me").json()
    assert body["username"] == "drrivera"
    assert not {"hashed_password", "totpSecret", "tokenVersion", "failedLoginCount"} & body.keys()


def test_weak_password_rejected_without_echoing_it(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "username": "weak",
            "email": "w@x.org",
            "password": "short-pw",
            "fullName": "W",
        },
    )
    assert response.status_code == 422
    assert "short-pw" not in response.text


def test_password_over_bcrypt_limit_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "username": "long",
            "email": "l@x.org",
            "password": "p" * 73,
            "fullName": "L",
        },
    )
    assert response.status_code == 422


def test_duplicate_username_and_email_are_case_insensitive(
    user_client: TestClient,
) -> None:
    body = {"password": PASSWORD, "fullName": "Other"}
    dup_name = {"username": "DrRivera", "email": "other@x.org", **body}
    dup_email = {"username": "other", "email": "DRRIVERA@hospital.org", **body}
    assert user_client.post("/api/auth/register", json=dup_name).status_code == 409
    assert user_client.post("/api/auth/register", json=dup_email).status_code == 409


def test_registration_can_be_disabled(make_client: Callable[..., TestClient]) -> None:
    client = make_client(allow_registration=False)
    body = {
        "username": "a1b2c3",
        "email": "a@x.org",
        "password": PASSWORD,
        "fullName": "A",
    }
    assert client.post("/api/auth/register", json=body).status_code == 403


def test_mass_assignment_rejected(user_client: TestClient) -> None:
    response = user_client.put("/api/auth/me", json={"hashed_password": "x", "tokenVersion": 9})
    assert response.status_code == 422


def test_login_errors_do_not_reveal_which_part_was_wrong(
    user_client: TestClient,
) -> None:
    user_client.cookies.clear()
    wrong_password = login(user_client, password="wrong-password-123")
    unknown_user = login(user_client, username="nobody")
    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()


def test_account_locks_after_repeated_failures(user_client: TestClient) -> None:
    user_client.cookies.clear()
    for _ in range(5):
        assert login(user_client, password="wrong-password-123").status_code == 401
    assert db_user(user_client).lockedUntil is not None
    locked = login(user_client)
    assert locked.status_code == 401
    assert "locked" in locked.json()["detail"]
    # Without the password an attacker learns nothing about the lock.
    assert "locked" not in login(user_client, password="wrong-password-123").json()["detail"]


def test_auth_endpoints_are_rate_limited(
    make_client: Callable[..., TestClient],
) -> None:
    client = make_client(auth_rate_limit_per_minute=3)
    codes = [login(client, username="ghost").status_code for _ in range(4)]
    assert codes == [401, 401, 401, 429]


def test_unsafe_request_without_csrf_header_is_rejected(
    user_client: TestClient,
) -> None:
    response = user_client.put(
        "/api/auth/me", json={"fullName": "X"}, headers={"X-Requested-With": ""}
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "forge",
    [
        lambda secret, now: jwt.encode({"sub": "1", "ver": 0}, key="", algorithm="none"),
        lambda secret, now: jwt.encode(
            {
                "sub": "1",
                "ver": 0,
                "iat": now,
                "exp": now + timedelta(hours=1),
                "iss": ISSUER,
                "aud": AUDIENCE,
            },
            "a-different-secret-that-is-long-enough",
            algorithm="HS256",
        ),
        lambda secret, now: jwt.encode(
            {
                "sub": "1",
                "ver": 0,
                "iat": now - timedelta(hours=2),
                "exp": now - timedelta(hours=1),
                "iss": ISSUER,
                "aud": AUDIENCE,
            },
            secret,
            algorithm="HS256",
        ),
        lambda secret, now: jwt.encode(
            {
                "sub": "1",
                "ver": 0,
                "iat": now,
                "exp": now + timedelta(hours=1),
                "iss": ISSUER,
                "aud": "other",
            },
            secret,
            algorithm="HS256",
        ),
    ],
    ids=["alg-none", "wrong-key", "expired", "wrong-audience"],
)
def test_forged_or_expired_tokens_rejected(user_client: TestClient, forge) -> None:  # type: ignore[no-untyped-def]
    secret = user_client.app.state.settings.jwt_secret  # type: ignore[attr-defined]
    user_client.cookies.clear()  # the forged token must be the only session cookie sent
    user_client.cookies.set(SESSION_COOKIE, forge(secret, datetime.now(UTC)), path="/api")
    assert user_client.get("/api/auth/me").status_code == 401


def test_logout_revokes_the_token(user_client: TestClient) -> None:
    stolen = user_client.cookies.get(SESSION_COOKIE)
    assert user_client.post("/api/auth/logout").status_code == 204
    user_client.cookies.set(SESSION_COOKIE, stolen, path="/api")
    assert user_client.get("/api/auth/me").status_code == 401


def test_password_change_requires_current_password_and_revokes_old_sessions(
    user_client: TestClient,
) -> None:
    old = user_client.cookies.get(SESSION_COOKIE)
    bad = {
        "currentPassword": "not-my-password",
        "newPassword": "a-brand-new-passphrase",
    }
    assert user_client.post("/api/auth/password", json=bad).status_code == 400
    good = {"currentPassword": PASSWORD, "newPassword": "a-brand-new-passphrase"}
    assert user_client.post("/api/auth/password", json=good).status_code == 200
    assert user_client.get("/api/auth/me").status_code == 200
    user_client.cookies.clear()  # send only the revoked token, not beside the new one
    user_client.cookies.set(SESSION_COOKIE, old, path="/api")
    assert user_client.get("/api/auth/me").status_code == 401
    user_client.cookies.clear()
    assert login(user_client, password="a-brand-new-passphrase").status_code == 200


def enable_2fa(client: TestClient) -> pyotp.TOTP:
    assert (
        client.post("/api/auth/2fa/setup", json={"password": "wrong-password-1"}).status_code == 400
    )
    setup = client.post("/api/auth/2fa/setup", json={"password": PASSWORD})
    assert setup.status_code == 200
    body = setup.json()
    assert body["qrSvg"].startswith("<?xml") and body["otpauthUri"].startswith("otpauth://totp/")
    totp = pyotp.TOTP(body["secret"])
    assert client.post("/api/auth/2fa/enable", json={"code": "000000"}).status_code == 400
    enabled = client.post("/api/auth/2fa/enable", json={"code": totp.now()})
    assert enabled.status_code == 200 and enabled.json()["user"]["totpEnabled"] is True
    stored = db_user(client).totpSecret
    assert stored is not None and body["secret"] not in stored
    return totp


def test_two_factor_login_flow_and_replay_protection(user_client: TestClient) -> None:
    totp = enable_2fa(user_client)
    user_client.cookies.clear()

    challenge = login(user_client)
    assert challenge.status_code == 401 and challenge.json()["otpRequired"] is True
    assert login(user_client, otp="123456").status_code == 401

    # The code used to enable 2FA was consumed; wait for the next step if needed.
    code = totp.at(datetime.now(UTC) + timedelta(seconds=30))
    assert login(user_client, otp=code).status_code == 200
    user_client.cookies.clear()
    assert login(user_client, otp=code).status_code == 401  # replay


def test_disable_two_factor_needs_password_and_code(user_client: TestClient) -> None:
    totp = enable_2fa(user_client)
    later = totp.at(datetime.now(UTC) + timedelta(seconds=30))
    bad = {"password": "wrong-password-1", "code": later}
    assert user_client.post("/api/auth/2fa/disable", json=bad).status_code == 400
    good = {"password": PASSWORD, "code": later}
    response = user_client.post("/api/auth/2fa/disable", json=good)
    assert response.status_code == 200 and response.json()["user"]["totpEnabled"] is False
    assert db_user(user_client).totpSecret is None


def test_profile_update_rejects_taken_email(client: TestClient) -> None:
    register(client, "first")
    register(client, "second")
    response = client.put("/api/auth/me", json={"email": "FIRST@hospital.org"})
    assert response.status_code == 409
    assert (
        client.put("/api/auth/me", json={"title": "Radiologist"}).json()["title"] == "Radiologist"
    )
