"""Signed session tokens (HS256 JWT) carried in an HttpOnly cookie."""

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt

ALGORITHM = "HS256"
ISSUER = "neuroscan-api"
AUDIENCE = "neuroscan-web"
SESSION_COOKIE = "neuroscan_session"


@dataclass(frozen=True)
class TokenClaims:
    user_id: int
    token_version: int


def create_token(user_id: int, token_version: int, secret: str, lifetime: timedelta) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "ver": token_version,
        "iat": now,
        "nbf": now,
        "exp": now + lifetime,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str) -> TokenClaims | None:
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["sub", "ver", "exp", "iat", "iss", "aud"]},
        )
        return TokenClaims(user_id=int(payload["sub"]), token_version=int(payload["ver"]))
    except (jwt.PyJWTError, ValueError, TypeError):
        return None
