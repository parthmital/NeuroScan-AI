"""TOTP two-factor authentication. Secrets are stored encrypted with Fernet (AES-128-CBC + HMAC)."""

import io
import time

import pyotp
import qrcode
import qrcode.image.svg
from cryptography.fernet import Fernet, InvalidToken

ISSUER_NAME = "NeuroScan AI"
_STEP_SECONDS = 30


def new_secret() -> str:
    return pyotp.random_base32()


def encrypt_secret(secret: str, key: str) -> str:
    return Fernet(key.encode()).encrypt(secret.encode()).decode()


def decrypt_secret(token: str, key: str) -> str | None:
    try:
        return Fernet(key.encode()).decrypt(token.encode()).decode()
    except InvalidToken:
        return None


def provisioning_uri(secret: str, account_name: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=account_name, issuer_name=ISSUER_NAME)


def qr_svg(uri: str) -> str:
    image = qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    buffer = io.BytesIO()
    image.save(buffer)
    return buffer.getvalue().decode("utf-8")


def verify_code(secret: str, code: str, last_step: int | None) -> int | None:
    """Return the matched time step or None. Allows one step of drift; rejects reused steps."""
    totp = pyotp.TOTP(secret)
    current = int(time.time()) // _STEP_SECONDS
    for step in (current - 1, current, current + 1):
        if last_step is not None and step <= last_step:
            continue
        if pyotp.utils.strings_equal(totp.generate_otp(step), code):
            return step
    return None
