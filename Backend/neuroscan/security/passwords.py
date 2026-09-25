"""bcrypt password hashing with a constant-cost path for unknown users."""

import bcrypt

BCRYPT_ROUNDS = 12
# Checked when the username does not exist, so response time does not reveal valid usernames.
_DUMMY_HASH = bcrypt.hashpw(b"neuroscan-timing-equaliser", bcrypt.gensalt(BCRYPT_ROUNDS))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, hashed: str | None) -> bool:
    candidate = password.encode("utf-8")
    if len(candidate) > 72:
        # bcrypt rejects inputs over 72 bytes; no stored password can be that long.
        bcrypt.checkpw(b"x", _DUMMY_HASH)
        return False
    if hashed is None:
        bcrypt.checkpw(candidate, _DUMMY_HASH)
        return False
    return bcrypt.checkpw(candidate, hashed.encode("utf-8"))
