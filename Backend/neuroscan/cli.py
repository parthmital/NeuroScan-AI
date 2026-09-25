"""Operator commands for account recovery, run on the server (there is no email-based reset).

    python -m neuroscan.cli reset-password <username>   # prompts for the new password
    python -m neuroscan.cli unlock <username>           # clears a login lockout
    python -m neuroscan.cli disable-2fa <username>      # for a lost authenticator device

Every command signs the user out of all sessions.
"""

import argparse
import getpass
import sys
from collections.abc import Callable

from pydantic import ValidationError
from sqlmodel import Session, select

from neuroscan.config import get_settings
from neuroscan.db import build_engine, migrate
from neuroscan.models import User
from neuroscan.schemas import PasswordChange
from neuroscan.security.passwords import hash_password


def _prompt_password() -> str:
    first = getpass.getpass("New password: ")
    if first != getpass.getpass("Repeat new password: "):
        raise SystemExit("Passwords do not match")
    try:
        PasswordChange(currentPassword="", newPassword=first)
    except ValidationError as exc:
        raise SystemExit(exc.errors()[0]["msg"]) from exc
    return first


def run(argv: list[str], password_prompt: Callable[[], str] = _prompt_password) -> str:
    parser = argparse.ArgumentParser(prog="python -m neuroscan.cli")
    parser.add_argument("command", choices=["reset-password", "unlock", "disable-2fa"])
    parser.add_argument("username")
    args = parser.parse_args(argv)

    engine = build_engine(get_settings().database_url)
    migrate(engine)
    with Session(engine) as session:
        user = session.exec(select(User).where(User.username == args.username)).first()
        if user is None:
            raise SystemExit(f"No user named {args.username!r}")
        if args.command == "reset-password":
            user.hashed_password = hash_password(password_prompt())
        if args.command == "disable-2fa":
            user.totpEnabled = False
            user.totpSecret = None
            user.totpLastStep = None
        user.failedLoginCount = 0
        user.lockedUntil = None
        user.tokenVersion += 1
        session.add(user)
        session.commit()
    return f"{args.command}: done for {args.username}; all of their sessions were signed out"


if __name__ == "__main__":
    print(run(sys.argv[1:]))
