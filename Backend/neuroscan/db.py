"""Database engine, sessions, and schema migrations."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import event, inspect
from sqlalchemy.engine import Engine
from sqlmodel import create_engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
BASELINE_REVISION = "0001_baseline"


def build_engine(database_url: str) -> Engine:
    is_sqlite = database_url.startswith("sqlite")
    connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            # WAL lets the background pipeline write while API requests read.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def _alembic_config(engine: Engine) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.attributes["engine"] = engine
    return config


def migrate(engine: Engine) -> None:
    """Bring the schema to the latest revision.

    Databases created before migrations existed already hold the baseline tables,
    so they are stamped at the baseline before upgrading.
    """
    config = _alembic_config(engine)
    tables = set(inspect(engine).get_table_names())
    if "alembic_version" not in tables and {"user", "scan"} <= tables:
        command.stamp(config, BASELINE_REVISION)
    command.upgrade(config, "head")
