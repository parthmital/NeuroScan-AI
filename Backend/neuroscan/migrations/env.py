"""Alembic environment. Uses the engine from neuroscan.db.migrate, else DATABASE_URL."""

from alembic import context
from sqlmodel import SQLModel

from neuroscan import models  # noqa: F401  (registers tables on SQLModel.metadata)

config = context.config
target_metadata = SQLModel.metadata


def _engine():  # type: ignore[no-untyped-def]
    engine = config.attributes.get("engine")
    if engine is not None:
        return engine
    from neuroscan.config import get_settings
    from neuroscan.db import build_engine

    return build_engine(get_settings().database_url)


def run_migrations_online() -> None:
    with _engine().connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        if is_sqlite:
            # Batch migrations rebuild tables (copy, drop, rename); FK checks would block the drop.
            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()
        if is_sqlite:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            connection.commit()


run_migrations_online()
