"""Session revocation, login lockout, TOTP two-factor auth, scan job errors, scan owner index.

Additive only: every new column is nullable or has a server default.

Revision ID: 0002_auth_hardening_and_jobs
Revises: 0001_baseline
Create Date: 2026-09-26
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_auth_hardening_and_jobs"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("user") as batch:
        batch.add_column(
            sa.Column("tokenVersion", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(
            sa.Column("failedLoginCount", sa.Integer(), nullable=False, server_default="0")
        )
        batch.add_column(sa.Column("lockedUntil", sa.DateTime(), nullable=True))
        batch.add_column(sa.Column("totpSecret", sa.String(), nullable=True))
        batch.add_column(
            sa.Column("totpEnabled", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch.add_column(sa.Column("totpLastStep", sa.Integer(), nullable=True))
    with op.batch_alter_table("scan") as batch:
        batch.add_column(sa.Column("errorMessage", sa.String(), nullable=True))
        batch.create_index("ix_scan_userId", ["userId"])


def downgrade() -> None:
    with op.batch_alter_table("scan") as batch:
        batch.drop_index("ix_scan_userId")
        batch.drop_column("errorMessage")
    with op.batch_alter_table("user") as batch:
        batch.drop_column("totpLastStep")
        batch.drop_column("totpEnabled")
        batch.drop_column("totpSecret")
        batch.drop_column("lockedUntil")
        batch.drop_column("failedLoginCount")
        batch.drop_column("tokenVersion")
