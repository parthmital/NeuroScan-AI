"""Baseline schema, identical to the tables created before migrations were introduced.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-09-26
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("fullName", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("department", sa.String(), nullable=True),
        sa.Column("institution", sa.String(), nullable=True),
        sa.Column("createdAt", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_email", "user", ["email"], unique=True)
    op.create_index("ix_user_username", "user", ["username"], unique=True)
    op.create_table(
        "scan",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("patientId", sa.String(), nullable=False),
        sa.Column("patientName", sa.String(), nullable=False),
        sa.Column("scanDate", sa.String(), nullable=False),
        sa.Column("modalities", sa.JSON(), nullable=True),
        sa.Column("filePaths", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("pipelineStep", sa.String(), nullable=False),
        sa.Column("results", sa.JSON(), nullable=True),
        sa.Column("createdAt", sa.DateTime(), nullable=False),
        sa.Column("userId", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["userId"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("scan")
    op.drop_index("ix_user_username", table_name="user")
    op.drop_index("ix_user_email", table_name="user")
    op.drop_table("user")
