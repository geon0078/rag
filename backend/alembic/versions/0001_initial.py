"""initial schema — chunks + chunk_history + indexing_jobs (운영웹통합명세서 §9)

Revision ID: 0001
Revises:
Create Date: 2026-04-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chunks",
        sa.Column("doc_id", sa.String(), primary_key=True),
        sa.Column("parent_doc_id", sa.String(), nullable=True),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("schema_version", sa.String(), server_default="v3"),
        sa.Column("source_collection", sa.String(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=False),
        sa.Column("contents", sa.Text(), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="Draft"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_chunks_status", "chunks", ["status"])
    op.create_index("idx_chunks_collection", "chunks", ["source_collection"])
    op.create_index("idx_chunks_parent", "chunks", ["parent_doc_id"])
    op.create_index("idx_chunks_path", "chunks", ["path"])
    op.create_index(
        "idx_chunks_metadata_gin",
        "chunks",
        ["metadata"],
        postgresql_using="gin",
    )

    op.create_table(
        "chunk_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "doc_id",
            sa.String(),
            sa.ForeignKey("chunks.doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("diff", JSONB(), nullable=False),
    )
    op.create_index(
        "idx_chunk_history_doc_version", "chunk_history", ["doc_id", "version"]
    )

    op.create_table(
        "indexing_jobs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("status", sa.String(), server_default="queued"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("chunks_total", sa.Integer(), nullable=True),
        sa.Column("chunks_processed", sa.Integer(), server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("indexing_jobs")
    op.drop_index("idx_chunk_history_doc_version", table_name="chunk_history")
    op.drop_table("chunk_history")
    op.drop_index("idx_chunks_metadata_gin", table_name="chunks")
    op.drop_index("idx_chunks_path", table_name="chunks")
    op.drop_index("idx_chunks_parent", table_name="chunks")
    op.drop_index("idx_chunks_collection", table_name="chunks")
    op.drop_index("idx_chunks_status", table_name="chunks")
    op.drop_table("chunks")
