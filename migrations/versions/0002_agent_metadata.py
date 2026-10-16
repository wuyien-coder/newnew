"""Persist public ReAct tool traces alongside assistant messages."""
from alembic import op
import sqlalchemy as sa

revision = "0002_agent_metadata"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("messages", sa.Column("agent_metadata", sa.JSON(), nullable=False,
                                      server_default=sa.text("'{}'")))


def downgrade():
    op.drop_column("messages", "agent_metadata")
