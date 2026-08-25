"""
Alembic env.py — async-compatible migration runner.

Key design decisions
────────────────────
1.  We use run_async_migrations() so the same asyncpg connection used by the
    running application is also used for migrations.
2.  target_metadata is imported from app.models (all ORM classes) so Alembic
    can auto-generate accurate migration scripts with `alembic revision --autogenerate`.
3.  The DATABASE_URL is read from pydantic-settings (which reads backend/.env)
    so there is no secret duplication.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# ── Make app importable when running alembic from backend/ directory ───────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database import Base

# Import every model so Alembic can see them
import app.models  # noqa: F401

# ── Alembic config ────────────────────────────────────────────────────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── Offline migrations (generate SQL without a live DB connection) ────────────
def run_migrations_offline() -> None:
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online async migrations ───────────────────────────────────────────────────
def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


# ── Entry point ───────────────────────────────────────────────────────────────
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
