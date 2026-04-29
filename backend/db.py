"""SQLAlchemy 비동기 세션 팩토리 (운영웹통합명세서 §8 — backend API 공용).

FastAPI Depends 로 세션을 주입받아 사용한다.
``DB_URL`` 환경변수가 우선, 미설정 시 alembic.ini 기본값 폴백.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DEFAULT_DB_URL = "postgresql+psycopg://rag:rag@postgres:5432/euljiu_rag"

DB_URL = os.environ.get("DB_URL", DEFAULT_DB_URL)

engine = create_async_engine(DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
