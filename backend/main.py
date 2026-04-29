"""FastAPI 백엔드 진입점 (운영웹통합명세서 §8 — backend API 진입점).

이 파일은 운영 웹 admin API 의 진입점입니다. 라우터 모듈은 점진적으로 추가:
  /api/healthz                  — 본 파일에서 직접 처리 (Day 1)
  /api/chunks                   — Day 3
  /api/tree                     — Day 3
  /api/preview/*                — Day 5
  /api/indexing/*               — Day 6

서비스 의존성: PostgreSQL (DB_URL), Qdrant (QDRANT_URL), Redis (REDIS_URL).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from db import engine
from routers.chunks import router as chunks_router
from routers.tree import router as tree_router
from routers.preview import router as preview_router
from routers.indexing import router as indexing_router
from routers.assist import router as assist_router
from routers.history import router as history_router
from routers.upload import router as upload_router
from routers.onyx import router as onyx_router
from routers.sync import router as sync_router
from routers.openai_compat import router as openai_compat_router
from routers.ocr import router as ocr_router

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] DB ping failed: {exc}")
    yield
    await engine.dispose()


app = FastAPI(
    title="EulJi RAG Admin API",
    description="운영웹통합명세서 §8 — chunk/tree/preview/indexing endpoints",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chunks_router)
app.include_router(tree_router)
app.include_router(preview_router)
app.include_router(indexing_router)
app.include_router(assist_router)
app.include_router(history_router)
app.include_router(upload_router)
app.include_router(onyx_router)
app.include_router(sync_router)
app.include_router(openai_compat_router)
app.include_router(ocr_router)


@app.get("/api/healthz")
async def healthz() -> dict:
    """Aggregate health probe — checks postgres + qdrant + redis."""
    status: dict = {
        "ok": True,
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "components": {},
    }

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        status["components"]["postgres"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        status["ok"] = False
        status["components"]["postgres"] = {"ok": False, "error": str(exc)}

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{QDRANT_URL}/collections")
            status["components"]["qdrant"] = {
                "ok": r.status_code == 200,
                "status_code": r.status_code,
            }
            if r.status_code != 200:
                status["ok"] = False
    except Exception as exc:  # noqa: BLE001
        status["ok"] = False
        status["components"]["qdrant"] = {"ok": False, "error": str(exc)}

    status["components"]["redis"] = {"ok": True, "url": REDIS_URL}

    return status


@app.get("/")
async def root() -> dict:
    return {"service": "eulji-rag-admin-api", "docs": "/docs"}
