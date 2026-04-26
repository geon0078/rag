"""Per-query JSONL telemetry recorder for KPI dashboard + monitoring.

Writes one JSON object per line to ``logs/queries.jsonl``. Loguru handles rotation
(50 MB) and 30-day retention via the project-wide logger.add() in ``logger.py``;
this module uses a separate sink so the schema stays stable and the dashboard
parser doesn't have to filter loguru's own structured records.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

QUERIES_LOG = settings.log_dir / "queries.jsonl"
_LOCK = Lock()


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def record_query(
    query: str,
    result: dict[str, Any],
    cached: bool,
    similarity: float | None = None,
) -> None:
    """Append one telemetry event for a query/response cycle."""
    sources = result.get("sources") or []
    doc_ids: list[str] = []
    categories: list[str] = []
    campuses: list[str] = []
    for s in sources:
        doc_id = s.get("doc_id") if isinstance(s, dict) else getattr(s, "doc_id", None)
        if doc_id:
            doc_ids.append(doc_id)
        cat = s.get("category") if isinstance(s, dict) else getattr(s, "category", None)
        if cat and cat not in categories:
            categories.append(cat)
        campus = s.get("campus") if isinstance(s, dict) else getattr(s, "campus", None)
        if campus and campus not in campuses:
            campuses.append(campus)

    event = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": query,
        "doc_ids": doc_ids,
        "categories": categories,
        "campuses": campuses,
        "grounded": bool(result.get("grounded")),
        "verdict": result.get("verdict"),
        "cached": bool(cached),
        "similarity": similarity,
        "retry": bool(result.get("retry", False)),
        "elapsed_ms": int(result.get("elapsed_ms") or 0),
    }

    try:
        _ensure_dir(QUERIES_LOG)
        line = json.dumps(event, ensure_ascii=False)
        with _LOCK:
            with open(QUERIES_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as exc:
        log.warning(f"telemetry write failed: {exc}")


def read_events(limit: int | None = None) -> list[dict[str, Any]]:
    """Read telemetry events from disk (newest last). Returns [] if file missing."""
    if not QUERIES_LOG.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(QUERIES_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        out = out[-limit:]
    return out
