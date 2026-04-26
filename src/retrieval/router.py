"""Query → routing decision (campus filter + collection boosts).

Reads `configs/routing_rules.yaml`. Per CLAUDE.md, keywords live in YAML, not in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml
from qdrant_client.http import models as qm

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

ROUTING_YAML = settings.project_root / "configs" / "routing_rules.yaml"
CAMPUS_ALL = "전체"


@dataclass(frozen=True)
class RoutingDecision:
    campus: str | None
    boosts: dict[str, float] = field(default_factory=dict)
    qdrant_filter: qm.Filter | None = None
    bm25_dense_weights: tuple[float, float] | None = None  # (bm25, dense)
    sparse_filter: dict[str, list[str]] | None = None  # campus filter for BM25
    # When True, ``campus`` came from settings.default_campus rather than from
    # the query itself. The pipeline surfaces this so the API/UI can warn the
    # user that the answer is scoped to the default campus.
    campus_was_inferred: bool = False


@lru_cache(maxsize=1)
def _load_rules(path: Path = ROUTING_YAML) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"routing rules not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_campus(query: str, rules: dict | None = None) -> str | None:
    """Return the explicit campus signal from the query, or None if absent.

    Pure detection — no defaulting. Use ``resolve_campus()`` when you want
    the default-campus fallback applied.
    """
    rules = rules if rules is not None else _load_rules()
    campus_keywords: dict[str, list[str]] = rules.get("campus_keywords", {})
    for campus, keywords in campus_keywords.items():
        for kw in keywords:
            if kw in query:
                return campus
    return None


def resolve_campus(query: str, rules: dict | None = None) -> tuple[str, bool]:
    """Return ``(campus, was_inferred)``.

    ``was_inferred=True`` means the query had no explicit signal and we fell
    back to ``settings.default_campus``. Callers can use the flag to mark
    answers as "scoped to 성남캠퍼스 by default" in the user-facing UI.
    """
    explicit = extract_campus(query, rules)
    if explicit is not None:
        return explicit, False
    return settings.default_campus, True


# Back-compat shim — older call sites still import _detect_campus.
_detect_campus = extract_campus


def _detect_boosts(query: str, rules: dict) -> dict[str, float]:
    boosts: dict[str, float] = {}
    for entry in rules.get("collection_priority", []):
        keywords = entry.get("keywords", [])
        boost_col = entry.get("boost_collection")
        weight = float(entry.get("weight", 1.0))
        if not boost_col:
            continue
        if any(kw in query for kw in keywords):
            boosts[boost_col] = max(boosts.get(boost_col, 0.0), weight)
    return boosts


def _bm25_dense_weights(boosts: dict[str, float], rules: dict) -> tuple[float, float] | None:
    weights_map: dict[str, dict] = rules.get("collection_weights", {})
    if not boosts or not weights_map:
        return None
    top_collection = max(boosts.items(), key=lambda kv: kv[1])[0]
    cfg = weights_map.get(top_collection)
    if not cfg:
        return None
    return float(cfg.get("bm25", 0.5)), float(cfg.get("dense", 0.5))


def _build_filter(campus: str | None) -> qm.Filter | None:
    if campus is None:
        return None
    return qm.Filter(
        should=[
            qm.FieldCondition(key="campus", match=qm.MatchValue(value=campus)),
            qm.FieldCondition(key="campus", match=qm.MatchValue(value=CAMPUS_ALL)),
        ]
    )


def _build_sparse_filter(campus: str | None) -> dict[str, list[str]] | None:
    """Same campus semantics as Qdrant filter, but as a plain dict for BM25."""
    if campus is None:
        return None
    return {"campus": [campus, CAMPUS_ALL]}


def build_metadata_filter(campus: str | None) -> dict[str, list[str]] | None:
    """Public alias of ``_build_sparse_filter`` for use outside the router.

    Useful when callers (e.g., the API layer or eval scripts) need to
    construct the campus filter independently of routing.
    """
    return _build_sparse_filter(campus)


def route(query: str) -> RoutingDecision:
    rules = _load_rules()
    campus, was_inferred = resolve_campus(query, rules)
    boosts = _detect_boosts(query, rules)
    qfilter = _build_filter(campus)
    sparse_filter = _build_sparse_filter(campus)
    weights = _bm25_dense_weights(boosts, rules)
    decision = RoutingDecision(
        campus=campus,
        boosts=boosts,
        qdrant_filter=qfilter,
        bm25_dense_weights=weights,
        sparse_filter=sparse_filter,
        campus_was_inferred=was_inferred,
    )
    log.debug(
        f"route: campus={campus} (inferred={was_inferred}) "
        f"boosts={boosts} weights={weights}"
    )
    return decision
