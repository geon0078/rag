"""Streamlit KPI dashboard for the EulJi RAG system.

Run with: ``streamlit run src/dashboard/kpi.py``

Reads ``logs/queries.jsonl`` written by ``src.utils.telemetry`` and renders the
spec's Phase 7 Task 7.2 metrics: daily query count, grounded ratio (target ≥0.80),
mean/p95 latency, cache hit ratio, and per-category distribution.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import settings
from src.utils.telemetry import read_events

GROUNDED_TARGET = 0.80
LATENCY_P95_TARGET_MS = 3000


@st.cache_data(ttl=30)
def _load_df() -> pd.DataFrame:
    events = read_events()
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["date"] = df["ts"].dt.tz_convert("Asia/Seoul").dt.date
    return df


def _kpi_row(df: pd.DataFrame) -> None:
    total = len(df)
    grounded_ratio = float(df["grounded"].mean()) if total else 0.0
    cache_hit_ratio = float(df["cached"].mean()) if total else 0.0
    mean_latency = float(df["elapsed_ms"].mean()) if total else 0.0
    p95_latency = float(df["elapsed_ms"].quantile(0.95)) if total else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("총 쿼리 수", f"{total:,}")
    c2.metric(
        "Grounded 비율",
        f"{grounded_ratio:.1%}",
        delta=f"{(grounded_ratio - GROUNDED_TARGET):+.1%} vs 목표",
        delta_color="normal" if grounded_ratio >= GROUNDED_TARGET else "inverse",
    )
    c3.metric("캐시 히트율", f"{cache_hit_ratio:.1%}")
    c4.metric("평균 지연 (ms)", f"{mean_latency:,.0f}")
    c5.metric(
        "p95 지연 (ms)",
        f"{p95_latency:,.0f}",
        delta=f"{(p95_latency - LATENCY_P95_TARGET_MS):+,.0f} vs 목표",
        delta_color="inverse" if p95_latency > LATENCY_P95_TARGET_MS else "normal",
    )


def _daily_chart(df: pd.DataFrame) -> None:
    daily = (
        df.groupby("date")
        .agg(
            queries=("query", "count"),
            grounded_ratio=("grounded", "mean"),
            cache_hit_ratio=("cached", "mean"),
            mean_ms=("elapsed_ms", "mean"),
            p95_ms=("elapsed_ms", lambda s: float(s.quantile(0.95))),
        )
        .reset_index()
    )
    st.subheader("일별 추이")
    st.dataframe(daily, use_container_width=True, hide_index=True)
    if not daily.empty:
        st.line_chart(
            daily.set_index("date")[["grounded_ratio", "cache_hit_ratio"]]
        )
        st.line_chart(daily.set_index("date")[["mean_ms", "p95_ms"]])


def _category_chart(df: pd.DataFrame) -> None:
    st.subheader("카테고리 분포")
    exploded = df.explode("categories")
    counts = (
        exploded.dropna(subset=["categories"])
        .groupby("categories")["query"]
        .count()
        .sort_values(ascending=False)
    )
    if counts.empty:
        st.info("아직 카테고리 데이터가 없습니다.")
        return
    st.bar_chart(counts)


def _campus_chart(df: pd.DataFrame) -> None:
    st.subheader("캠퍼스 분포")
    exploded = df.explode("campuses")
    counts = (
        exploded.dropna(subset=["campuses"])
        .groupby("campuses")["query"]
        .count()
        .sort_values(ascending=False)
    )
    if counts.empty:
        st.info("아직 캠퍼스 데이터가 없습니다.")
        return
    st.bar_chart(counts)


def _recent_table(df: pd.DataFrame) -> None:
    st.subheader("최근 쿼리 50건")
    recent = df.sort_values("ts", ascending=False).head(50)
    columns = [
        "ts",
        "query",
        "grounded",
        "verdict",
        "cached",
        "similarity",
        "retry",
        "elapsed_ms",
        "categories",
        "campuses",
    ]
    st.dataframe(recent[columns], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="을지대 RAG KPI", layout="wide")
    st.title("을지대 RAG 운영 대시보드")
    st.caption(
        f"Source: `{settings.log_dir / 'queries.jsonl'}` | "
        f"Targets: grounded ≥ {GROUNDED_TARGET:.0%}, p95 ≤ {LATENCY_P95_TARGET_MS} ms"
    )

    df = _load_df()
    if df.empty:
        st.warning(
            "logs/queries.jsonl 에 데이터가 아직 없습니다. "
            "API 서버에 쿼리를 보내면 이 대시보드에 누적됩니다."
        )
        return

    with st.sidebar:
        st.header("필터")
        min_d, max_d = df["date"].min(), df["date"].max()
        date_range = st.date_input(
            "기간", value=(min_d, max_d), min_value=min_d, max_value=max_d
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
        else:
            start_d, end_d = min_d, max_d
        only_grounded = st.checkbox("grounded만 보기", value=False)
        exclude_cached = st.checkbox("캐시 히트 제외", value=False)

    filtered = df[(df["date"] >= start_d) & (df["date"] <= end_d)]
    if only_grounded:
        filtered = filtered[filtered["grounded"]]
    if exclude_cached:
        filtered = filtered[~filtered["cached"]]

    _kpi_row(filtered)
    st.divider()
    _daily_chart(filtered)
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        _category_chart(filtered)
    with col_b:
        _campus_chart(filtered)
    st.divider()
    _recent_table(filtered)


if __name__ == "__main__":
    main()
