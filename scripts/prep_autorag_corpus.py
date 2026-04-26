"""Add `last_modified_datetime` to corpus metadata for AutoRAG compatibility.

AutoRAG requires `metadata.last_modified_datetime: datetime.datetime` whenever
metadata is non-empty. Our preprocessed corpus does not include it, so we
materialize a patched copy at `data/corpus_autorag.parquet`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


def _patch_metadata(meta: dict | None, ts: datetime) -> dict:
    out = dict(meta) if isinstance(meta, dict) else {}
    out.setdefault("last_modified_datetime", ts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(settings.corpus_path))
    parser.add_argument("--output", default=str(settings.data_dir / "corpus_autorag.parquet"))
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    log.info(f"loaded {len(df)} rows from {args.input}")

    ts = datetime.now()
    df["metadata"] = df["metadata"].apply(lambda m: _patch_metadata(m, ts))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info(f"wrote patched corpus to {out}")


if __name__ == "__main__":
    main()
