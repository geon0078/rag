"""Run AutoRAG evaluator with a Solar-compatible LLM registered.

Upstage Solar exposes only chat completions; OpenAI's `complete()` path
hits `/completions` and 404s. We subclass `OpenAILike` with
`is_chat_model=True` and register it into `autorag.generator_models`
so the YAML can reference `llm: solar_openailike`.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import os
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-21.0.8.9-hotspot")

import autorag  # noqa: E402
from llama_index.llms.openai_like import OpenAILike  # noqa: E402
from llama_index.embeddings.openai_like import OpenAILikeEmbedding  # noqa: E402
from autorag import LazyInit  # noqa: E402
from autorag.embedding.base import embedding_models  # noqa: E402
from autorag.evaluator import Evaluator  # noqa: E402

from src.config import settings  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)


# Patch OpenAILike so AutoRAG's pop_params sees the full named signature
# (subclassing with *args/**kwargs would cause pop_params to strip all kwargs)
# and so every instance defaults to chat-completions (Upstage has no /completions).
_OPENAILIKE_ORIG_INIT = OpenAILike.__init__


@functools.wraps(_OPENAILIKE_ORIG_INIT)
def _solar_patched_init(self, *args, **kwargs):
    kwargs.setdefault("is_chat_model", True)
    return _OPENAILIKE_ORIG_INIT(self, *args, **kwargs)


_solar_patched_init.__signature__ = inspect.signature(_OPENAILIKE_ORIG_INIT)
OpenAILike.__init__ = _solar_patched_init


def _register_solar_embeddings() -> None:
    """Register Solar passage/query embeddings into AutoRAG's model registry.

    AutoRAG's load_from_dict rejects `type: openai_like` (it is in the
    embedding_map but not the allowed-types whitelist), so the only path is
    a string key into `embedding_models`. We add `solar_passage` (corpus
    indexing) and `solar_query` (query-time embedding).
    """
    api_key = os.environ.get("UPSTAGE_API_KEY") or settings.upstage_api_key
    if not api_key:
        raise RuntimeError("UPSTAGE_API_KEY missing — cannot register Solar embeddings")

    embedding_models["solar_passage"] = LazyInit(
        OpenAILikeEmbedding,
        model_name="solar-embedding-1-large-passage",
        api_base=settings.upstage_base_url,
        api_key=api_key,
        embed_batch_size=settings.embed_batch_size,
    )
    embedding_models["solar_query"] = LazyInit(
        OpenAILikeEmbedding,
        model_name="solar-embedding-1-large-query",
        api_base=settings.upstage_base_url,
        api_key=api_key,
        embed_batch_size=settings.embed_batch_size,
    )


def _expand_env_yaml(src: Path) -> Path:
    """Expand ${VAR} placeholders in YAML and write to a temp copy."""
    text = src.read_text(encoding="utf-8")
    expanded = os.path.expandvars(text)
    tmp = Path(tempfile.mkdtemp(prefix="autorag_cfg_")) / src.name
    tmp.write_text(expanded, encoding="utf-8")
    log.info(f"expanded config -> {tmp}")
    return tmp


# AutoRAG embeds the expanded config (with the live API key) into trial
# artefacts (summary.csv, config.yaml). Redact those after the run so that
# benchmark output never leaves a secret on disk.
_API_KEY_PATTERNS = [
    re.compile(r"up_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
]


def _sanitize_benchmark_outputs(project_dir: Path) -> int:
    redacted = 0
    for path in project_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".csv", ".yaml", ".yml", ".json", ".log"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new = text
        for pat in _API_KEY_PATTERNS:
            new = pat.sub("REDACTED_API_KEY", new)
        if new != text:
            path.write_text(new, encoding="utf-8")
            redacted += 1
    return redacted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "autorag.yaml"))
    p.add_argument("--qa", default=str(settings.data_dir / "qa.parquet"))
    p.add_argument("--corpus", default=str(settings.data_dir / "corpus_autorag.parquet"))
    p.add_argument("--project_dir", default=str(PROJECT_ROOT / "benchmark"))
    p.add_argument("--skip_validation", action="store_true")
    args = p.parse_args()

    if not settings.upstage_api_key:
        raise SystemExit("UPSTAGE_API_KEY missing — set it in .env")
    os.environ["UPSTAGE_API_KEY"] = settings.upstage_api_key

    _register_solar_embeddings()
    log.info("registered Solar embeddings: solar_passage, solar_query")

    Path(args.project_dir).mkdir(parents=True, exist_ok=True)

    expanded_config = _expand_env_yaml(Path(args.config))

    log.info(f"yaml={expanded_config}")
    log.info(f"qa={args.qa}")
    log.info(f"corpus={args.corpus}")
    log.info(f"project_dir={args.project_dir}")

    evaluator = Evaluator(
        qa_data_path=args.qa,
        corpus_data_path=args.corpus,
        project_dir=args.project_dir,
    )
    try:
        evaluator.start_trial(str(expanded_config), skip_validation=args.skip_validation)
    finally:
        # Run sanitize even on failure — partial outputs may still embed the key.
        n = _sanitize_benchmark_outputs(Path(args.project_dir))
        if n:
            log.info(f"sanitize: redacted API keys in {n} file(s) under {args.project_dir}")
        else:
            log.info("sanitize: no API key patterns found in benchmark outputs")


if __name__ == "__main__":
    main()
