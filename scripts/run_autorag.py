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
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("JAVA_HOME", r"C:\Program Files\Eclipse Adoptium\jdk-21.0.8.9-hotspot")

import autorag  # noqa: E402
from llama_index.llms.openai_like import OpenAILike  # noqa: E402
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


def _expand_env_yaml(src: Path) -> Path:
    """Expand ${VAR} placeholders in YAML and write to a temp copy."""
    text = src.read_text(encoding="utf-8")
    expanded = os.path.expandvars(text)
    tmp = Path(tempfile.mkdtemp(prefix="autorag_cfg_")) / src.name
    tmp.write_text(expanded, encoding="utf-8")
    log.info(f"expanded config -> {tmp}")
    return tmp


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
    evaluator.start_trial(str(expanded_config), skip_validation=args.skip_validation)


if __name__ == "__main__":
    main()
