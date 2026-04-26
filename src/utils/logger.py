"""Project-wide logger using loguru with JSON Lines sink."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.config import settings


_INITIALIZED = False


def _init_once() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    logger.remove()

    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "euljiu.jsonl",
        level=settings.log_level,
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        serialize=True,
        enqueue=True,
    )

    _INITIALIZED = True


def get_logger(name: str | None = None):
    _init_once()
    return logger.bind(name=name) if name else logger
