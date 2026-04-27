"""Scan files for accidentally committed API keys / secrets.

Designed to be wired into pre-commit (see .pre-commit-config.yaml) so the
repo cannot accept a commit that contains a live key. Also runnable
standalone:

    python scripts/check_secrets.py                       # scan whole repo
    python scripts/check_secrets.py path/to/file ...      # scan given paths
    python scripts/check_secrets.py --staged              # scan only files staged for commit

Patterns are deliberately specific to providers used by this project to keep
false positives low. Add a new pattern if a new vendor is integrated.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Each tuple: (label, regex). Labels show up in the report.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("upstage_api_key", re.compile(r"up_[A-Za-z0-9]{20,}")),
    ("openai_api_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("anthropic_api_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(r"gh[opsu]_[A-Za-z0-9]{30,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._-]{30,}")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]

# Don't scan binary blobs or generated artefacts.
SKIP_SUFFIXES = {
    ".parquet", ".pkl", ".pyc", ".bin", ".pt", ".pth", ".onnx",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar",
    ".woff", ".woff2", ".ttf", ".ico",
}
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "benchmark",  # AutoRAG outputs — sanitize hook handles those at write time
    "logs",
}

# Known-safe placeholders so .env.example etc. don't trip the scanner.
ALLOWED = {"REDACTED_API_KEY", "up_xxxxxxxxxxxxxxxxxxxxx"}


def _gitignored_set(root: Path) -> set[str]:
    """Return relative paths that git would ignore (so we don't flag them)."""
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        return {line for line in out.decode().splitlines() if line}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()


def _iter_repo_files(root: Path):
    ignored = _gitignored_set(root)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        if rel.as_posix() in ignored:
            continue
        yield path


def _staged_files(root: Path) -> list[Path]:
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=root,
    )
    files: list[Path] = []
    for line in out.decode().splitlines():
        if not line.strip():
            continue
        p = root / line
        if p.is_file() and p.suffix not in SKIP_SUFFIXES:
            if not any(part in SKIP_DIRS for part in p.relative_to(root).parts):
                files.append(p)
    return files


def _scan_file(path: Path) -> list[tuple[str, int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    findings: list[tuple[str, int, str]] = []
    for label, pat in PATTERNS:
        for match in pat.finditer(text):
            if match.group(0) in ALLOWED:
                continue
            line_no = text.count("\n", 0, match.start()) + 1
            snippet = match.group(0)
            findings.append((label, line_no, snippet))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path,
                        help="Specific files to scan (default: whole repo).")
    parser.add_argument("--staged", action="store_true",
                        help="Scan only files staged for commit (pre-commit mode).")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.staged:
        targets = _staged_files(repo_root)
    elif args.paths:
        targets = [p for p in args.paths if p.is_file() and p.suffix not in SKIP_SUFFIXES]
    else:
        targets = list(_iter_repo_files(repo_root))

    if not targets:
        print("[check_secrets] no files to scan")
        return 0

    total_findings = 0
    for path in targets:
        for label, line_no, snippet in _scan_file(path):
            rel = path.relative_to(repo_root) if path.is_relative_to(repo_root) else path
            # Mask middle of the secret in the report.
            masked = snippet[:6] + "..." + snippet[-4:] if len(snippet) > 12 else snippet
            print(f"[check_secrets] FAIL {rel}:{line_no} {label}={masked}")
            total_findings += 1

    if total_findings:
        print(f"[check_secrets] {total_findings} potential secret(s) found.")
        print("[check_secrets] Either remove the secret, or add it to ALLOWED in scripts/check_secrets.py if it is a known placeholder.")
        return 1

    print(f"[check_secrets] scanned {len(targets)} file(s), 0 findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
