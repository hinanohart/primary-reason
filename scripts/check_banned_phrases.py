#!/usr/bin/env python3
"""Reject banned over-claim phrases in repo files (mosaicraft/chrono-domains regression guard)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

BANNED = [
    "永続的",
    "完全自動",
    "完全な自動",
    "世界初",
    "forever",
    "never-fail",
    "never fails",
    "100% reliable",
    "完璧な",
]

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "dist",
    "build",
}
SKIP_SUFFIXES = {".lock", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf"}
ALLOWED_FILES = {"scripts/check_banned_phrases.py"}


def scan() -> int:
    repo = Path(__file__).resolve().parent.parent
    findings: list[tuple[Path, int, str, str]] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        rel = path.relative_to(repo).as_posix()
        if rel in ALLOWED_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for phrase in BANNED:
            for m in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE):
                line_no = text[: m.start()].count("\n") + 1
                line_content = (
                    text.splitlines()[line_no - 1] if line_no - 1 < len(text.splitlines()) else ""
                )
                findings.append((path, line_no, phrase, line_content.strip()))
    if findings:
        print("Banned phrases detected:", file=sys.stderr)
        for path, line_no, phrase, content in findings:
            print(f"  {path}:{line_no}: {phrase!r} in {content[:120]!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(scan())
