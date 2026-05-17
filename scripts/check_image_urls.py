#!/usr/bin/env python3
"""Reject README images hardcoded to /main/ raw URLs (mosaicraft trap regression guard)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

HARDCODED_MAIN = re.compile(
    r"https?://raw\.githubusercontent\.com/[^/]+/[^/]+/main/",
    flags=re.IGNORECASE,
)

# Also reject github.com/.../blob/main/<image> for image-like extensions
HARDCODED_BLOB_MAIN_IMG = re.compile(
    r"https?://github\.com/[^/]+/[^/]+/(blob|raw)/main/[^\s)]*\.(?:png|jpg|jpeg|gif|svg|webp)",
    flags=re.IGNORECASE,
)

REPO = Path(__file__).resolve().parent.parent
TARGET_GLOBS = ["README.md", "docs/**/*.md", "src/**/*.py"]


def scan() -> int:
    findings: list[tuple[Path, int, str]] = []
    for pattern in TARGET_GLOBS:
        for path in REPO.glob(pattern):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for regex in (HARDCODED_MAIN, HARDCODED_BLOB_MAIN_IMG):
                for m in regex.finditer(text):
                    line_no = text[: m.start()].count("\n") + 1
                    findings.append((path, line_no, m.group(0)))
    if findings:
        print("Hardcoded /main/ image URLs detected (mosaicraft trap):", file=sys.stderr)
        for path, line_no, url in findings:
            print(f"  {path}:{line_no}: {url}", file=sys.stderr)
        print("Use relative paths in docs/ or commit SHA-pinned URLs instead.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(scan())
