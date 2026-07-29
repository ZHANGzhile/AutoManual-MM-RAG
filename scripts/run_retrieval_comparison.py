#!/usr/bin/env python3
"""Rebuild Dense and reproduce BM25/Dense/RRF evaluation plus audit."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for script_name in (
        "build_dense_index.py",
        "evaluate_retrieval.py",
        "audit_retrieval.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script_name)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
