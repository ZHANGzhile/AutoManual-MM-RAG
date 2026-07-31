#!/usr/bin/env python3
"""Launch the AutoManual-MM-RAG FastAPI service."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import uvicorn

from automanual_rag.api import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the JSON API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        print("Error: --port must be from 1 to 65535", file=sys.stderr)
        return 2
    uvicorn.run(
        create_app(project_root=PROJECT_ROOT),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
