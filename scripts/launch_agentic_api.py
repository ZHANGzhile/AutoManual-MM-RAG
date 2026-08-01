#!/usr/bin/env python3
"""Launch the independent Agentic GraphRAG FastAPI service."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import uvicorn

from automanual_rag.agentic_api import create_agentic_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the Agentic GraphRAG JSON API."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--graph-index", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        print("Error: --port must be from 1 to 65535", file=sys.stderr)
        return 2
    uvicorn.run(
        create_agentic_app(
            project_root=args.project_root,
            data_root=args.data_root,
            graph_index_path=args.graph_index,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
