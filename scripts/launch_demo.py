#!/usr/bin/env python3
"""Launch the local AutoManual-MM-RAG Gradio demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.ui import DemoService, create_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local Gradio demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "corpus.csv",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.sqlite3",
    )
    parser.add_argument(
        "--visual-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "visual_traditional.npz",
    )
    parser.add_argument(
        "--table-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "tables.sqlite3",
    )
    parser.add_argument(
        "--table-row-index",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "indexes"
        / "table_rows.sqlite3",
    )
    parser.add_argument(
        "--graph-index",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "indexes"
        / "manual_graph.sqlite3",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        service = DemoService(
            project_root=PROJECT_ROOT,
            manifest_path=args.manifest,
            bm25_index_path=args.bm25_index,
            table_index_path=args.table_index,
            table_row_index_path=args.table_row_index,
            visual_index_path=args.visual_index,
            graph_index_path=args.graph_index,
        )
        demo = create_demo(service)
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
