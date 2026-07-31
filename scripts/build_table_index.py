#!/usr/bin/env python3
"""Build the local table-context FTS5 index."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.retrieval.table import build_table_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the table evidence index.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "tables.sqlite3",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    element_paths = sorted(args.processed_root.resolve().glob("*/elements.jsonl"))
    try:
        summary = build_table_index(
            index_path=args.output.resolve(),
            element_paths=element_paths,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Tables: {summary['table_count']}")
    print(f"Documents: {summary['document_count']}")
    print(f"Structured tables: {summary['structured_table_count']}")
    print(f"Index: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
