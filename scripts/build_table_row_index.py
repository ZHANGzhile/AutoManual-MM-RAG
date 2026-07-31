#!/usr/bin/env python3
"""Build the curated, source-verified table-row index."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.retrieval.table_rows import build_table_row_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build curated table rows.")
    parser.add_argument(
        "--rows",
        type=Path,
        default=PROJECT_ROOT / "data" / "curated" / "table_rows.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "indexes"
        / "table_rows.sqlite3",
    )
    parser.add_argument(
        "--skip-asset-verification",
        action="store_true",
        help="Build without checking source image SHA-256.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = build_table_row_index(
            index_path=args.output.resolve(),
            rows_path=args.rows.resolve(),
            project_root=PROJECT_ROOT,
            verify_assets=not args.skip_asset_verification,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Rows: {summary['row_count']}")
    print(f"Source tables: {summary['source_table_count']}")
    print(f"Documents: {summary['document_count']}")
    print(f"Asset verification: {summary['asset_verification']}")
    print(f"Index: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
