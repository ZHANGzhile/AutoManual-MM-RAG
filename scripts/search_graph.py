#!/usr/bin/env python3
"""Search deterministic graph paths for one exact vehicle manual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.graphrag import GraphRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match graph entities and expand one/two-hop paths."
    )
    parser.add_argument("query")
    parser.add_argument(
        "--graph-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "manual_graph.sqlite3",
    )
    parser.add_argument("--doc-id")
    parser.add_argument("--brand", default="Ford")
    parser.add_argument("--model", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--region", default="North America")
    parser.add_argument("--language", default="en")
    parser.add_argument("--manual-type", default="owner_manual")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-hops", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    filters = {
        key: value
        for key, value in {
            "doc_id": args.doc_id,
            "brand": args.brand,
            "model": args.model,
            "year": args.year,
            "region": args.region,
            "language": args.language,
            "manual_type": args.manual_type,
        }.items()
        if value
    }
    try:
        results = GraphRetriever(args.graph_index).search(
            args.query,
            filters=filters,
            limit=args.limit,
            max_hops=args.max_hops,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
