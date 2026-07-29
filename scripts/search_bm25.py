#!/usr/bin/env python3
"""Query the local BM25 baseline with optional metadata hard filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.retrieval.bm25 import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the BM25 chunk index.")
    parser.add_argument("query")
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.sqlite3",
    )
    parser.add_argument("--doc-id")
    parser.add_argument("--brand")
    parser.add_argument("--model")
    parser.add_argument("--year")
    parser.add_argument("--region")
    parser.add_argument("--language")
    parser.add_argument("--manual-type")
    parser.add_argument("--chunk-type")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print complete JSON results instead of a compact table.",
    )
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
            "chunk_type": args.chunk_type,
        }.items()
        if value
    }
    try:
        results = BM25Index(args.index.resolve()).search(
            args.query,
            filters=filters,
            limit=args.limit,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print("rank\tscore\tdoc_id\tpages\tsection\tcontent")
    for result in results:
        content = " ".join(result["content"].split())
        if len(content) > 160:
            content = content[:157] + "..."
        print(
            "\t".join(
                (
                    str(result["rank"]),
                    f"{result['score']:.4f}",
                    result["doc_id"],
                    ",".join(str(page) for page in result["page_nos"]),
                    " > ".join(result["section_path"]),
                    content,
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
