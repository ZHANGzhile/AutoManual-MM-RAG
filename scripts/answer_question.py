#!/usr/bin/env python3
"""Answer one manual question using filtered BM25 evidence and citations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.answering import answer_question
from automanual_rag.generation import (
    configured_backend,
    generate_or_fallback,
)
from automanual_rag.retrieval.bm25 import BM25Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an evidence-constrained extractive answer."
    )
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
    parser.add_argument("--region", default="North America")
    parser.add_argument("--language", default="en")
    parser.add_argument("--manual-type", default="owner_manual")
    parser.add_argument("--retrieval-limit", type=int, default=10)
    parser.add_argument("--max-evidence", type=int, default=3)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete answer and Evidence Pack as JSON.",
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
        }.items()
        if value
    }
    try:
        result = answer_question(
            BM25Index(args.index.resolve()),
            args.query,
            filters,
            retrieval_limit=args.retrieval_limit,
            max_evidence=args.max_evidence,
        )
        result = generate_or_fallback(
            result,
            question=args.query,
            backend=configured_backend(),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
