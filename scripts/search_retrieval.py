#!/usr/bin/env python3
"""Query the Dense or BM25/Dense RRF index with metadata hard filters."""

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
from automanual_rag.retrieval.dense import DenseIndex
from automanual_rag.retrieval.hybrid import HybridIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local retrieval indexes.")
    parser.add_argument("query")
    parser.add_argument(
        "--backend",
        choices=("dense", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.sqlite3",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "dense_lsa.npz",
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
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
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
        dense = DenseIndex(args.dense_index.resolve())
        index = (
            dense
            if args.backend == "dense"
            else HybridIndex(
                bm25=BM25Index(args.bm25_index.resolve()),
                dense=dense,
                candidate_limit=args.candidate_limit,
                rrf_k=args.rrf_k,
                bm25_weight=args.bm25_weight,
                dense_weight=args.dense_weight,
            )
        )
        results = index.search(
            args.query,
            filters=filters,
            limit=args.limit,
        )
    except (OSError, RuntimeError, ValueError) as exc:
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
                    f"{result['score']:.6f}",
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
