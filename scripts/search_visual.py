#!/usr/bin/env python3
"""Search the visual MVP with optional page-level BM25 text-hint fusion."""

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
from automanual_rag.retrieval.visual import (
    VisualIndex,
    VisualTextFusionIndex,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the visual index.")
    parser.add_argument("query_image", type=Path)
    parser.add_argument(
        "--backend",
        choices=("visual", "fusion"),
        default="visual",
    )
    parser.add_argument(
        "--query-text",
        help="Optional user hint required by the fusion backend.",
    )
    parser.add_argument(
        "--visual-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "visual_traditional.npz",
    )
    parser.add_argument(
        "--bm25-index",
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
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
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
        }.items()
        if value
    }
    try:
        visual = VisualIndex(
            args.visual_index.resolve(),
            project_root=PROJECT_ROOT,
        )
        if args.backend == "visual":
            results = visual.search(
                args.query_image.resolve(),
                filters=filters,
                limit=args.limit,
            )
        else:
            if not args.query_text:
                raise ValueError("--query-text is required for fusion")
            fusion = VisualTextFusionIndex(
                visual=visual,
                bm25=BM25Index(args.bm25_index.resolve()),
                candidate_limit=args.candidate_limit,
                rrf_k=args.rrf_k,
            )
            results = fusion.search(
                args.query_image.resolve(),
                query_text=args.query_text,
                filters=filters,
                limit=args.limit,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print("rank\tscore\telement_id\tdoc_id\tpage\tsection\tasset_path")
    for result in results:
        print(
            "\t".join(
                (
                    str(result["rank"]),
                    f"{result['score']:.6f}",
                    result["element_id"],
                    result["doc_id"],
                    str(result["page_no"]),
                    " > ".join(result["section_path"]),
                    result["asset_path"],
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
