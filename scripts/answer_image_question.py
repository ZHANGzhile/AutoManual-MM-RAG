#!/usr/bin/env python3
"""Answer an image-grounded manual question with traceable visual evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.generation import (
    LabeledImage,
    configured_backend,
    generate_or_fallback,
)
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.visual import (
    VisualIndex,
    VisualTextFusionIndex,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve visual evidence and optionally generate an answer."
    )
    parser.add_argument("query_image", type=Path)
    parser.add_argument(
        "--question",
        default="Identify the uploaded image.",
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
    parser.add_argument("--brand", default="Ford")
    parser.add_argument("--model", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--region", default="North America")
    parser.add_argument("--language", default="en")
    parser.add_argument("--manual-type", default="owner_manual")
    parser.add_argument("--limit", type=int, default=5)
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
        if args.question.strip():
            retriever = VisualTextFusionIndex(
                visual=visual,
                bm25=BM25Index(args.bm25_index.resolve()),
            )
            results = retriever.search(
                args.query_image.resolve(),
                query_text=args.question,
                filters=filters,
                limit=args.limit,
            )
            retrieval_backend = "visual_text_rrf"
        else:
            results = visual.search(
                args.query_image.resolve(),
                filters=filters,
                limit=args.limit,
            )
            retrieval_backend = "visual_only"
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    evidence: list[dict[str, object]] = []
    images = [
        LabeledImage("Uploaded query image", args.query_image.resolve())
    ]
    for position, item in enumerate(results, start=1):
        source = (
            f"{item['brand']} {item['model']} {item['year']}, "
            f"PDF page {item['page_no']}, "
            f"{' > '.join(item['section_path'])}"
        )
        evidence.append(
            {
                "citation_id": position,
                **item,
                "source": source,
            }
        )
        if position <= 3:
            images.append(
                LabeledImage(
                    f"Evidence [{position}]: {source}",
                    (PROJECT_ROOT / item["asset_path"]).resolve(),
                )
            )

    if evidence:
        base = {
            "status": "answered",
            "reason": "visual_evidence_retrieved",
            "answer": "\n".join(
                [
                    "Retrieved matching manual figures:",
                    *[
                        f"[{item['citation_id']}] {item['source']}"
                        for item in evidence
                    ],
                ]
            ),
            "evidence": evidence,
        }
    else:
        base = {
            "status": "insufficient_evidence",
            "reason": "no_visual_results",
            "answer": (
                "The current manual evidence is insufficient to answer safely."
            ),
            "evidence": [],
        }

    result = generate_or_fallback(
        base,
        question=args.question or "Identify the uploaded image.",
        backend=configured_backend(),
        images=images,
    )
    result["retrieval_backend"] = retrieval_backend
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
