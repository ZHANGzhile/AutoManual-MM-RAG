#!/usr/bin/env python3
"""Evaluate visual-only and optional page-level visual/text fusion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.evaluation.visual import (
    evaluate_visual,
    load_visual_questions,
)
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.visual import (
    VisualIndex,
    VisualTextFusionIndex,
)
from automanual_rag.serialization import relativize_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual retrieval.")
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
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "visual_questions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics",
    )
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            relativize_project_paths(value, PROJECT_ROOT),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        questions = load_visual_questions(
            args.questions.resolve(),
            project_root=PROJECT_ROOT,
        )
        visual = VisualIndex(
            args.visual_index.resolve(),
            project_root=PROJECT_ROOT,
        )
        fusion = VisualTextFusionIndex(
            visual=visual,
            bm25=BM25Index(args.bm25_index.resolve()),
            candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k,
        )
        visual_result = evaluate_visual(
            index=visual,
            questions=questions,
            backend="traditional_multifeature_visual_v1",
            use_text_hint=False,
            limit=args.limit,
        )
        fusion_result = evaluate_visual(
            index=fusion,
            questions=questions,
            backend="visual_page_bm25_rrf",
            use_text_hint=True,
            limit=args.limit,
        )
        output_dir = args.output_dir.resolve()
        _write_json(
            output_dir / "visual_traditional_baseline.json",
            visual_result,
        )
        _write_json(
            output_dir / "visual_text_fusion.json",
            fusion_result,
        )
        comparison = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "questions": args.questions.resolve().as_posix(),
            "query_count": len(questions),
            "fusion": {
                "candidate_limit": args.candidate_limit,
                "k": args.rrf_k,
                "visual_weight": 1.0,
                "text_weight": 1.0,
                "text_input": "user-supplied query_text hint",
            },
            "metrics": {
                "visual": visual_result["metrics"],
                "fusion": fusion_result["metrics"],
            },
            "split_metrics": {
                "visual": visual_result["split_metrics"],
                "fusion": fusion_result["split_metrics"],
            },
        }
        _write_json(output_dir / "visual_comparison.json", comparison)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("backend\tsplit\trecall@1\trecall@5\trecall@10\tmrr@10\tviolations")
    for name, result in (
        ("visual", visual_result),
        ("fusion", fusion_result),
    ):
        for split, metrics in (
            ("all", result["metrics"]),
            ("test", result["split_metrics"]["test"]),
        ):
            print(
                f"{name}\t{split}\t{metrics['recall_at_1']:.4f}\t"
                f"{metrics['recall_at_5']:.4f}\t"
                f"{metrics['recall_at_10']:.4f}\t"
                f"{metrics['mrr_at_10']:.4f}\t"
                f"{result['metrics']['metadata_filter_violations']}"
            )
    print(f"Metrics: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
