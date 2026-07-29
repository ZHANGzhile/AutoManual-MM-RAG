#!/usr/bin/env python3
"""Evaluate the filtered BM25 baseline against Gold Evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.evaluation.retrieval import evaluate_bm25, load_questions
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.serialization import relativize_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the BM25 baseline.")
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.sqlite3",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "questions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "bm25_baseline.json",
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        questions = load_questions(args.questions.resolve())
        result = evaluate_bm25(
            index=BM25Index(args.index.resolve()),
            questions=questions,
            limit=args.limit,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                relativize_project_paths(result, PROJECT_ROOT),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    metrics = result["metrics"]
    print(f"Questions: {result['questions_total']}")
    print(f"Answerable: {result['answerable_questions']}")
    print(f"Recall@5: {metrics['recall_at_5']:.4f}")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"MRR@10: {metrics['mrr_at_10']:.4f}")
    print(
        "Metadata filter violations: "
        f"{metrics['metadata_filter_violations']}"
    )
    print(
        "No-answer cases returning candidates: "
        f"{result['no_answer_diagnostic']['cases_returning_candidates']}"
    )
    print(f"Metrics: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
