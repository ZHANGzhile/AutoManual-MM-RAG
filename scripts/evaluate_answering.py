#!/usr/bin/env python3
"""Evaluate answer/refusal decisions and citation grounding."""

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

from automanual_rag.answering import (
    DEFAULT_MIN_BM25_SCORE,
    DEFAULT_MIN_CONFIDENCE,
    answer_question,
)
from automanual_rag.evaluation.retrieval import load_questions
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.serialization import relativize_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate extractive answers, refusals, and citations."
    )
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
        default=PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "answering_baseline.json",
    )
    parser.add_argument("--retrieval-limit", type=int, default=10)
    parser.add_argument("--max-evidence", type=int, default=3)
    parser.add_argument(
        "--min-bm25-score",
        type=float,
        default=DEFAULT_MIN_BM25_SCORE,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
    )
    return parser.parse_args()


def _metadata_violations(
    evidence: list[dict[str, Any]],
    filters: dict[str, str],
) -> int:
    return sum(
        any(
            str(item.get(field, "")).casefold() != str(expected).casefold()
            for field, expected in filters.items()
        )
        for item in evidence
    )


def main() -> int:
    args = parse_args()
    try:
        questions = load_questions(args.questions.resolve())
        index = BM25Index(args.index.resolve())
        details: list[dict[str, Any]] = []
        answerable_count = 0
        answered_answerable = 0
        citation_hits = 0
        no_answer_count = 0
        correct_refusals = 0
        decision_correct = 0
        metadata_violations = 0

        for question in questions:
            result = answer_question(
                index,
                question["question"],
                question["filters"],
                retrieval_limit=args.retrieval_limit,
                max_evidence=args.max_evidence,
                min_bm25_score=args.min_bm25_score,
                min_confidence=args.min_confidence,
            )
            predicted_answerable = result["status"] == "answered"
            expected_answerable = bool(question["answerable"])
            decision_correct += predicted_answerable == expected_answerable
            selected_elements = {
                element_id
                for item in result["evidence"]
                for element_id in item["element_ids"]
            }
            gold_elements = {
                item["element_id"] for item in question["gold_evidence"]
            }
            citation_hit = bool(selected_elements.intersection(gold_elements))
            if expected_answerable:
                answerable_count += 1
                answered_answerable += predicted_answerable
                citation_hits += citation_hit
            else:
                no_answer_count += 1
                correct_refusals += not predicted_answerable
            violations = _metadata_violations(
                result["evidence"],
                question["filters"],
            )
            metadata_violations += violations
            details.append(
                {
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "expected_answerable": expected_answerable,
                    "status": result["status"],
                    "reason": result["reason"],
                    "confidence": result["confidence"],
                    "citation_hit": citation_hit,
                    "metadata_violations": violations,
                    "evidence": [
                        {
                            "citation_id": item["citation_id"],
                            "chunk_id": item["chunk_id"],
                            "retrieval_rank": item["retrieval_rank"],
                            "rerank_score": item["rerank_score"],
                            "page_nos": item["page_nos"],
                        }
                        for item in result["evidence"]
                    ],
                }
            )

        metrics = {
            "decision_accuracy": decision_correct / len(questions),
            "answerable_response_rate": (
                answered_answerable / answerable_count
                if answerable_count
                else 0.0
            ),
            "citation_recall": (
                citation_hits / answerable_count if answerable_count else 0.0
            ),
            "no_answer_accuracy": (
                correct_refusals / no_answer_count if no_answer_count else 0.0
            ),
            "metadata_filter_violations": metadata_violations,
        }
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backend": "extractive_evidence_v1",
            "index_path": args.index.resolve().as_posix(),
            "questions": args.questions.resolve().as_posix(),
            "question_count": len(questions),
            "answerable_questions": answerable_count,
            "no_answer_questions": no_answer_count,
            "configuration": {
                "retrieval_limit": args.retrieval_limit,
                "max_evidence": args.max_evidence,
                "min_bm25_score": args.min_bm25_score,
                "min_confidence": args.min_confidence,
            },
            "metrics": metrics,
            "limitations": [
                "Answers are extractive evidence quotes, not LLM generation.",
                "Citation recall checks Gold element membership.",
                "Answer semantic accuracy is not scored in this baseline.",
            ],
            "details": details,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                relativize_project_paths(report, PROJECT_ROOT),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Decision accuracy: {metrics['decision_accuracy']:.4f}")
    print(
        "Answerable response rate: "
        f"{metrics['answerable_response_rate']:.4f}"
    )
    print(f"Citation recall: {metrics['citation_recall']:.4f}")
    print(f"No-answer accuracy: {metrics['no_answer_accuracy']:.4f}")
    print(
        "Metadata filter violations: "
        f"{metrics['metadata_filter_violations']}"
    )
    print(f"Metrics: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
