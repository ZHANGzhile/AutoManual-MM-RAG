#!/usr/bin/env python3
"""Evaluate curated table-row retrieval and exact-value answers."""

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

from automanual_rag.retrieval.table_rows import TableRowIndex
from automanual_rag.serialization import relativize_project_paths
from automanual_rag.table_answering import answer_table_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate curated table rows.")
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "indexes"
        / "table_rows.sqlite3",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "eval"
        / "table_row_questions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "table_row_answering.json",
    )
    return parser.parse_args()


def _load(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            values.append(value)
    if not values:
        raise ValueError("No table-row questions found")
    return values


def main() -> int:
    args = parse_args()
    try:
        questions = _load(args.questions.resolve())
        index = TableRowIndex(args.index.resolve())
        top1_hits = 0
        top5_hits = 0
        exact_value_hits = 0
        answered = 0
        decision_hits = 0
        no_answer_hits = 0
        metadata_violations = 0
        answerable_count = 0
        no_answer_count = 0
        details: list[dict[str, Any]] = []
        for question in questions:
            expected_answerable = question.get("answerable", True)
            answerable_count += expected_answerable
            no_answer_count += not expected_answerable
            result = answer_table_question(
                index,
                question["question"],
                question["filters"],
                retrieval_limit=5,
            )
            evidence = result["evidence"]
            ranks = {
                item["row_id"]: item["rank"] for item in evidence
            }
            gold_row_id = question.get("gold_row_id")
            gold_rank = ranks.get(gold_row_id) if gold_row_id else None
            if expected_answerable:
                top1_hits += gold_rank == 1
                top5_hits += gold_rank is not None and gold_rank <= 5
            is_answered = result["status"] == "answered"
            answered += is_answered
            decision_hit = is_answered == expected_answerable
            decision_hits += decision_hit
            if not expected_answerable:
                no_answer_hits += not is_answered
            value_hit = (
                expected_answerable
                and is_answered
                and all(
                    expected.casefold() in result["answer"].casefold()
                    for expected in question.get("expected_values", [])
                )
            )
            exact_value_hits += value_hit
            violations = sum(
                any(
                    str(item.get(field, "")).casefold()
                    != str(expected).casefold()
                    for field, expected in question["filters"].items()
                    if field in item
                )
                for item in evidence
            )
            metadata_violations += violations
            details.append(
                {
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "expected_answerable": expected_answerable,
                    "status": result["status"],
                    "reason": result["reason"],
                    "decision_hit": decision_hit,
                    "gold_row_id": gold_row_id,
                    "gold_rank": gold_rank,
                    "exact_value_hit": value_hit,
                    "metadata_filter_violations": violations,
                    "top_row_id": (
                        evidence[0]["row_id"] if evidence else None
                    ),
                }
            )
        total = len(questions)
        metrics = {
            "row_recall_at_1": top1_hits / answerable_count,
            "row_recall_at_5": top5_hits / answerable_count,
            "answer_rate": answered / total,
            "decision_accuracy": decision_hits / total,
            "no_answer_accuracy": (
                no_answer_hits / no_answer_count
                if no_answer_count
                else None
            ),
            "exact_value_coverage": (
                exact_value_hits / answerable_count
            ),
            "metadata_filter_violations": metadata_violations,
        }
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backend": index.metadata["backend"],
            "index_path": args.index.resolve().as_posix(),
            "questions": args.questions.resolve().as_posix(),
            "question_count": total,
            "answerable_question_count": answerable_count,
            "no_answer_question_count": no_answer_count,
            "curated_row_count": index.count(),
            "source_table_count": int(
                index.metadata["source_table_count"]
            ),
            "asset_verification": index.metadata["asset_verification"],
            "metrics": metrics,
            "limitations": [
                "Rows cover nine manually selected source tables.",
                "This is a curated development benchmark, not broad OCR.",
                "Exact-value coverage checks expected strings in the answer.",
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
    print(f"Questions: {total}")
    print(f"Row Recall@1: {metrics['row_recall_at_1']:.4f}")
    print(f"Row Recall@5: {metrics['row_recall_at_5']:.4f}")
    print(f"Answer rate: {metrics['answer_rate']:.4f}")
    print(f"Decision accuracy: {metrics['decision_accuracy']:.4f}")
    if metrics["no_answer_accuracy"] is not None:
        print(
            "No-answer accuracy: "
            f"{metrics['no_answer_accuracy']:.4f}"
        )
    print(
        "Exact value coverage: "
        f"{metrics['exact_value_coverage']:.4f}"
    )
    print(
        "Metadata filter violations: "
        f"{metrics['metadata_filter_violations']}"
    )
    print(f"Metrics: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
