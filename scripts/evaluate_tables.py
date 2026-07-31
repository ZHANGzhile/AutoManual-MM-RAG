#!/usr/bin/env python3
"""Evaluate table-crop evidence retrieval."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.retrieval.table import TableIndex
from automanual_rag.serialization import relativize_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate table retrieval.")
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "tables.sqlite3",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "table_questions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "table_retrieval_baseline.json",
    )
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def _load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            required = {
                "query_id",
                "query",
                "filters",
                "gold_element_id",
                "gold_page_no",
                "category",
            }
            missing = required.difference(value)
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing {', '.join(sorted(missing))}"
                )
            query_id = str(value["query_id"])
            if query_id in seen:
                raise ValueError(f"Duplicate query_id: {query_id}")
            seen.add(query_id)
            questions.append(value)
    if not questions:
        raise ValueError("No table questions found")
    return questions


def _summary(ranks: list[int | None]) -> dict[str, float]:
    total = len(ranks)
    return {
        "recall_at_1": sum(rank == 1 for rank in ranks) / total,
        "recall_at_5": sum(
            rank is not None and rank <= 5 for rank in ranks
        )
        / total,
        "recall_at_10": sum(
            rank is not None and rank <= 10 for rank in ranks
        )
        / total,
        "mrr_at_10": sum(
            1.0 / rank
            for rank in ranks
            if rank is not None and rank <= 10
        )
        / total,
    }


def main() -> int:
    args = parse_args()
    try:
        questions = _load_questions(args.questions.resolve())
        index = TableIndex(args.index.resolve())
        ranks: list[int | None] = []
        category_ranks: dict[str, list[int | None]] = defaultdict(list)
        details: list[dict[str, Any]] = []
        violations = 0
        for question in questions:
            results = index.search(
                question["query"],
                filters=question["filters"],
                limit=args.limit,
            )
            rank = next(
                (
                    result["rank"]
                    for result in results
                    if result["element_id"] == question["gold_element_id"]
                ),
                None,
            )
            ranks.append(rank)
            category_ranks[question["category"]].append(rank)
            result_violations = sum(
                any(
                    str(result.get(field, "")).casefold()
                    != str(expected).casefold()
                    for field, expected in question["filters"].items()
                )
                for result in results
            )
            violations += result_violations
            details.append(
                {
                    "query_id": question["query_id"],
                    "category": question["category"],
                    "gold_element_id": question["gold_element_id"],
                    "gold_page_no": question["gold_page_no"],
                    "first_relevant_rank": rank,
                    "metadata_filter_violations": result_violations,
                    "top_result": (
                        {
                            "element_id": results[0]["element_id"],
                            "page_no": results[0]["page_no"],
                            "section_path": results[0]["section_path"],
                            "score": results[0]["score"],
                        }
                        if results
                        else None
                    ),
                }
            )
        metrics = _summary(ranks)
        metrics["metadata_filter_violations"] = violations
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "backend": index.metadata["backend"],
            "index_path": args.index.resolve().as_posix(),
            "questions": args.questions.resolve().as_posix(),
            "query_count": len(questions),
            "indexed_tables": index.count(),
            "structured_tables": int(
                index.metadata["structured_element_count"]
            ),
            "metrics": metrics,
            "category_metrics": {
                category: _summary(values)
                for category, values in sorted(category_ranks.items())
            },
            "limitations": [
                "The index searches section, caption, and adjacent text.",
                "Table cells are not structured and row values are not indexed.",
                "The benchmark measures table-crop localization, not value QA.",
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

    print(f"Queries: {len(questions)}")
    print(f"Recall@1: {metrics['recall_at_1']:.4f}")
    print(f"Recall@5: {metrics['recall_at_5']:.4f}")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"MRR@10: {metrics['mrr_at_10']:.4f}")
    print(f"Metadata filter violations: {violations}")
    print(f"Metrics: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
