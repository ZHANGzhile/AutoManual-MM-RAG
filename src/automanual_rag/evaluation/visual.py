"""Gold Evidence evaluation for visual and visual/text fusion retrieval."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from automanual_rag.retrieval.visual import (
    VISUAL_FILTER_FIELDS,
    VisualIndex,
    VisualTextFusionIndex,
)


REQUIRED_FIELDS = {
    "query_id",
    "category",
    "split",
    "query_image",
    "query_text",
    "filters",
    "gold_evidence",
    "source",
    "transform",
}


def load_visual_questions(
    path: Path,
    *,
    project_root: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Visual evaluation set not found: {path}")
    project_root = project_root.resolve()
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: query must be an object")
            missing = REQUIRED_FIELDS.difference(value)
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields "
                    + ", ".join(sorted(missing))
                )
            query_id = value["query_id"]
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"{path}:{line_number}: invalid query_id")
            if query_id in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate query_id {query_id!r}"
                )
            seen.add(query_id)
            if value["split"] not in {"dev", "test"}:
                raise ValueError(f"{path}:{line_number}: invalid split")
            filters = value["filters"]
            if not isinstance(filters, dict) or not filters:
                raise ValueError(f"{path}:{line_number}: filters must be non-empty")
            unknown = set(filters).difference(VISUAL_FILTER_FIELDS)
            if unknown:
                raise ValueError(
                    f"{path}:{line_number}: unsupported filters "
                    + ", ".join(sorted(unknown))
                )
            evidence = value["gold_evidence"]
            if not isinstance(evidence, dict) or not {
                "element_id",
                "doc_id",
                "page_no",
                "asset_path",
            }.issubset(evidence):
                raise ValueError(
                    f"{path}:{line_number}: malformed gold_evidence"
                )
            query_path = (project_root / value["query_image"]).resolve()
            try:
                query_path.relative_to(project_root)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: query image escapes project root"
                ) from exc
            if not query_path.is_file():
                raise FileNotFoundError(
                    f"{path}:{line_number}: query image not found: {query_path}"
                )
            value["_query_path"] = query_path
            questions.append(value)
    if not questions:
        raise ValueError(f"Visual evaluation set is empty: {path}")
    return questions


def _filter_violation(
    result: Mapping[str, Any],
    filters: Mapping[str, str],
) -> bool:
    return any(
        str(result.get(field, "")).casefold() != str(expected).casefold()
        for field, expected in filters.items()
    )


def _summary(ranks: Sequence[int | None]) -> dict[str, float | int]:
    if not ranks:
        return {
            "queries": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
        }
    count = len(ranks)
    return {
        "queries": count,
        "recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "recall_at_5": sum(
            rank is not None and rank <= 5 for rank in ranks
        )
        / count,
        "recall_at_10": sum(
            rank is not None and rank <= 10 for rank in ranks
        )
        / count,
        "mrr_at_10": sum(
            1.0 / rank for rank in ranks if rank is not None and rank <= 10
        )
        / count,
    }


def evaluate_visual(
    *,
    index: VisualIndex | VisualTextFusionIndex,
    questions: Sequence[Mapping[str, Any]],
    backend: str,
    use_text_hint: bool,
    limit: int = 10,
) -> dict[str, Any]:
    if limit < 10:
        raise ValueError("Visual evaluation limit must be at least 10")
    all_ranks: list[int | None] = []
    split_ranks: dict[str, list[int | None]] = defaultdict(list)
    category_ranks: dict[str, list[int | None]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    violations = 0
    retrieved = 0

    for question in questions:
        if use_text_hint:
            assert isinstance(index, VisualTextFusionIndex)
            results = index.search(
                question["_query_path"],
                query_text=question["query_text"],
                filters=question["filters"],
                limit=limit,
            )
        else:
            assert isinstance(index, VisualIndex)
            results = index.search(
                question["_query_path"],
                filters=question["filters"],
                limit=limit,
            )
        retrieved += len(results)
        violations += sum(
            _filter_violation(result, question["filters"])
            for result in results
        )
        gold_id = question["gold_evidence"]["element_id"]
        rank = next(
            (
                int(result["rank"])
                for result in results
                if result["element_id"] == gold_id
            ),
            None,
        )
        all_ranks.append(rank)
        split_ranks[question["split"]].append(rank)
        category_ranks[question["category"]].append(rank)
        details.append(
            {
                "query_id": question["query_id"],
                "category": question["category"],
                "split": question["split"],
                "first_relevant_rank": rank,
                "gold_element_id": gold_id,
                "top_result": (
                    {
                        "element_id": results[0]["element_id"],
                        "doc_id": results[0]["doc_id"],
                        "page_no": results[0]["page_no"],
                        "section_path": results[0]["section_path"],
                        "asset_path": results[0]["asset_path"],
                        "score": results[0]["score"],
                    }
                    if results
                    else None
                ),
            }
        )

    metrics = _summary(all_ranks)
    metrics.update(
        {
            "metadata_filter_violations": violations,
            "metadata_filter_violation_rate": (
                violations / retrieved if retrieved else 0.0
            ),
        }
    )
    return {
        "backend": backend,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "use_text_hint": use_text_hint,
        "query_count": len(questions),
        "metrics": metrics,
        "split_metrics": {
            split: _summary(ranks)
            for split, ranks in sorted(split_ranks.items())
        },
        "category_metrics": {
            category: _summary(ranks)
            for category, ranks in sorted(category_ranks.items())
        },
        "details": details,
    }
