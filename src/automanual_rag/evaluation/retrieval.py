"""Evidence-retrieval metrics shared by local retrieval backends."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from automanual_rag.retrieval.bm25 import BM25Index, FILTER_FIELDS
from automanual_rag.retrieval.dense import DenseIndex
from automanual_rag.retrieval.hybrid import HybridIndex


REQUIRED_QUESTION_FIELDS = {
    "question_id",
    "category",
    "question",
    "filters",
    "gold_evidence",
    "reference_answer",
    "answerable",
}


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation set not found: {path}")
    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: question must be an object")
            missing = REQUIRED_QUESTION_FIELDS.difference(value)
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields "
                    + ", ".join(sorted(missing))
                )
            question_id = value["question_id"]
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"{path}:{line_number}: invalid question_id")
            if question_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_number}: duplicate question_id {question_id!r}"
                )
            seen_ids.add(question_id)
            if not isinstance(value["answerable"], bool):
                raise ValueError(f"{path}:{line_number}: answerable must be boolean")
            if not isinstance(value["filters"], dict) or not value["filters"]:
                raise ValueError(f"{path}:{line_number}: filters must be non-empty")
            unknown_filters = set(value["filters"]).difference(FILTER_FIELDS)
            if unknown_filters:
                raise ValueError(
                    f"{path}:{line_number}: unsupported filters "
                    + ", ".join(sorted(unknown_filters))
                )
            if not isinstance(value["gold_evidence"], list):
                raise ValueError(
                    f"{path}:{line_number}: gold_evidence must be a list"
                )
            if value["answerable"] and not value["gold_evidence"]:
                raise ValueError(
                    f"{path}:{line_number}: answerable question requires evidence"
                )
            if not value["answerable"] and value["gold_evidence"]:
                raise ValueError(
                    f"{path}:{line_number}: no-answer question must not have evidence"
                )
            for evidence in value["gold_evidence"]:
                if not isinstance(evidence, dict) or not {
                    "doc_id",
                    "page_no",
                    "element_id",
                }.issubset(evidence):
                    raise ValueError(
                        f"{path}:{line_number}: malformed gold evidence"
                    )
            questions.append(value)
    if not questions:
        raise ValueError(f"Evaluation set is empty: {path}")
    return questions


class RetrievalIndex(Protocol):
    path: Any

    def count(self) -> int: ...

    def element_membership(self) -> dict[str, dict[str, Any]]: ...

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...


def _validate_gold_evidence(
    questions: Sequence[Mapping[str, Any]],
    membership: Mapping[str, Mapping[str, Any]],
) -> None:
    for question in questions:
        for evidence in question["gold_evidence"]:
            element_id = evidence["element_id"]
            location = membership.get(element_id)
            if location is None:
                raise ValueError(
                    f"{question['question_id']}: gold element not indexed: {element_id}"
                )
            if location["doc_id"] != evidence["doc_id"]:
                raise ValueError(
                    f"{question['question_id']}: gold doc_id mismatch for {element_id}"
                )
            if evidence["page_no"] not in location["page_nos"]:
                raise ValueError(
                    f"{question['question_id']}: gold page mismatch for {element_id}"
                )


def _filter_violation(
    result: Mapping[str, Any],
    filters: Mapping[str, str],
) -> bool:
    for field, expected in filters.items():
        actual = result.get(field)
        if actual is None or str(actual).casefold() != str(expected).casefold():
            return True
    return False


def _metric_summary(first_ranks: Sequence[int | None]) -> dict[str, float | int]:
    count = len(first_ranks)
    if not count:
        return {
            "questions": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
        }
    return {
        "questions": count,
        "recall_at_5": sum(
            rank is not None and rank <= 5 for rank in first_ranks
        )
        / count,
        "recall_at_10": sum(
            rank is not None and rank <= 10 for rank in first_ranks
        )
        / count,
        "mrr_at_10": sum(
            1.0 / rank
            for rank in first_ranks
            if rank is not None and rank <= 10
        )
        / count,
    }


def evaluate_retriever(
    *,
    index: RetrievalIndex,
    backend: str,
    questions: Sequence[Mapping[str, Any]],
    limit: int = 10,
) -> dict[str, Any]:
    if limit < 10:
        raise ValueError("Evaluation limit must be at least 10")
    membership = index.element_membership()
    _validate_gold_evidence(questions, membership)

    answerable_ranks: list[int | None] = []
    category_ranks: dict[str, list[int | None]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    metadata_violations = 0
    retrieved_results = 0
    no_answer_with_results = 0

    for question in questions:
        results = index.search(
            question["question"],
            filters=question["filters"],
            limit=limit,
        )
        retrieved_results += len(results)
        metadata_violations += sum(
            _filter_violation(result, question["filters"])
            for result in results
        )

        gold_ids = {
            evidence["element_id"] for evidence in question["gold_evidence"]
        }
        first_rank: int | None = None
        if question["answerable"]:
            for result in results:
                if gold_ids.intersection(result["element_ids"]):
                    first_rank = result["rank"]
                    break
            answerable_ranks.append(first_rank)
            category_ranks[question["category"]].append(first_rank)
        elif results:
            no_answer_with_results += 1

        details.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "answerable": question["answerable"],
                "first_relevant_rank": first_rank,
                "top_result": (
                    {
                        "chunk_id": results[0]["chunk_id"],
                        "doc_id": results[0]["doc_id"],
                        "page_nos": results[0]["page_nos"],
                        "section_path": results[0]["section_path"],
                        "score": results[0]["score"],
                    }
                    if results
                    else None
                ),
            }
        )

    metrics = _metric_summary(answerable_ranks)
    metrics.update(
        {
            "metadata_filter_violations": metadata_violations,
            "metadata_filter_violation_rate": (
                metadata_violations / retrieved_results
                if retrieved_results
                else 0.0
            ),
        }
    )
    return {
        "backend": backend,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_path": (
            index.path.as_posix()
            if isinstance(index.path, Path)
            else index.path
        ),
        "index_chunks": index.count(),
        "questions_total": len(questions),
        "answerable_questions": len(answerable_ranks),
        "no_answer_questions": sum(
            not question["answerable"] for question in questions
        ),
        "metrics": metrics,
        "category_metrics": {
            category: _metric_summary(ranks)
            for category, ranks in sorted(category_ranks.items())
        },
        "no_answer_diagnostic": {
            "cases_returning_candidates": no_answer_with_results,
            "abstention_threshold_configured": False,
            "included_in_recall_metrics": False,
        },
        "details": details,
    }


def evaluate_bm25(
    *,
    index: BM25Index,
    questions: Sequence[Mapping[str, Any]],
    limit: int = 10,
) -> dict[str, Any]:
    return evaluate_retriever(
        index=index,
        backend="sqlite_fts5_bm25",
        questions=questions,
        limit=limit,
    )


def evaluate_dense(
    *,
    index: DenseIndex,
    questions: Sequence[Mapping[str, Any]],
    limit: int = 10,
) -> dict[str, Any]:
    return evaluate_retriever(
        index=index,
        backend="hashed_tfidf_randomized_lsa",
        questions=questions,
        limit=limit,
    )


def evaluate_hybrid(
    *,
    index: HybridIndex,
    questions: Sequence[Mapping[str, Any]],
    limit: int = 10,
) -> dict[str, Any]:
    return evaluate_retriever(
        index=index,
        backend="rrf_bm25_dense",
        questions=questions,
        limit=limit,
    )
