"""Exact answers from manually verified table rows."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


BACKEND_NAME = "curated_table_row_answer_v1"
DEFAULT_MIN_SCORE = 3.0
DEFAULT_MIN_QUERY_COVERAGE = 0.6
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "much",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _vehicle_context_present(filters: Mapping[str, str]) -> bool:
    if str(filters.get("doc_id", "")).strip():
        return True
    return bool(
        str(filters.get("model", "")).strip()
        and str(filters.get("year", "")).strip()
    )


def _source(result: Mapping[str, Any]) -> str:
    return (
        f"{result['brand']} {result['model']} Owner's Manual "
        f"({result['year']}), "
        f"{' > '.join(result['section_path'])}, "
        f"physical PDF p.{result['page_no']}"
    )


def _markdown_row(cells: Mapping[str, str]) -> str:
    headers = list(cells)
    values = [str(cells[header]) for header in headers]
    return "\n".join(
        (
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            "| " + " | ".join(values) + " |",
        )
    )


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if token.casefold() not in STOPWORDS and len(token) > 1
    }


def _query_coverage(query: str, result: Mapping[str, Any]) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    row_text = " ".join(
        (
            str(result.get("model", "")),
            " ".join(result.get("section_path", [])),
            " ".join(
                f"{header} {value}"
                for header, value in result.get("cells", {}).items()
            ),
            " ".join(result.get("aliases", [])),
        )
    )
    return len(query_tokens.intersection(_tokens(row_text))) / len(query_tokens)


def _applicability_conflict(
    query: str,
    result: Mapping[str, Any],
) -> bool:
    section = " ".join(result.get("section_path", [])).casefold()
    query_text = query.casefold()
    excludes_hybrid = "excluding" in section and "hybrid" in section
    asks_hybrid = "hybrid" in query_text and not re.search(
        r"\b(?:non[- ]hybrid|excluding\s+(?:the\s+)?hybrid)\b",
        query_text,
    )
    return excludes_hybrid and asks_hybrid


def answer_table_from_results(
    query: str,
    filters: Mapping[str, str],
    results: Sequence[Mapping[str, Any]],
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    min_query_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
    max_evidence: int = 3,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if min_score < 0:
        raise ValueError("min_score must not be negative")
    if not 0 <= min_query_coverage <= 1:
        raise ValueError("min_query_coverage must be from 0 to 1")
    if not 1 <= max_evidence <= 10:
        raise ValueError("max_evidence must be from 1 to 10")
    base = {
        "backend": BACKEND_NAME,
        "query": query,
        "filters": dict(filters),
    }
    if not _vehicle_context_present(filters):
        return {
            **base,
            "status": "needs_context",
            "reason": "model_and_year_required",
            "answer": (
                "Please select a vehicle model and year before searching "
                "table values."
            ),
            "evidence": [],
        }
    evidence = [
        {
            "citation_id": index + 1,
            "row_id": result["row_id"],
            "element_id": result["element_id"],
            "doc_id": result["doc_id"],
            "brand": result["brand"],
            "model": result["model"],
            "year": result["year"],
            "region": result["region"],
            "language": result["language"],
            "manual_type": result["manual_type"],
            "page_no": result["page_no"],
            "section_path": list(result["section_path"]),
            "cells": dict(result["cells"]),
            "asset_path": result["asset_path"],
            "asset_sha256": result["asset_sha256"],
            "score": float(result["score"]),
            "rank": int(result["rank"]),
            "transcription_method": result["transcription_method"],
            "query_coverage": round(_query_coverage(query, result), 6),
            "applicability_conflict": _applicability_conflict(
                query,
                result,
            ),
            "source": _source(result),
        }
        for index, result in enumerate(results[:max_evidence])
    ]
    if not evidence:
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": "no_curated_row",
            "answer": (
                "No manually verified table row covers this question. "
                "Check the matching table crop instead."
            ),
            "evidence": [],
        }
    if evidence[0]["score"] < min_score:
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": "row_below_threshold",
            "answer": (
                "The curated table-row evidence is too weak for an exact "
                "value answer. Check the source table crop."
            ),
            "evidence": evidence,
        }
    if (
        evidence[0]["query_coverage"] < min_query_coverage
        or evidence[0]["applicability_conflict"]
    ):
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": (
                "applicability_conflict"
                if evidence[0]["applicability_conflict"]
                else "row_semantic_coverage_too_low"
            ),
            "answer": (
                "A related curated row was found, but it does not safely "
                "cover the requested value or vehicle variant. Check the "
                "source table crop."
            ),
            "evidence": evidence,
        }
    strongest = evidence[0]
    answer = "\n\n".join(
        (
            "Manually verified table row:",
            _markdown_row(strongest["cells"]),
            f"Source: [1] {strongest['source']}",
            (
                "Verification: manual visual transcription; source image "
                f"SHA-256 `{strongest['asset_sha256']}`."
            ),
        )
    )
    return {
        **base,
        "status": "answered",
        "reason": "verified_table_row",
        "answer": answer,
        "evidence": evidence,
    }


def answer_table_question(
    index: Any,
    query: str,
    filters: Mapping[str, str],
    *,
    retrieval_limit: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
    min_query_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
    max_evidence: int = 3,
) -> dict[str, Any]:
    if not _vehicle_context_present(filters):
        return answer_table_from_results(
            query,
            filters,
            [],
            min_score=min_score,
            min_query_coverage=min_query_coverage,
            max_evidence=max_evidence,
        )
    results = index.search(
        query,
        filters=filters,
        limit=retrieval_limit,
    )
    return answer_table_from_results(
        query,
        filters,
        results,
        min_score=min_score,
        min_query_coverage=min_query_coverage,
        max_evidence=max_evidence,
    )
