"""Evidence-constrained extractive answers with portable citations."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


ANSWER_BACKEND = "extractive_evidence_v1"
DEFAULT_MIN_BM25_SCORE = 12.0
DEFAULT_MIN_CONFIDENCE = 0.55
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
STOP_WORDS = {
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
    "my",
    "of",
    "on",
    "should",
    "the",
    "to",
    "use",
    "what",
    "when",
    "where",
    "which",
    "with",
}
PROCEDURE_QUERY_RE = re.compile(
    r"^\s*(?:how\s+(?:do|can|should)\b|which\s+tools?\b|what\s+steps?\b)",
    re.IGNORECASE,
)
PROCEDURE_CONTENT_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+[.)]|step\s+\d+)\s+"
    r"|(?:^|[.!?]\s+)(?:open|press|remove|install|select|pull|push|turn|"
    r"unlock|lock|move|connect|disconnect|insert|lift|lower|check|make sure|"
    r"use)\b",
    re.IGNORECASE,
)
PROCEDURE_SECTION_RE = re.compile(
    r"\b(?:adjusting|enabling|installing|opening|removing|replacing|"
    r"switching|using)\b",
    re.IGNORECASE,
)


def _normalize_token(token: str) -> str:
    value = token.casefold()
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    elif len(value) > 4 and value.endswith("es"):
        value = value[:-2]
    elif len(value) > 3 and value.endswith("s"):
        value = value[:-1]
    return value


def _content_terms(text: str) -> set[str]:
    return {
        normalized
        for token in TOKEN_RE.findall(text)
        if (normalized := _normalize_token(token)) not in STOP_WORDS
        and len(normalized) > 1
    }


def _coverage(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms.intersection(_content_terms(text))) / len(query_terms)


def _vehicle_context_present(filters: Mapping[str, str]) -> bool:
    if str(filters.get("doc_id", "")).strip():
        return True
    return bool(
        str(filters.get("model", "")).strip()
        and str(filters.get("year", "")).strip()
    )


def _is_procedural_query(query: str) -> bool:
    return PROCEDURE_QUERY_RE.search(query) is not None


def _has_procedure_support(result: Mapping[str, Any]) -> bool:
    if str(result.get("chunk_type", "")) == "steps":
        return True
    section = " ".join(str(item) for item in result.get("section_path", []))
    content = str(result.get("content", ""))
    return bool(
        PROCEDURE_CONTENT_RE.search(content)
        or PROCEDURE_SECTION_RE.search(section)
    )


def _source_label(result: Mapping[str, Any]) -> str:
    section = " > ".join(str(item) for item in result["section_path"])
    pages = [int(page) for page in result["page_nos"]]
    page_label = (
        str(pages[0])
        if len(pages) == 1
        else f"{min(pages)}–{max(pages)}"
    )
    return (
        f"{result['brand']} {result['model']} Owner's Manual "
        f"({result['year']}), {section}, physical PDF p.{page_label}"
    )


def _compact_content(content: str, limit: int = 900) -> str:
    value = content.strip()
    if len(value) <= limit:
        return value
    shortened = value[:limit].rsplit(" ", 1)[0].rstrip()
    return shortened + "…"


def build_evidence_pack(
    query: str,
    results: Sequence[Mapping[str, Any]],
    *,
    max_evidence: int = 3,
) -> list[dict[str, Any]]:
    """Rerank retrieval results and create citation-ready evidence records."""
    if not 1 <= max_evidence <= 10:
        raise ValueError("max_evidence must be from 1 to 10")

    query_terms = _content_terms(query)
    scored: list[
        tuple[float, int, Mapping[str, Any], float, bool]
    ] = []
    for result in results:
        content = str(result.get("content", ""))
        section = " ".join(str(item) for item in result.get("section_path", []))
        content_coverage = _coverage(query_terms, f"{section} {content}")
        section_coverage = _coverage(query_terms, section)
        bm25_score = max(float(result.get("score", 0.0)), 0.0)
        score_strength = min(bm25_score / 20.0, 1.0)
        procedure_support = _has_procedure_support(result)
        procedure_bonus = (
            0.08
            if _is_procedural_query(query) and procedure_support
            else 0.0
        )
        rerank_score = min(
            0.55 * score_strength
            + 0.35 * content_coverage
            + 0.10 * section_coverage
            + procedure_bonus,
            1.0,
        )
        scored.append(
            (
                rerank_score,
                int(result.get("rank", len(scored) + 1)),
                result,
                content_coverage,
                procedure_support,
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]))
    evidence: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for (
        rerank_score,
        retrieval_rank,
        result,
        coverage,
        procedure_support,
    ) in scored:
        chunk_id = str(result["chunk_id"])
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        citation_id = len(evidence) + 1
        evidence.append(
            {
                "citation_id": citation_id,
                "chunk_id": chunk_id,
                "element_ids": list(result.get("element_ids", [])),
                "doc_id": result["doc_id"],
                "brand": result["brand"],
                "model": result["model"],
                "year": result["year"],
                "region": result["region"],
                "language": result["language"],
                "manual_type": result["manual_type"],
                "page_nos": list(result["page_nos"]),
                "section_path": list(result["section_path"]),
                "chunk_type": result["chunk_type"],
                "content": _compact_content(str(result["content"])),
                "retrieval_rank": retrieval_rank,
                "retrieval_score": float(result["score"]),
                "query_term_coverage": round(coverage, 6),
                "procedure_support": procedure_support,
                "rerank_score": round(rerank_score, 6),
                "source": _source_label(result),
            }
        )
        if len(evidence) == max_evidence:
            break
    return evidence


def answer_from_results(
    query: str,
    filters: Mapping[str, str],
    results: Sequence[Mapping[str, Any]],
    *,
    max_evidence: int = 3,
    min_bm25_score: float = DEFAULT_MIN_BM25_SCORE,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Create an extractive answer or refuse when evidence is insufficient."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= max_evidence <= 10:
        raise ValueError("max_evidence must be from 1 to 10")
    if min_bm25_score < 0:
        raise ValueError("min_bm25_score must not be negative")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be from 0 to 1")

    base = {
        "backend": ANSWER_BACKEND,
        "query": query,
        "filters": dict(filters),
    }
    if not _vehicle_context_present(filters):
        return {
            **base,
            "status": "needs_context",
            "reason": "model_and_year_required",
            "confidence": 0.0,
            "answer": (
                "Please select a vehicle model and year before searching. "
                "Evidence from different manuals must not be mixed."
            ),
            "evidence": [],
        }

    evidence = build_evidence_pack(
        query,
        results,
        max_evidence=max_evidence,
    )
    if not evidence:
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": "no_retrieval_results",
            "confidence": 0.0,
            "answer": "Current manual evidence is insufficient.",
            "evidence": [],
        }

    strongest = evidence[0]
    confidence = float(strongest["rerank_score"])
    if _is_procedural_query(query) and not strongest["procedure_support"]:
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": "procedure_not_supported",
            "confidence": round(confidence, 6),
            "answer": (
                "Current manual evidence mentions related terms but does not "
                "provide the requested procedure."
            ),
            "evidence": evidence,
        }
    if (
        float(strongest["retrieval_score"]) < min_bm25_score
        or confidence < min_confidence
    ):
        return {
            **base,
            "status": "insufficient_evidence",
            "reason": "evidence_below_threshold",
            "confidence": round(confidence, 6),
            "answer": (
                "Current manual evidence is insufficient. Try a more specific "
                "question or verify the selected model and year."
            ),
            "evidence": evidence,
        }

    answer_parts = ["According to the selected owner-manual evidence:"]
    for item in evidence:
        answer_parts.append(f"[{item['citation_id']}] {item['content']}")
    if any(
        item["chunk_type"] in {"warning", "caution"} for item in evidence
    ):
        answer_parts.append(
            "Safety note: follow the quoted Warning/Caution and the vehicle's "
            "current messages; use professional service when required."
        )
    answer_parts.append("Sources:")
    answer_parts.extend(
        f"[{item['citation_id']}] {item['source']}" for item in evidence
    )
    return {
        **base,
        "status": "answered",
        "reason": "sufficient_retrieved_evidence",
        "confidence": round(confidence, 6),
        "answer": "\n\n".join(answer_parts),
        "evidence": evidence,
    }


def answer_question(
    index: Any,
    query: str,
    filters: Mapping[str, str],
    *,
    retrieval_limit: int = 10,
    max_evidence: int = 3,
    min_bm25_score: float = DEFAULT_MIN_BM25_SCORE,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, Any]:
    """Retrieve filtered chunks and produce a grounded answer."""
    if not _vehicle_context_present(filters):
        return answer_from_results(
            query,
            filters,
            [],
            max_evidence=max_evidence,
            min_bm25_score=min_bm25_score,
            min_confidence=min_confidence,
        )
    results = index.search(
        query,
        filters=filters,
        limit=retrieval_limit,
    )
    return answer_from_results(
        query,
        filters,
        results,
        max_evidence=max_evidence,
        min_bm25_score=min_bm25_score,
        min_confidence=min_confidence,
    )
