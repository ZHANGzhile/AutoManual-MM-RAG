"""Evaluation helpers for Baseline RAG, GraphRAG, and Agentic GraphRAG."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Mapping, Sequence

from automanual_rag.agentic import AgenticWorkflow
from automanual_rag.answering import answer_question
from automanual_rag.generation import validate_grounded_text


REQUIRED_FIELDS = {
    "question_id",
    "question",
    "filters",
    "answerable",
    "gold_evidence",
    "gold_path",
    "expected_route",
}


def load_agentic_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            missing = REQUIRED_FIELDS.difference(value)
            if missing:
                raise ValueError(
                    f"{path}:{line_no} missing: {', '.join(sorted(missing))}"
                )
            if not isinstance(value["filters"], dict):
                raise ValueError(f"{path}:{line_no} filters must be an object")
            if not isinstance(value["answerable"], bool):
                raise ValueError(
                    f"{path}:{line_no} answerable must be boolean"
                )
            questions.append(value)
    if not questions:
        raise ValueError("Agentic evaluation set is empty")
    return questions


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(
        0,
        min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1),
    )
    return float(ordered[position])


def _metadata_violations(
    evidence: Sequence[Mapping[str, Any]],
    filters: Mapping[str, str],
) -> int:
    violations = 0
    for item in evidence:
        for field in (
            "doc_id",
            "brand",
            "model",
            "year",
            "region",
            "language",
            "manual_type",
        ):
            expected = str(filters.get(field, "")).strip()
            if (
                expected
                and str(item.get(field, "")).casefold()
                != expected.casefold()
            ):
                violations += 1
                break
    return violations


def _citation_faithful(result: Mapping[str, Any]) -> bool | None:
    if result.get("status") != "answered":
        return None
    try:
        validate_grounded_text(
            str(result.get("answer", "")),
            len(result.get("evidence", [])),
        )
    except (RuntimeError, ValueError):
        return False
    return True


def _evidence_recall(
    gold: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    evidence_ids = {
        str(value)
        for item in evidence
        for value in item.get(
            "element_ids",
            item.get("evidence_ids", []),
        )
    }
    pages = {
        int(value)
        for item in evidence
        for value in (
            item.get("page_nos", [])
            or (
                [item["page_no"]]
                if item.get("page_no") is not None
                else []
            )
        )
    }
    hits = 0
    for item in gold:
        if str(item.get("element_id", "")) in evidence_ids:
            hits += 1
        elif item.get("page_no") is not None and int(item["page_no"]) in pages:
            hits += 1
    return hits, len(gold)


def _path_correct(
    gold_path: Mapping[str, Any],
    paths: Sequence[Mapping[str, Any]],
) -> bool | None:
    required_types = set(gold_path.get("required_node_types", []))
    required_relations = set(gold_path.get("required_relations", []))
    required_pages = set(int(value) for value in gold_path.get("page_nos", []))
    if not required_types and not required_relations and not required_pages:
        return None
    for path in paths:
        node_types = set(path.get("node_types", []))
        relations = set(path.get("relations", []))
        pages = set(int(value) for value in path.get("page_nos", []))
        if (
            required_types.issubset(node_types)
            and required_relations.issubset(relations)
            and required_pages.issubset(pages)
        ):
            return True
    return False


def _graph_result(
    paths: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for position, path in enumerate(paths[:3], start=1):
        evidence.append(
            {
                "citation_id": position,
                "graph_path_id": path["path_id"],
                "element_ids": list(path.get("evidence_ids", [])),
                "doc_id": path["doc_id"],
                "brand": path["brand"],
                "model": path["model"],
                "year": path["year"],
                "region": path["region"],
                "language": path["language"],
                "manual_type": path["manual_type"],
                "page_nos": list(path.get("page_nos", [])),
                "section_path": list(path.get("section_path", [])),
                "content": path["content"],
                "source": path["source"],
            }
        )
    answered = bool(
        paths
        and float(paths[0]["score"]) >= 1.2
        and float(paths[0]["entity_score"]) >= 0.45
    )
    if not answered:
        return {
            "status": "insufficient_evidence",
            "answer": "The current manual evidence is insufficient to answer safely.",
            "evidence": evidence,
        }
    return {
        "status": "answered",
        "answer": "\n\n".join(
            [
                "According to the selected graph evidence:",
                *[
                    f"[{item['citation_id']}] {item['content']}"
                    for item in evidence
                ],
                "Sources:",
                *[
                    f"[{item['citation_id']}] {item['source']}"
                    for item in evidence
                ],
            ]
        ),
        "evidence": evidence,
    }


def _record(
    *,
    system: str,
    question: Mapping[str, Any],
    result: Mapping[str, Any],
    paths: Sequence[Mapping[str, Any]],
    latency_ms: float,
    route: Mapping[str, Any] | None = None,
    supports_paths: bool = True,
) -> dict[str, Any]:
    answerable = bool(question["answerable"])
    answered = result.get("status") == "answered"
    hits, total = _evidence_recall(
        question["gold_evidence"], result.get("evidence", [])
    )
    path_correct = (
        _path_correct(question["gold_path"], paths)
        if supports_paths
        else None
    )
    route_correct: bool | None = None
    if route is not None:
        route_correct = all(
            bool(route.get(key)) == bool(expected)
            for key, expected in question["expected_route"].items()
        )
        if not bool(question["expected_route"].get("graph")):
            path_correct = None
    return {
        "system": system,
        "question_id": question["question_id"],
        "answerable": answerable,
        "status": result.get("status"),
        "decision_correct": answered == answerable,
        "refusal_correct": (not answered) if not answerable else None,
        "evidence_hits": hits,
        "evidence_total": total,
        "path_correct": path_correct,
        "citation_faithful": _citation_faithful(result),
        "route_correct": route_correct,
        "metadata_violations": _metadata_violations(
            result.get("evidence", []), question["filters"]
        ),
        "latency_ms": round(latency_ms, 3),
        "retry_count": int(result.get("retry_count", 0)),
        "graph_path_ids": [
            path["path_id"] for path in paths[:3]
        ],
        "failure": (
            None
            if answered == answerable
            and (path_correct in {None, True})
            and (route_correct in {None, True})
            else {
                "decision": answered == answerable,
                "path": path_correct,
                "route": route_correct,
            }
        ),
    }


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    evidence_hits = sum(int(item["evidence_hits"]) for item in records)
    evidence_total = sum(int(item["evidence_total"]) for item in records)
    path_values = [
        bool(item["path_correct"])
        for item in records
        if item["path_correct"] is not None
    ]
    citation_values = [
        bool(item["citation_faithful"])
        for item in records
        if item["citation_faithful"] is not None
    ]
    route_values = [
        bool(item["route_correct"])
        for item in records
        if item["route_correct"] is not None
    ]
    refusal_values = [
        bool(item["refusal_correct"])
        for item in records
        if item["refusal_correct"] is not None
    ]
    latencies = [float(item["latency_ms"]) for item in records]
    return {
        "questions": len(records),
        "multi_hop_evidence_recall": (
            round(evidence_hits / evidence_total, 6)
            if evidence_total
            else None
        ),
        "path_accuracy": (
            round(mean(path_values), 6) if path_values else None
        ),
        "citation_faithfulness": (
            round(mean(citation_values), 6) if citation_values else None
        ),
        "route_accuracy": (
            round(mean(route_values), 6) if route_values else None
        ),
        "decision_accuracy": round(
            mean(bool(item["decision_correct"]) for item in records), 6
        ),
        "refusal_accuracy": (
            round(mean(refusal_values), 6) if refusal_values else None
        ),
        "metadata_violations": sum(
            int(item["metadata_violations"]) for item in records
        ),
        "retry_rate": round(
            mean(int(item["retry_count"]) > 0 for item in records), 6
        ),
        "latency_ms": {
            "mean": round(mean(latencies), 3),
            "median": round(median(latencies), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "tokens": None,
        "cost_usd": 0.0,
        "cost_note": "Offline extractive evaluation; no remote model calls.",
    }


def evaluate_agentic_comparison(
    *,
    questions: Sequence[Mapping[str, Any]],
    text_index: Any,
    graph_retriever: Any,
    table_row_index: Any | None = None,
    table_index: Any | None = None,
) -> dict[str, Any]:
    agentic_graph = type(graph_retriever)(graph_retriever.path)
    workflow = AgenticWorkflow(
        text_index=text_index,
        graph_retriever=agentic_graph,
        table_row_index=table_row_index,
        table_index=table_index,
        generation_backend=None,
    )
    records: list[dict[str, Any]] = []
    for question in questions:
        started = perf_counter()
        baseline = answer_question(
            text_index,
            question["question"],
            question["filters"],
            retrieval_limit=12,
            max_evidence=3,
        )
        records.append(
            _record(
                system="Baseline RAG",
                question=question,
                result=baseline,
                paths=[],
                latency_ms=(perf_counter() - started) * 1000,
                supports_paths=False,
            )
        )

        started = perf_counter()
        paths = graph_retriever.search(
            question["question"],
            filters=question["filters"],
            limit=6,
            max_hops=2,
        )
        graph_result = _graph_result(paths)
        records.append(
            _record(
                system="GraphRAG",
                question=question,
                result=graph_result,
                paths=paths,
                latency_ms=(perf_counter() - started) * 1000,
            )
        )

        started = perf_counter()
        agentic = workflow.run(
            query=question["question"],
            filters=question["filters"],
        )
        records.append(
            _record(
                system="Agentic GraphRAG",
                question=question,
                result=agentic,
                paths=agentic["graph_paths"],
                latency_ms=(perf_counter() - started) * 1000,
                route=agentic["initial_route"],
            )
        )

    systems = {}
    for system in ("Baseline RAG", "GraphRAG", "Agentic GraphRAG"):
        selected = [item for item in records if item["system"] == system]
        systems[system] = _summary(selected)
    return {
        "evaluation": "agentic_graphrag_multihop_v1",
        "split": "development",
        "questions": len(questions),
        "systems": systems,
        "records": records,
        "limitations": [
            "The questions are derived from the existing development corpus.",
            "Graph construction and final answers are deterministic and offline.",
            "Path accuracy requires one returned path to contain every Gold node type, relation, and page.",
            "Latency uses independent per-system graph caches; the first query for each vehicle is a cold load.",
            "Cost is zero only because this run does not call a remote generator.",
        ],
    }


def comparison_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Agentic GraphRAG evaluation",
        "",
        "This report compares the unchanged BM25 answer baseline, standalone "
        "deterministic GraphRAG, and the explicit-state Agentic GraphRAG "
        "workflow on the checked-in multi-hop development set.",
        "",
        "## Results",
        "",
        "| System | Evidence recall | Path accuracy | Citation faithfulness | "
        "Route accuracy | Decision accuracy | Refusal accuracy | Metadata "
        "violations | Mean latency | P95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in result["systems"].items():
        metric = lambda key: (
            "N/A"
            if summary[key] is None
            else f"{float(summary[key]):.4f}"
        )
        lines.append(
            f"| {name} | {metric('multi_hop_evidence_recall')} | "
            f"{metric('path_accuracy')} | "
            f"{metric('citation_faithfulness')} | "
            f"{metric('route_accuracy')} | "
            f"{metric('decision_accuracy')} | "
            f"{metric('refusal_accuracy')} | "
            f"{summary['metadata_violations']} | "
            f"{summary['latency_ms']['mean']:.1f} ms | "
            f"{summary['latency_ms']['p95']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "No remote model was called in this comparison, so measured token "
            "usage is unavailable and API cost is USD 0.00.",
            "",
            "## Failure cases",
            "",
        ]
    )
    failures = [
        item for item in result["records"] if item["failure"] is not None
    ]
    if not failures:
        lines.append("No recorded decision, path, or route failures.")
    else:
        lines.extend(
            f"- `{item['system']}` / `{item['question_id']}`: "
            f"{json.dumps(item['failure'], ensure_ascii=False)}"
            for item in failures
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)
