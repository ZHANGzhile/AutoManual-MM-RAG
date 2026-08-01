"""Explicit-state Agentic GraphRAG orchestration.

The workflow uses a TypedDict state and named graph nodes. Retrieval specialists
can run concurrently; a critic decides whether evidence is sufficient; at most
one broadened replan is allowed; synthesis remains constrained to a normalized
Evidence Pack and a deterministic citation/metadata guard.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence, TypedDict

from automanual_rag.answering import (
    answer_from_results,
    build_evidence_pack,
)
from automanual_rag.generation import (
    GenerationBackend,
    LabeledImage,
    REFUSAL_TEXT,
    generate_or_fallback,
    validate_grounded_text,
)
from automanual_rag.table_answering import answer_table_from_results


TABLE_RE = re.compile(
    r"\b(?:capacity|specification|specifications|torque|pressure|"
    r"how many|how much|weight|load|voltage|fluid|gallon|liter|litre|"
    r"lb|kg|nm|n-m|psi)\b",
    re.IGNORECASE,
)
GRAPH_RE = re.compile(
    r"\b(?:and then|before|after|requires?|steps?|procedure|warning|"
    r"caution|symbol|indicator|means?|related|shown|illustrated|"
    r"what happens|how does)\b",
    re.IGNORECASE,
)
PROCEDURE_RE = re.compile(
    r"^\s*(?:how|which tools?|what steps?|where)\b|"
    r"\b(?:adjust|open|close|remove|install|replace|switch|reset|"
    r"connect|disconnect|charge|start|stop|lock|unlock)\b",
    re.IGNORECASE,
)
MULTI_CLAUSE_RE = re.compile(r"\b(?:and|then|also|before|after)\b", re.IGNORECASE)


class AgenticState(TypedDict, total=False):
    query: str
    filters: dict[str, str]
    image_path: str | None
    route: dict[str, Any]
    initial_route: dict[str, Any]
    retry_count: int
    retrievers_called: list[str]
    text_results: list[dict[str, Any]]
    visual_results: list[dict[str, Any]]
    table_row_results: list[dict[str, Any]]
    table_results: list[dict[str, Any]]
    graph_paths: list[dict[str, Any]]
    critic: dict[str, Any]
    evidence: list[dict[str, Any]]
    answer_result: dict[str, Any]
    guard: dict[str, Any]
    trace: list[dict[str, Any]]
    usage: dict[str, Any]
    total_latency_ms: float


def _event(
    state: AgenticState,
    *,
    node: str,
    started: float,
    status: str = "ok",
    details: Mapping[str, Any] | None = None,
) -> None:
    state.setdefault("trace", []).append(
        {
            "sequence": len(state.get("trace", [])) + 1,
            "node": node,
            "status": status,
            "latency_ms": round((perf_counter() - started) * 1000, 3),
            "details": dict(details or {}),
        }
    )


def _compact(text: str, limit: int = 800) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def _same_manual(
    item: Mapping[str, Any],
    filters: Mapping[str, str],
) -> bool:
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
        if expected and str(item.get(field, "")).casefold() != expected.casefold():
            return False
    return True


class DeterministicPlanner:
    """Auditable router that can be replaced by a remote planner later."""

    name = "deterministic_agentic_router_v1"

    def plan(
        self,
        *,
        query: str,
        image_path: str | None,
        available: Mapping[str, bool],
        broaden: bool = False,
    ) -> dict[str, Any]:
        table = bool(available.get("table") and TABLE_RE.search(query))
        visual = bool(available.get("visual") and image_path)
        multi_hop = bool(
            GRAPH_RE.search(query)
            or PROCEDURE_RE.search(query)
            or MULTI_CLAUSE_RE.search(query)
        )
        graph = bool(available.get("graph") and (multi_hop or broaden))
        if broaden:
            table = bool(available.get("table"))
            graph = bool(available.get("graph"))
        return {
            "planner": self.name,
            "intent": (
                "visual"
                if visual
                else "exact_value"
                if table
                else "procedure"
                if PROCEDURE_RE.search(query)
                else "fact"
            ),
            "multi_hop": multi_hop,
            "text": bool(available.get("text")),
            "visual": visual,
            "table": table,
            "graph": graph,
            "broadened": broaden,
            "subqueries": {
                "text": query.strip(),
                "visual": query.strip(),
                "table": query.strip(),
                "graph": query.strip(),
            },
        }


class AgenticWorkflow:
    """Small explicit state graph over existing retrieval/generation tools."""

    def __init__(
        self,
        *,
        text_index: Any,
        graph_retriever: Any | None = None,
        visual_index: Any | None = None,
        table_row_index: Any | None = None,
        table_index: Any | None = None,
        generation_backend: GenerationBackend | None = None,
        planner: DeterministicPlanner | None = None,
        asset_root: Path | None = None,
    ) -> None:
        self.text_index = text_index
        self.graph_retriever = graph_retriever
        self.visual_index = visual_index
        self.table_row_index = table_row_index
        self.table_index = table_index
        self.generation_backend = generation_backend
        self.planner = planner or DeterministicPlanner()
        self.asset_root = asset_root.resolve() if asset_root else None

    def _available(self) -> dict[str, bool]:
        return {
            "text": self.text_index is not None,
            "graph": self.graph_retriever is not None,
            "visual": self.visual_index is not None,
            "table": (
                self.table_row_index is not None
                or self.table_index is not None
            ),
        }

    def _planner_node(
        self,
        state: AgenticState,
        *,
        broaden: bool,
    ) -> None:
        started = perf_counter()
        state["route"] = self.planner.plan(
            query=state["query"],
            image_path=state.get("image_path"),
            available=self._available(),
            broaden=broaden,
        )
        if not broaden:
            state["initial_route"] = dict(state["route"])
        _event(
            state,
            node="Planner/Router",
            started=started,
            details={"route": state["route"]},
        )

    def _retrieval_jobs(
        self,
        state: AgenticState,
    ) -> dict[str, Callable[[], list[dict[str, Any]]]]:
        route = state["route"]
        query = state["query"]
        filters = state["filters"]
        jobs: dict[str, Callable[[], list[dict[str, Any]]]] = {}
        if route["text"]:
            jobs["Text Retrieval"] = lambda: self.text_index.search(
                route["subqueries"]["text"],
                filters=filters,
                limit=12,
            )
        if route["graph"]:
            jobs["Graph Retrieval"] = lambda: self.graph_retriever.search(
                route["subqueries"]["graph"],
                filters=filters,
                limit=6,
                max_hops=2,
            )
        if route["visual"]:
            jobs["Visual Retrieval"] = lambda: self.visual_index.search(
                Path(str(state["image_path"])).resolve(),
                filters=filters,
                limit=5,
            )
        if route["table"] and self.table_row_index is not None:
            jobs["Table Row Retrieval"] = lambda: self.table_row_index.search(
                route["subqueries"]["table"],
                filters=filters,
                limit=5,
            )
        if route["table"] and self.table_index is not None:
            jobs["Table Crop Retrieval"] = lambda: self.table_index.search(
                route["subqueries"]["table"],
                filters=filters,
                limit=5,
            )
        return jobs

    def _retrieval_node(self, state: AgenticState) -> None:
        jobs = self._retrieval_jobs(state)
        key_by_node = {
            "Text Retrieval": "text_results",
            "Graph Retrieval": "graph_paths",
            "Visual Retrieval": "visual_results",
            "Table Row Retrieval": "table_row_results",
            "Table Crop Retrieval": "table_results",
        }
        for key in key_by_node.values():
            state[key] = []
        state["retrievers_called"] = []
        started_all = perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as executor:
            futures = {
                executor.submit(job): (name, perf_counter())
                for name, job in jobs.items()
            }
            for future in as_completed(futures):
                name, started = futures[future]
                try:
                    results = future.result()
                    state[key_by_node[name]] = results
                    state["retrievers_called"].append(name)
                    _event(
                        state,
                        node=name,
                        started=started,
                        details={"results": len(results)},
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    state[key_by_node[name]] = []
                    state["retrievers_called"].append(name)
                    _event(
                        state,
                        node=name,
                        started=started,
                        status="error",
                        details={"error_type": type(exc).__name__},
                    )
        _event(
            state,
            node="Parallel Retrieval Join",
            started=started_all,
            details={
                "retrievers": sorted(state["retrievers_called"]),
                "graph_path_ids": [
                    item["path_id"] for item in state["graph_paths"]
                ],
            },
        )

    def _critic_node(self, state: AgenticState) -> None:
        started = perf_counter()
        text_answer = answer_from_results(
            state["query"],
            state["filters"],
            state["text_results"],
            max_evidence=3,
        )
        table_answer = answer_table_from_results(
            state["query"],
            state["filters"],
            state["table_row_results"],
            max_evidence=3,
        )
        metadata_violations = 0
        for collection in (
            state["text_results"],
            state["visual_results"],
            state["table_row_results"],
            state["table_results"],
            state["graph_paths"],
        ):
            metadata_violations += sum(
                not _same_manual(item, state["filters"])
                for item in collection
            )
        support = {
            "text": text_answer["status"] == "answered",
            "graph": bool(
                state["graph_paths"]
                and float(state["graph_paths"][0]["score"]) >= 1.0
            ),
            "visual": bool(state["visual_results"]),
            "table": table_answer["status"] == "answered",
        }
        route = state["route"]
        required = ["text"]
        if route["graph"]:
            required.append("graph")
        if route["visual"]:
            required.append("visual")
        # Table values may safely fall back to text/graph but cannot be
        # presented as exact verified values without curated-row support.
        missing = [
            modality
            for modality in required
            if not support.get(modality, False)
        ]
        if metadata_violations:
            decision = "refuse"
            reason = "metadata_violation"
        elif not missing:
            decision = "accept"
            reason = "required_modalities_supported"
        elif state["retry_count"] == 0:
            decision = "replan"
            reason = "missing_required_evidence"
        else:
            decision = "refuse"
            reason = "evidence_still_insufficient_after_retry"
        state["critic"] = {
            "decision": decision,
            "reason": reason,
            "support": support,
            "missing": missing,
            "metadata_violations": metadata_violations,
            "text_decision": {
                "status": text_answer["status"],
                "reason": text_answer["reason"],
                "confidence": text_answer.get("confidence"),
            },
            "table_decision": {
                "status": table_answer["status"],
                "reason": table_answer["reason"],
            },
        }
        _event(
            state,
            node="Evidence Critic",
            started=started,
            status=decision,
            details=state["critic"],
        )

    @staticmethod
    def _graph_evidence(path: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "graph_path_id": path["path_id"],
            "chunk_id": path["path_id"],
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
            "chunk_type": "graph_path",
            "content": path["content"],
            "retrieval_rank": int(path["rank"]),
            "retrieval_score": float(path["score"]),
            "rerank_score": min(float(path["score"]) / 3.0, 1.0),
            "graph_nodes": list(path["node_labels"]),
            "graph_relations": list(path["relations"]),
            "source": path["source"],
        }

    @staticmethod
    def _table_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
        page_no = int(item["page_no"])
        cells = item.get("cells", {})
        content = " | ".join(
            f"{key}: {value}" for key, value in cells.items()
        )
        return {
            "row_id": item["row_id"],
            "element_ids": [item["element_id"]],
            "doc_id": item["doc_id"],
            "brand": item["brand"],
            "model": item["model"],
            "year": item["year"],
            "region": item["region"],
            "language": item["language"],
            "manual_type": item["manual_type"],
            "page_nos": [page_no],
            "section_path": list(item["section_path"]),
            "chunk_type": "verified_table_row",
            "content": content,
            "retrieval_rank": int(item["rank"]),
            "retrieval_score": float(item["score"]),
            "source": (
                f"{item['brand']} {item['model']} Owner's Manual "
                f"({item['year']}), {' > '.join(item['section_path'])}, "
                f"physical PDF p.{page_no}"
            ),
        }

    @staticmethod
    def _visual_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
        page_no = int(item["page_no"])
        return {
            "element_ids": [item["element_id"]],
            "doc_id": item["doc_id"],
            "brand": item["brand"],
            "model": item["model"],
            "year": item["year"],
            "region": item["region"],
            "language": item["language"],
            "manual_type": item["manual_type"],
            "page_nos": [page_no],
            "section_path": list(item["section_path"]),
            "chunk_type": "image",
            "content": str(item.get("content", "")),
            "asset_path": item.get("asset_path"),
            "retrieval_rank": int(item["rank"]),
            "retrieval_score": float(item["score"]),
            "source": (
                f"{item['brand']} {item['model']} Owner's Manual "
                f"({item['year']}), {' > '.join(item['section_path'])}, "
                f"physical PDF p.{page_no}"
            ),
        }

    def _build_evidence(self, state: AgenticState) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        candidates.extend(
            self._graph_evidence(path) for path in state["graph_paths"][:3]
        )
        candidates.extend(
            build_evidence_pack(
                state["query"], state["text_results"], max_evidence=3
            )
        )
        candidates.extend(
            self._table_evidence(item)
            for item in state["table_row_results"][:2]
        )
        candidates.extend(
            self._visual_evidence(item)
            for item in state["visual_results"][:2]
        )
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[int, ...], str]] = set()
        for item in candidates:
            identity = (
                ",".join(sorted(str(value) for value in item["element_ids"]))
                if item.get("graph_path_id")
                else str(item.get("chunk_id", item.get("row_id", "")))
            )
            key = (
                str(item["doc_id"]),
                tuple(int(page) for page in item.get("page_nos", [])),
                identity,
            )
            if key in seen:
                continue
            seen.add(key)
            value = dict(item)
            value["citation_id"] = len(evidence) + 1
            evidence.append(value)
            if len(evidence) == 8:
                break
        return evidence

    def _synthesis_node(self, state: AgenticState) -> None:
        started = perf_counter()
        if state["critic"]["decision"] != "accept":
            state["evidence"] = self._build_evidence(state)
            state["answer_result"] = {
                "status": "insufficient_evidence",
                "reason": state["critic"]["reason"],
                "answer": REFUSAL_TEXT,
                "evidence": state["evidence"],
                "generation": {
                    "status": "not_used",
                    "backend": "extractive_evidence_v1",
                },
            }
            _event(
                state,
                node="Answer/Synthesis",
                started=started,
                status="refused",
                details={"evidence": len(state["evidence"])},
            )
            return
        state["evidence"] = self._build_evidence(state)
        answer_lines = ["According to the selected manual evidence:"]
        for item in state["evidence"]:
            answer_lines.append(
                f"[{item['citation_id']}] "
                f"{_compact(str(item.get('content', '')), 650)}"
            )
        answer_lines.append("Sources:")
        answer_lines.extend(
            f"[{item['citation_id']}] {item['source']}"
            for item in state["evidence"]
        )
        base = {
            "status": "answered",
            "reason": "agentic_evidence_accepted",
            "answer": "\n\n".join(answer_lines),
            "evidence": state["evidence"],
        }
        images: list[LabeledImage] = []
        if state.get("image_path"):
            images.append(
                LabeledImage(
                    "Uploaded query image",
                    Path(str(state["image_path"])).resolve(),
                )
            )
        if self.asset_root is not None:
            images.extend(
                LabeledImage(
                    f"Evidence image [{position}]",
                    self.asset_root / str(item["asset_path"]),
                )
                for position, item in enumerate(
                    state["visual_results"][:3],
                    start=1,
                )
                if item.get("asset_path")
            )
        state["answer_result"] = generate_or_fallback(
            base,
            question=state["query"],
            backend=self.generation_backend,
            images=images,
        )
        usage = getattr(self.generation_backend, "last_usage", None)
        state["usage"] = {
            "tokens": usage if isinstance(usage, Mapping) else None,
            "cost": None,
            "cost_reason": (
                "provider_usage_not_available"
                if usage is None
                else "pricing_not_configured"
            ),
        }
        _event(
            state,
            node="Answer/Synthesis",
            started=started,
            details={
                "evidence": len(state["evidence"]),
                "generation": state["answer_result"].get("generation"),
                "usage": state["usage"],
            },
        )

    def _guard_node(self, state: AgenticState) -> bool:
        started = perf_counter()
        result = state["answer_result"]
        errors: list[str] = []
        if result["status"] == "answered":
            try:
                validate_grounded_text(
                    str(result["answer"]), len(result["evidence"])
                )
            except (RuntimeError, ValueError) as exc:
                errors.append(str(exc))
        violations = sum(
            not _same_manual(item, state["filters"])
            for item in result["evidence"]
        )
        if violations:
            errors.append(f"{violations} metadata filter violation(s)")
        passed = not errors
        state["guard"] = {
            "passed": passed,
            "citation_check": not any(
                "citation" in error.casefold() for error in errors
            ),
            "metadata_violations": violations,
            "errors": errors,
        }
        _event(
            state,
            node="Citation/Metadata Guard",
            started=started,
            status="passed" if passed else "failed",
            details=state["guard"],
        )
        return passed

    def run(
        self,
        *,
        query: str,
        filters: Mapping[str, str],
        image_path: str | None = None,
    ) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("Agentic query must not be empty")
        state: AgenticState = {
            "query": query.strip(),
            "filters": {
                key: str(value)
                for key, value in filters.items()
                if value is not None and str(value).strip()
            },
            "image_path": image_path,
            "retry_count": 0,
            "trace": [],
        }
        total_started = perf_counter()
        self._planner_node(state, broaden=False)
        for attempt in range(2):
            self._retrieval_node(state)
            self._critic_node(state)
            if state["critic"]["decision"] == "replan" and attempt == 0:
                state["retry_count"] = 1
                started = perf_counter()
                _event(
                    state,
                    node="Conditional Replan",
                    started=started,
                    status="retry",
                    details={"retry_count": 1},
                )
                self._planner_node(state, broaden=True)
                continue
            self._synthesis_node(state)
            if self._guard_node(state):
                break
            if attempt == 0:
                state["retry_count"] = 1
                started = perf_counter()
                _event(
                    state,
                    node="Conditional Replan",
                    started=started,
                    status="retry",
                    details={
                        "retry_count": 1,
                        "trigger": "validation_failure",
                    },
                )
                self._planner_node(state, broaden=True)
                continue
            state["answer_result"] = {
                "status": "insufficient_evidence",
                "reason": "final_validation_failed",
                "answer": REFUSAL_TEXT,
                "evidence": state.get("evidence", []),
                "generation": {
                    "status": "not_used",
                    "backend": "extractive_evidence_v1",
                },
            }
            break
        state["total_latency_ms"] = round(
            (perf_counter() - total_started) * 1000, 3
        )
        result = dict(state["answer_result"])
        result.update(
            {
                "backend": "agentic_graphrag_state_graph_v1",
                "query": state["query"],
                "filters": state["filters"],
                "route": state["route"],
                "initial_route": state["initial_route"],
                "retrievers_called": sorted(state["retrievers_called"]),
                "graph_paths": state["graph_paths"],
                "critic": state["critic"],
                "retry_count": state["retry_count"],
                "guard": state["guard"],
                "trace": state["trace"],
                "latency_ms": state["total_latency_ms"],
                "usage": state.get(
                    "usage",
                    {
                        "tokens": None,
                        "cost": None,
                        "cost_reason": "generation_not_used",
                    },
                ),
            }
        )
        return result
