"""Deterministic automotive-manual graph construction and retrieval.

The graph is derived only from normalized elements and section-aware chunks.
It is intentionally model-free: graph construction is reproducible, every
node and edge retains source provenance, and retrieval is always constrained
to exactly one vehicle manual before entity matching or path expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from automanual_rag.ingestion.mineru import load_manifest


GRAPH_SCHEMA_VERSION = 1
GRAPH_BACKEND = "deterministic_automotive_manual_graph_v1"
NODE_TYPES = frozenset(
    {
        "Vehicle",
        "Section",
        "Component",
        "Symbol",
        "Procedure",
        "Step",
        "Warning",
        "Caution",
        "Specification",
        "Image",
        "Table",
        "EvidencePage",
    }
)
EDGE_TYPES = frozenset(
    {
        "APPLIES_TO",
        "LOCATED_IN",
        "SYMBOL_MEANS",
        "EXPLAINED_BY",
        "REQUIRES_STEP",
        "NEXT_STEP",
        "HAS_WARNING",
        "HAS_SPECIFICATION",
        "ILLUSTRATED_BY",
        "REFERENCES",
    }
)
FILTER_FIELDS = frozenset(
    {
        "doc_id",
        "brand",
        "model",
        "year",
        "region",
        "language",
        "manual_type",
    }
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
ACTION_RE = re.compile(
    r"\b(?:adjust(?:ing)?|charg(?:e|ing)|check(?:ing)?|clos(?:e|ing)|"
    r"connect(?:ing)?|disconnect(?:ing)?|fasten(?:ing)?|install(?:ing)?|"
    r"locat(?:e|ing)|lock(?:ing)?|open(?:ing)?|operat(?:e|ing)|"
    r"program(?:ming)?|remov(?:e|ing)|replac(?:e|ing)|reset(?:ting)?|"
    r"select(?:ing)?|set(?:ting)?|start(?:ing)?|stop(?:ping)?|"
    r"switch(?:ing)?|unlock(?:ing)?|using)\b",
    re.IGNORECASE,
)
SPEC_RE = re.compile(
    r"\b(?:capacity|capacities|specification|specifications|torque|"
    r"pressure|dimension|fluid|weight|load|voltage|amperage)\b",
    re.IGNORECASE,
)
SYMBOL_RE = re.compile(
    r"\b(?:symbol|symbols|indicator|indicators|warning light|tell-?tale)\b",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"^\s*(?:step\s+)?(\d+)[.)]\s+(.+?)\s*$", re.IGNORECASE)
ACTION_PREFIX_RE = re.compile(
    r"^(?:how (?:do|does) |what is |adjusting |charging |checking |closing |"
    r"connecting |disconnecting |fastening and unfastening |identifying |"
    r"installing |locating |locking and unlocking |opening |operating |"
    r"programming |removing and installing |removing |replacing |resetting |"
    r"selecting |setting |starting and stopping |starting |stopping |"
    r"switching |unlocking |using )",
    re.IGNORECASE,
)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
        "you",
        "your",
    }
)
RELATION_WEIGHTS = {
    "SYMBOL_MEANS": 1.00,
    "REQUIRES_STEP": 0.98,
    "NEXT_STEP": 0.96,
    "HAS_WARNING": 0.95,
    "HAS_SPECIFICATION": 0.93,
    "EXPLAINED_BY": 0.90,
    "ILLUSTRATED_BY": 0.88,
    "LOCATED_IN": 0.82,
    "REFERENCES": 0.78,
    "APPLIES_TO": 0.55,
}
NODE_TYPE_WEIGHTS = {
    "Symbol": 1.00,
    "Component": 0.98,
    "Procedure": 0.97,
    "Specification": 0.96,
    "Warning": 0.95,
    "Caution": 0.95,
    "Step": 0.92,
    "Section": 0.88,
    "Table": 0.84,
    "Image": 0.82,
    "EvidencePage": 0.55,
    "Vehicle": 0.45,
}


def _stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if token.casefold() not in STOPWORDS and len(token) > 1
    }


def _compact(text: str, limit: int = 1200) -> str:
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def _section_label(path: Sequence[str]) -> str:
    return str(path[-1]).strip() if path else "Unsectioned evidence"


def _component_label(section_path: Sequence[str]) -> str:
    label = _section_label(section_path)
    label = re.sub(
        r"\s*-\s*(?:vehicles?|excluding|including|with|without)\b.*$",
        "",
        label,
        flags=re.IGNORECASE,
    )
    label = ACTION_PREFIX_RE.sub("", label).strip(" -:")
    return label or _section_label(section_path)


def _source_label(node: Mapping[str, Any]) -> str:
    section = " > ".join(node.get("section_path", []))
    page_no = node.get("page_no")
    page_nos = node.get("page_nos", [])
    if page_no:
        page = f", physical PDF p.{page_no}"
    elif page_nos:
        page = ", physical PDF p." + ",".join(
            str(value) for value in page_nos
        )
    else:
        page = ""
    return (
        f"{node['brand']} {node['model']} Owner's Manual "
        f"({node['year']}), {section}{page}"
    )


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str
    text: str
    doc_id: str
    brand: str
    model: str
    year: str
    region: str
    language: str
    manual_type: str
    page_no: int | None
    section_path: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    asset_path: str | None = None
    properties: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(f"Unsupported graph node type: {self.node_type}")
        if not self.node_id or not self.label or not self.doc_id:
            raise ValueError("Graph nodes require node_id, label, and doc_id")
        if self.page_no is not None and self.page_no < 1:
            raise ValueError("Graph node page_no must be positive")


@dataclass(frozen=True, slots=True)
class GraphEdge:
    edge_id: str
    edge_type: str
    source_id: str
    target_id: str
    doc_id: str
    brand: str
    model: str
    year: str
    region: str
    language: str
    manual_type: str
    page_no: int | None
    evidence_ids: tuple[str, ...]
    confidence: float = 1.0
    properties: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"Unsupported graph edge type: {self.edge_type}")
        if not self.edge_id or not self.source_id or not self.target_id:
            raise ValueError("Graph edges require IDs")
        if self.source_id == self.target_id:
            raise ValueError("Graph edges must not be self-referential")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Graph edge confidence must be from 0 to 1")


class _GraphWriter:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.node_ids: set[str] = set()
        self.edge_ids: set[str] = set()
        self.node_counts = {node_type: 0 for node_type in sorted(NODE_TYPES)}
        self.edge_counts = {edge_type: 0 for edge_type in sorted(EDGE_TYPES)}

    def add_node(self, node: GraphNode) -> str:
        node.validate()
        if node.node_id in self.node_ids:
            return node.node_id
        self.connection.execute(
            """
            INSERT INTO nodes (
                node_id, node_type, label, text, doc_id, brand, model, year,
                region, language, manual_type, page_no, section_path,
                evidence_ids, asset_path, properties
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.node_id,
                node.node_type,
                node.label,
                node.text,
                node.doc_id,
                node.brand,
                node.model,
                node.year,
                node.region,
                node.language,
                node.manual_type,
                node.page_no,
                _json(list(node.section_path)),
                _json(list(node.evidence_ids)),
                node.asset_path,
                _json(dict(node.properties or {})),
            ),
        )
        self.node_ids.add(node.node_id)
        self.node_counts[node.node_type] += 1
        return node.node_id

    def add_edge(self, edge: GraphEdge) -> str:
        edge.validate()
        if edge.edge_id in self.edge_ids:
            return edge.edge_id
        if edge.source_id not in self.node_ids or edge.target_id not in self.node_ids:
            raise ValueError(
                f"Graph edge endpoint missing: {edge.source_id} -> {edge.target_id}"
            )
        self.connection.execute(
            """
            INSERT INTO edges (
                edge_id, edge_type, source_id, target_id, doc_id, brand,
                model, year, region, language, manual_type, page_no,
                evidence_ids, confidence, properties
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.edge_id,
                edge.edge_type,
                edge.source_id,
                edge.target_id,
                edge.doc_id,
                edge.brand,
                edge.model,
                edge.year,
                edge.region,
                edge.language,
                edge.manual_type,
                edge.page_no,
                _json(list(edge.evidence_ids)),
                edge.confidence,
                _json(dict(edge.properties or {})),
            ),
        )
        self.edge_ids.add(edge.edge_id)
        self.edge_counts[edge.edge_type] += 1
        return edge.edge_id


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            year TEXT NOT NULL,
            region TEXT NOT NULL,
            language TEXT NOT NULL,
            manual_type TEXT NOT NULL
        );
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            node_type TEXT NOT NULL,
            label TEXT NOT NULL,
            text TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            year TEXT NOT NULL,
            region TEXT NOT NULL,
            language TEXT NOT NULL,
            manual_type TEXT NOT NULL,
            page_no INTEGER,
            section_path TEXT NOT NULL,
            evidence_ids TEXT NOT NULL,
            asset_path TEXT,
            properties TEXT NOT NULL
        );
        CREATE TABLE edges (
            edge_id TEXT PRIMARY KEY,
            edge_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            year TEXT NOT NULL,
            region TEXT NOT NULL,
            language TEXT NOT NULL,
            manual_type TEXT NOT NULL,
            page_no INTEGER,
            evidence_ids TEXT NOT NULL,
            confidence REAL NOT NULL,
            properties TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES nodes(node_id),
            FOREIGN KEY(target_id) REFERENCES nodes(node_id)
        );
        CREATE INDEX idx_nodes_doc_type ON nodes(doc_id, node_type);
        CREATE INDEX idx_nodes_vehicle
            ON nodes(model, year, region, language, manual_type);
        CREATE INDEX idx_edges_source ON edges(doc_id, source_id);
        CREATE INDEX idx_edges_target ON edges(doc_id, target_id);
        CREATE VIRTUAL TABLE nodes_fts USING fts5(
            node_id UNINDEXED,
            label,
            text,
            section_text,
            tokenize='unicode61'
        );
        """
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield value


def _metadata(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        field: str(value[field])
        for field in (
            "doc_id",
            "brand",
            "model",
            "year",
            "region",
            "language",
            "manual_type",
        )
    }


def _node(
    *,
    node_id: str,
    node_type: str,
    label: str,
    text: str,
    record: Mapping[str, Any],
    page_no: int | None,
    section_path: Sequence[str],
    evidence_ids: Sequence[str],
    asset_path: str | None = None,
    properties: Mapping[str, Any] | None = None,
) -> GraphNode:
    metadata = _metadata(record)
    return GraphNode(
        node_id=node_id,
        node_type=node_type,
        label=_compact(label, 240),
        text=_compact(text),
        page_no=page_no,
        section_path=tuple(str(item) for item in section_path),
        evidence_ids=tuple(str(item) for item in evidence_ids if item),
        asset_path=asset_path,
        properties=properties,
        **metadata,
    )


def _edge(
    *,
    edge_type: str,
    source_id: str,
    target_id: str,
    record: Mapping[str, Any],
    page_no: int | None,
    evidence_ids: Sequence[str],
    confidence: float = 1.0,
    properties: Mapping[str, Any] | None = None,
) -> GraphEdge:
    metadata = _metadata(record)
    return GraphEdge(
        edge_id=_stable_id(
            "edge",
            metadata["doc_id"],
            edge_type,
            source_id,
            target_id,
            list(evidence_ids),
            page_no,
        ),
        edge_type=edge_type,
        source_id=source_id,
        target_id=target_id,
        page_no=page_no,
        evidence_ids=tuple(str(item) for item in evidence_ids if item),
        confidence=confidence,
        properties=properties,
        **metadata,
    )


def _extract_steps(content: str, *, force: bool) -> list[str]:
    numbered: list[tuple[int, str]] = []
    for line in content.splitlines():
        match = STEP_RE.match(line)
        if match:
            numbered.append((int(match.group(1)), match.group(2).strip()))
    if numbered:
        numbered.sort(key=lambda item: item[0])
        return [text for _, text in numbered[:20] if text]
    if not force:
        return []
    lines = [
        re.sub(r"^\s*[-*\u2022]\s*", "", line).strip()
        for line in content.splitlines()
        if line.strip()
    ]
    if 1 <= len(lines) <= 20:
        return lines
    return [_compact(content, 500)] if content.strip() else []


def build_manual_graph(
    *,
    manifest_path: Path,
    processed_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build one auditable graph index for every document in the manifest."""

    manifest_path = manifest_path.resolve()
    processed_root = processed_root.resolve()
    output_path = output_path.resolve()
    documents = load_manifest(manifest_path)
    if not documents:
        raise ValueError("Corpus manifest is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".building")
    if temp_path.exists():
        temp_path.unlink()

    connection = sqlite3.connect(temp_path)
    connection.execute("PRAGMA foreign_keys=ON")
    writer = _GraphWriter(connection)
    try:
        _create_schema(connection)
    except Exception:
        connection.close()
        if temp_path.exists():
            temp_path.unlink()
        raise

    for document in documents:
        metadata = document.element_metadata()
        connection.execute(
            """
            INSERT INTO documents (
                doc_id, brand, model, year, region, language, manual_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata["doc_id"],
                metadata["brand"],
                metadata["model"],
                metadata["year"],
                metadata["region"],
                metadata["language"],
                metadata["manual_type"],
            ),
        )

    try:
        for document in documents:
            doc_dir = processed_root / document.doc_id
            elements_path = doc_dir / "elements.jsonl"
            chunks_path = doc_dir / "chunks.jsonl"
            if not elements_path.is_file() or not chunks_path.is_file():
                raise FileNotFoundError(
                    f"Processed elements/chunks missing for {document.doc_id}"
                )
            metadata = document.element_metadata()
            vehicle_id = _stable_id("node", document.doc_id, "Vehicle")
            writer.add_node(
                _node(
                    node_id=vehicle_id,
                    node_type="Vehicle",
                    label=f"{document.brand} {document.model} {document.year}",
                    text=(
                        f"{document.brand} {document.model} {document.year} "
                        f"{document.region} {document.language} "
                        f"{document.manual_type}"
                    ),
                    record=metadata,
                    page_no=None,
                    section_path=(),
                    evidence_ids=(document.doc_id,),
                    properties={"source_url": document.source_url},
                )
            )
            section_ids: dict[tuple[str, ...], str] = {}
            component_ids: dict[tuple[str, ...], str] = {}
            page_ids: dict[int, str] = {}
            procedure_ids: dict[tuple[str, ...], list[str]] = {}
            safety_ids: dict[tuple[str, ...], list[str]] = {}
            safety_evidence: dict[str, tuple[str, ...]] = {}
            symbol_ids: dict[tuple[str, ...], list[str]] = {}

            def ensure_page(
                record: Mapping[str, Any],
                page_no: int | None,
                evidence_id: str,
            ) -> str | None:
                if page_no is None:
                    return None
                if page_no in page_ids:
                    return page_ids[page_no]
                page_id = _stable_id(
                    "node", document.doc_id, "EvidencePage", page_no
                )
                writer.add_node(
                    _node(
                        node_id=page_id,
                        node_type="EvidencePage",
                        label=f"Physical PDF page {page_no}",
                        text=f"Evidence page {page_no} of {document.doc_id}",
                        record=record,
                        page_no=page_no,
                        section_path=(),
                        evidence_ids=(evidence_id,),
                    )
                )
                writer.add_edge(
                    _edge(
                        edge_type="APPLIES_TO",
                        source_id=page_id,
                        target_id=vehicle_id,
                        record=record,
                        page_no=page_no,
                        evidence_ids=(evidence_id,),
                    )
                )
                page_ids[page_no] = page_id
                return page_id

            def ensure_section(
                record: Mapping[str, Any],
                section_path: Sequence[str],
                page_no: int | None,
                evidence_id: str,
            ) -> str:
                path = tuple(str(item).strip() for item in section_path if str(item).strip())
                if not path:
                    path = ("Unsectioned evidence",)
                if path in section_ids:
                    section_id = section_ids[path]
                else:
                    section_id = _stable_id(
                        "node", document.doc_id, "Section", path
                    )
                    writer.add_node(
                        _node(
                            node_id=section_id,
                            node_type="Section",
                            label=_section_label(path),
                            text=" > ".join(path),
                            record=record,
                            page_no=page_no,
                            section_path=path,
                            evidence_ids=(evidence_id,),
                        )
                    )
                    writer.add_edge(
                        _edge(
                            edge_type="APPLIES_TO",
                            source_id=section_id,
                            target_id=vehicle_id,
                            record=record,
                            page_no=page_no,
                            evidence_ids=(evidence_id,),
                        )
                    )
                    if len(path) > 1:
                        parent_id = ensure_section(
                            record, path[:-1], page_no, evidence_id
                        )
                        writer.add_edge(
                            _edge(
                                edge_type="LOCATED_IN",
                                source_id=section_id,
                                target_id=parent_id,
                                record=record,
                                page_no=page_no,
                                evidence_ids=(evidence_id,),
                            )
                        )
                    section_ids[path] = section_id
                page_id = ensure_page(record, page_no, evidence_id)
                if page_id:
                    writer.add_edge(
                        _edge(
                            edge_type="REFERENCES",
                            source_id=section_id,
                            target_id=page_id,
                            record=record,
                            page_no=page_no,
                            evidence_ids=(evidence_id,),
                        )
                    )
                return section_id

            def ensure_component(
                record: Mapping[str, Any],
                section_path: Sequence[str],
                page_no: int | None,
                evidence_id: str,
            ) -> str:
                path = tuple(section_path) or ("Unsectioned evidence",)
                if path in component_ids:
                    return component_ids[path]
                label = _component_label(path)
                component_id = _stable_id(
                    "node", document.doc_id, "Component", path, label
                )
                section_id = ensure_section(
                    record, path, page_no, evidence_id
                )
                writer.add_node(
                    _node(
                        node_id=component_id,
                        node_type="Component",
                        label=label,
                        text=" > ".join(path),
                        record=record,
                        page_no=page_no,
                        section_path=path,
                        evidence_ids=(evidence_id,),
                    )
                )
                for edge_type, target in (
                    ("LOCATED_IN", section_id),
                    ("EXPLAINED_BY", section_id),
                    ("APPLIES_TO", vehicle_id),
                ):
                    writer.add_edge(
                        _edge(
                            edge_type=edge_type,
                            source_id=component_id,
                            target_id=target,
                            record=record,
                            page_no=page_no,
                            evidence_ids=(evidence_id,),
                        )
                    )
                page_id = ensure_page(record, page_no, evidence_id)
                if page_id:
                    writer.add_edge(
                        _edge(
                            edge_type="REFERENCES",
                            source_id=component_id,
                            target_id=page_id,
                            record=record,
                            page_no=page_no,
                            evidence_ids=(evidence_id,),
                        )
                    )
                component_ids[path] = component_id
                return component_id

            for chunk in _read_jsonl(chunks_path):
                chunk_id = str(chunk["chunk_id"])
                page_no = int(chunk["page_start"])
                section_path = tuple(chunk.get("section_path") or ())
                section_id = ensure_section(
                    chunk, section_path, page_no, chunk_id
                )
                component_id = ensure_component(
                    chunk, section_path, page_no, chunk_id
                )
                content = str(chunk.get("content", ""))
                label = _section_label(section_path)
                chunk_type = str(chunk.get("chunk_type", "text")).casefold()
                evidence_ids = tuple(chunk.get("element_ids") or (chunk_id,))
                page_id = ensure_page(chunk, page_no, chunk_id)

                if chunk_type in {"warning", "caution"}:
                    node_type = (
                        "Warning" if chunk_type == "warning" else "Caution"
                    )
                    safety_id = _stable_id(
                        "node", document.doc_id, node_type, chunk_id
                    )
                    writer.add_node(
                        _node(
                            node_id=safety_id,
                            node_type=node_type,
                            label=f"{node_type}: {label}",
                            text=content,
                            record=chunk,
                            page_no=page_no,
                            section_path=section_path,
                            evidence_ids=evidence_ids,
                            properties={"chunk_id": chunk_id},
                        )
                    )
                    for edge_type, source, target in (
                        ("LOCATED_IN", safety_id, section_id),
                        ("HAS_WARNING", section_id, safety_id),
                        ("HAS_WARNING", component_id, safety_id),
                        ("APPLIES_TO", safety_id, vehicle_id),
                    ):
                        writer.add_edge(
                            _edge(
                                edge_type=edge_type,
                                source_id=source,
                                target_id=target,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                    if page_id:
                        writer.add_edge(
                            _edge(
                                edge_type="REFERENCES",
                                source_id=safety_id,
                                target_id=page_id,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                    safety_ids.setdefault(section_path, []).append(safety_id)
                    safety_evidence[safety_id] = evidence_ids

                steps = _extract_steps(
                    content, force=(chunk_type == "steps")
                )
                is_procedure = bool(
                    chunk_type == "steps"
                    or steps
                    or ACTION_RE.search(label)
                )
                if is_procedure:
                    procedure_id = _stable_id(
                        "node", document.doc_id, "Procedure", chunk_id
                    )
                    writer.add_node(
                        _node(
                            node_id=procedure_id,
                            node_type="Procedure",
                            label=label,
                            text=content,
                            record=chunk,
                            page_no=page_no,
                            section_path=section_path,
                            evidence_ids=evidence_ids,
                            properties={
                                "chunk_id": chunk_id,
                                "chunk_type": chunk_type,
                            },
                        )
                    )
                    for edge_type, source, target in (
                        ("LOCATED_IN", procedure_id, section_id),
                        ("EXPLAINED_BY", component_id, procedure_id),
                        ("APPLIES_TO", procedure_id, vehicle_id),
                    ):
                        writer.add_edge(
                            _edge(
                                edge_type=edge_type,
                                source_id=source,
                                target_id=target,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                    if page_id:
                        writer.add_edge(
                            _edge(
                                edge_type="REFERENCES",
                                source_id=procedure_id,
                                target_id=page_id,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                    previous_step_id: str | None = None
                    for position, step_text in enumerate(steps, start=1):
                        step_id = _stable_id(
                            "node",
                            document.doc_id,
                            "Step",
                            chunk_id,
                            position,
                        )
                        writer.add_node(
                            _node(
                                node_id=step_id,
                                node_type="Step",
                                label=f"Step {position}",
                                text=step_text,
                                record=chunk,
                                page_no=page_no,
                                section_path=section_path,
                                evidence_ids=evidence_ids,
                                properties={
                                    "chunk_id": chunk_id,
                                    "position": position,
                                },
                            )
                        )
                        writer.add_edge(
                            _edge(
                                edge_type="REQUIRES_STEP",
                                source_id=procedure_id,
                                target_id=step_id,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                        writer.add_edge(
                            _edge(
                                edge_type="APPLIES_TO",
                                source_id=step_id,
                                target_id=vehicle_id,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                        if page_id:
                            writer.add_edge(
                                _edge(
                                    edge_type="REFERENCES",
                                    source_id=step_id,
                                    target_id=page_id,
                                    record=chunk,
                                    page_no=page_no,
                                    evidence_ids=evidence_ids,
                                )
                            )
                        if previous_step_id:
                            writer.add_edge(
                                _edge(
                                    edge_type="NEXT_STEP",
                                    source_id=previous_step_id,
                                    target_id=step_id,
                                    record=chunk,
                                    page_no=page_no,
                                    evidence_ids=evidence_ids,
                                )
                            )
                        previous_step_id = step_id
                    procedure_ids.setdefault(section_path, []).append(
                        procedure_id
                    )

                if SPEC_RE.search(f"{label} {content}"):
                    specification_id = _stable_id(
                        "node", document.doc_id, "Specification", chunk_id
                    )
                    writer.add_node(
                        _node(
                            node_id=specification_id,
                            node_type="Specification",
                            label=label,
                            text=content,
                            record=chunk,
                            page_no=page_no,
                            section_path=section_path,
                            evidence_ids=evidence_ids,
                            properties={"chunk_id": chunk_id},
                        )
                    )
                    for source in (section_id, component_id):
                        writer.add_edge(
                            _edge(
                                edge_type="HAS_SPECIFICATION",
                                source_id=source,
                                target_id=specification_id,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                    writer.add_edge(
                        _edge(
                            edge_type="APPLIES_TO",
                            source_id=specification_id,
                            target_id=vehicle_id,
                            record=chunk,
                            page_no=page_no,
                            evidence_ids=evidence_ids,
                        )
                    )
                    if page_id:
                        writer.add_edge(
                            _edge(
                                edge_type="REFERENCES",
                                source_id=specification_id,
                                target_id=page_id,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )

                if SYMBOL_RE.search(f"{' '.join(section_path)} {content}"):
                    symbol_id = _stable_id(
                        "node", document.doc_id, "Symbol", chunk_id
                    )
                    writer.add_node(
                        _node(
                            node_id=symbol_id,
                            node_type="Symbol",
                            label=label,
                            text=content,
                            record=chunk,
                            page_no=page_no,
                            section_path=section_path,
                            evidence_ids=evidence_ids,
                            properties={"chunk_id": chunk_id},
                        )
                    )
                    for edge_type, target in (
                        ("LOCATED_IN", section_id),
                        ("SYMBOL_MEANS", component_id),
                        ("APPLIES_TO", vehicle_id),
                    ):
                        writer.add_edge(
                            _edge(
                                edge_type=edge_type,
                                source_id=symbol_id,
                                target_id=target,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                    if page_id:
                        writer.add_edge(
                            _edge(
                                edge_type="REFERENCES",
                                source_id=symbol_id,
                                target_id=page_id,
                                record=chunk,
                                page_no=page_no,
                                evidence_ids=evidence_ids,
                            )
                        )
                    symbol_ids.setdefault(section_path, []).append(symbol_id)

            for element in _read_jsonl(elements_path):
                element_type = str(element.get("element_type", ""))
                if element_type not in {"image", "table"}:
                    continue
                element_id = str(element["element_id"])
                page_no_value = element.get("page_no")
                page_no = (
                    int(page_no_value) if page_no_value is not None else None
                )
                section_path = tuple(element.get("section_path") or ())
                section_id = ensure_section(
                    element, section_path, page_no, element_id
                )
                component_id = ensure_component(
                    element, section_path, page_no, element_id
                )
                graph_type = "Image" if element_type == "image" else "Table"
                visual_id = _stable_id(
                    "node", document.doc_id, graph_type, element_id
                )
                writer.add_node(
                    _node(
                        node_id=visual_id,
                        node_type=graph_type,
                        label=f"{graph_type}: {_section_label(section_path)}",
                        text=str(element.get("content", "")),
                        record=element,
                        page_no=page_no,
                        section_path=section_path,
                        evidence_ids=(element_id,),
                        asset_path=element.get("asset_path"),
                        properties={
                            "element_id": element_id,
                            "source_locator": element.get(
                                "source_locator", {}
                            ),
                        },
                    )
                )
                for edge_type, source, target in (
                    ("LOCATED_IN", visual_id, section_id),
                    ("APPLIES_TO", visual_id, vehicle_id),
                ):
                    writer.add_edge(
                        _edge(
                            edge_type=edge_type,
                            source_id=source,
                            target_id=target,
                            record=element,
                            page_no=page_no,
                            evidence_ids=(element_id,),
                        )
                    )
                if graph_type == "Image":
                    writer.add_edge(
                        _edge(
                            edge_type="ILLUSTRATED_BY",
                            source_id=section_id,
                            target_id=visual_id,
                            record=element,
                            page_no=page_no,
                            evidence_ids=(element_id,),
                        )
                    )
                    for symbol_id in symbol_ids.get(section_path, []):
                        writer.add_edge(
                            _edge(
                                edge_type="ILLUSTRATED_BY",
                                source_id=symbol_id,
                                target_id=visual_id,
                                record=element,
                                page_no=page_no,
                                evidence_ids=(element_id,),
                            )
                        )
                else:
                    writer.add_edge(
                        _edge(
                            edge_type="HAS_SPECIFICATION",
                            source_id=component_id,
                            target_id=visual_id,
                            record=element,
                            page_no=page_no,
                            evidence_ids=(element_id,),
                            confidence=0.75,
                        )
                    )
                page_id = ensure_page(element, page_no, element_id)
                if page_id:
                    writer.add_edge(
                        _edge(
                            edge_type="REFERENCES",
                            source_id=visual_id,
                            target_id=page_id,
                            record=element,
                            page_no=page_no,
                            evidence_ids=(element_id,),
                        )
                    )

            # Connect procedures and safety blocks after both have been seen.
            for section_path, procedures in procedure_ids.items():
                for procedure_id in procedures:
                    for safety_id in safety_ids.get(section_path, []):
                        writer.add_edge(
                            _edge(
                                edge_type="HAS_WARNING",
                                source_id=procedure_id,
                                target_id=safety_id,
                                record=metadata,
                                page_no=None,
                                evidence_ids=safety_evidence[safety_id],
                            )
                        )

            connection.commit()

        connection.execute(
            """
            INSERT INTO nodes_fts (node_id, label, text, section_text)
            SELECT node_id, label, text, section_path FROM nodes
            """
        )
        summary = {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "backend": GRAPH_BACKEND,
            "documents": len(documents),
            "node_count": len(writer.node_ids),
            "edge_count": len(writer.edge_ids),
            "node_counts": writer.node_counts,
            "edge_counts": writer.edge_counts,
            "manifest_path": manifest_path.as_posix(),
            "processed_root": processed_root.as_posix(),
        }
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            [
                ("schema_version", str(GRAPH_SCHEMA_VERSION)),
                ("backend", GRAPH_BACKEND),
                ("summary", _json(summary)),
            ],
        )
        connection.commit()
    except Exception:
        connection.close()
        if temp_path.exists():
            temp_path.unlink()
        raise
    connection.close()
    temp_path.replace(output_path)
    return summary


def _decode_node(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    for field in ("section_path", "evidence_ids", "properties"):
        value[field] = json.loads(value[field])
    return value


def _decode_edge(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    for field in ("evidence_ids", "properties"):
        value[field] = json.loads(value[field])
    value["confidence"] = float(value["confidence"])
    return value


class GraphRetriever:
    """Entity matching plus one/two-hop expansion under exact manual filters."""

    def __init__(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Graph index not found: {path}")
        self.path = path
        connection = sqlite3.connect(path)
        try:
            metadata = dict(
                connection.execute("SELECT key, value FROM metadata")
            )
        finally:
            connection.close()
        if metadata.get("schema_version") != str(GRAPH_SCHEMA_VERSION):
            raise ValueError("Unsupported graph index schema version")
        if metadata.get("backend") != GRAPH_BACKEND:
            raise ValueError("Unexpected graph index backend")
        self.summary = json.loads(metadata["summary"])
        self._doc_graph_cache: dict[
            str,
            tuple[
                dict[str, dict[str, Any]],
                dict[str, list[dict[str, Any]]],
            ],
        ] = {}

    def count(self) -> dict[str, int]:
        return {
            "nodes": int(self.summary["node_count"]),
            "edges": int(self.summary["edge_count"]),
        }

    def _matching_doc_id(
        self,
        connection: sqlite3.Connection,
        filters: Mapping[str, str] | None,
    ) -> str:
        values = {
            key: str(value).strip()
            for key, value in dict(filters or {}).items()
            if value is not None and str(value).strip()
        }
        unknown = set(values).difference(FILTER_FIELDS)
        if unknown:
            raise ValueError(
                "Unsupported graph filter(s): "
                + ", ".join(sorted(unknown))
            )
        if not values.get("doc_id") and not (
            values.get("model") and values.get("year")
        ):
            raise ValueError(
                "Graph retrieval requires doc_id or exact model and year"
            )
        conditions: list[str] = []
        parameters: list[str] = []
        for field, value in values.items():
            conditions.append(f"LOWER({field}) = LOWER(?)")
            parameters.append(value)
        rows = connection.execute(
            "SELECT doc_id FROM documents WHERE " + " AND ".join(conditions),
            parameters,
        ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                "Graph filters do not identify exactly one vehicle manual"
            )
        return str(rows[0][0])

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = list(dict.fromkeys(_terms(query)))
        if not tokens:
            raise ValueError("Graph query contains no searchable terms")
        return " OR ".join(f'"{token}"' for token in tokens[:30])

    def _document_graph(
        self,
        connection: sqlite3.Connection,
        doc_id: str,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, list[dict[str, Any]]],
    ]:
        cached = self._doc_graph_cache.get(doc_id)
        if cached is not None:
            return cached
        nodes = {
            node["node_id"]: node
            for node in (
                _decode_node(row)
                for row in connection.execute(
                    "SELECT * FROM nodes WHERE doc_id = ?",
                    (doc_id,),
                )
            )
        }
        adjacency: dict[str, list[dict[str, Any]]] = {}
        for row in connection.execute(
            """
            SELECT * FROM edges
            WHERE doc_id = ?
              AND edge_type NOT IN ('APPLIES_TO', 'REFERENCES')
            ORDER BY edge_type, edge_id
            """,
            (doc_id,),
        ):
            edge = _decode_edge(row)
            adjacency.setdefault(edge["source_id"], []).append(edge)
            adjacency.setdefault(edge["target_id"], []).append(edge)
        cached = (nodes, adjacency)
        self._doc_graph_cache[doc_id] = cached
        return cached

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 5,
        max_hops: int = 2,
        entity_limit: int = 8,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Graph query must not be empty")
        if not 1 <= limit <= 50:
            raise ValueError("Graph result limit must be from 1 to 50")
        if max_hops not in {1, 2}:
            raise ValueError("Graph max_hops must be 1 or 2")
        if not 1 <= entity_limit <= 50:
            raise ValueError("entity_limit must be from 1 to 50")

        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            doc_id = self._matching_doc_id(connection, filters)
            anchor_rows = connection.execute(
                """
                SELECT n.*, bm25(nodes_fts, 0.0, 2.2, 1.0, 0.6) AS fts_raw
                FROM nodes_fts
                JOIN nodes AS n ON n.node_id = nodes_fts.node_id
                WHERE nodes_fts MATCH ? AND n.doc_id = ?
                ORDER BY fts_raw ASC, n.node_id ASC
                LIMIT ?
                """,
                (self._fts_query(query), doc_id, entity_limit),
            ).fetchall()
            query_terms = _terms(query)
            anchors: list[tuple[dict[str, Any], float]] = []
            for row in anchor_rows:
                node = _decode_node(row)
                raw = abs(float(node.pop("fts_raw")))
                overlap = len(
                    query_terms.intersection(
                        _terms(f"{node['label']} {node['text']}")
                    )
                ) / max(len(query_terms), 1)
                score = (
                    0.48 * min(raw / 12.0, 1.0)
                    + 0.37 * overlap
                    + 0.15 * NODE_TYPE_WEIGHTS[node["node_type"]]
                )
                anchors.append((node, score))

            paths: dict[str, dict[str, Any]] = {}
            node_cache, adjacency = self._document_graph(
                connection, doc_id
            )
            for anchor, entity_score in anchors:
                frontier: list[
                    tuple[list[dict[str, Any]], list[dict[str, Any]], float]
                ] = [([anchor], [], entity_score)]
                for hop in range(1, max_hops + 1):
                    next_frontier: list[
                        tuple[
                            list[dict[str, Any]],
                            list[dict[str, Any]],
                            float,
                        ]
                    ] = []
                    for nodes, edges, score in frontier:
                        current = nodes[-1]
                        for edge in adjacency.get(
                            current["node_id"], []
                        )[:40]:
                            target_id = (
                                edge["target_id"]
                                if edge["source_id"] == current["node_id"]
                                else edge["source_id"]
                            )
                            if any(
                                item["node_id"] == target_id for item in nodes
                            ):
                                continue
                            target = node_cache.get(target_id)
                            if target is None:
                                continue
                            overlap = len(
                                query_terms.intersection(
                                    _terms(
                                        f"{target['label']} {target['text']}"
                                    )
                                )
                            ) / max(len(query_terms), 1)
                            relation = RELATION_WEIGHTS[edge["edge_type"]]
                            target_weight = NODE_TYPE_WEIGHTS[
                                target["node_type"]
                            ]
                            path_score = (
                                score
                                + relation * edge["confidence"] / (hop + 0.5)
                                + 0.20 * overlap
                                + 0.08 * target_weight
                            )
                            new_nodes = [*nodes, target]
                            new_edges = [*edges, edge]
                            path_id = _stable_id(
                                "path",
                                doc_id,
                                [item["node_id"] for item in new_nodes],
                                [item["edge_id"] for item in new_edges],
                            )
                            pages = sorted(
                                {
                                    int(item["page_no"])
                                    for item in new_nodes
                                    if item.get("page_no") is not None
                                }
                            )
                            evidence_ids = list(
                                dict.fromkeys(
                                    evidence_id
                                    for item in new_nodes
                                    for evidence_id in item["evidence_ids"]
                                )
                            )
                            path = {
                                "path_id": path_id,
                                "rank": 0,
                                "score": round(path_score, 6),
                                "entity_score": round(entity_score, 6),
                                "hops": hop,
                                "doc_id": doc_id,
                                "brand": anchor["brand"],
                                "model": anchor["model"],
                                "year": anchor["year"],
                                "region": anchor["region"],
                                "language": anchor["language"],
                                "manual_type": anchor["manual_type"],
                                "page_nos": pages,
                                "section_path": list(
                                    max(
                                        (
                                            item["section_path"]
                                            for item in new_nodes
                                        ),
                                        key=len,
                                        default=[],
                                    )
                                ),
                                "node_ids": [
                                    item["node_id"] for item in new_nodes
                                ],
                                "node_types": [
                                    item["node_type"] for item in new_nodes
                                ],
                                "node_labels": [
                                    item["label"] for item in new_nodes
                                ],
                                "edge_ids": [
                                    item["edge_id"] for item in new_edges
                                ],
                                "relations": [
                                    item["edge_type"] for item in new_edges
                                ],
                                "nodes": new_nodes,
                                "edges": new_edges,
                                "evidence_ids": evidence_ids,
                                "content": _compact(
                                    " | ".join(
                                        f"{item['node_type']} "
                                        f"{item['label']}: {item['text']}"
                                        for item in new_nodes
                                        if item["node_type"]
                                        not in {"Vehicle", "EvidencePage"}
                                    ),
                                    1800,
                                ),
                            }
                            path["source"] = _source_label(path)
                            existing = paths.get(path_id)
                            if (
                                existing is None
                                or path["score"] > existing["score"]
                            ):
                                paths[path_id] = path
                            next_frontier.append(
                                (new_nodes, new_edges, path_score)
                            )
                    frontier = sorted(
                        next_frontier,
                        key=lambda item: (
                            -item[2],
                            item[0][-1]["node_id"],
                        ),
                    )[:30]

            ordered = sorted(
                paths.values(),
                key=lambda item: (-item["score"], item["path_id"]),
            )[:limit]
            for rank, path in enumerate(ordered, start=1):
                path["rank"] = rank
            return ordered
        finally:
            connection.close()
