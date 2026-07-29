"""Build stable, citation-ready text chunks from normalized manual elements."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from automanual_rag.schema import ManualElement


CHUNK_SCHEMA_VERSION = 1
CHUNK_TYPES = frozenset({"text", "steps", "warning", "caution", "note"})
_SAFETY_RE = re.compile(r"^\s*(WARNING|CAUTION|NOTE)\s*:", re.IGNORECASE)
_STEP_RE = re.compile(r"^\s*(?:STEP\s+)?\d+[.)]\s+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    brand: str
    model: str
    year: str
    region: str
    language: str
    manual_type: str
    page_start: int | None
    page_end: int | None
    page_nos: tuple[int, ...]
    section_path: tuple[str, ...]
    chunk_type: str
    content: str
    indexed_text: str
    element_ids: tuple[str, ...]
    previous_chunk_id: str | None
    next_chunk_id: str | None

    def validate(self) -> None:
        for name in (
            "chunk_id",
            "doc_id",
            "brand",
            "model",
            "year",
            "region",
            "language",
            "manual_type",
            "content",
            "indexed_text",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.chunk_type not in CHUNK_TYPES:
            raise ValueError(f"Unsupported chunk_type: {self.chunk_type!r}")
        if not self.element_ids or any(
            not isinstance(value, str) or not value
            for value in self.element_ids
        ):
            raise ValueError("element_ids must contain non-empty IDs")
        if tuple(sorted(set(self.page_nos))) != self.page_nos:
            raise ValueError("page_nos must be unique and sorted")
        if self.page_nos:
            if self.page_start != self.page_nos[0]:
                raise ValueError("page_start does not match page_nos")
            if self.page_end != self.page_nos[-1]:
                raise ValueError("page_end does not match page_nos")
        elif self.page_start is not None or self.page_end is not None:
            raise ValueError("page range requires page_nos")
        if any(page < 1 for page in self.page_nos):
            raise ValueError("page_nos must be one-based positive integers")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.section_path
        ):
            raise ValueError("section_path must contain non-empty strings")
        for name in ("previous_chunk_id", "next_chunk_id"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty string or None")
            if value == self.chunk_id:
                raise ValueError(f"{name} must not refer to the chunk itself")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "brand": self.brand,
            "model": self.model,
            "year": self.year,
            "region": self.region,
            "language": self.language,
            "manual_type": self.manual_type,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "page_nos": list(self.page_nos),
            "page_number_basis": "1-based physical PDF page",
            "section_path": list(self.section_path),
            "chunk_type": self.chunk_type,
            "content": self.content,
            "indexed_text": self.indexed_text,
            "element_ids": list(self.element_ids),
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TextChunk":
        chunk = cls(
            chunk_id=str(value.get("chunk_id", "")),
            doc_id=str(value.get("doc_id", "")),
            brand=str(value.get("brand", "")),
            model=str(value.get("model", "")),
            year=str(value.get("year", "")),
            region=str(value.get("region", "")),
            language=str(value.get("language", "")),
            manual_type=str(value.get("manual_type", "")),
            page_start=value.get("page_start"),
            page_end=value.get("page_end"),
            page_nos=tuple(value.get("page_nos") or ()),
            section_path=tuple(value.get("section_path") or ()),
            chunk_type=str(value.get("chunk_type", "")),
            content=str(value.get("content", "")),
            indexed_text=str(value.get("indexed_text", "")),
            element_ids=tuple(value.get("element_ids") or ()),
            previous_chunk_id=value.get("previous_chunk_id"),
            next_chunk_id=value.get("next_chunk_id"),
        )
        chunk.validate()
        return chunk


def _chunk_type(element: ManualElement) -> str:
    safety = _SAFETY_RE.match(element.content)
    if safety:
        return safety.group(1).lower()
    if _STEP_RE.match(element.content):
        return "steps"
    return "text"


def _stable_chunk_id(
    *, doc_id: str, chunk_type: str, element_ids: Sequence[str]
) -> str:
    canonical = json.dumps(
        {
            "doc_id": doc_id,
            "chunk_type": chunk_type,
            "element_ids": list(element_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:chunk:{chunk_type}:{digest}"


def _make_chunk(
    elements: Sequence[ManualElement],
    chunk_type: str,
) -> TextChunk:
    first = elements[0]
    page_nos = tuple(
        sorted({element.page_no for element in elements if element.page_no is not None})
    )
    content = "\n".join(element.content.strip() for element in elements)
    section = " > ".join(first.section_path)
    indexed_parts = [
        f"{first.brand} {first.model} {first.year}",
        section,
        content,
    ]
    indexed_text = "\n".join(part for part in indexed_parts if part)
    element_ids = tuple(element.element_id for element in elements)
    chunk = TextChunk(
        chunk_id=_stable_chunk_id(
            doc_id=first.doc_id,
            chunk_type=chunk_type,
            element_ids=element_ids,
        ),
        doc_id=first.doc_id,
        brand=first.brand,
        model=first.model,
        year=first.year,
        region=first.region,
        language=first.language,
        manual_type=first.manual_type,
        page_start=page_nos[0] if page_nos else None,
        page_end=page_nos[-1] if page_nos else None,
        page_nos=page_nos,
        section_path=first.section_path,
        chunk_type=chunk_type,
        content=content,
        indexed_text=indexed_text,
        element_ids=element_ids,
        previous_chunk_id=None,
        next_chunk_id=None,
    )
    chunk.validate()
    return chunk


def build_text_chunks(
    elements: Sequence[ManualElement],
    *,
    max_chars: int = 1200,
) -> list[TextChunk]:
    """Group text elements without crossing pages or section boundaries.

    Warnings, cautions, and notes remain standalone chunks. Consecutive numbered
    steps are grouped in source order. MinerU title elements are represented by
    ``section_path`` and are not duplicated in chunk bodies.
    """

    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    text_elements = [element for element in elements if element.element_type == "text"]
    if not text_elements:
        return []

    chunks: list[TextChunk] = []
    pending: list[ManualElement] = []
    pending_type: str | None = None
    pending_key: tuple[Any, ...] | None = None
    pending_chars = 0

    def flush() -> None:
        nonlocal pending, pending_type, pending_key, pending_chars
        if pending and pending_type:
            chunks.append(_make_chunk(pending, pending_type))
        pending = []
        pending_type = None
        pending_key = None
        pending_chars = 0

    for element in text_elements:
        kind = _chunk_type(element)
        if kind in {"warning", "caution", "note"}:
            flush()
            chunks.append(_make_chunk([element], kind))
            continue
        if element.source_locator.get("title_level") is not None:
            flush()
            continue

        key = (element.doc_id, element.page_no, element.section_path, kind)
        added_chars = len(element.content) + (1 if pending else 0)
        if pending and (
            pending_key != key or pending_chars + added_chars > max_chars
        ):
            flush()
        if not pending:
            pending_type = kind
            pending_key = key
        pending.append(element)
        pending_chars += added_chars
    flush()

    linked: list[TextChunk] = []
    for index, chunk in enumerate(chunks):
        linked_chunk = replace(
            chunk,
            previous_chunk_id=chunks[index - 1].chunk_id if index else None,
            next_chunk_id=(
                chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
            ),
        )
        linked_chunk.validate()
        linked.append(linked_chunk)
    return linked


def read_elements(path: Path) -> list[ManualElement]:
    elements: list[ManualElement] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
                elements.append(ManualElement.from_dict(value))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return elements


def write_chunks(path: Path, chunks: Iterable[TextChunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    chunk.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def build_corpus_chunks(
    *,
    element_paths: Mapping[str, Path],
    max_chars: int = 1200,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    total_types: Counter[str] = Counter()
    total_chunks = 0
    total_elements = 0
    oversized_chunks = 0

    for doc_id, element_path in element_paths.items():
        elements = read_elements(element_path)
        chunks = build_text_chunks(elements, max_chars=max_chars)
        output_path = element_path.with_name("chunks.jsonl")
        write_chunks(output_path, chunks)
        counts = Counter(chunk.chunk_type for chunk in chunks)
        used_elements = sum(len(chunk.element_ids) for chunk in chunks)
        oversized = sum(len(chunk.content) > max_chars for chunk in chunks)
        summaries.append(
            {
                "doc_id": doc_id,
                "source_file": element_path.as_posix(),
                "output_file": output_path.as_posix(),
                "source_elements": len(elements),
                "text_elements_used": used_elements,
                "chunks": len(chunks),
                "chunk_type_counts": {
                    kind: counts.get(kind, 0)
                    for kind in ("text", "steps", "warning", "caution", "note")
                },
                "oversized_single_element_chunks": oversized,
            }
        )
        total_types.update(counts)
        total_chunks += len(chunks)
        total_elements += used_elements
        oversized_chunks += oversized

    return {
        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
        "max_chars": max_chars,
        "documents": summaries,
        "totals": {
            "documents": len(summaries),
            "chunks": total_chunks,
            "text_elements_used": total_elements,
            "text": total_types["text"],
            "steps": total_types["steps"],
            "warning": total_types["warning"],
            "caution": total_types["caution"],
            "note": total_types["note"],
            "oversized_single_element_chunks": oversized_chunks,
        },
    }
