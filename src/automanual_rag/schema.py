"""Dependency-free schemas for corpus metadata and normalized manual elements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


SCHEMA_VERSION = 1
ELEMENT_TYPES = frozenset({"text", "table", "image"})
_DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def page_idx_to_page_no(page_idx: int) -> int:
    """Convert MinerU's zero-based page index to a one-based PDF page number."""

    if isinstance(page_idx, bool) or not isinstance(page_idx, int) or page_idx < 0:
        raise ValueError("page_idx must be a non-negative integer")
    return page_idx + 1


def _require_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _validate_bbox(
    bbox: Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    if isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise ValueError("bbox must contain exactly four numbers")
    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox values must be finite")
    if values[0] > values[2] or values[1] > values[3]:
        raise ValueError("bbox must use [x0, y0, x1, y1] order")
    return values


def _validate_source_span(
    source_span: Sequence[int] | None,
) -> tuple[int, int] | None:
    if source_span is None:
        return None
    if isinstance(source_span, (str, bytes)) or len(source_span) != 2:
        raise ValueError("source_span must contain exactly two integers")
    start, end = source_span
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise ValueError("source_span must use non-negative [start, end] order")
    return start, end


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One row from ``data/manifests/corpus.csv``."""

    doc_id: str
    brand: str
    model: str
    year: str
    region: str
    language: str
    manual_type: str
    source_url: str
    downloaded_at: str
    local_filename: str | None = None

    def validate(self) -> None:
        for name in (
            "doc_id",
            "brand",
            "model",
            "year",
            "region",
            "language",
            "manual_type",
            "source_url",
            "downloaded_at",
        ):
            _require_string(name, getattr(self, name))

        if not _DOC_ID_RE.fullmatch(self.doc_id):
            raise ValueError(f"Unsafe doc_id: {self.doc_id!r}")

        parsed_url = urlparse(self.source_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(f"source_url is not an HTTP(S) URL: {self.source_url!r}")

        try:
            datetime.fromisoformat(self.downloaded_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"downloaded_at is not ISO 8601: {self.downloaded_at!r}"
            ) from exc

        if self.local_filename is not None:
            _require_string("local_filename", self.local_filename)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "CorpusDocument":
        document = cls(
            doc_id=str(row.get("doc_id", "")).strip(),
            brand=str(row.get("brand", "")).strip(),
            model=str(row.get("model", "")).strip(),
            year=str(row.get("year", "")).strip(),
            region=str(row.get("region", "")).strip(),
            language=str(row.get("language", "")).strip(),
            manual_type=str(row.get("manual_type", "")).strip(),
            source_url=str(row.get("source_url", "")).strip(),
            downloaded_at=str(row.get("downloaded_at", "")).strip(),
            local_filename=(
                str(row["local_filename"]).strip()
                if row.get("local_filename")
                else None
            ),
        )
        document.validate()
        return document

    def element_metadata(self) -> dict[str, str]:
        return {
            "doc_id": self.doc_id,
            "brand": self.brand,
            "model": self.model,
            "year": self.year,
            "region": self.region,
            "language": self.language,
            "manual_type": self.manual_type,
        }


@dataclass(frozen=True, slots=True)
class ManualElement:
    """Normalized evidence element emitted as one JSONL record.

    ``page_no`` is always the one-based physical PDF page number when the
    source provides a valid MinerU ``page_idx``. It is ``None`` only for a
    malformed source element, which the importer also records as an anomaly.
    """

    element_id: str
    doc_id: str
    brand: str
    model: str
    year: str
    region: str
    language: str
    manual_type: str
    page_no: int | None
    section_path: tuple[str, ...]
    element_type: str
    content: str
    asset_path: str | None
    bbox: tuple[float, float, float, float] | None
    source_span: tuple[int, int] | None
    previous_element_id: str | None
    next_element_id: str | None
    source_locator: dict[str, Any]

    def validate(self) -> None:
        for name in (
            "element_id",
            "doc_id",
            "brand",
            "model",
            "year",
            "region",
            "language",
            "manual_type",
        ):
            _require_string(name, getattr(self, name))

        if not _DOC_ID_RE.fullmatch(self.doc_id):
            raise ValueError(f"Unsafe doc_id: {self.doc_id!r}")
        if self.element_type not in ELEMENT_TYPES:
            raise ValueError(f"Unsupported element_type: {self.element_type!r}")
        if self.page_no is not None and (
            isinstance(self.page_no, bool)
            or not isinstance(self.page_no, int)
            or self.page_no < 1
        ):
            raise ValueError("page_no must be a positive integer or None")
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        if not isinstance(self.section_path, tuple) or any(
            not isinstance(part, str) or not part.strip()
            for part in self.section_path
        ):
            raise ValueError("section_path must be a tuple of non-empty strings")
        if self.asset_path is not None and (
            not isinstance(self.asset_path, str) or not self.asset_path.strip()
        ):
            raise ValueError("asset_path must be a non-empty string or None")
        if self.element_type == "text" and self.asset_path is not None:
            raise ValueError("text elements must not have asset_path")

        _validate_bbox(self.bbox)
        _validate_source_span(self.source_span)

        for name in ("previous_element_id", "next_element_id"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"{name} must be a non-empty string or None")
            if value == self.element_id:
                raise ValueError(f"{name} must not refer to the element itself")

        if not isinstance(self.source_locator, dict):
            raise ValueError("source_locator must be a dictionary")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "element_id": self.element_id,
            "doc_id": self.doc_id,
            "brand": self.brand,
            "model": self.model,
            "year": self.year,
            "region": self.region,
            "language": self.language,
            "manual_type": self.manual_type,
            "page_no": self.page_no,
            "page_number_basis": "1-based physical PDF page",
            "section_path": list(self.section_path),
            "element_type": self.element_type,
            "content": self.content,
            "asset_path": self.asset_path,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "source_span": (
                list(self.source_span) if self.source_span is not None else None
            ),
            "previous_element_id": self.previous_element_id,
            "next_element_id": self.next_element_id,
            "source_locator": self.source_locator,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManualElement":
        element = cls(
            element_id=str(value.get("element_id", "")),
            doc_id=str(value.get("doc_id", "")),
            brand=str(value.get("brand", "")),
            model=str(value.get("model", "")),
            year=str(value.get("year", "")),
            region=str(value.get("region", "")),
            language=str(value.get("language", "")),
            manual_type=str(value.get("manual_type", "")),
            page_no=value.get("page_no"),
            section_path=tuple(value.get("section_path") or ()),
            element_type=str(value.get("element_type", "")),
            content=value.get("content", ""),
            asset_path=value.get("asset_path"),
            bbox=(
                _validate_bbox(value.get("bbox"))
                if value.get("bbox") is not None
                else None
            ),
            source_span=(
                _validate_source_span(value.get("source_span"))
                if value.get("source_span") is not None
                else None
            ),
            previous_element_id=value.get("previous_element_id"),
            next_element_id=value.get("next_element_id"),
            source_locator=dict(value.get("source_locator") or {}),
        )
        element.validate()
        return element


def stable_element_id(
    *,
    doc_id: str,
    page_idx: int | None,
    source_index: int,
    element_type: str,
    fingerprint: Mapping[str, Any],
) -> str:
    """Build a deterministic, cross-document unique element identifier."""

    if not _DOC_ID_RE.fullmatch(doc_id):
        raise ValueError(f"Unsafe doc_id: {doc_id!r}")
    if element_type not in ELEMENT_TYPES:
        raise ValueError(f"Unsupported element_type: {element_type!r}")
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise ValueError("source_index must be an integer")

    if page_idx is None:
        page_token = "unknown"
    else:
        page_token = f"{page_idx_to_page_no(page_idx):04d}"

    canonical = json.dumps(
        {
            "doc_id": doc_id,
            "page_idx": page_idx,
            "source_index": source_index,
            "element_type": element_type,
            "fingerprint": fingerprint,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return (
        f"{doc_id}:p{page_token}:{element_type}:"
        f"{source_index:06d}:{digest}"
    )
