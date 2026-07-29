"""Import native MinerU content lists into normalized JSONL elements."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from automanual_rag.schema import (
    CorpusDocument,
    ManualElement,
    SCHEMA_VERSION,
    page_idx_to_page_no,
    stable_element_id,
)


REQUIRED_MANIFEST_FIELDS = {
    "doc_id",
    "brand",
    "model",
    "year",
    "region",
    "language",
    "manual_type",
    "source_url",
    "downloaded_at",
}
SKIPPED_LAYOUT_TYPES = {"header", "footer", "page_header", "page_footer", "page_number"}
TEXT_SOURCE_TYPES = {"text", "page_footnote"}
IMAGE_SOURCE_TYPES = {"image", "chart", "equation"}


@dataclass(slots=True)
class ImportResult:
    document: CorpusDocument
    elements: list[ManualElement]
    summary: dict[str, Any]


def load_manifest(path: Path) -> list[CorpusDocument]:
    """Read and validate corpus metadata without third-party dependencies."""

    if not path.is_file():
        raise FileNotFoundError(f"Corpus manifest not found: {path}")

    documents: list[CorpusDocument] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_FIELDS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Manifest is missing required columns: "
                + ", ".join(sorted(missing))
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                document = CorpusDocument.from_mapping(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if document.doc_id in seen:
                raise ValueError(
                    f"{path}:{line_number}: duplicate doc_id {document.doc_id!r}"
                )
            seen.add(document.doc_id)
            documents.append(document)

    if not documents:
        raise ValueError(f"Manifest contains no documents: {path}")
    return documents


def _safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        pieces: list[str] = []
        preferred_keys = (
            "text",
            "content",
            "caption",
            "title_content",
            "paragraph_content",
        )
        matched = False
        for key in preferred_keys:
            if key in value:
                matched = True
                pieces.extend(_flatten_text(value[key]))
        if not matched:
            for child in value.values():
                pieces.extend(_flatten_text(child))
        return pieces
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        pieces = []
        for child in value:
            pieces.extend(_flatten_text(child))
        return pieces
    return []


def _normalized_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        return None
    try:
        bbox = tuple(float(part) for part in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(part) for part in bbox):
        return None
    if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
        return None
    return bbox


def _valid_page_idx(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class MinerUImporter:
    """Normalize one MinerU ``*_content_list.json`` at a time."""

    def __init__(self, *, project_root: Path, parsed_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.parsed_root = parsed_root.resolve()

    def find_content_list(self, doc_id: str) -> Path:
        doc_root = self.parsed_root / doc_id
        preferred = (
            doc_root
            / doc_id
            / "txt"
            / f"{doc_id}_content_list.json"
        )
        if preferred.is_file():
            return preferred
        if not doc_root.is_dir():
            raise FileNotFoundError(f"Parsed document directory not found: {doc_root}")
        matches = sorted(doc_root.rglob(f"{doc_id}_content_list.json"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one content list for {doc_id}, found {len(matches)}"
            )
        return matches[0]

    def _parser_version(self, document: CorpusDocument) -> str | None:
        marker_path = self.parsed_root / document.doc_id / "_SUCCESS.json"
        if not marker_path.is_file():
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = marker.get("mineru_version")
        return str(value) if value else None

    @staticmethod
    def _section_paths(
        source_elements: Sequence[Any],
    ) -> list[tuple[str, ...]]:
        page_headers: dict[int, str] = {}
        for item in source_elements:
            if not isinstance(item, Mapping) or item.get("type") not in {
                "header",
                "page_header",
            }:
                continue
            page_idx = _valid_page_idx(item.get("page_idx"))
            text = str(item.get("text") or "").strip()
            if page_idx is not None and text and page_idx not in page_headers:
                page_headers[page_idx] = text

        stack: list[str] = []
        paths: list[tuple[str, ...]] = []
        current_page_idx: int | None = None
        for item in source_elements:
            if isinstance(item, Mapping):
                page_idx = _valid_page_idx(item.get("page_idx"))
                if page_idx != current_page_idx:
                    current_page_idx = page_idx
                    if page_idx in page_headers:
                        stack = [page_headers[page_idx]]

            if isinstance(item, Mapping) and item.get("type") in TEXT_SOURCE_TYPES:
                text = str(item.get("text") or "").strip()
                level = item.get("text_level")
                if (
                    text
                    and isinstance(level, int)
                    and not isinstance(level, bool)
                    and level > 0
                ):
                    page_header = page_headers.get(page_idx)
                    if level == 1:
                        if page_header and page_header.casefold() != text.casefold():
                            stack = [page_header]
                        else:
                            stack = []
                    else:
                        stack = stack[: max(1, level - 1)]
                    while len(stack) < level - 1:
                        stack.append("")
                    stack.append(text)
            paths.append(tuple(part for part in stack if part))
        return paths

    @staticmethod
    def _page_labels(source_elements: Sequence[Any]) -> dict[int, str]:
        labels: dict[int, str] = {}
        for item in source_elements:
            if not isinstance(item, Mapping) or item.get("type") != "page_number":
                continue
            page_idx = _valid_page_idx(item.get("page_idx"))
            text = str(item.get("text") or "").strip()
            if page_idx is not None and text and page_idx not in labels:
                labels[page_idx] = text
        return labels

    @staticmethod
    def _nearest_text(
        source_elements: Sequence[Any],
        source_index: int,
        page_idx: int | None,
        *,
        direction: int,
    ) -> str | None:
        cursor = source_index + direction
        while 0 <= cursor < len(source_elements):
            item = source_elements[cursor]
            cursor += direction
            if not isinstance(item, Mapping):
                continue
            candidate_page = _valid_page_idx(item.get("page_idx"))
            if candidate_page != page_idx:
                break
            if item.get("type") not in TEXT_SOURCE_TYPES:
                continue
            text = str(item.get("text") or "").strip()
            if text:
                return text
        return None

    def _asset_path(
        self,
        *,
        content_list_path: Path,
        native_asset_path: Any,
        anomalies: list[dict[str, Any]],
        source_index: int,
    ) -> tuple[str | None, bool]:
        if not isinstance(native_asset_path, str) or not native_asset_path.strip():
            anomalies.append(
                {
                    "source_index": source_index,
                    "code": "missing_asset_path",
                    "message": "Visual element has no native img_path",
                }
            )
            return None, False

        native = Path(native_asset_path.replace("\\", "/"))
        if native.is_absolute():
            anomalies.append(
                {
                    "source_index": source_index,
                    "code": "absolute_asset_path",
                    "message": f"Rejected absolute native asset path: {native_asset_path}",
                }
            )
            return None, False

        candidate = (content_list_path.parent / native).resolve()
        try:
            relative = candidate.relative_to(self.project_root).as_posix()
        except ValueError:
            anomalies.append(
                {
                    "source_index": source_index,
                    "code": "unsafe_asset_path",
                    "message": f"Asset escapes project root: {native_asset_path}",
                }
            )
            return None, False

        if not candidate.is_file():
            anomalies.append(
                {
                    "source_index": source_index,
                    "code": "asset_not_found",
                    "message": f"Asset file does not exist: {relative}",
                }
            )
            return relative, False
        return relative, True

    @staticmethod
    def _visual_content(
        *,
        source: Mapping[str, Any],
        source_type: str,
        section_path: tuple[str, ...],
        previous_text: str | None,
        next_text: str | None,
        page_no: int | None,
    ) -> str:
        if source_type == "table":
            caption = _flatten_text(source.get("table_caption"))
            footnote = _flatten_text(source.get("table_footnote"))
        elif source_type == "chart":
            caption = _flatten_text(source.get("chart_caption"))
            footnote = _flatten_text(source.get("chart_footnote"))
        else:
            caption = _flatten_text(source.get("image_caption"))
            footnote = _flatten_text(source.get("image_footnote"))

        recognized = _flatten_text(
            source.get("ocr_text")
            or source.get("content")
            or source.get("text")
        )
        parts: list[str] = []
        if section_path:
            parts.append("Section: " + " > ".join(section_path))
        if caption:
            parts.append("Caption: " + " ".join(caption))
        if footnote:
            parts.append("Footnote: " + " ".join(footnote))
        if recognized:
            parts.append("Recognized text: " + " ".join(recognized))
        if previous_text:
            parts.append("Previous text: " + previous_text)
        if next_text:
            parts.append("Next text: " + next_text)
        if not parts:
            page_label = str(page_no) if page_no is not None else "unknown"
            parts.append(f"Visual element on PDF page {page_label}.")
        return "\n".join(parts)

    def import_document(self, document: CorpusDocument) -> ImportResult:
        document.validate()
        content_list_path = self.find_content_list(document.doc_id)
        try:
            source_elements = json.loads(
                content_list_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {content_list_path}: {exc}") from exc
        if not isinstance(source_elements, list):
            raise ValueError(f"Content list must be a JSON array: {content_list_path}")

        parser_version = self._parser_version(document)
        source_file = _safe_relative(content_list_path, self.project_root)
        section_paths = self._section_paths(source_elements)
        page_labels = self._page_labels(source_elements)
        source_page_indices = {
            page_idx
            for item in source_elements
            if isinstance(item, Mapping)
            for page_idx in [_valid_page_idx(item.get("page_idx"))]
            if page_idx is not None
        }
        anomalies: list[dict[str, Any]] = []
        skipped_layout = 0
        skipped_empty_text = 0
        skipped_unsupported = 0
        missing_page_no = 0
        invalid_asset_paths = 0
        table_structured_count = 0
        source_type_counts: Counter[str] = Counter()
        output: list[ManualElement] = []
        observed_page_indices: set[int] = set()

        for source_index, source in enumerate(source_elements):
            if not isinstance(source, Mapping):
                anomalies.append(
                    {
                        "source_index": source_index,
                        "code": "invalid_element",
                        "message": "Source element is not an object",
                    }
                )
                skipped_unsupported += 1
                continue

            source_type = str(source.get("type") or "").strip()
            source_type_counts[source_type or "<missing>"] += 1
            if source_type in SKIPPED_LAYOUT_TYPES:
                skipped_layout += 1
                continue
            if source_type in TEXT_SOURCE_TYPES:
                element_type = "text"
            elif source_type == "table":
                element_type = "table"
            elif source_type in IMAGE_SOURCE_TYPES:
                element_type = "image"
            else:
                anomalies.append(
                    {
                        "source_index": source_index,
                        "code": "unsupported_type",
                        "message": f"Unsupported or missing source type: {source_type!r}",
                    }
                )
                skipped_unsupported += 1
                continue

            page_idx = _valid_page_idx(source.get("page_idx"))
            if page_idx is None:
                page_no = None
                missing_page_no += 1
                anomalies.append(
                    {
                        "source_index": source_index,
                        "code": "invalid_page_idx",
                        "message": f"Invalid page_idx: {source.get('page_idx')!r}",
                    }
                )
            else:
                page_no = page_idx_to_page_no(page_idx)
                observed_page_indices.add(page_idx)

            raw_bbox = source.get("bbox")
            bbox = _normalized_bbox(raw_bbox)
            if raw_bbox is not None and bbox is None:
                anomalies.append(
                    {
                        "source_index": source_index,
                        "code": "invalid_bbox",
                        "message": f"Invalid bbox: {raw_bbox!r}",
                    }
                )

            section_path = section_paths[source_index]
            asset_path: str | None = None
            native_asset_path = source.get("img_path")
            previous_text: str | None = None
            next_text: str | None = None
            caption: list[str] = []
            footnote: list[str] = []

            if element_type == "text":
                content = str(source.get("text") or "").strip()
                if not content:
                    skipped_empty_text += 1
                    continue
            else:
                asset_path, asset_valid = self._asset_path(
                    content_list_path=content_list_path,
                    native_asset_path=native_asset_path,
                    anomalies=anomalies,
                    source_index=source_index,
                )
                if not asset_valid:
                    invalid_asset_paths += 1
                previous_text = self._nearest_text(
                    source_elements,
                    source_index,
                    page_idx,
                    direction=-1,
                )
                next_text = self._nearest_text(
                    source_elements,
                    source_index,
                    page_idx,
                    direction=1,
                )
                content = self._visual_content(
                    source=source,
                    source_type=source_type,
                    section_path=section_path,
                    previous_text=previous_text,
                    next_text=next_text,
                    page_no=page_no,
                )
                if source_type == "table":
                    caption = _flatten_text(source.get("table_caption"))
                    footnote = _flatten_text(source.get("table_footnote"))
                    structured = bool(
                        _flatten_text(source.get("html"))
                        or _flatten_text(source.get("table_body"))
                    )
                    table_structured_count += int(structured)
                elif source_type == "chart":
                    caption = _flatten_text(source.get("chart_caption"))
                    footnote = _flatten_text(source.get("chart_footnote"))
                else:
                    caption = _flatten_text(source.get("image_caption"))
                    footnote = _flatten_text(source.get("image_footnote"))

            title_level = source.get("text_level")
            fingerprint = {
                "source_type": source_type,
                "text": str(source.get("text") or "").strip(),
                "native_asset_path": (
                    str(native_asset_path) if native_asset_path is not None else None
                ),
                "bbox": list(bbox) if bbox is not None else None,
                "caption": caption,
                "footnote": footnote,
                "title_level": title_level,
            }
            element_id = stable_element_id(
                doc_id=document.doc_id,
                page_idx=page_idx,
                source_index=source_index,
                element_type=element_type,
                fingerprint=fingerprint,
            )
            source_locator = {
                "parser": "MinerU",
                "parser_version": parser_version,
                "source_file": source_file,
                "source_index": source_index,
                "source_type": source_type,
                "source_page_idx": page_idx,
                "source_page_no": page_no,
                "source_page_label": page_labels.get(page_idx),
                "native_asset_path": native_asset_path,
                "title_level": title_level,
                "caption": caption,
                "footnote": footnote,
                "structured_table_content_available": (
                    bool(
                        _flatten_text(source.get("html"))
                        or _flatten_text(source.get("table_body"))
                    )
                    if source_type == "table"
                    else None
                ),
            }
            element = ManualElement(
                element_id=element_id,
                **document.element_metadata(),
                page_no=page_no,
                section_path=section_path,
                element_type=element_type,
                content=content,
                asset_path=asset_path,
                bbox=bbox,
                source_span=None,
                previous_element_id=None,
                next_element_id=None,
                source_locator=source_locator,
            )
            element.validate()
            output.append(element)

        linked: list[ManualElement] = []
        for index, element in enumerate(output):
            linked.append(
                replace(
                    element,
                    previous_element_id=(
                        output[index - 1].element_id if index > 0 else None
                    ),
                    next_element_id=(
                        output[index + 1].element_id
                        if index + 1 < len(output)
                        else None
                    ),
                )
            )

        element_counts = Counter(element.element_type for element in linked)
        max_source_page_idx = max(source_page_indices, default=-1)
        source_page_count = max_source_page_idx + 1
        summary = {
            "doc_id": document.doc_id,
            "source_file": source_file,
            "source_elements": len(source_elements),
            "source_type_counts": dict(sorted(source_type_counts.items())),
            "output_elements": len(linked),
            "element_counts": {
                kind: element_counts.get(kind, 0)
                for kind in ("text", "image", "table")
            },
            "page_number_basis": "1-based physical PDF page (MinerU page_idx + 1)",
            "source_page_count": source_page_count,
            "pages_with_source_elements": len(source_page_indices),
            "pages_with_output_elements": len(observed_page_indices),
            "blank_or_unrepresented_pages": (
                source_page_count - len(source_page_indices)
                if max_source_page_idx >= 0
                else 0
            ),
            "pages_without_imported_elements": (
                source_page_count - len(observed_page_indices)
                if max_source_page_idx >= 0
                else 0
            ),
            "missing_page_no": missing_page_no,
            "invalid_asset_paths": invalid_asset_paths,
            "table_structured_count": table_structured_count,
            "table_image_only_count": (
                element_counts.get("table", 0) - table_structured_count
            ),
            "skipped_elements": (
                skipped_layout + skipped_empty_text + skipped_unsupported
            ),
            "skipped_layout_elements": skipped_layout,
            "skipped_empty_text": skipped_empty_text,
            "skipped_unsupported_elements": skipped_unsupported,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
        }
        return ImportResult(document=document, elements=linked, summary=summary)


def write_jsonl(path: Path, elements: Iterable[ManualElement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for element in elements:
            handle.write(
                json.dumps(
                    element.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)


def import_corpus(
    *,
    project_root: Path,
    parsed_root: Path,
    output_root: Path,
    documents: Sequence[CorpusDocument],
) -> dict[str, Any]:
    importer = MinerUImporter(
        project_root=project_root,
        parsed_root=parsed_root,
    )
    summaries: list[dict[str, Any]] = []
    total_counts: Counter[str] = Counter()
    total_missing_pages = 0
    total_invalid_assets = 0
    total_anomalies = 0

    for document in documents:
        result = importer.import_document(document)
        output_path = output_root / document.doc_id / "elements.jsonl"
        write_jsonl(output_path, result.elements)
        summary = dict(result.summary)
        summary["output_file"] = _safe_relative(output_path, project_root)
        summaries.append(summary)
        total_counts.update(summary["element_counts"])
        total_missing_pages += summary["missing_page_no"]
        total_invalid_assets += summary["invalid_asset_paths"]
        total_anomalies += summary["anomaly_count"]

    corpus_summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": summaries,
        "totals": {
            "documents": len(summaries),
            "elements": sum(total_counts.values()),
            "text": total_counts["text"],
            "image": total_counts["image"],
            "table": total_counts["table"],
            "missing_page_no": total_missing_pages,
            "invalid_asset_paths": total_invalid_assets,
            "anomalies": total_anomalies,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "import_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(corpus_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return corpus_summary
