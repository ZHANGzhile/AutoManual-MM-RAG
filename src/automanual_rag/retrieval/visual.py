"""Traceable traditional visual retrieval using NumPy and Pillow.

The feature extractor combines normalized thumbnails, color/intensity
histograms, edge orientation, projection profiles, and low-frequency FFT
magnitudes. It is deliberately labelled as a traditional baseline, not a
semantic or neural image embedding.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.schema import ManualElement


INDEX_SCHEMA_VERSION = 1
BACKEND_NAME = "traditional_multifeature_visual_v1"
FEATURE_DIMENSIONS = 1296
VISUAL_FILTER_FIELDS = frozenset(
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


def _dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Visual retrieval requires NumPy and Pillow. Install "
            "requirements-visual.txt or use an environment containing both."
        ) from exc
    return np, Image, ImageOps


def _normalize_block(block: Any, weight: float) -> Any:
    np, _, _ = _dependencies()
    values = np.asarray(block, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm > 0:
        values /= norm
    return values * weight


def _trim_background(image: Any) -> Any:
    np, _, _ = _dependencies()
    rgb = np.asarray(image, dtype=np.uint8)
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    channel_range = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2)
    foreground = (gray < 247) | (channel_range > 12)
    if not foreground.any():
        return image
    rows, columns = np.nonzero(foreground)
    margin_x = max(2, int(image.width * 0.015))
    margin_y = max(2, int(image.height * 0.015))
    left = max(0, int(columns.min()) - margin_x)
    top = max(0, int(rows.min()) - margin_y)
    right = min(image.width, int(columns.max()) + margin_x + 1)
    bottom = min(image.height, int(rows.max()) + margin_y + 1)
    return image.crop((left, top, right, bottom))


def _fit_canvas(image: Any, size: int = 96) -> Any:
    _, Image, _ = _dependencies()
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    resized = image.copy()
    resized.thumbnail((size, size), resampling)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(
        resized,
        ((size - resized.width) // 2, (size - resized.height) // 2),
    )
    return canvas


def extract_visual_features(path: Path) -> Any:
    """Extract a deterministic, L2-normalized 1,296-dimensional vector."""

    np, Image, ImageOps = _dependencies()
    if not path.is_file():
        raise FileNotFoundError(f"Image asset not found: {path}")
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except OSError as exc:
        raise ValueError(f"Unreadable image asset: {path}: {exc}") from exc
    if image.width < 1 or image.height < 1:
        raise ValueError(f"Image has invalid dimensions: {path}")

    image = _fit_canvas(_trim_background(image))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ).astype(np.float32)
    darkness = 1.0 - gray

    # Coarse normalized shape/intensity thumbnail: 24 x 24.
    thumbnail = darkness.reshape(24, 4, 24, 4).mean(axis=(1, 3))

    # Gradient magnitude thumbnail and spatial orientation histograms.
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.hypot(gradient_x, gradient_y).astype(np.float32)
    gradient_thumbnail = magnitude.reshape(16, 6, 16, 6).mean(axis=(1, 3))
    orientation = np.mod(np.arctan2(gradient_y, gradient_x), np.pi)
    orientation_bins = np.minimum(
        (orientation * (8.0 / np.pi)).astype(np.int32),
        7,
    )
    orientation_histograms: list[Any] = []
    for row in range(4):
        for column in range(4):
            row_slice = slice(row * 24, (row + 1) * 24)
            column_slice = slice(column * 24, (column + 1) * 24)
            histogram = np.bincount(
                orientation_bins[row_slice, column_slice].reshape(-1),
                weights=magnitude[row_slice, column_slice].reshape(-1),
                minlength=8,
            )
            orientation_histograms.append(histogram)
    edge_orientation = np.concatenate(orientation_histograms)

    # RGB, HSV, and grayscale distributions.
    histograms: list[Any] = []
    for channel in range(3):
        histogram, _ = np.histogram(
            rgb[..., channel],
            bins=16,
            range=(0.0, 1.0),
        )
        histograms.append(histogram)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    for channel in range(3):
        histogram, _ = np.histogram(
            hsv[..., channel],
            bins=16,
            range=(0.0, 1.0),
        )
        histograms.append(histogram)
    gray_histogram, _ = np.histogram(gray, bins=32, range=(0.0, 1.0))
    histograms.append(gray_histogram)
    color_intensity = np.concatenate(histograms)

    # Row/column ink profiles over a 32 x 32 average-pooled image.
    profile_image = darkness.reshape(32, 3, 32, 3).mean(axis=(1, 3))
    projections = np.concatenate(
        (profile_image.mean(axis=0), profile_image.mean(axis=1))
    )

    # Low-frequency magnitude spectrum is translation tolerant.
    frequency_image = darkness.reshape(48, 2, 48, 2).mean(axis=(1, 3))
    spectrum = np.log1p(np.abs(np.fft.rfft2(frequency_image))[:16, :9])
    spectrum[0, 0] = 0.0

    vector = np.concatenate(
        (
            _normalize_block(thumbnail, 2.0),
            _normalize_block(gradient_thumbnail, 1.25),
            _normalize_block(color_intensity, 0.75),
            _normalize_block(edge_orientation, 1.25),
            _normalize_block(projections, 0.75),
            _normalize_block(spectrum, 1.0),
        )
    ).astype(np.float32)
    if vector.shape != (FEATURE_DIMENSIONS,):
        raise AssertionError(
            f"Unexpected visual feature shape: {vector.shape}"
        )
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"Image produced an invalid visual vector: {path}")
    return vector / norm


def _read_elements(path: Path) -> list[ManualElement]:
    elements: list[ManualElement] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                elements.append(ManualElement.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return elements


def _record(element: ManualElement) -> dict[str, Any]:
    return {
        "element_id": element.element_id,
        "doc_id": element.doc_id,
        "brand": element.brand,
        "model": element.model,
        "year": element.year,
        "region": element.region,
        "language": element.language,
        "manual_type": element.manual_type,
        "page_no": element.page_no,
        "section_path": list(element.section_path),
        "element_type": element.element_type,
        "content": element.content,
        "asset_path": element.asset_path,
        "source_locator": element.source_locator,
    }


def _bytes_array(payload: bytes) -> Any:
    np, _, _ = _dependencies()
    return np.frombuffer(payload, dtype=np.uint8)


def build_visual_index(
    *,
    project_root: Path,
    index_path: Path,
    element_paths: Sequence[Path],
) -> dict[str, Any]:
    """Build an atomic, pickle-free index over normalized image elements."""

    np, _, _ = _dependencies()
    project_root = project_root.resolve()
    image_elements: list[ManualElement] = []
    table_elements = 0
    for path in element_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Element file not found: {path}")
        for element in _read_elements(path):
            if element.element_type == "table":
                table_elements += 1
            elif element.element_type == "image":
                image_elements.append(element)
    if not image_elements:
        raise ValueError("No normalized image elements found")
    if len({element.element_id for element in image_elements}) != len(
        image_elements
    ):
        raise ValueError("Duplicate visual element_id detected")

    records: list[dict[str, Any]] = []
    vectors: list[Any] = []
    asset_paths: set[Path] = set()
    for element in image_elements:
        if element.page_no is None:
            raise ValueError(f"Visual element has no page: {element.element_id}")
        if not element.asset_path:
            raise ValueError(
                f"Visual element has no asset_path: {element.element_id}"
            )
        asset_path = (project_root / element.asset_path).resolve()
        try:
            asset_path.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(
                f"Visual asset escapes project root: {element.asset_path}"
            ) from exc
        if asset_path in asset_paths:
            raise ValueError(f"Duplicate visual asset path: {element.asset_path}")
        asset_paths.add(asset_path)
        vectors.append(extract_visual_features(asset_path))
        records.append(_record(element))

    embeddings = np.vstack(vectors).astype(np.float32)
    records_payload = (
        "\n".join(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for value in records
        )
        + "\n"
    ).encode("utf-8")
    metadata = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "backend": BACKEND_NAME,
        "semantic_embedding": False,
        "neural_embedding": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "element_count": len(records),
        "documents": len({record["doc_id"] for record in records}),
        "dimensions": int(embeddings.shape[1]),
        "indexed_element_types": ["image"],
        "excluded_table_crops": table_elements,
        "records_sha256": hashlib.sha256(records_payload).hexdigest(),
        "numpy_version": np.__version__,
    }
    metadata_payload = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(index_path.name + ".tmp.npz")
    temporary.unlink(missing_ok=True)
    try:
        np.savez_compressed(
            temporary,
            metadata_json=_bytes_array(metadata_payload),
            records_jsonl=_bytes_array(records_payload),
            embeddings=embeddings,
        )
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **metadata,
        "index_path": index_path.as_posix(),
        "index_size_bytes": index_path.stat().st_size,
    }


class VisualIndex:
    def __init__(self, path: Path, *, project_root: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Visual index not found: {path}")
        np, _, _ = _dependencies()
        self.path = path
        self.project_root = project_root.resolve()
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(
                archive["metadata_json"].tobytes().decode("utf-8")
            )
            records_payload = archive["records_jsonl"].tobytes()
            self.embeddings = archive["embeddings"].astype(np.float32)
        if self.metadata.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported visual index schema version")
        if self.metadata.get("backend") != BACKEND_NAME:
            raise ValueError("Unexpected visual index backend")
        if (
            hashlib.sha256(records_payload).hexdigest()
            != self.metadata["records_sha256"]
        ):
            raise ValueError("Visual index records checksum mismatch")
        self.records = [
            json.loads(line)
            for line in records_payload.decode("utf-8").splitlines()
        ]
        expected_shape = (
            len(self.records),
            int(self.metadata["dimensions"]),
        )
        if self.embeddings.shape != expected_shape:
            raise ValueError(
                f"Visual embedding shape mismatch: "
                f"{self.embeddings.shape} != {expected_shape}"
            )

    def count(self) -> int:
        return len(self.records)

    def search(
        self,
        query_image: Path,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        np, _, _ = _dependencies()
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        filters = dict(filters or {})
        unknown = set(filters).difference(VISUAL_FILTER_FIELDS)
        if unknown:
            raise ValueError(
                "Unsupported visual filter(s): "
                + ", ".join(sorted(unknown))
            )
        normalized_filters = {
            key: str(value).strip().casefold()
            for key, value in filters.items()
            if value is not None and str(value).strip()
        }
        candidate_indexes = [
            index
            for index, record in enumerate(self.records)
            if all(
                str(record.get(field, "")).casefold() == expected
                for field, expected in normalized_filters.items()
            )
        ]
        if not candidate_indexes:
            return []
        query = extract_visual_features(query_image.resolve())
        scores = self.embeddings[candidate_indexes] @ query
        ordered = sorted(
            range(len(candidate_indexes)),
            key=lambda position: (
                -float(scores[position]),
                self.records[candidate_indexes[position]]["element_id"],
            ),
        )[:limit]
        results: list[dict[str, Any]] = []
        for rank, position in enumerate(ordered, start=1):
            record = dict(self.records[candidate_indexes[position]])
            record.update(
                {
                    "rank": rank,
                    "score": float(scores[position]),
                    "visual_score": float(scores[position]),
                }
            )
            results.append(record)
        return results


class VisualTextFusionIndex:
    """Fuse image rank with optional user-supplied text hint at page level."""

    def __init__(
        self,
        *,
        visual: VisualIndex,
        bm25: BM25Index,
        candidate_limit: int = 50,
        rrf_k: int = 60,
        visual_weight: float = 1.0,
        text_weight: float = 1.0,
    ) -> None:
        if not 1 <= candidate_limit <= 100:
            raise ValueError("candidate_limit must be from 1 to 100")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if visual_weight <= 0 or text_weight <= 0:
            raise ValueError("Fusion weights must be positive")
        self.visual = visual
        self.bm25 = bm25
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k
        self.visual_weight = visual_weight
        self.text_weight = text_weight
        self.path = {
            "visual": visual.path.as_posix(),
            "bm25": bm25.path.as_posix(),
        }

    def search(
        self,
        query_image: Path,
        *,
        query_text: str,
        filters: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not query_text.strip():
            raise ValueError("Visual/text fusion requires a non-empty text hint")
        visual_results = self.visual.search(
            query_image,
            filters=filters,
            limit=max(limit, self.candidate_limit),
        )
        text_results = self.bm25.search(
            query_text,
            filters=filters,
            limit=self.candidate_limit,
        )
        page_hits: dict[tuple[str, int], dict[str, Any]] = {}
        for result in text_results:
            for page_no in result["page_nos"]:
                key = (result["doc_id"], page_no)
                page_hits.setdefault(key, result)

        fused: list[dict[str, Any]] = []
        for result in visual_results:
            text_result = page_hits.get((result["doc_id"], result["page_no"]))
            visual_rank = int(result["rank"])
            text_rank = int(text_result["rank"]) if text_result else None
            score = self.visual_weight / (self.rrf_k + visual_rank)
            if text_rank is not None:
                score += self.text_weight / (self.rrf_k + text_rank)
            value = dict(result)
            value.update(
                {
                    "score": score,
                    "fusion_score": score,
                    "visual_rank": visual_rank,
                    "visual_score": float(result["score"]),
                    "text_rank": text_rank,
                    "text_score": (
                        float(text_result["score"]) if text_result else None
                    ),
                }
            )
            fused.append(value)
        fused.sort(key=lambda value: (-value["score"], value["element_id"]))
        fused = fused[:limit]
        for rank, value in enumerate(fused, start=1):
            value["rank"] = rank
        return fused
