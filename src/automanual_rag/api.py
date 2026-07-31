"""FastAPI service for grounded text, image, and table manual retrieval."""

from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from automanual_rag.answering import answer_question
from automanual_rag.generation import (
    LabeledImage,
    configured_backend,
    generate_or_fallback,
)
from automanual_rag.ingestion.mineru import load_manifest
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.table import TableIndex
from automanual_rag.retrieval.table_rows import TableRowIndex
from automanual_rag.retrieval.visual import (
    VisualIndex,
    VisualTextFusionIndex,
)
from automanual_rag.table_answering import answer_table_question


MAX_IMAGE_BYTES = 12 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class VehicleFilter(BaseModel):
    brand: str = "Ford"
    model: str
    year: str
    region: str = "North America"
    language: str = "en"
    manual_type: str = "owner_manual"
    doc_id: str | None = None

    def retrieval_filters(self) -> dict[str, str]:
        return {
            key: value
            for key, value in self.model_dump().items()
            if value is not None and str(value).strip()
        }


class TextQuestion(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    vehicle: VehicleFilter
    retrieval_limit: int = Field(default=10, ge=1, le=100)
    max_evidence: int = Field(default=3, ge=1, le=10)


class TableQuestion(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    vehicle: VehicleFilter
    retrieval_limit: int = Field(default=5, ge=1, le=100)
    max_evidence: int = Field(default=3, ge=1, le=10)


class ImageQuestion(BaseModel):
    image_base64: str = Field(min_length=4)
    filename: str = "query.jpg"
    question: str = Field(
        default="Identify the uploaded image.",
        max_length=2000,
    )
    vehicle: VehicleFilter
    limit: int = Field(default=5, ge=1, le=25)


def _decode_image(payload: str) -> bytes:
    value = payload.strip()
    if value.startswith("data:"):
        try:
            header, value = value.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid image data URL") from exc
        if ";base64" not in header or not header.startswith("data:image/"):
            raise ValueError("Only base64 image data URLs are supported")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not decoded:
        raise ValueError("Decoded image is empty")
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("Decoded image exceeds the 12 MiB limit")
    return decoded


class ManualRAGService:
    """Load all runtime indexes once and expose JSON-friendly operations."""

    def __init__(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        bm25_index_path: Path,
        visual_index_path: Path,
        table_index_path: Path,
        table_row_index_path: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        self.documents = load_manifest(manifest_path.resolve())
        self.bm25 = BM25Index(bm25_index_path.resolve())
        self.visual = VisualIndex(
            visual_index_path.resolve(),
            project_root=self.project_root,
        )
        self.tables = TableIndex(table_index_path.resolve())
        self.table_rows = TableRowIndex(table_row_index_path.resolve())
        self.generation = configured_backend()

    def _validated_filters(
        self,
        vehicle: VehicleFilter,
    ) -> dict[str, str]:
        filters = vehicle.retrieval_filters()
        matching = [
            document
            for document in self.documents
            if all(
                str(getattr(document, field)).casefold()
                == str(value).casefold()
                for field, value in filters.items()
            )
        ]
        if len(matching) != 1:
            raise ValueError(
                "Vehicle metadata does not identify exactly one corpus manual"
            )
        document = matching[0]
        return {
            "doc_id": document.doc_id,
            "brand": document.brand,
            "model": document.model,
            "year": document.year,
            "region": document.region,
            "language": document.language,
            "manual_type": document.manual_type,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "manuals": len(self.documents),
            "indexes": {
                "text_chunks": self.bm25.count(),
                "visual_elements": self.visual.count(),
                "table_crops": self.tables.count(),
                "verified_table_rows": self.table_rows.count(),
            },
            "generation_backend": (
                self.generation.name
                if self.generation is not None
                else "extractive_evidence_v1"
            ),
        }

    def answer_text(self, payload: TextQuestion) -> dict[str, Any]:
        result = answer_question(
            self.bm25,
            payload.query,
            self._validated_filters(payload.vehicle),
            retrieval_limit=payload.retrieval_limit,
            max_evidence=payload.max_evidence,
        )
        return generate_or_fallback(
            result,
            question=payload.query,
            backend=self.generation,
        )

    def answer_table(self, payload: TableQuestion) -> dict[str, Any]:
        return answer_table_question(
            self.table_rows,
            payload.query,
            self._validated_filters(payload.vehicle),
            retrieval_limit=payload.retrieval_limit,
            max_evidence=payload.max_evidence,
        )

    def answer_image(self, payload: ImageQuestion) -> dict[str, Any]:
        suffix = Path(payload.filename).suffix.casefold()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(
                "filename must end in jpg, jpeg, png, webp, or bmp"
            )
        content = _decode_image(payload.image_base64)
        with tempfile.TemporaryDirectory(prefix="automanual-query-") as temp:
            query_path = Path(temp) / f"query{suffix}"
            query_path.write_bytes(content)
            filters = self._validated_filters(payload.vehicle)
            if payload.question.strip():
                retriever: Any = VisualTextFusionIndex(
                    visual=self.visual,
                    bm25=self.bm25,
                )
                results = retriever.search(
                    query_path,
                    query_text=payload.question,
                    filters=filters,
                    limit=payload.limit,
                )
                retrieval_backend = "visual_text_rrf"
            else:
                results = self.visual.search(
                    query_path,
                    filters=filters,
                    limit=payload.limit,
                )
                retrieval_backend = "visual_only"
            result = self._visual_result(
                payload.question,
                results,
                query_path,
            )
            result["retrieval_backend"] = retrieval_backend
            return result

    def _visual_result(
        self,
        question: str,
        results: list[Mapping[str, Any]],
        query_path: Path,
    ) -> dict[str, Any]:
        evidence: list[dict[str, Any]] = []
        images = [LabeledImage("Uploaded query image", query_path)]
        for position, item in enumerate(results, start=1):
            source = (
                f"{item['brand']} {item['model']} Owner's Manual "
                f"({item['year']}), {' > '.join(item['section_path'])}, "
                f"physical PDF p.{item['page_no']}"
            )
            evidence.append(
                {"citation_id": position, **item, "source": source}
            )
            if position <= 3:
                images.append(
                    LabeledImage(
                        f"Evidence image [{position}]",
                        self.project_root / item["asset_path"],
                    )
                )
        if not evidence:
            return {
                "status": "insufficient_evidence",
                "reason": "no_visual_results",
                "answer": (
                    "The current manual evidence is insufficient to answer safely."
                ),
                "evidence": [],
                "generation": {
                    "status": "not_used",
                    "backend": "extractive_evidence_v1",
                },
            }
        base = {
            "status": "answered",
            "reason": "visual_evidence_retrieved",
            "answer": "\n".join(
                [
                    "Retrieved matching manual figures:",
                    *[
                        f"[{item['citation_id']}] {item['source']}"
                        for item in evidence
                    ],
                ]
            ),
            "evidence": evidence,
        }
        return generate_or_fallback(
            base,
            question=question or "Identify the uploaded image.",
            backend=self.generation,
            images=images,
        )


def default_service(project_root: Path) -> ManualRAGService:
    root = project_root.resolve()
    indexes = root / "data" / "indexes"
    return ManualRAGService(
        project_root=root,
        manifest_path=root / "data" / "manifests" / "corpus.csv",
        bm25_index_path=indexes / "bm25.sqlite3",
        visual_index_path=indexes / "visual_traditional.npz",
        table_index_path=indexes / "tables.sqlite3",
        table_row_index_path=indexes / "table_rows.sqlite3",
    )


def create_app(
    *,
    project_root: Path | None = None,
    service: Any | None = None,
) -> FastAPI:
    root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or default_service(root)
        yield

    app = FastAPI(
        title="AutoManual-MM-RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    def current(request: Request) -> Any:
        return request.app.state.service

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return current(request).health()

    @app.post("/v1/text")
    def text(
        payload: TextQuestion,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return current(request).answer_text(payload)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/table")
    def table(
        payload: TableQuestion,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return current(request).answer_table(payload)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/image")
    def image(
        payload: ImageQuestion,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return current(request).answer_image(payload)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
