"""Independent FastAPI service for the Agentic GraphRAG workflow."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import tempfile
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from automanual_rag.agentic import AgenticWorkflow
from automanual_rag.api import (
    IMAGE_SUFFIXES,
    VehicleFilter,
    _decode_image,
)
from automanual_rag.generation import configured_backend
from automanual_rag.graphrag import GraphRetriever
from automanual_rag.ingestion.mineru import load_manifest
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.table import TableIndex
from automanual_rag.retrieval.table_rows import TableRowIndex
from automanual_rag.retrieval.visual import VisualIndex


class AgenticQuestion(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    vehicle: VehicleFilter
    image_base64: str | None = None
    filename: str = "query.jpg"


class AgenticRAGService:
    """Load one shared explicit-state workflow and validate exact manuals."""

    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path,
        graph_index_path: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        self.data_root = data_root.resolve()
        self.documents = load_manifest(
            self.data_root / "manifests" / "corpus.csv"
        )
        indexes = self.data_root / "indexes"
        graph = GraphRetriever(graph_index_path.resolve())
        visual_path = indexes / "visual_traditional.npz"
        table_rows_path = indexes / "table_rows.sqlite3"
        tables_path = indexes / "tables.sqlite3"
        self.workflow = AgenticWorkflow(
            text_index=BM25Index(indexes / "bm25.sqlite3"),
            graph_retriever=graph,
            visual_index=(
                VisualIndex(
                    visual_path,
                    project_root=self.data_root.parent,
                )
                if visual_path.is_file()
                else None
            ),
            table_row_index=(
                TableRowIndex(table_rows_path)
                if table_rows_path.is_file()
                else None
            ),
            table_index=(
                TableIndex(tables_path) if tables_path.is_file() else None
            ),
            generation_backend=configured_backend(),
            asset_root=self.data_root.parent,
        )

    def _validated_filters(
        self, vehicle: VehicleFilter
    ) -> dict[str, str]:
        requested = vehicle.retrieval_filters()
        matching = [
            document
            for document in self.documents
            if all(
                str(getattr(document, field)).casefold()
                == str(value).casefold()
                for field, value in requested.items()
            )
        ]
        if len(matching) != 1:
            raise ValueError(
                "Vehicle metadata does not identify exactly one corpus manual"
            )
        return matching[0].element_metadata()

    def health(self) -> dict[str, Any]:
        graph = self.workflow.graph_retriever.count()
        return {
            "status": "ok",
            "backend": "agentic_graphrag_state_graph_v1",
            "manuals": len(self.documents),
            "graph": graph,
            "generation_backend": (
                self.workflow.generation_backend.name
                if self.workflow.generation_backend is not None
                else "extractive_evidence_v1"
            ),
        }

    def answer(self, payload: AgenticQuestion) -> dict[str, Any]:
        filters = self._validated_filters(payload.vehicle)
        if payload.image_base64 is None:
            return self.workflow.run(
                query=payload.query,
                filters=filters,
            )
        suffix = Path(payload.filename).suffix.casefold()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(
                "filename must end in jpg, jpeg, png, webp, or bmp"
            )
        content = _decode_image(payload.image_base64)
        with tempfile.TemporaryDirectory(prefix="agentic-query-") as temp:
            image_path = Path(temp) / f"query{suffix}"
            image_path.write_bytes(content)
            return self.workflow.run(
                query=payload.query,
                filters=filters,
                image_path=str(image_path),
            )


def default_agentic_service(
    *,
    project_root: Path,
    data_root: Path | None = None,
    graph_index_path: Path | None = None,
) -> AgenticRAGService:
    project_root = project_root.resolve()
    data_root = (
        data_root.resolve()
        if data_root is not None
        else project_root / "data"
    )
    graph_index_path = (
        graph_index_path.resolve()
        if graph_index_path is not None
        else project_root / "data" / "indexes" / "manual_graph.sqlite3"
    )
    return AgenticRAGService(
        project_root=project_root,
        data_root=data_root,
        graph_index_path=graph_index_path,
    )


def create_agentic_app(
    *,
    project_root: Path | None = None,
    data_root: Path | None = None,
    graph_index_path: Path | None = None,
    service: Any | None = None,
) -> FastAPI:
    root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or default_agentic_service(
            project_root=root,
            data_root=data_root,
            graph_index_path=graph_index_path,
        )
        yield

    app = FastAPI(
        title="AutoManual Agentic GraphRAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return request.app.state.service.health()

    @app.post("/v1/agentic")
    def answer(
        payload: AgenticQuestion,
        request: Request,
    ) -> dict[str, Any]:
        try:
            return request.app.state.service.answer(payload)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
