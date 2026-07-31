"""Gradio demo for grounded text answers and visual page retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
from automanual_rag.schema import CorpusDocument
from automanual_rag.table_answering import answer_table_question


TEXT_COLUMNS = [
    "citation",
    "vehicle",
    "page",
    "section",
    "type",
    "retrieval rank",
    "confidence",
]
VISUAL_COLUMNS = [
    "rank",
    "vehicle",
    "page",
    "section",
    "score",
    "element ID",
]
TABLE_COLUMNS = [
    "citation",
    "vehicle",
    "page",
    "section",
    "verified row",
    "retrieval score",
    "verification",
]


def _document_label(document: CorpusDocument) -> str:
    return f"{document.brand} {document.model} {document.year}"


def _filters(document: CorpusDocument) -> dict[str, str]:
    return {
        "doc_id": document.doc_id,
        "brand": document.brand,
        "model": document.model,
        "year": document.year,
        "region": document.region,
        "language": document.language,
        "manual_type": document.manual_type,
    }


class DemoService:
    """Share one set of local indexes across Gradio callbacks."""

    def __init__(
        self,
        *,
        project_root: Path,
        manifest_path: Path,
        bm25_index_path: Path,
        table_index_path: Path,
        table_row_index_path: Path,
        visual_index_path: Path,
    ) -> None:
        self.project_root = project_root.resolve()
        documents = load_manifest(manifest_path.resolve())
        self.documents = {
            _document_label(document): document for document in documents
        }
        self.bm25 = BM25Index(bm25_index_path.resolve())
        self.tables = TableIndex(table_index_path.resolve())
        self.table_rows = TableRowIndex(table_row_index_path.resolve())
        self.visual = VisualIndex(
            visual_index_path.resolve(),
            project_root=self.project_root,
        )
        self.generation = configured_backend()

    @property
    def vehicle_choices(self) -> list[str]:
        return sorted(self.documents)

    def _document(self, label: str) -> CorpusDocument:
        if label not in self.documents:
            raise ValueError("Select one vehicle before searching")
        return self.documents[label]

    def answer_text(
        self,
        vehicle: str,
        question: str,
    ) -> tuple[str, list[list[Any]]]:
        try:
            document = self._document(vehicle)
            result = answer_question(
                self.bm25,
                question,
                _filters(document),
                retrieval_limit=10,
                max_evidence=3,
            )
            result = generate_or_fallback(
                result,
                question=question,
                backend=self.generation,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return f"Error: {exc}", []

        rows = [
            [
                item["citation_id"],
                f"{item['brand']} {item['model']} {item['year']}",
                ", ".join(str(page) for page in item["page_nos"]),
                " > ".join(item["section_path"]),
                item["chunk_type"],
                item["retrieval_rank"],
                f"{item['rerank_score']:.3f}",
            ]
            for item in result["evidence"]
        ]
        return result["answer"], rows

    def search_image(
        self,
        vehicle: str,
        image_path: str | None,
        text_hint: str,
    ) -> tuple[list[tuple[str, str]], list[list[Any]], str]:
        if not image_path:
            return [], [], "Upload an image before searching."
        try:
            document = self._document(vehicle)
            filters = _filters(document)
            if text_hint.strip():
                index: Any = VisualTextFusionIndex(
                    visual=self.visual,
                    bm25=self.bm25,
                )
                results = index.search(
                    Path(image_path),
                    query_text=text_hint,
                    filters=filters,
                    limit=5,
                )
                backend = "visual + text-hint RRF"
            else:
                results = self.visual.search(
                    Path(image_path),
                    filters=filters,
                    limit=5,
                )
                backend = "visual-only"
        except (OSError, RuntimeError, ValueError) as exc:
            return [], [], f"Error: {exc}"

        gallery: list[tuple[str, str]] = []
        rows: list[list[Any]] = []
        for item in results:
            asset = (self.project_root / item["asset_path"]).resolve()
            caption = (
                f"#{item['rank']} · p.{item['page_no']} · "
                f"{' > '.join(item['section_path'])}"
            )
            gallery.append((str(asset), caption))
            rows.append(
                [
                    item["rank"],
                    f"{item['brand']} {item['model']} {item['year']}",
                    item["page_no"],
                    " > ".join(item["section_path"]),
                    f"{float(item['score']):.4f}",
                    item["element_id"],
                ]
            )
        status = f"Backend: {backend}; returned {len(rows)} results."
        if self.generation is not None and results:
            evidence = [
                {
                    "citation_id": index + 1,
                    **item,
                    "source": (
                        f"{item['brand']} {item['model']} "
                        f"Owner's Manual ({item['year']}), "
                        f"{' > '.join(item['section_path'])}, "
                        f"physical PDF p.{item['page_no']}"
                    ),
                }
                for index, item in enumerate(results[:3])
            ]
            images = [
                LabeledImage("User query image", Path(image_path)),
                *[
                    LabeledImage(
                        f"Evidence image [{index + 1}]",
                        self.project_root / item["asset_path"],
                    )
                    for index, item in enumerate(results[:3])
                ],
            ]
            generated = generate_or_fallback(
                {
                    "status": "answered",
                    "reason": "visual_evidence_retrieved",
                    "answer": status,
                    "evidence": evidence,
                },
                question=text_hint or "Identify the uploaded image.",
                backend=self.generation,
                images=images,
            )
            status = (
                generated["answer"]
                + "\n\n"
                + status
                + "; generation="
                + generated["generation"]["status"]
            )
        return gallery, rows, status

    def search_table(
        self,
        vehicle: str,
        query: str,
    ) -> tuple[str, list[tuple[str, str]], list[list[Any]], str]:
        if not query.strip():
            return "", [], [], "Enter a table topic before searching."
        try:
            document = self._document(vehicle)
            filters = _filters(document)
            answer_result = answer_table_question(
                self.table_rows,
                query,
                filters,
                retrieval_limit=5,
            )
            crop_results = self.tables.search(
                query,
                filters=filters,
                limit=5,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return "", [], [], f"Error: {exc}"

        gallery: list[tuple[str, str]] = []
        shown_assets: set[str] = set()
        for item in answer_result["evidence"]:
            asset_path = str(item["asset_path"])
            if asset_path in shown_assets:
                continue
            shown_assets.add(asset_path)
            asset = (self.project_root / asset_path).resolve()
            caption = (
                f"Verified source [{item['citation_id']}] · "
                f"p.{item['page_no']} · "
                f"{' > '.join(item['section_path'])}"
            )
            gallery.append((str(asset), caption))
        for item in crop_results:
            asset_path = str(item["asset_path"])
            if asset_path in shown_assets:
                continue
            shown_assets.add(asset_path)
            asset = (self.project_root / asset_path).resolve()
            caption = (
                f"#{item['rank']} · p.{item['page_no']} · "
                f"{' > '.join(item['section_path'])}"
            )
            gallery.append((str(asset), caption))
            if len(gallery) >= 5:
                break
        rows = [
            [
                item["citation_id"],
                f"{item['brand']} {item['model']} {item['year']}",
                item["page_no"],
                " > ".join(item["section_path"]),
                " | ".join(
                    f"{header}: {value}"
                    for header, value in item["cells"].items()
                ),
                f"{item['score']:.4f}",
                item["transcription_method"],
            ]
            for item in answer_result["evidence"]
        ]
        if answer_result["status"] == "answered":
            status = (
                f"Exact value returned from {len(rows)} curated candidates; "
                f"{len(gallery)} source table crops shown."
            )
        else:
            status = (
                f"No strong curated row; showing {len(gallery)} table crops "
                "for source verification."
            )
        return (
            answer_result["answer"],
            gallery,
            rows,
            status,
        )


def create_demo(service: DemoService) -> Any:
    """Build the Gradio Blocks app without starting a server."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Run: "
            "python -m pip install -r requirements-demo.txt"
        ) from exc

    choices = service.vehicle_choices
    default_vehicle = choices[0] if choices else None
    with gr.Blocks(title="AutoManual-MM-RAG") as demo:
        gr.Markdown(
            "# AutoManual-MM-RAG\n"
            "Evidence-constrained answers and image-to-manual-page retrieval. "
            "Select the exact vehicle before searching."
        )
        with gr.Tab("Text question"):
            text_vehicle = gr.Dropdown(
                choices=choices,
                value=default_vehicle,
                label="Vehicle",
            )
            question = gr.Textbox(
                label="Question",
                placeholder="How do I adjust the steering wheel?",
                lines=2,
            )
            answer_button = gr.Button("Answer", variant="primary")
            answer = gr.Markdown()
            text_evidence = gr.Dataframe(
                headers=TEXT_COLUMNS,
                datatype=[
                    "number",
                    "str",
                    "str",
                    "str",
                    "str",
                    "number",
                    "str",
                ],
                interactive=False,
                label="Evidence Pack",
            )
            answer_button.click(
                service.answer_text,
                inputs=[text_vehicle, question],
                outputs=[answer, text_evidence],
            )

        with gr.Tab("Image search"):
            image_vehicle = gr.Dropdown(
                choices=choices,
                value=default_vehicle,
                label="Vehicle",
            )
            query_image = gr.Image(type="filepath", label="Query image")
            text_hint = gr.Textbox(
                label="Question / optional text hint",
                placeholder="What does this instrument-cluster image show?",
            )
            image_button = gr.Button("Search image", variant="primary")
            image_status = gr.Markdown()
            gallery = gr.Gallery(
                label="Matching manual images",
                columns=3,
                object_fit="contain",
            )
            visual_evidence = gr.Dataframe(
                headers=VISUAL_COLUMNS,
                datatype=["number", "str", "number", "str", "str", "str"],
                interactive=False,
                label="Visual evidence",
            )
            image_button.click(
                service.search_image,
                inputs=[image_vehicle, query_image, text_hint],
                outputs=[gallery, visual_evidence, image_status],
            )

        with gr.Tab("Table search"):
            table_vehicle = gr.Dropdown(
                choices=choices,
                value=default_vehicle,
                label="Vehicle",
            )
            table_query = gr.Textbox(
                label="Table topic",
                placeholder="roof rack load capacity",
            )
            table_button = gr.Button("Search tables", variant="primary")
            table_status = gr.Markdown()
            table_answer = gr.Markdown()
            table_gallery = gr.Gallery(
                label="Matching table crops",
                columns=2,
                height=520,
                object_fit="contain",
            )
            table_evidence = gr.Dataframe(
                headers=TABLE_COLUMNS,
                datatype=[
                    "number",
                    "str",
                    "number",
                    "str",
                    "str",
                    "str",
                    "str",
                ],
                interactive=False,
                label="Table evidence",
            )
            table_button.click(
                service.search_table,
                inputs=[table_vehicle, table_query],
                outputs=[
                    table_answer,
                    table_gallery,
                    table_evidence,
                    table_status,
                ],
            )
    return demo
