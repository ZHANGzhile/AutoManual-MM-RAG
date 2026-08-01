#!/usr/bin/env python3
"""Run one Agentic GraphRAG question with a complete execution trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.agentic import AgenticWorkflow
from automanual_rag.generation import configured_backend
from automanual_rag.graphrag import GraphRetriever
from automanual_rag.ingestion.mineru import load_manifest
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.table import TableIndex
from automanual_rag.retrieval.table_rows import TableRowIndex
from automanual_rag.retrieval.visual import VisualIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer through the explicit Agentic GraphRAG state graph."
    )
    parser.add_argument("query")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Read-only data root; may point to the main project data.",
    )
    parser.add_argument("--graph-index", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--doc-id")
    parser.add_argument("--brand", default="Ford")
    parser.add_argument("--model", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--region", default="North America")
    parser.add_argument("--language", default="en")
    parser.add_argument("--manual-type", default="owner_manual")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        project_root = args.project_root.resolve()
        data_root = (
            args.data_root.resolve()
            if args.data_root is not None
            else project_root / "data"
        )
        requested = {
            key: value
            for key, value in {
                "doc_id": args.doc_id,
                "brand": args.brand,
                "model": args.model,
                "year": args.year,
                "region": args.region,
                "language": args.language,
                "manual_type": args.manual_type,
            }.items()
            if value
        }
        matches = [
            document
            for document in load_manifest(
                data_root / "manifests" / "corpus.csv"
            )
            if all(
                str(getattr(document, field)).casefold()
                == str(value).casefold()
                for field, value in requested.items()
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "Vehicle metadata does not identify exactly one corpus manual"
            )
        filters = matches[0].element_metadata()
        indexes = data_root / "indexes"
        graph_index = (
            args.graph_index.resolve()
            if args.graph_index is not None
            else project_root
            / "data"
            / "indexes"
            / "manual_graph.sqlite3"
        )
        visual_path = indexes / "visual_traditional.npz"
        row_path = indexes / "table_rows.sqlite3"
        table_path = indexes / "tables.sqlite3"
        workflow = AgenticWorkflow(
            text_index=BM25Index(indexes / "bm25.sqlite3"),
            graph_retriever=GraphRetriever(graph_index),
            visual_index=(
                VisualIndex(visual_path, project_root=data_root.parent)
                if visual_path.is_file() and args.image is not None
                else None
            ),
            table_row_index=(
                TableRowIndex(row_path) if row_path.is_file() else None
            ),
            table_index=(
                TableIndex(table_path) if table_path.is_file() else None
            ),
            generation_backend=configured_backend(),
            asset_root=data_root.parent,
        )
        result = workflow.run(
            query=args.query,
            filters=filters,
            image_path=(
                str(args.image.resolve()) if args.image is not None else None
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["answer"])
        print(
            "\nTrace: "
            + " -> ".join(event["node"] for event in result["trace"])
        )
        print(
            f"Latency: {result['latency_ms']:.1f} ms; "
            f"retry={result['retry_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
