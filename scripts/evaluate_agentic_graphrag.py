#!/usr/bin/env python3
"""Compare Baseline RAG, GraphRAG, and Agentic GraphRAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.evaluation.agentic import (
    comparison_markdown,
    evaluate_agentic_comparison,
    load_agentic_questions,
)
from automanual_rag.graphrag import GraphRetriever
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.table import TableIndex
from automanual_rag.retrieval.table_rows import TableRowIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline Agentic GraphRAG comparison."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--graph-index", type=Path)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "eval"
        / "agentic_multihop_questions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "agentic_graphrag_comparison.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "reports"
        / "agentic_graphrag_evaluation.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = (
        args.data_root.resolve()
        if args.data_root is not None
        else project_root / "data"
    )
    graph_path = (
        args.graph_index.resolve()
        if args.graph_index is not None
        else project_root / "data" / "indexes" / "manual_graph.sqlite3"
    )
    indexes = data_root / "indexes"
    try:
        row_path = indexes / "table_rows.sqlite3"
        table_path = indexes / "tables.sqlite3"
        result = evaluate_agentic_comparison(
            questions=load_agentic_questions(args.questions.resolve()),
            text_index=BM25Index(indexes / "bm25.sqlite3"),
            graph_retriever=GraphRetriever(graph_path),
            table_row_index=(
                TableRowIndex(row_path) if row_path.is_file() else None
            ),
            table_index=(
                TableIndex(table_path) if table_path.is_file() else None
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    output = args.output.resolve()
    report = args.report.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report.write_text(comparison_markdown(result), encoding="utf-8")
    print(json.dumps(result["systems"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
