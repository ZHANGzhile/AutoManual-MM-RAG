#!/usr/bin/env python3
"""Build the traditional visual feature index over normalized image elements."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.ingestion.mineru import load_manifest
from automanual_rag.retrieval.visual import build_visual_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the visual MVP index.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "corpus.csv",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "visual_traditional.npz",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = load_manifest(args.manifest.resolve())
        element_paths = [
            args.processed_root.resolve() / document.doc_id / "elements.jsonl"
            for document in documents
        ]
        summary = build_visual_index(
            project_root=PROJECT_ROOT,
            index_path=args.index.resolve(),
            element_paths=element_paths,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Built {summary['element_count']} image elements from "
        f"{summary['documents']} documents"
    )
    print(
        f"Backend: {summary['backend']} "
        f"({summary['dimensions']} dimensions, semantic=False, neural=False)"
    )
    print(f"Excluded table crops: {summary['excluded_table_crops']}")
    print(f"Index: {args.index.resolve()}")
    print(f"Size: {summary['index_size_bytes'] / (1024 * 1024):.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
