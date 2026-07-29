#!/usr/bin/env python3
"""Build the offline hashed TF-IDF + randomized LSA dense index."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.ingestion.mineru import load_manifest
from automanual_rag.retrieval.dense import build_dense_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the local Dense index.")
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
        default=PROJECT_ROOT / "data" / "indexes" / "dense_lsa.npz",
    )
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--oversamples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = load_manifest(args.manifest.resolve())
        chunk_paths = [
            args.processed_root.resolve() / document.doc_id / "chunks.jsonl"
            for document in documents
        ]
        summary = build_dense_index(
            index_path=args.index.resolve(),
            chunk_paths=chunk_paths,
            feature_count=args.features,
            dimensions=args.dimensions,
            oversamples=args.oversamples,
            seed=args.seed,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Built {summary['chunk_count']} chunks from "
        f"{summary['documents']} documents"
    )
    print(
        f"Backend: {summary['backend']} "
        f"({summary['dimensions']} dimensions, neural=False)"
    )
    print(f"Index: {args.index.resolve()}")
    print(f"Size: {summary['index_size_bytes'] / (1024 * 1024):.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
