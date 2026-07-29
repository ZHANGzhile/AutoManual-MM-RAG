#!/usr/bin/env python3
"""Build citation-ready text chunks from normalized manual elements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.chunking import build_corpus_chunks
from automanual_rag.ingestion.mineru import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build section-aware text chunks for retrieval."
    )
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
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum characters for grouped text/step chunks (default: 1200).",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        help="Build only this doc_id; repeat to select more documents.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = load_manifest(args.manifest.resolve())
        if args.doc_id:
            requested = set(args.doc_id)
            known = {document.doc_id for document in documents}
            unknown = sorted(requested - known)
            if unknown:
                raise ValueError("Unknown doc_id(s): " + ", ".join(unknown))
            documents = [
                document
                for document in documents
                if document.doc_id in requested
            ]

        element_paths = {
            document.doc_id: (
                args.processed_root.resolve()
                / document.doc_id
                / "elements.jsonl"
            )
            for document in documents
        }
        missing = [str(path) for path in element_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Normalized element file(s) not found: " + ", ".join(missing)
            )

        summary = build_corpus_chunks(
            element_paths=element_paths,
            max_chars=args.max_chars,
        )
        summary_path = args.processed_root.resolve() / "chunk_summary.json"
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(summary_path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("doc_id\tchunks\ttext\tsteps\twarning\tcaution\tnote")
    for document in summary["documents"]:
        counts = document["chunk_type_counts"]
        print(
            "\t".join(
                str(value)
                for value in (
                    document["doc_id"],
                    document["chunks"],
                    counts["text"],
                    counts["steps"],
                    counts["warning"],
                    counts["caution"],
                    counts["note"],
                )
            )
        )
    totals = summary["totals"]
    print(
        "\t".join(
            str(value)
            for value in (
                "TOTAL",
                totals["chunks"],
                totals["text"],
                totals["steps"],
                totals["warning"],
                totals["caution"],
                totals["note"],
            )
        )
    )
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
