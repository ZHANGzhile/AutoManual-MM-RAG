#!/usr/bin/env python3
"""Convert MinerU output for the corpus into normalized JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.ingestion.mineru import import_corpus, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import native MinerU content lists into normalized JSONL."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "corpus.csv",
        help="Corpus CSV containing authoritative document metadata.",
    )
    parser.add_argument(
        "--parsed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "parsed",
        help="Root containing one MinerU output directory per doc_id.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Destination for per-document elements.jsonl files.",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        help="Import only this doc_id. Repeat to select multiple documents.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a failure status if pages, assets, or source fields are invalid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        documents = load_manifest(args.manifest.resolve())
        if args.doc_id:
            selected = set(args.doc_id)
            known = {document.doc_id for document in documents}
            unknown = sorted(selected - known)
            if unknown:
                raise ValueError("Unknown doc_id(s): " + ", ".join(unknown))
            documents = [
                document
                for document in documents
                if document.doc_id in selected
            ]

        summary = import_corpus(
            project_root=PROJECT_ROOT,
            parsed_root=args.parsed_root.resolve(),
            output_root=args.output_dir.resolve(),
            documents=documents,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    header = (
        "doc_id",
        "text",
        "image",
        "table",
        "missing_page_no",
        "invalid_asset_paths",
        "anomalies",
    )
    print("\t".join(header))
    for document in summary["documents"]:
        counts = document["element_counts"]
        print(
            "\t".join(
                str(value)
                for value in (
                    document["doc_id"],
                    counts["text"],
                    counts["image"],
                    counts["table"],
                    document["missing_page_no"],
                    document["invalid_asset_paths"],
                    document["anomaly_count"],
                )
            )
        )

    totals = summary["totals"]
    print(
        "\t".join(
            str(value)
            for value in (
                "TOTAL",
                totals["text"],
                totals["image"],
                totals["table"],
                totals["missing_page_no"],
                totals["invalid_asset_paths"],
                totals["anomalies"],
            )
        )
    )
    print(f"Summary: {args.output_dir.resolve() / 'import_summary.json'}")

    if args.strict and (
        totals["missing_page_no"]
        or totals["invalid_asset_paths"]
        or totals["anomalies"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
