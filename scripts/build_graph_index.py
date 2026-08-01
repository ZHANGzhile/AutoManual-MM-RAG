#!/usr/bin/env python3
"""Build the deterministic automotive-manual graph index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.graphrag import build_manual_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the evidence-provenance GraphRAG index."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Code/output root; defaults to this repository.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help=(
            "Data root containing manifests/ and processed/. It may point "
            "to the main project's ignored local data."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Graph SQLite path; defaults to <project-root>/data/indexes/.",
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
    output = (
        args.output.resolve()
        if args.output is not None
        else project_root / "data" / "indexes" / "manual_graph.sqlite3"
    )
    try:
        summary = build_manual_graph(
            manifest_path=data_root / "manifests" / "corpus.csv",
            processed_root=data_root / "processed",
            output_path=output,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": output.as_posix(), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
