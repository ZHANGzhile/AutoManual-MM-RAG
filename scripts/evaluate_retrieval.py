#!/usr/bin/env python3
"""Evaluate BM25, local Dense, and RRF hybrid on one Gold Evidence set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.evaluation.retrieval import (
    evaluate_bm25,
    evaluate_dense,
    evaluate_hybrid,
    load_questions,
)
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.dense import DenseIndex
from automanual_rag.retrieval.hybrid import HybridIndex
from automanual_rag.serialization import relativize_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare retrieval baselines.")
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.sqlite3",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "dense_lsa.npz",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "questions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    return parser.parse_args()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            relativize_project_paths(value, PROJECT_ROOT),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    try:
        questions = load_questions(args.questions.resolve())
        bm25 = BM25Index(args.bm25_index.resolve())
        dense = DenseIndex(args.dense_index.resolve())
        hybrid = HybridIndex(
            bm25=bm25,
            dense=dense,
            candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
        )
        results = {
            "bm25": evaluate_bm25(
                index=bm25,
                questions=questions,
                limit=args.limit,
            ),
            "dense": evaluate_dense(
                index=dense,
                questions=questions,
                limit=args.limit,
            ),
            "hybrid": evaluate_hybrid(
                index=hybrid,
                questions=questions,
                limit=args.limit,
            ),
        }
        output_dir = args.output_dir.resolve()
        _write_json(output_dir / "bm25_baseline.json", results["bm25"])
        _write_json(output_dir / "dense_lsa_baseline.json", results["dense"])
        _write_json(output_dir / "hybrid_rrf.json", results["hybrid"])
        comparison = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "questions": args.questions.resolve().as_posix(),
            "answerable_questions": results["bm25"]["answerable_questions"],
            "rrf": {
                "candidate_limit": args.candidate_limit,
                "k": args.rrf_k,
                "bm25_weight": args.bm25_weight,
                "dense_weight": args.dense_weight,
            },
            "metrics": {
                name: result["metrics"] for name, result in results.items()
            },
        }
        _write_json(output_dir / "retrieval_comparison.json", comparison)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("backend\trecall@5\trecall@10\tmrr@10\tfilter_violations")
    for name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{name}\t{metrics['recall_at_5']:.4f}\t"
            f"{metrics['recall_at_10']:.4f}\t"
            f"{metrics['mrr_at_10']:.4f}\t"
            f"{metrics['metadata_filter_violations']}"
        )
    print(f"Metrics: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
