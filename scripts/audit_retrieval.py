#!/usr/bin/env python3
"""Audit Dense index integrity and BM25/Dense RRF provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.evaluation.retrieval import load_questions
from automanual_rag.ingestion.mineru import load_manifest
from automanual_rag.retrieval.bm25 import BM25Index, read_chunks
from automanual_rag.retrieval.dense import DenseIndex
from automanual_rag.retrieval.hybrid import HybridIndex
from automanual_rag.serialization import relativize_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local retrieval indexes.")
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
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "metrics" / "retrieval_audit.json",
    )
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    return parser.parse_args()


def _filter_violations(
    results: list[dict[str, Any]],
    filters: Mapping[str, str],
) -> int:
    return sum(
        any(
            str(result.get(field, "")).casefold()
            != str(expected).casefold()
            for field, expected in filters.items()
        )
        for result in results
    )


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
        import numpy as np

        documents = load_manifest(args.manifest.resolve())
        expected_chunks = []
        for document in documents:
            expected_chunks.extend(
                read_chunks(
                    args.processed_root.resolve()
                    / document.doc_id
                    / "chunks.jsonl"
                )
            )

        dense = DenseIndex(args.dense_index.resolve())
        dense_reloaded = DenseIndex(args.dense_index.resolve())
        bm25 = BM25Index(args.bm25_index.resolve())
        hybrid = HybridIndex(
            bm25=bm25,
            dense=dense,
            candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
        )
        questions = load_questions(args.questions.resolve())

        expected_metadata = [chunk.to_dict() for chunk in expected_chunks]
        actual_metadata = [chunk.to_dict() for chunk in dense._chunks]
        metadata_mismatches = sum(
            expected != actual
            for expected, actual in zip(expected_metadata, actual_metadata)
        ) + abs(len(expected_metadata) - len(actual_metadata))

        norms = np.linalg.norm(dense.embeddings, axis=1)
        nonfinite_values = int(
            np.size(dense.embeddings)
            - np.count_nonzero(np.isfinite(dense.embeddings))
        )
        zero_embeddings = int(np.count_nonzero(norms == 0))
        normalized_embedding_failures = int(
            np.count_nonzero(
                (norms != 0) & (~np.isclose(norms, 1.0, atol=1e-5))
            )
        )
        reload_equal = bool(
            dense.metadata == dense_reloaded.metadata
            and np.array_equal(dense.idf, dense_reloaded.idf)
            and np.array_equal(dense.components, dense_reloaded.components)
            and np.array_equal(dense.embeddings, dense_reloaded.embeddings)
            and [
                chunk.chunk_id for chunk in dense._chunks
            ]
            == [chunk.chunk_id for chunk in dense_reloaded._chunks]
        )

        hard_filter_violations = 0
        hard_filter_cases = 0
        for document in documents:
            filters = {
                "doc_id": document.doc_id,
                "year": document.year,
                "region": document.region,
            }
            results = dense.search(
                "How do I operate this vehicle?",
                filters=filters,
                limit=10,
            )
            hard_filter_cases += 1
            hard_filter_violations += _filter_violations(results, filters)

        source_rank_mismatches = 0
        source_score_mismatches = 0
        rrf_formula_mismatches = 0
        hybrid_filter_violations = 0
        audited_hybrid_results = 0
        for question in questions:
            filters = question["filters"]
            bm25_results = bm25.search(
                question["question"],
                filters=filters,
                limit=args.candidate_limit,
            )
            dense_results = dense.search(
                question["question"],
                filters=filters,
                limit=args.candidate_limit,
            )
            hybrid_results = hybrid.search(
                question["question"],
                filters=filters,
                limit=10,
            )
            bm25_by_id = {
                result["chunk_id"]: result for result in bm25_results
            }
            dense_by_id = {
                result["chunk_id"]: result for result in dense_results
            }
            hybrid_filter_violations += _filter_violations(
                hybrid_results,
                filters,
            )
            audited_hybrid_results += len(hybrid_results)

            for result in hybrid_results:
                expected_score = 0.0
                for name, weight, source in (
                    ("bm25", args.bm25_weight, bm25_by_id),
                    ("dense", args.dense_weight, dense_by_id),
                ):
                    source_result = source.get(result["chunk_id"])
                    expected_rank = (
                        source_result["rank"] if source_result else None
                    )
                    expected_backend_score = (
                        source_result["score"] if source_result else None
                    )
                    if result[f"{name}_rank"] != expected_rank:
                        source_rank_mismatches += 1
                    actual_backend_score = result[f"{name}_score"]
                    if expected_backend_score is None:
                        if actual_backend_score is not None:
                            source_score_mismatches += 1
                    elif actual_backend_score is None or not np.isclose(
                        actual_backend_score,
                        expected_backend_score,
                        atol=1e-10,
                    ):
                        source_score_mismatches += 1
                    if expected_rank is not None:
                        expected_score += weight / (
                            args.rrf_k + expected_rank
                        )
                if not np.isclose(
                    result["rrf_score"],
                    expected_score,
                    atol=1e-12,
                ):
                    rrf_formula_mismatches += 1

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "dense": {
                "backend": dense.metadata["backend"],
                "neural_embedding": dense.metadata["neural_embedding"],
                "index_path": dense.path.as_posix(),
                "index_size_bytes": dense.path.stat().st_size,
                "chunk_count": dense.count(),
                "document_count": dense.metadata["documents"],
                "dimensions": dense.metadata["dimensions"],
                "feature_count": dense.metadata["feature_count"],
                "embedding_shape": list(dense.embeddings.shape),
                "metadata_mismatches": metadata_mismatches,
                "nonfinite_vector_values": nonfinite_values,
                "zero_embeddings": zero_embeddings,
                "normalization_failures": normalized_embedding_failures,
                "repeat_load_equal": reload_equal,
                "hard_filter_cases": hard_filter_cases,
                "hard_filter_violations": hard_filter_violations,
            },
            "rrf": {
                "questions_audited": len(questions),
                "hybrid_results_audited": audited_hybrid_results,
                "candidate_limit": args.candidate_limit,
                "k": args.rrf_k,
                "bm25_weight": args.bm25_weight,
                "dense_weight": args.dense_weight,
                "same_filters_forwarded": True,
                "metadata_filter_violations": hybrid_filter_violations,
                "source_rank_mismatches": source_rank_mismatches,
                "source_score_mismatches": source_score_mismatches,
                "formula_mismatches": rrf_formula_mismatches,
            },
        }
        failures = [
            metadata_mismatches,
            nonfinite_values,
            zero_embeddings,
            normalized_embedding_failures,
            int(not reload_equal),
            hard_filter_violations,
            hybrid_filter_violations,
            source_rank_mismatches,
            source_score_mismatches,
            rrf_formula_mismatches,
            int(dense.count() != bm25.count()),
            int(dense.metadata["dimensions"] != 128),
        ]
        if any(failures):
            report["status"] = "fail"
        _write_json(args.output.resolve(), report)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Status: {report['status']}")
    print(
        "Dense: "
        f"{report['dense']['embedding_shape']} "
        f"metadata_mismatches={metadata_mismatches} "
        f"filter_violations={hard_filter_violations}"
    )
    print(
        "RRF: "
        f"audited={audited_hybrid_results} "
        f"rank_mismatches={source_rank_mismatches} "
        f"score_mismatches={source_score_mismatches} "
        f"formula_mismatches={rrf_formula_mismatches} "
        f"filter_violations={hybrid_filter_violations}"
    )
    print(f"Audit: {args.output.resolve()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
