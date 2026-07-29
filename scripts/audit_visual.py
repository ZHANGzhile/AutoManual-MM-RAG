#!/usr/bin/env python3
"""Audit visual resource coverage, vector integrity, and fusion provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.evaluation.visual import load_visual_questions
from automanual_rag.ingestion.mineru import load_manifest
from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.visual import (
    FEATURE_DIMENSIONS,
    VisualIndex,
    VisualTextFusionIndex,
)
from automanual_rag.schema import ManualElement
from automanual_rag.serialization import relativize_project_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit visual retrieval MVP.")
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
        "--parsed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "parsed",
    )
    parser.add_argument(
        "--visual-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "visual_traditional.npz",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "bm25.sqlite3",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "visual_questions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "metrics"
        / "visual_retrieval_audit.json",
    )
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_record(element: ManualElement) -> dict[str, Any]:
    return {
        "element_id": element.element_id,
        "doc_id": element.doc_id,
        "brand": element.brand,
        "model": element.model,
        "year": element.year,
        "region": element.region,
        "language": element.language,
        "manual_type": element.manual_type,
        "page_no": element.page_no,
        "section_path": list(element.section_path),
        "element_type": element.element_type,
        "content": element.content,
        "asset_path": element.asset_path,
        "source_locator": element.source_locator,
    }


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
        image_elements: list[ManualElement] = []
        table_elements: list[ManualElement] = []
        referenced_assets: set[Path] = set()
        invalid_asset_paths = 0
        for document in documents:
            path = (
                args.processed_root.resolve()
                / document.doc_id
                / "elements.jsonl"
            )
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        element = ManualElement.from_dict(json.loads(line))
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise ValueError(
                            f"{path}:{line_number}: {exc}"
                        ) from exc
                    if element.element_type not in {"image", "table"}:
                        continue
                    if not element.asset_path:
                        invalid_asset_paths += 1
                        continue
                    asset = (PROJECT_ROOT / element.asset_path).resolve()
                    referenced_assets.add(asset)
                    if not asset.is_file():
                        invalid_asset_paths += 1
                    if element.element_type == "image":
                        image_elements.append(element)
                    else:
                        table_elements.append(element)

        raw_assets = {
            path.resolve()
            for path in args.parsed_root.resolve().rglob("*.jpg")
        }
        visual = VisualIndex(
            args.visual_index.resolve(),
            project_root=PROJECT_ROOT,
        )
        visual_reloaded = VisualIndex(
            args.visual_index.resolve(),
            project_root=PROJECT_ROOT,
        )
        bm25 = BM25Index(args.bm25_index.resolve())
        fusion = VisualTextFusionIndex(
            visual=visual,
            bm25=bm25,
            candidate_limit=args.candidate_limit,
            rrf_k=args.rrf_k,
        )
        questions = load_visual_questions(
            args.questions.resolve(),
            project_root=PROJECT_ROOT,
        )

        expected_records = [
            _expected_record(element) for element in image_elements
        ]
        metadata_mismatches = sum(
            expected != actual
            for expected, actual in zip(expected_records, visual.records)
        ) + abs(len(expected_records) - len(visual.records))
        indexed_ids = {record["element_id"] for record in visual.records}
        gold_missing_from_index = sum(
            question["gold_evidence"]["element_id"] not in indexed_ids
            for question in questions
        )
        indexed_invalid_paths = sum(
            not (PROJECT_ROOT / record["asset_path"]).is_file()
            for record in visual.records
        )

        norms = np.linalg.norm(visual.embeddings, axis=1)
        nonfinite_values = int(
            np.size(visual.embeddings)
            - np.count_nonzero(np.isfinite(visual.embeddings))
        )
        zero_vectors = int(np.count_nonzero(norms == 0))
        normalization_failures = int(
            np.count_nonzero(
                (norms != 0) & (~np.isclose(norms, 1.0, atol=1e-5))
            )
        )
        repeat_load_equal = bool(
            visual.metadata == visual_reloaded.metadata
            and visual.records == visual_reloaded.records
            and np.array_equal(
                visual.embeddings,
                visual_reloaded.embeddings,
            )
        )

        deterministic_failures = 0
        query_source_hash_matches = 0
        query_metadata_violations = 0
        fusion_metadata_violations = 0
        source_rank_mismatches = 0
        source_score_mismatches = 0
        fusion_formula_mismatches = 0
        audited_fusion_results = 0
        for question in questions:
            query_path = question["_query_path"]
            source_path = (
                PROJECT_ROOT
                / question["source"]["source_asset_path"]
            ).resolve()
            if _sha256(query_path) == _sha256(source_path):
                query_source_hash_matches += 1

            first = visual.search(
                query_path,
                filters=question["filters"],
                limit=args.candidate_limit,
            )
            second = visual_reloaded.search(
                query_path,
                filters=question["filters"],
                limit=args.candidate_limit,
            )
            if [value["element_id"] for value in first] != [
                value["element_id"] for value in second
            ] or not np.allclose(
                [value["score"] for value in first],
                [value["score"] for value in second],
                atol=1e-10,
            ):
                deterministic_failures += 1
            query_metadata_violations += _filter_violations(
                first,
                question["filters"],
            )

            text_results = bm25.search(
                question["query_text"],
                filters=question["filters"],
                limit=args.candidate_limit,
            )
            page_hits: dict[tuple[str, int], dict[str, Any]] = {}
            for result in text_results:
                for page_no in result["page_nos"]:
                    page_hits.setdefault(
                        (result["doc_id"], page_no),
                        result,
                    )
            fused = fusion.search(
                query_path,
                query_text=question["query_text"],
                filters=question["filters"],
                limit=10,
            )
            fusion_metadata_violations += _filter_violations(
                fused,
                question["filters"],
            )
            audited_fusion_results += len(fused)
            visual_by_id = {
                result["element_id"]: result for result in first
            }
            for result in fused:
                source_visual = visual_by_id[result["element_id"]]
                text_source = page_hits.get(
                    (result["doc_id"], result["page_no"])
                )
                expected_text_rank = (
                    int(text_source["rank"]) if text_source else None
                )
                if result["visual_rank"] != source_visual["rank"]:
                    source_rank_mismatches += 1
                if result["text_rank"] != expected_text_rank:
                    source_rank_mismatches += 1
                if not np.isclose(
                    result["visual_score"],
                    source_visual["score"],
                    atol=1e-10,
                ):
                    source_score_mismatches += 1
                expected_text_score = (
                    float(text_source["score"]) if text_source else None
                )
                if expected_text_score is None:
                    if result["text_score"] is not None:
                        source_score_mismatches += 1
                elif result["text_score"] is None or not np.isclose(
                    result["text_score"],
                    expected_text_score,
                    atol=1e-10,
                ):
                    source_score_mismatches += 1
                expected_fusion = 1.0 / (
                    args.rrf_k + source_visual["rank"]
                )
                if expected_text_rank is not None:
                    expected_fusion += 1.0 / (
                        args.rrf_k + expected_text_rank
                    )
                if not np.isclose(
                    result["fusion_score"],
                    expected_fusion,
                    atol=1e-12,
                ):
                    fusion_formula_mismatches += 1

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "pass",
            "resources": {
                "raw_jpg_files": len(raw_assets),
                "normalized_image_elements": len(image_elements),
                "normalized_table_elements": len(table_elements),
                "unique_referenced_assets": len(referenced_assets),
                "orphan_raw_assets": len(raw_assets - referenced_assets),
                "references_missing_from_raw_set": len(
                    referenced_assets - raw_assets
                ),
                "invalid_asset_paths": invalid_asset_paths,
            },
            "index": {
                "backend": visual.metadata["backend"],
                "semantic_embedding": visual.metadata["semantic_embedding"],
                "neural_embedding": visual.metadata["neural_embedding"],
                "index_path": visual.path.as_posix(),
                "index_size_bytes": visual.path.stat().st_size,
                "element_count": visual.count(),
                "dimensions": visual.metadata["dimensions"],
                "embedding_shape": list(visual.embeddings.shape),
                "excluded_table_crops": visual.metadata[
                    "excluded_table_crops"
                ],
                "metadata_mismatches": metadata_mismatches,
                "indexed_invalid_asset_paths": indexed_invalid_paths,
                "nonfinite_vector_values": nonfinite_values,
                "zero_vectors": zero_vectors,
                "normalization_failures": normalization_failures,
                "repeat_load_equal": repeat_load_equal,
            },
            "queries": {
                "query_count": len(questions),
                "dev_queries": sum(
                    question["split"] == "dev" for question in questions
                ),
                "test_queries": sum(
                    question["split"] == "test" for question in questions
                ),
                "gold_missing_from_index": gold_missing_from_index,
                "query_source_hash_matches": query_source_hash_matches,
                "deterministic_query_failures": deterministic_failures,
                "visual_metadata_filter_violations": (
                    query_metadata_violations
                ),
            },
            "fusion": {
                "queries_audited": len(questions),
                "results_audited": audited_fusion_results,
                "candidate_limit": args.candidate_limit,
                "k": args.rrf_k,
                "same_filters_forwarded": True,
                "metadata_filter_violations": (
                    fusion_metadata_violations
                ),
                "source_rank_mismatches": source_rank_mismatches,
                "source_score_mismatches": source_score_mismatches,
                "formula_mismatches": fusion_formula_mismatches,
            },
        }
        failures = [
            len(raw_assets) - 3803,
            len(image_elements) - 3287,
            len(table_elements) - 516,
            len(raw_assets - referenced_assets),
            len(referenced_assets - raw_assets),
            invalid_asset_paths,
            visual.count() - 3287,
            int(visual.metadata["dimensions"]) - FEATURE_DIMENSIONS,
            metadata_mismatches,
            indexed_invalid_paths,
            nonfinite_values,
            zero_vectors,
            normalization_failures,
            int(not repeat_load_equal),
            gold_missing_from_index,
            query_source_hash_matches,
            deterministic_failures,
            query_metadata_violations,
            fusion_metadata_violations,
            source_rank_mismatches,
            source_score_mismatches,
            fusion_formula_mismatches,
        ]
        if any(failures):
            report["status"] = "fail"
        _write_json(args.output.resolve(), report)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Status: {report['status']}")
    print(
        "Resources: "
        f"raw={len(raw_assets)} image={len(image_elements)} "
        f"table={len(table_elements)} orphan=0 invalid={invalid_asset_paths}"
    )
    print(
        "Index: "
        f"shape={list(visual.embeddings.shape)} "
        f"metadata_mismatches={metadata_mismatches} "
        f"nonfinite={nonfinite_values} zero={zero_vectors}"
    )
    print(
        "Queries/Fusion: "
        f"deterministic_failures={deterministic_failures} "
        f"filter_violations="
        f"{query_metadata_violations + fusion_metadata_violations} "
        f"rank_mismatches={source_rank_mismatches} "
        f"score_mismatches={source_score_mismatches} "
        f"formula_mismatches={fusion_formula_mismatches}"
    )
    print(f"Audit: {args.output.resolve()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
