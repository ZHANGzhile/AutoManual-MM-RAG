"""Reciprocal-rank fusion for BM25 and dense retrieval."""

from __future__ import annotations

from typing import Any, Mapping

from automanual_rag.retrieval.bm25 import BM25Index
from automanual_rag.retrieval.dense import DenseIndex


class HybridIndex:
    def __init__(
        self,
        *,
        bm25: BM25Index,
        dense: DenseIndex,
        candidate_limit: int = 50,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        if not 1 <= candidate_limit <= 100:
            raise ValueError("candidate_limit must be from 1 to 100")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if bm25_weight <= 0 or dense_weight <= 0:
            raise ValueError("RRF weights must be positive")
        if bm25.count() != dense.count():
            raise ValueError("BM25 and dense indexes contain different chunk counts")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_limit = candidate_limit
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.path = {
            "bm25": bm25.path.as_posix(),
            "dense": dense.path.as_posix(),
        }

    def count(self) -> int:
        return self.bm25.count()

    def element_membership(self) -> dict[str, dict[str, Any]]:
        return self.bm25.element_membership()

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        candidate_limit = max(limit, self.candidate_limit)
        bm25_results = self.bm25.search(
            query,
            filters=filters,
            limit=candidate_limit,
        )
        dense_results = self.dense.search(
            query,
            filters=filters,
            limit=candidate_limit,
        )

        fused: dict[str, dict[str, Any]] = {}
        for backend, weight, results in (
            ("bm25", self.bm25_weight, bm25_results),
            ("dense", self.dense_weight, dense_results),
        ):
            for result in results:
                chunk_id = result["chunk_id"]
                value = fused.setdefault(
                    chunk_id,
                    {
                        **result,
                        "score": 0.0,
                        "bm25_rank": None,
                        "bm25_score": None,
                        "dense_rank": None,
                        "dense_score": None,
                    },
                )
                backend_rank = int(result["rank"])
                value["score"] += weight / (self.rrf_k + backend_rank)
                value[f"{backend}_rank"] = backend_rank
                value[f"{backend}_score"] = float(result["score"])

        ordered = sorted(
            fused.values(),
            key=lambda value: (-value["score"], value["chunk_id"]),
        )[:limit]
        for rank, value in enumerate(ordered, start=1):
            value["rank"] = rank
            value["rrf_score"] = value["score"]
        return ordered
