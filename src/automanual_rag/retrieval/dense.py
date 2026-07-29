"""Local hashed TF-IDF + randomized LSA dense retrieval.

This is an auditable, offline dense baseline rather than a neural embedding
model. It needs NumPy at build/query time and stores no pickle payloads.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from automanual_rag.chunking import TextChunk
from automanual_rag.retrieval.bm25 import FILTER_FIELDS, read_chunks


INDEX_SCHEMA_VERSION = 1
BACKEND_NAME = "hashed_tfidf_randomized_lsa"
TOKENIZER_VERSION = "word_unigram_bigram_v1"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_HASH_PERSON = b"amrag-lsa-v1"
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "should",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
        "you",
        "your",
    }
)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Dense retrieval requires NumPy. Install requirements-retrieval.txt "
            "or run with a Python environment that already provides NumPy."
        ) from exc
    return np


def _terms(text: str) -> list[str]:
    tokens = [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS
        and (len(token) > 1 or token.isdigit())
    ]
    return tokens + [
        f"{left}_{right}" for left, right in zip(tokens, tokens[1:])
    ]


def _feature(term: str, feature_count: int) -> tuple[int, int]:
    digest = hashlib.blake2b(
        term.encode("utf-8"),
        digest_size=8,
        person=_HASH_PERSON,
    ).digest()
    value = int.from_bytes(digest, "little")
    return value % feature_count, 1 if value & (1 << 63) else -1


def _hashed_counts(text: str, feature_count: int) -> dict[int, int]:
    values: Counter[int] = Counter()
    for term in _terms(text):
        bucket, sign = _feature(term, feature_count)
        values[bucket] += sign
    return {bucket: count for bucket, count in values.items() if count}


def _normalized_rows(matrix: Any) -> Any:
    np = _numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


def _build_tfidf(
    chunks: Sequence[TextChunk],
    *,
    feature_count: int,
) -> tuple[Any, Any]:
    np = _numpy()
    document_frequency = np.zeros(feature_count, dtype=np.int32)
    row_counts: list[dict[int, int]] = []
    for chunk in chunks:
        counts = _hashed_counts(chunk.indexed_text, feature_count)
        row_counts.append(counts)
        if counts:
            document_frequency[list(counts)] += 1

    idf = (
        np.log(
            (1.0 + len(chunks))
            / (1.0 + document_frequency.astype(np.float32))
        )
        + 1.0
    ).astype(np.float32)
    matrix = np.zeros((len(chunks), feature_count), dtype=np.float32)
    for row_index, counts in enumerate(row_counts):
        for bucket, signed_count in counts.items():
            magnitude = 1.0 + math.log(abs(signed_count))
            matrix[row_index, bucket] = (
                math.copysign(magnitude, signed_count) * idf[bucket]
            )
    return _normalized_rows(matrix), idf


def _randomized_lsa(
    matrix: Any,
    *,
    dimensions: int,
    oversamples: int,
    seed: int,
) -> tuple[Any, Any]:
    np = _numpy()
    max_rank = min(matrix.shape)
    actual_dimensions = min(dimensions, max_rank)
    sample_size = min(max_rank, actual_dimensions + oversamples)
    random = np.random.default_rng(seed)
    projection = random.standard_normal(
        (matrix.shape[1], sample_size)
    ).astype(np.float32)
    sample = matrix @ projection
    basis, _ = np.linalg.qr(sample, mode="reduced")
    compressed = basis.T @ matrix
    _, _, right_vectors = np.linalg.svd(compressed, full_matrices=False)
    components = right_vectors[:actual_dimensions].astype(np.float32)
    embeddings = (matrix @ components.T).astype(np.float32)
    return _normalized_rows(embeddings), components


def _json_bytes(value: Any) -> Any:
    np = _numpy()
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return np.frombuffer(payload, dtype=np.uint8)


def _chunks_bytes(chunks: Sequence[TextChunk]) -> tuple[Any, str]:
    np = _numpy()
    payload = (
        "\n".join(
            json.dumps(
                chunk.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for chunk in chunks
        )
        + "\n"
    ).encode("utf-8")
    return (
        np.frombuffer(payload, dtype=np.uint8),
        hashlib.sha256(payload).hexdigest(),
    )


def build_dense_index(
    *,
    index_path: Path,
    chunk_paths: Sequence[Path],
    feature_count: int = 2048,
    dimensions: int = 128,
    oversamples: int = 16,
    seed: int = 2026,
) -> dict[str, Any]:
    """Build an atomic, pickle-free local LSA dense index."""

    np = _numpy()
    if not 128 <= feature_count <= 65536:
        raise ValueError("feature_count must be from 128 to 65536")
    if not 1 <= dimensions <= feature_count:
        raise ValueError("dimensions must be from 1 to feature_count")
    if not 0 <= oversamples <= 256:
        raise ValueError("oversamples must be from 0 to 256")

    chunks: list[TextChunk] = []
    for path in chunk_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Chunk file not found: {path}")
        chunks.extend(read_chunks(path))
    if not chunks:
        raise ValueError("No chunks found")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk_id detected")

    tfidf, idf = _build_tfidf(chunks, feature_count=feature_count)
    embeddings, components = _randomized_lsa(
        tfidf,
        dimensions=dimensions,
        oversamples=oversamples,
        seed=seed,
    )
    del tfidf

    chunk_payload, chunk_sha256 = _chunks_bytes(chunks)
    metadata = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "backend": BACKEND_NAME,
        "tokenizer_version": TOKENIZER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(chunks),
        "documents": len({chunk.doc_id for chunk in chunks}),
        "feature_count": feature_count,
        "dimensions": int(embeddings.shape[1]),
        "requested_dimensions": dimensions,
        "oversamples": oversamples,
        "seed": seed,
        "chunk_payload_sha256": chunk_sha256,
        "numpy_version": np.__version__,
        "neural_embedding": False,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(index_path.name + ".tmp.npz")
    temporary.unlink(missing_ok=True)
    try:
        np.savez_compressed(
            temporary,
            metadata_json=_json_bytes(metadata),
            chunks_jsonl=chunk_payload,
            idf=idf,
            components=components,
            embeddings=embeddings,
        )
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        **metadata,
        "index_path": index_path.as_posix(),
        "index_size_bytes": index_path.stat().st_size,
    }


class DenseIndex:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Dense index not found: {path}")
        np = _numpy()
        self.path = path
        with np.load(path, allow_pickle=False) as archive:
            self.metadata = json.loads(
                archive["metadata_json"].tobytes().decode("utf-8")
            )
            chunk_payload = archive["chunks_jsonl"].tobytes()
            if (
                hashlib.sha256(chunk_payload).hexdigest()
                != self.metadata["chunk_payload_sha256"]
            ):
                raise ValueError("Dense index chunk payload checksum mismatch")
            self.idf = archive["idf"].astype(np.float32)
            self.components = archive["components"].astype(np.float32)
            self.embeddings = archive["embeddings"].astype(np.float32)

        if self.metadata.get("index_schema_version") != INDEX_SCHEMA_VERSION:
            raise ValueError("Unsupported dense index schema version")
        if self.metadata.get("backend") != BACKEND_NAME:
            raise ValueError("Unexpected dense index backend")
        self._chunks = self._decode_chunks(chunk_payload)
        expected_shape = (
            len(self._chunks),
            int(self.metadata["dimensions"]),
        )
        if self.embeddings.shape != expected_shape:
            raise ValueError(
                "Dense embedding shape mismatch: "
                f"{self.embeddings.shape} != {expected_shape}"
            )
        if self.components.shape != (
            int(self.metadata["dimensions"]),
            int(self.metadata["feature_count"]),
        ):
            raise ValueError("Dense LSA component shape mismatch")
        if self.idf.shape != (int(self.metadata["feature_count"]),):
            raise ValueError("Dense IDF shape mismatch")

    @staticmethod
    def _decode_chunks(payload: bytes) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for line_number, line in enumerate(
            payload.decode("utf-8").splitlines(),
            start=1,
        ):
            try:
                chunks.append(TextChunk.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Dense index chunk payload line {line_number}: {exc}"
                ) from exc
        return chunks

    def count(self) -> int:
        return len(self._chunks)

    def element_membership(self) -> dict[str, dict[str, Any]]:
        membership: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks:
            for element_id in chunk.element_ids:
                if element_id in membership:
                    raise ValueError(
                        f"Element belongs to multiple chunks: {element_id}"
                    )
                membership[element_id] = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page_nos": list(chunk.page_nos),
                }
        return membership

    def _query_embedding(self, query: str) -> Any:
        np = _numpy()
        counts = _hashed_counts(query, len(self.idf))
        if not counts:
            raise ValueError("Query contains no searchable tokens")
        vector = np.zeros(len(self.idf), dtype=np.float32)
        for bucket, signed_count in counts.items():
            magnitude = 1.0 + math.log(abs(signed_count))
            vector[bucket] = (
                math.copysign(magnitude, signed_count) * self.idf[bucket]
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Query produced a zero TF-IDF vector")
        vector /= norm
        embedding = vector @ self.components.T
        embedding_norm = float(np.linalg.norm(embedding))
        if embedding_norm == 0.0:
            raise ValueError("Query produced a zero LSA embedding")
        return embedding / embedding_norm

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        filters = dict(filters or {})
        unknown_filters = set(filters).difference(FILTER_FIELDS)
        if unknown_filters:
            raise ValueError(
                "Unsupported filter(s): " + ", ".join(sorted(unknown_filters))
            )
        normalized_filters = {
            field: str(value).strip().casefold()
            for field, value in filters.items()
            if value is not None and str(value).strip()
        }
        candidate_indexes = [
            index
            for index, chunk in enumerate(self._chunks)
            if all(
                str(getattr(chunk, field)).casefold() == expected
                for field, expected in normalized_filters.items()
            )
        ]
        if not candidate_indexes:
            return []

        query_embedding = self._query_embedding(query)
        scores = self.embeddings[candidate_indexes] @ query_embedding
        ordered = sorted(
            range(len(candidate_indexes)),
            key=lambda position: (
                -float(scores[position]),
                self._chunks[candidate_indexes[position]].chunk_id,
            ),
        )[:limit]

        results: list[dict[str, Any]] = []
        for rank, position in enumerate(ordered, start=1):
            chunk = self._chunks[candidate_indexes[position]]
            value = chunk.to_dict()
            value.update(
                {
                    "rank": rank,
                    "score": float(scores[position]),
                    "dense_score": float(scores[position]),
                }
            )
            results.append(value)
        return results
