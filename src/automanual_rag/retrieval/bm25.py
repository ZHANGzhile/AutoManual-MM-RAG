"""SQLite FTS5 BM25 retrieval with pre-retrieval metadata hard filters."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from automanual_rag.chunking import TextChunk


INDEX_SCHEMA_VERSION = 1
FILTER_FIELDS = frozenset(
    {
        "doc_id",
        "brand",
        "model",
        "year",
        "region",
        "language",
        "manual_type",
        "chunk_type",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
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


def _fts_query(query: str) -> str:
    all_tokens = [token.lower() for token in _TOKEN_RE.findall(query)]
    tokens = [
        token
        for token in all_tokens
        if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())
    ]
    if not tokens:
        tokens = all_tokens
    unique_tokens = list(dict.fromkeys(tokens))
    if not unique_tokens:
        raise ValueError("Query contains no searchable tokens")
    return " OR ".join(f'"{token}"' for token in unique_tokens)


def _metadata_text(chunk: TextChunk) -> str:
    model_variants = {
        chunk.model,
        chunk.model.replace("-", " "),
        chunk.model.replace("-", ""),
        chunk.model.replace(" ", ""),
    }
    if chunk.model == "Mustang Mach-E":
        model_variants.update({"Mach-E", "Mach E", "MachE"})
    if chunk.model == "F-150 Lightning":
        model_variants.update({"F150 Lightning", "F 150 Lightning", "Lightning"})
    values = [
        chunk.brand,
        *sorted(model_variants),
        chunk.year,
        chunk.region,
        chunk.language,
        chunk.manual_type.replace("_", " "),
        chunk.doc_id.replace("_", " "),
    ]
    return " ".join(value for value in values if value)


def read_chunks(path: Path) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                chunks.append(TextChunk.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return chunks


def build_bm25_index(
    *,
    index_path: Path,
    chunk_paths: Sequence[Path],
) -> dict[str, Any]:
    """Build an atomic SQLite FTS5 index from normalized chunk JSONL files."""

    chunks: list[TextChunk] = []
    for path in chunk_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Chunk file not found: {path}")
        chunks.extend(read_chunks(path))
    if not chunks:
        raise ValueError("No chunks found")

    chunk_ids = {chunk.chunk_id for chunk in chunks}
    if len(chunk_ids) != len(chunks):
        raise ValueError("Duplicate chunk_id detected")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                year TEXT NOT NULL,
                region TEXT NOT NULL,
                language TEXT NOT NULL,
                manual_type TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                page_nos TEXT NOT NULL,
                section_path TEXT NOT NULL,
                section_text TEXT NOT NULL,
                content TEXT NOT NULL,
                indexed_text TEXT NOT NULL,
                element_ids TEXT NOT NULL
            );
            CREATE INDEX chunks_doc_id_idx ON chunks(doc_id);
            CREATE INDEX chunks_model_idx ON chunks(model);
            CREATE INDEX chunks_year_idx ON chunks(year);
            CREATE INDEX chunks_region_idx ON chunks(region);
            CREATE INDEX chunks_language_idx ON chunks(language);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                content,
                section,
                metadata,
                tokenize='porter unicode61 remove_diacritics 2'
            );
            """
        )
    except sqlite3.OperationalError as exc:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "SQLite FTS5 is required but unavailable in this Python runtime"
        ) from exc

    try:
        with connection:
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(INDEX_SCHEMA_VERSION)),
                    ("chunk_count", str(len(chunks))),
                ),
            )
            for chunk in chunks:
                section_text = " > ".join(chunk.section_path)
                connection.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, doc_id, brand, model, year, region, language,
                        manual_type, chunk_type, page_start, page_end, page_nos,
                        section_path, section_text, content, indexed_text,
                        element_ids
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.brand,
                        chunk.model,
                        chunk.year,
                        chunk.region,
                        chunk.language,
                        chunk.manual_type,
                        chunk.chunk_type,
                        chunk.page_start,
                        chunk.page_end,
                        json.dumps(chunk.page_nos),
                        json.dumps(chunk.section_path, ensure_ascii=False),
                        section_text,
                        chunk.content,
                        chunk.indexed_text,
                        json.dumps(chunk.element_ids),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO chunks_fts(chunk_id, content, section, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.content,
                        section_text,
                        _metadata_text(chunk),
                    ),
                )
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        connection.commit()
    finally:
        connection.close()

    temporary.replace(index_path)
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "index_path": index_path.as_posix(),
        "chunk_count": len(chunks),
        "documents": len({chunk.doc_id for chunk in chunks}),
    }


class BM25Index:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        self.path = path

    def count(self) -> int:
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        finally:
            connection.close()
        return int(row[0])

    def element_membership(self) -> dict[str, dict[str, Any]]:
        """Return the indexed chunk location for every source element."""

        membership: dict[str, dict[str, Any]] = {}
        connection = sqlite3.connect(self.path)
        try:
            rows = connection.execute(
                "SELECT chunk_id, doc_id, page_nos, element_ids FROM chunks"
            )
            for chunk_id, doc_id, page_nos_json, element_ids_json in rows:
                page_nos = json.loads(page_nos_json)
                for element_id in json.loads(element_ids_json):
                    if element_id in membership:
                        raise ValueError(
                            f"Element belongs to multiple chunks: {element_id}"
                        )
                    membership[element_id] = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "page_nos": page_nos,
                    }
        finally:
            connection.close()
        return membership

    def search(
        self,
        query: str,
        *,
        filters: Mapping[str, str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        match_query = _fts_query(query)
        filters = dict(filters or {})
        unknown_filters = set(filters).difference(FILTER_FIELDS)
        if unknown_filters:
            raise ValueError(
                "Unsupported filter(s): " + ", ".join(sorted(unknown_filters))
            )

        conditions = ["chunks_fts MATCH ?"]
        parameters: list[Any] = [match_query]
        for field, value in filters.items():
            if value is None or str(value).strip() == "":
                continue
            conditions.append(f"LOWER(c.{field}) = LOWER(?)")
            parameters.append(str(value).strip())
        parameters.append(limit)

        sql = f"""
            SELECT
                c.*,
                bm25(chunks_fts, 0.0, 1.0, 1.8, 0.6) AS bm25_raw
            FROM chunks_fts
            JOIN chunks AS c ON c.chunk_id = chunks_fts.chunk_id
            WHERE {' AND '.join(conditions)}
            ORDER BY bm25_raw ASC, c.chunk_id ASC
            LIMIT ?
        """
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            value = dict(row)
            raw_score = float(value.pop("bm25_raw"))
            value.update(
                {
                    "rank": rank,
                    "score": -raw_score,
                    "page_nos": json.loads(value["page_nos"]),
                    "section_path": json.loads(value["section_path"]),
                    "element_ids": json.loads(value["element_ids"]),
                }
            )
            results.append(value)
        return results
