"""SQLite FTS5 retrieval over traceable table-crop evidence."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping, Sequence

from automanual_rag.schema import ManualElement


INDEX_SCHEMA_VERSION = 1
BACKEND_NAME = "sqlite_fts5_table_context"
FILTER_FIELDS = frozenset(
    {
        "doc_id",
        "brand",
        "model",
        "year",
        "region",
        "language",
        "manual_type",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
)


def _fts_query(query: str) -> str:
    all_tokens = [token.casefold() for token in _TOKEN_RE.findall(query)]
    tokens = [
        token
        for token in all_tokens
        if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())
    ]
    if not tokens:
        tokens = all_tokens
    unique = list(dict.fromkeys(tokens))
    if not unique:
        raise ValueError("Query contains no searchable tokens")
    return " OR ".join(f'"{token}"' for token in unique)


def _metadata_text(element: ManualElement) -> str:
    model_variants = {
        element.model,
        element.model.replace("-", " "),
        element.model.replace("-", ""),
        element.model.replace(" ", ""),
    }
    return " ".join(
        (
            element.brand,
            *sorted(model_variants),
            element.year,
            element.region,
            element.language,
            element.manual_type.replace("_", " "),
            element.doc_id.replace("_", " "),
        )
    )


def read_table_elements(paths: Sequence[Path]) -> list[ManualElement]:
    tables: list[ManualElement] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Element file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    element = ManualElement.from_dict(json.loads(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if element.element_type == "table":
                    tables.append(element)
    return tables


def build_table_index(
    *,
    index_path: Path,
    element_paths: Sequence[Path],
) -> dict[str, Any]:
    """Build an atomic FTS index over table context and source metadata."""
    tables = read_table_elements(element_paths)
    if not tables:
        raise ValueError("No table elements found")
    if len({table.element_id for table in tables}) != len(tables):
        raise ValueError("Duplicate table element_id detected")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
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
            CREATE TABLE table_elements (
                element_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                year TEXT NOT NULL,
                region TEXT NOT NULL,
                language TEXT NOT NULL,
                manual_type TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                section_path TEXT NOT NULL,
                section_text TEXT NOT NULL,
                content TEXT NOT NULL,
                asset_path TEXT NOT NULL,
                structured_content_available INTEGER NOT NULL,
                source_locator TEXT NOT NULL
            );
            CREATE INDEX table_doc_id_idx ON table_elements(doc_id);
            CREATE INDEX table_model_idx ON table_elements(model);
            CREATE INDEX table_year_idx ON table_elements(year);
            CREATE VIRTUAL TABLE table_fts USING fts5(
                element_id UNINDEXED,
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
                    ("backend", BACKEND_NAME),
                    ("element_count", str(len(tables))),
                    (
                        "structured_element_count",
                        str(
                            sum(
                                bool(
                                    table.source_locator.get(
                                        "structured_table_content_available"
                                    )
                                )
                                for table in tables
                            )
                        ),
                    ),
                ),
            )
            for table in tables:
                if table.page_no is None or not table.asset_path:
                    raise ValueError(
                        f"Table lacks page or asset: {table.element_id}"
                    )
                section_text = " > ".join(table.section_path)
                structured = bool(
                    table.source_locator.get(
                        "structured_table_content_available"
                    )
                )
                connection.execute(
                    """
                    INSERT INTO table_elements(
                        element_id, doc_id, brand, model, year, region,
                        language, manual_type, page_no, section_path,
                        section_text, content, asset_path,
                        structured_content_available, source_locator
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        table.element_id,
                        table.doc_id,
                        table.brand,
                        table.model,
                        table.year,
                        table.region,
                        table.language,
                        table.manual_type,
                        table.page_no,
                        json.dumps(table.section_path, ensure_ascii=False),
                        section_text,
                        table.content,
                        table.asset_path,
                        int(structured),
                        json.dumps(
                            table.source_locator,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO table_fts(element_id, content, section, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        table.element_id,
                        table.content,
                        section_text,
                        _metadata_text(table),
                    ),
                )
            connection.execute("INSERT INTO table_fts(table_fts) VALUES('optimize')")
    finally:
        connection.close()

    temporary.replace(index_path)
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "backend": BACKEND_NAME,
        "index_path": index_path.as_posix(),
        "table_count": len(tables),
        "document_count": len({table.doc_id for table in tables}),
        "structured_table_count": sum(
            bool(
                table.source_locator.get(
                    "structured_table_content_available"
                )
            )
            for table in tables
        ),
    }


class TableIndex:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Table index not found: {path}")
        self.path = path
        connection = sqlite3.connect(path)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        finally:
            connection.close()
        if metadata.get("schema_version") != str(INDEX_SCHEMA_VERSION):
            raise ValueError("Unsupported table index schema version")
        if metadata.get("backend") != BACKEND_NAME:
            raise ValueError("Unexpected table index backend")
        self.metadata = metadata

    def count(self) -> int:
        return int(self.metadata["element_count"])

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
        unknown = set(filters).difference(FILTER_FIELDS)
        if unknown:
            raise ValueError(
                "Unsupported filter(s): " + ", ".join(sorted(unknown))
            )

        conditions = ["table_fts MATCH ?"]
        parameters: list[Any] = [_fts_query(query)]
        for field, value in filters.items():
            if value is None or not str(value).strip():
                continue
            conditions.append(f"LOWER(t.{field}) = LOWER(?)")
            parameters.append(str(value).strip())
        parameters.append(limit)
        sql = f"""
            SELECT
                t.*,
                bm25(table_fts, 0.0, 1.0, 2.0, 0.5) AS bm25_raw
            FROM table_fts
            JOIN table_elements AS t
              ON t.element_id = table_fts.element_id
            WHERE {' AND '.join(conditions)}
            ORDER BY bm25_raw ASC, t.element_id ASC
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
                    "section_path": json.loads(value["section_path"]),
                    "structured_content_available": bool(
                        value["structured_content_available"]
                    ),
                    "source_locator": json.loads(value["source_locator"]),
                }
            )
            results.append(value)
        return results
