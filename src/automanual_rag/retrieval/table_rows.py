"""Search manually verified table rows with source-image integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping, Sequence


INDEX_SCHEMA_VERSION = 1
BACKEND_NAME = "sqlite_fts5_curated_table_rows"
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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = frozenset(
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


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class CuratedTableRow:
    row_id: str
    element_id: str
    doc_id: str
    brand: str
    model: str
    year: str
    region: str
    language: str
    manual_type: str
    page_no: int
    section_path: tuple[str, ...]
    cells: dict[str, str]
    aliases: tuple[str, ...]
    asset_path: str
    asset_sha256: str
    transcription_method: str
    verified_at: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CuratedTableRow":
        cells_value = value.get("cells")
        if not isinstance(cells_value, Mapping):
            raise ValueError("cells must be an object")
        row = cls(
            row_id=str(value.get("row_id", "")),
            element_id=str(value.get("element_id", "")),
            doc_id=str(value.get("doc_id", "")),
            brand=str(value.get("brand", "")),
            model=str(value.get("model", "")),
            year=str(value.get("year", "")),
            region=str(value.get("region", "")),
            language=str(value.get("language", "")),
            manual_type=str(value.get("manual_type", "")),
            page_no=value.get("page_no"),
            section_path=tuple(value.get("section_path") or ()),
            cells={
                str(key): str(cell_value)
                for key, cell_value in cells_value.items()
            },
            aliases=tuple(str(item) for item in value.get("aliases") or ()),
            asset_path=str(value.get("asset_path", "")),
            asset_sha256=str(value.get("asset_sha256", "")),
            transcription_method=str(
                value.get("transcription_method", "")
            ),
            verified_at=str(value.get("verified_at", "")),
        )
        row.validate()
        return row

    def validate(self) -> None:
        for name in (
            "row_id",
            "element_id",
            "doc_id",
            "brand",
            "model",
            "year",
            "region",
            "language",
            "manual_type",
            "asset_path",
            "transcription_method",
            "verified_at",
        ):
            _require_text(name, getattr(self, name))
        if (
            isinstance(self.page_no, bool)
            or not isinstance(self.page_no, int)
            or self.page_no < 1
        ):
            raise ValueError("page_no must be a positive integer")
        if not self.section_path or any(
            not isinstance(item, str) or not item.strip()
            for item in self.section_path
        ):
            raise ValueError("section_path must contain non-empty strings")
        if len(self.cells) < 2:
            raise ValueError("cells must contain at least two columns")
        for header, cell_value in self.cells.items():
            _require_text("cell header", header)
            _require_text("cell value", cell_value)
        if any(not alias.strip() for alias in self.aliases):
            raise ValueError("aliases must be non-empty strings")
        if not SHA256_RE.fullmatch(self.asset_sha256):
            raise ValueError("asset_sha256 must be lowercase SHA-256")
        if self.transcription_method != "manual_visual_verification":
            raise ValueError("unsupported transcription_method")

    @property
    def section_text(self) -> str:
        return " > ".join(self.section_path)

    @property
    def row_text(self) -> str:
        return " | ".join(
            f"{header}: {value}" for header, value in self.cells.items()
        )

    @property
    def searchable_text(self) -> str:
        return " ".join(
            (
                self.section_text,
                self.row_text,
                " ".join(self.aliases),
            )
        )


def _fts_query(query: str) -> str:
    all_tokens = [token.casefold() for token in TOKEN_RE.findall(query)]
    tokens = [
        token
        for token in all_tokens
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    ]
    if not tokens:
        tokens = all_tokens
    unique = list(dict.fromkeys(tokens))
    if not unique:
        raise ValueError("Query contains no searchable tokens")
    return " OR ".join(f'"{token}"' for token in unique)


def _metadata_text(row: CuratedTableRow) -> str:
    return " ".join(
        (
            row.brand,
            row.model,
            row.model.replace("-", " "),
            row.model.replace("-", ""),
            row.year,
            row.region,
            row.language,
            row.manual_type.replace("_", " "),
            row.doc_id.replace("_", " "),
        )
    )


def read_curated_rows(
    path: Path,
    *,
    project_root: Path | None = None,
    verify_assets: bool = False,
) -> list[CuratedTableRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Curated table rows not found: {path}")
    rows: list[CuratedTableRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = CuratedTableRow.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if verify_assets:
                if project_root is None:
                    raise ValueError(
                        "project_root is required when verify_assets is true"
                    )
                asset = (project_root.resolve() / row.asset_path).resolve()
                try:
                    asset.relative_to(project_root.resolve())
                except ValueError as exc:
                    raise ValueError(
                        f"Asset escapes project root: {row.asset_path}"
                    ) from exc
                if not asset.is_file():
                    raise FileNotFoundError(f"Table asset not found: {asset}")
                digest = hashlib.sha256(asset.read_bytes()).hexdigest()
                if digest != row.asset_sha256:
                    raise ValueError(
                        f"Table asset hash mismatch: {row.row_id}"
                    )
            rows.append(row)
    if not rows:
        raise ValueError("No curated table rows found")
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("Duplicate curated row_id")
    return rows


def build_table_row_index(
    *,
    index_path: Path,
    rows_path: Path,
    project_root: Path,
    verify_assets: bool = True,
) -> dict[str, Any]:
    rows = read_curated_rows(
        rows_path,
        project_root=project_root,
        verify_assets=verify_assets,
    )
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
            CREATE TABLE rows (
                row_id TEXT PRIMARY KEY,
                element_id TEXT NOT NULL,
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
                cells TEXT NOT NULL,
                aliases TEXT NOT NULL,
                row_text TEXT NOT NULL,
                asset_path TEXT NOT NULL,
                asset_sha256 TEXT NOT NULL,
                transcription_method TEXT NOT NULL,
                verified_at TEXT NOT NULL
            );
            CREATE INDEX row_doc_id_idx ON rows(doc_id);
            CREATE INDEX row_model_idx ON rows(model);
            CREATE INDEX row_year_idx ON rows(year);
            CREATE VIRTUAL TABLE rows_fts USING fts5(
                row_id UNINDEXED,
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
                    ("row_count", str(len(rows))),
                    (
                        "source_table_count",
                        str(len({row.element_id for row in rows})),
                    ),
                    (
                        "document_count",
                        str(len({row.doc_id for row in rows})),
                    ),
                    ("asset_verification", str(verify_assets).lower()),
                ),
            )
            for row in rows:
                connection.execute(
                    """
                    INSERT INTO rows(
                        row_id, element_id, doc_id, brand, model, year,
                        region, language, manual_type, page_no, section_path,
                        section_text, cells, aliases, row_text, asset_path,
                        asset_sha256, transcription_method, verified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.row_id,
                        row.element_id,
                        row.doc_id,
                        row.brand,
                        row.model,
                        row.year,
                        row.region,
                        row.language,
                        row.manual_type,
                        row.page_no,
                        json.dumps(row.section_path, ensure_ascii=False),
                        row.section_text,
                        json.dumps(
                            row.cells,
                            ensure_ascii=False,
                            sort_keys=False,
                        ),
                        json.dumps(row.aliases, ensure_ascii=False),
                        row.row_text,
                        row.asset_path,
                        row.asset_sha256,
                        row.transcription_method,
                        row.verified_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO rows_fts(row_id, content, section, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        row.row_id,
                        row.searchable_text,
                        row.section_text,
                        _metadata_text(row),
                    ),
                )
            connection.execute("INSERT INTO rows_fts(rows_fts) VALUES('optimize')")
    finally:
        connection.close()
    temporary.replace(index_path)
    return {
        "backend": BACKEND_NAME,
        "index_path": index_path.as_posix(),
        "row_count": len(rows),
        "source_table_count": len({row.element_id for row in rows}),
        "document_count": len({row.doc_id for row in rows}),
        "asset_verification": verify_assets,
    }


class TableRowIndex:
    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Table row index not found: {path}")
        self.path = path
        connection = sqlite3.connect(path)
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        finally:
            connection.close()
        if metadata.get("schema_version") != str(INDEX_SCHEMA_VERSION):
            raise ValueError("Unsupported table row index schema version")
        if metadata.get("backend") != BACKEND_NAME:
            raise ValueError("Unexpected table row index backend")
        self.metadata = metadata

    def count(self) -> int:
        return int(self.metadata["row_count"])

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
        conditions = ["rows_fts MATCH ?"]
        parameters: list[Any] = [_fts_query(query)]
        for field, value in filters.items():
            if value is None or not str(value).strip():
                continue
            conditions.append(f"LOWER(r.{field}) = LOWER(?)")
            parameters.append(str(value).strip())
        parameters.append(limit)
        sql = f"""
            SELECT
                r.*,
                bm25(rows_fts, 0.0, 1.0, 1.8, 0.5) AS bm25_raw
            FROM rows_fts
            JOIN rows AS r ON r.row_id = rows_fts.row_id
            WHERE {' AND '.join(conditions)}
            ORDER BY bm25_raw ASC, r.row_id ASC
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
                    "cells": json.loads(value["cells"]),
                    "aliases": json.loads(value["aliases"]),
                }
            )
            results.append(value)
        return results
