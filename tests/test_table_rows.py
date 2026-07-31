from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.retrieval.table_rows import (
    TableRowIndex,
    build_table_row_index,
)


def row(asset_path: str, digest: str) -> dict:
    return {
        "row_id": "maverick_fuel",
        "element_id": "maverick:table:1",
        "doc_id": "ford_maverick_2026_na_en",
        "brand": "Ford",
        "model": "Maverick",
        "year": "2026",
        "region": "North America",
        "language": "en",
        "manual_type": "owner_manual",
        "page_no": 169,
        "section_path": ["Fuel and Refueling", "Fuel Tank Capacity"],
        "cells": {"Variant": "All.", "Quantity": "16.5 gal (62.4 L)"},
        "aliases": ["non-hybrid fuel tank capacity"],
        "asset_path": asset_path,
        "asset_sha256": digest,
        "transcription_method": "manual_visual_verification",
        "verified_at": "2026-07-31",
    }


class TableRowIndexTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        asset = root / "table.jpg"
        asset.write_bytes(b"verified table image")
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        rows_path = root / "rows.jsonl"
        rows_path.write_text(
            json.dumps(row("table.jpg", digest)) + "\n",
            encoding="utf-8",
        )
        return rows_path, root / "rows.sqlite3"

    def test_build_verifies_asset_and_searches_exact_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows_path, index_path = self._fixture(root)

            summary = build_table_row_index(
                index_path=index_path,
                rows_path=rows_path,
                project_root=root,
            )
            results = TableRowIndex(index_path).search(
                "non-hybrid fuel tank capacity",
                filters={"model": "Maverick", "year": "2026"},
            )

            self.assertEqual(summary["row_count"], 1)
            self.assertTrue(summary["asset_verification"])
            self.assertEqual(results[0]["row_id"], "maverick_fuel")
            self.assertEqual(results[0]["cells"]["Quantity"], "16.5 gal (62.4 L)")

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows_path, index_path = self._fixture(root)
            value = json.loads(rows_path.read_text(encoding="utf-8"))
            value["asset_sha256"] = "0" * 64
            rows_path.write_text(json.dumps(value) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_table_row_index(
                    index_path=index_path,
                    rows_path=rows_path,
                    project_root=root,
                )

    def test_unknown_filter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows_path, index_path = self._fixture(root)
            build_table_row_index(
                index_path=index_path,
                rows_path=rows_path,
                project_root=root,
            )

            with self.assertRaisesRegex(ValueError, "Unsupported filter"):
                TableRowIndex(index_path).search(
                    "fuel capacity",
                    filters={"edition": "hybrid"},
                )


if __name__ == "__main__":
    unittest.main()
