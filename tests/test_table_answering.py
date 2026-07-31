from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.table_answering import answer_table_from_results


FILTERS = {
    "brand": "Ford",
    "model": "Maverick",
    "year": "2026",
    "region": "North America",
    "language": "en",
    "manual_type": "owner_manual",
}


def result(score: float = 12.0) -> dict:
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
        "asset_path": "data/parsed/table.jpg",
        "asset_sha256": "1" * 64,
        "transcription_method": "manual_visual_verification",
        "rank": 1,
        "score": score,
    }


class TableAnsweringTests(unittest.TestCase):
    def test_exact_answer_includes_values_source_and_hash(self) -> None:
        value = answer_table_from_results(
            "What is the fuel tank capacity?",
            FILTERS,
            [result()],
        )

        self.assertEqual(value["status"], "answered")
        self.assertIn("16.5 gal (62.4 L)", value["answer"])
        self.assertIn("physical PDF p.169", value["answer"])
        self.assertIn("1" * 64, value["answer"])

    def test_missing_vehicle_context_is_rejected(self) -> None:
        value = answer_table_from_results(
            "What is the fuel tank capacity?",
            {"brand": "Ford"},
            [result()],
        )

        self.assertEqual(value["status"], "needs_context")
        self.assertEqual(value["evidence"], [])

    def test_weak_row_is_not_presented_as_exact_value(self) -> None:
        value = answer_table_from_results(
            "What is the fuel tank capacity?",
            FILTERS,
            [result(score=0.5)],
        )

        self.assertEqual(value["status"], "insufficient_evidence")
        self.assertEqual(value["reason"], "row_below_threshold")

    def test_low_semantic_coverage_is_refused(self) -> None:
        value = answer_table_from_results(
            "What is the recommended cold tire pressure?",
            FILTERS,
            [result()],
        )

        self.assertEqual(value["status"], "insufficient_evidence")
        self.assertEqual(value["reason"], "row_semantic_coverage_too_low")

    def test_excluded_hybrid_variant_is_refused(self) -> None:
        excluded = result()
        excluded["section_path"] = [
            "Fuel and Refueling",
            "FUEL TANK CAPACITY - EXCLUDING: FULL HYBRID ELECTRIC VEHICLE",
        ]
        value = answer_table_from_results(
            "What is the hybrid Maverick fuel tank capacity?",
            FILTERS,
            [excluded],
        )

        self.assertEqual(value["status"], "insufficient_evidence")
        self.assertEqual(value["reason"], "applicability_conflict")


if __name__ == "__main__":
    unittest.main()
