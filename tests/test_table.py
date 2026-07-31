from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.retrieval.table import TableIndex, build_table_index
from automanual_rag.schema import ManualElement


def element(
    element_id: str,
    model: str,
    element_type: str,
    content: str,
    page_no: int,
) -> ManualElement:
    return ManualElement(
        element_id=element_id,
        doc_id=f"ford_{model.casefold().replace(' ', '_')}_2026_na_en",
        brand="Ford",
        model=model,
        year="2026",
        region="North America",
        language="en",
        manual_type="owner_manual",
        page_no=page_no,
        section_path=("Vehicle Specifications", "Wheel Nut Torque"),
        element_type=element_type,
        content=content,
        asset_path=(
            f"data/parsed/{element_id}.jpg"
            if element_type != "text"
            else None
        ),
        bbox=(0.0, 0.0, 100.0, 100.0),
        source_span=None,
        previous_element_id=None,
        next_element_id=None,
        source_locator={
            "structured_table_content_available": False,
            "source_page_no": page_no,
        },
    )


class TableIndexTests(unittest.TestCase):
    def _build(self, root: Path) -> TableIndex:
        source = root / "elements.jsonl"
        values = [
            element(
                "bronco-table",
                "Bronco",
                "table",
                "Section: Wheel Nut Torque Specifications",
                464,
            ),
            element(
                "maverick-table",
                "Maverick",
                "table",
                "Section: Washer Fluid Specification",
                89,
            ),
            element(
                "ignored-text",
                "Bronco",
                "text",
                "Wheel nut torque prose",
                463,
            ),
        ]
        source.write_text(
            "\n".join(
                json.dumps(value.to_dict(), ensure_ascii=False)
                for value in values
            )
            + "\n",
            encoding="utf-8",
        )
        index_path = root / "tables.sqlite3"
        build_table_index(index_path=index_path, element_paths=[source])
        return TableIndex(index_path)

    def test_build_indexes_only_table_elements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = self._build(Path(temporary))

            self.assertEqual(index.count(), 2)
            self.assertEqual(index.metadata["structured_element_count"], "0")

    def test_search_preserves_asset_and_applies_hard_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = self._build(Path(temporary))

            results = index.search(
                "wheel nut torque",
                filters={"model": "Bronco", "year": "2026"},
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["element_id"], "bronco-table")
            self.assertTrue(results[0]["asset_path"].endswith(".jpg"))
            self.assertFalse(results[0]["structured_content_available"])

    def test_unknown_filter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = self._build(Path(temporary))

            with self.assertRaisesRegex(ValueError, "Unsupported filter"):
                index.search("wheel torque", filters={"edition": "sport"})


if __name__ == "__main__":
    unittest.main()
