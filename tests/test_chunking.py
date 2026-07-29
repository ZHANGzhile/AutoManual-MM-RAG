from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.chunking import build_text_chunks
from automanual_rag.schema import ManualElement


def element(
    index: int,
    content: str,
    *,
    page_no: int = 10,
    section_path: tuple[str, ...] = ("Tailgate", "Manual Operation"),
    title_level: int | None = None,
) -> ManualElement:
    return ManualElement(
        element_id=f"ford_test:p{page_no:04d}:text:{index:06d}:{index:016x}",
        doc_id="ford_test",
        brand="Ford",
        model="Test Vehicle",
        year="2026",
        region="North America",
        language="en",
        manual_type="owner_manual",
        page_no=page_no,
        section_path=section_path,
        element_type="text",
        content=content,
        asset_path=None,
        bbox=(10.0, 20.0 + index, 200.0, 40.0 + index),
        source_span=None,
        previous_element_id=None,
        next_element_id=None,
        source_locator={"source_index": index, "title_level": title_level},
    )


class ChunkingTests(unittest.TestCase):
    def test_safety_blocks_and_ordered_steps_remain_intact(self) -> None:
        elements = [
            element(0, "MANUAL OPERATION", title_level=2),
            element(1, "Introductory paragraph."),
            element(
                2,
                "WARNING: Secure the vehicle before continuing.",
                title_level=2,
            ),
            element(3, "1. Open the access panel."),
            element(4, "2. Pull the release handle."),
            element(5, "Note: Reinstall the panel."),
            element(6, "Closing paragraph."),
        ]
        chunks = build_text_chunks(elements, max_chars=400)

        self.assertEqual(
            [chunk.chunk_type for chunk in chunks],
            ["text", "warning", "steps", "note", "text"],
        )
        self.assertEqual(
            chunks[1].content,
            "WARNING: Secure the vehicle before continuing.",
        )
        self.assertEqual(
            chunks[2].content,
            "1. Open the access panel.\n2. Pull the release handle.",
        )
        self.assertEqual(
            chunks[2].element_ids,
            (elements[3].element_id, elements[4].element_id),
        )
        self.assertIsNone(chunks[0].previous_chunk_id)
        self.assertEqual(chunks[0].next_chunk_id, chunks[1].chunk_id)
        self.assertEqual(chunks[2].previous_chunk_id, chunks[1].chunk_id)
        self.assertIsNone(chunks[-1].next_chunk_id)

    def test_chunks_do_not_cross_page_or_section_boundaries(self) -> None:
        elements = [
            element(0, "First paragraph.", page_no=10),
            element(1, "Second paragraph.", page_no=10),
            element(2, "Next page.", page_no=11),
            element(
                3,
                "New section.",
                page_no=11,
                section_path=("Tailgate", "Removal"),
            ),
        ]
        chunks = build_text_chunks(elements, max_chars=400)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page_nos, (10,))
        self.assertEqual(chunks[1].page_nos, (11,))
        self.assertEqual(chunks[2].section_path, ("Tailgate", "Removal"))

    def test_chunk_ids_are_stable_for_identical_membership(self) -> None:
        elements = [
            element(0, "First paragraph."),
            element(1, "Second paragraph."),
        ]
        first = build_text_chunks(elements)
        rebuilt = build_text_chunks(elements)
        self.assertEqual(
            [chunk.chunk_id for chunk in first],
            [chunk.chunk_id for chunk in rebuilt],
        )


if __name__ == "__main__":
    unittest.main()
