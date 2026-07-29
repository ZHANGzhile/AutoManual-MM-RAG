from __future__ import annotations

from importlib.util import find_spec
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.chunking import TextChunk
from automanual_rag.evaluation.visual import evaluate_visual
from automanual_rag.retrieval.bm25 import BM25Index, build_bm25_index
from automanual_rag.retrieval.visual import (
    FEATURE_DIMENSIONS,
    VisualIndex,
    VisualTextFusionIndex,
    build_visual_index,
    extract_visual_features,
)
from automanual_rag.schema import ManualElement


DEPENDENCIES_AVAILABLE = (
    find_spec("numpy") is not None and find_spec("PIL") is not None
)


def element(
    *,
    element_id: str,
    doc_id: str,
    model: str,
    page_no: int,
    asset_path: str,
    element_type: str = "image",
) -> ManualElement:
    return ManualElement(
        element_id=element_id,
        doc_id=doc_id,
        brand="Ford",
        model=model,
        year="2026",
        region="North America",
        language="en",
        manual_type="owner_manual",
        page_no=page_no,
        section_path=("Maintenance",),
        element_type=element_type,
        content=f"Section: Maintenance Page {page_no}",
        asset_path=asset_path,
        bbox=(0.0, 0.0, 100.0, 100.0),
        source_span=None,
        previous_element_id=None,
        next_element_id=None,
        source_locator={"source_page_no": page_no},
    )


def chunk(
    *,
    chunk_id: str,
    doc_id: str,
    model: str,
    page_no: int,
    content: str,
) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        brand="Ford",
        model=model,
        year="2026",
        region="North America",
        language="en",
        manual_type="owner_manual",
        page_start=page_no,
        page_end=page_no,
        page_nos=(page_no,),
        section_path=("Maintenance",),
        chunk_type="text",
        content=content,
        indexed_text=f"Ford {model} 2026 Maintenance {content}",
        element_ids=(f"{chunk_id}:text",),
        previous_chunk_id=None,
        next_chunk_id=None,
    )


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "NumPy and Pillow are required for visual tests",
)
class VisualRetrievalTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
    ) -> tuple[VisualIndex, BM25Index, Path, Path]:
        from PIL import Image, ImageDraw

        assets = root / "assets"
        assets.mkdir()
        first_path = assets / "first.jpg"
        second_path = assets / "second.jpg"
        table_path = assets / "table.jpg"

        first = Image.new("RGB", (160, 100), "white")
        draw = ImageDraw.Draw(first)
        draw.rectangle((25, 20, 135, 80), fill="black")
        draw.ellipse((55, 30, 105, 70), fill="white")
        first.save(first_path)

        second = Image.new("RGB", (160, 100), "white")
        draw = ImageDraw.Draw(second)
        draw.polygon(((80, 10), (145, 90), (15, 90)), fill=(20, 80, 180))
        second.save(second_path)
        Image.new("RGB", (160, 100), "gray").save(table_path)

        values = [
            element(
                element_id="doc_a:p0001:image:1",
                doc_id="doc_a",
                model="Model A",
                page_no=1,
                asset_path="assets/first.jpg",
            ),
            element(
                element_id="doc_b:p0002:image:1",
                doc_id="doc_b",
                model="Model B",
                page_no=2,
                asset_path="assets/second.jpg",
            ),
            element(
                element_id="doc_a:p0003:table:1",
                doc_id="doc_a",
                model="Model A",
                page_no=3,
                asset_path="assets/table.jpg",
                element_type="table",
            ),
        ]
        elements_path = root / "elements.jsonl"
        elements_path.write_text(
            "".join(json.dumps(value.to_dict()) + "\n" for value in values),
            encoding="utf-8",
        )
        visual_path = root / "visual.npz"
        summary = build_visual_index(
            project_root=root,
            index_path=visual_path,
            element_paths=[elements_path],
        )
        self.assertEqual(summary["element_count"], 2)
        self.assertEqual(summary["excluded_table_crops"], 1)

        chunks = [
            chunk(
                chunk_id="doc_a:chunk:1",
                doc_id="doc_a",
                model="Model A",
                page_no=1,
                content="Open the hood release latch.",
            ),
            chunk(
                chunk_id="doc_b:chunk:1",
                doc_id="doc_b",
                model="Model B",
                page_no=2,
                content="Inspect the blue warning triangle.",
            ),
        ]
        chunks_path = root / "chunks.jsonl"
        chunks_path.write_text(
            "".join(json.dumps(value.to_dict()) + "\n" for value in chunks),
            encoding="utf-8",
        )
        bm25_path = root / "bm25.sqlite3"
        build_bm25_index(
            index_path=bm25_path,
            chunk_paths=[chunks_path],
        )
        return (
            VisualIndex(visual_path, project_root=root),
            BM25Index(bm25_path),
            first_path,
            visual_path,
        )

    def test_features_are_finite_normalized_and_deterministic(self) -> None:
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shape.jpg"
            Image.new("RGB", (120, 80), "black").save(path)
            first = extract_visual_features(path)
            second = extract_visual_features(path)

        self.assertEqual(first.shape, (FEATURE_DIMENSIONS,))
        self.assertTrue(np.isfinite(first).all())
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=6)
        self.assertTrue(np.array_equal(first, second))

    def test_index_search_uses_hard_filter_and_returns_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visual, _, query, visual_path = self._fixture(root)
            results = visual.search(
                query,
                filters={
                    "model": "Model A",
                    "year": "2026",
                    "region": "North America",
                },
            )
            reloaded = VisualIndex(visual_path, project_root=root)
            repeated = reloaded.search(
                query,
                filters={
                    "model": "Model A",
                    "year": "2026",
                    "region": "North America",
                },
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["element_id"], "doc_a:p0001:image:1")
        self.assertEqual(results[0]["page_no"], 1)
        self.assertEqual(results[0]["asset_path"], "assets/first.jpg")
        self.assertEqual(
            [value["element_id"] for value in results],
            [value["element_id"] for value in repeated],
        )

    def test_unknown_filter_and_missing_asset_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visual, _, query, _ = self._fixture(root)
            with self.assertRaises(ValueError):
                visual.search(query, filters={"unsupported": "value"})

            bad_path = root / "bad.jsonl"
            bad = element(
                element_id="bad:p0001:image:1",
                doc_id="bad",
                model="Model Bad",
                page_no=1,
                asset_path="assets/missing.jpg",
            )
            bad_path.write_text(
                json.dumps(bad.to_dict()) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                build_visual_index(
                    project_root=root,
                    index_path=root / "bad.npz",
                    element_paths=[bad_path],
                )

    def test_visual_evaluation_and_text_fusion_keep_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visual, bm25, query, _ = self._fixture(root)
            questions = [
                {
                    "query_id": "shape",
                    "category": "operation_diagram",
                    "split": "test",
                    "query_image": "assets/first.jpg",
                    "_query_path": query,
                    "query_text": "open hood release latch",
                    "filters": {
                        "model": "Model A",
                        "year": "2026",
                        "region": "North America",
                    },
                    "gold_evidence": {
                        "element_id": "doc_a:p0001:image:1",
                        "doc_id": "doc_a",
                        "page_no": 1,
                        "asset_path": "assets/first.jpg",
                    },
                    "source": {},
                    "transform": {},
                }
            ]
            result = evaluate_visual(
                index=visual,
                questions=questions,
                backend="test_visual",
                use_text_hint=False,
            )
            fusion = VisualTextFusionIndex(
                visual=visual,
                bm25=bm25,
                candidate_limit=2,
            )
            fused = fusion.search(
                query,
                query_text="open hood release latch",
                filters={
                    "model": "Model A",
                    "year": "2026",
                    "region": "North America",
                },
            )

        self.assertEqual(result["metrics"]["recall_at_1"], 1.0)
        self.assertEqual(result["metrics"]["metadata_filter_violations"], 0)
        self.assertEqual(fused[0]["visual_rank"], 1)
        self.assertEqual(fused[0]["text_rank"], 1)
        self.assertIsNotNone(fused[0]["visual_score"])
        self.assertIsNotNone(fused[0]["text_score"])
        self.assertEqual(fused[0]["fusion_score"], fused[0]["score"])


if __name__ == "__main__":
    unittest.main()
