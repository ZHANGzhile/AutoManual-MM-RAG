from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = Path(
    os.environ.get("AUTOMANUAL_TEST_DATA_ROOT", PROJECT_ROOT / "data")
).resolve()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.ingestion.mineru import MinerUImporter, load_manifest
from automanual_rag.schema import (
    CorpusDocument,
    ManualElement,
    page_idx_to_page_no,
    stable_element_id,
)


def make_document(doc_id: str, model: str) -> CorpusDocument:
    return CorpusDocument(
        doc_id=doc_id,
        brand="Ford",
        model=model,
        year="2026",
        region="North America",
        language="en",
        manual_type="owner_manual",
        source_url=f"https://example.com/{doc_id}.pdf",
        downloaded_at="2026-07-29T12:00:00+02:00",
        local_filename=f"{doc_id}.pdf",
    )


def write_fixture(
    project_root: Path,
    document: CorpusDocument,
    elements: list[object],
) -> None:
    txt_dir = (
        project_root
        / "data"
        / "parsed"
        / document.doc_id
        / document.doc_id
        / "txt"
    )
    images_dir = txt_dir / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "figure.jpg").write_bytes(b"fixture-image")
    (images_dir / "table.jpg").write_bytes(b"fixture-table")
    (txt_dir / f"{document.doc_id}_content_list.json").write_text(
        json.dumps(elements),
        encoding="utf-8",
    )


class SchemaTests(unittest.TestCase):
    def test_page_conversion_and_schema_round_trip(self) -> None:
        self.assertEqual(page_idx_to_page_no(0), 1)
        self.assertEqual(page_idx_to_page_no(17), 18)
        with self.assertRaises(ValueError):
            page_idx_to_page_no(-1)
        with self.assertRaises(ValueError):
            page_idx_to_page_no(True)

        element = ManualElement(
            element_id="ford_test:p0001:text:000000:0123456789abcdef",
            doc_id="ford_test",
            brand="Ford",
            model="Test",
            year="2026",
            region="North America",
            language="en",
            manual_type="owner_manual",
            page_no=1,
            section_path=("Introduction",),
            element_type="text",
            content="WARNING: Test warning.",
            asset_path=None,
            bbox=(10.0, 20.0, 30.0, 40.0),
            source_span=None,
            previous_element_id=None,
            next_element_id=None,
            source_locator={"source_index": 0},
        )
        restored = ManualElement.from_dict(element.to_dict())
        self.assertEqual(restored, element)

    def test_stable_id_is_rebuildable_and_cross_document_unique(self) -> None:
        arguments = {
            "page_idx": 4,
            "source_index": 12,
            "element_type": "image",
            "fingerprint": {
                "source_type": "image",
                "native_asset_path": "images/a.jpg",
                "bbox": [1, 2, 3, 4],
            },
        }
        first = stable_element_id(doc_id="ford_one", **arguments)
        rebuilt = stable_element_id(doc_id="ford_one", **arguments)
        other_document = stable_element_id(doc_id="ford_two", **arguments)
        self.assertEqual(first, rebuilt)
        self.assertNotEqual(first, other_document)


class ImporterFixtureTests(unittest.TestCase):
    def test_four_document_metadata_isolation_and_existing_assets(self) -> None:
        documents = [
            make_document("ford_bronco_2026_na_en", "Bronco"),
            make_document("ford_f150_lightning_2026_na_en", "F-150 Lightning"),
            make_document("ford_mache_2026_na_en", "Mustang Mach-E"),
            make_document("ford_maverick_2026_na_en", "Maverick"),
        ]
        source_elements = [
            {
                "type": "text",
                "text": "CHARGING",
                "text_level": 1,
                "bbox": [10, 10, 100, 30],
                "page_idx": 0,
            },
            {
                "type": "header",
                "text": "Charging Your Vehicle",
                "bbox": [10, 1, 100, 9],
                "page_idx": 0,
            },
            {
                "type": "text",
                "text": "WARNING: Follow the manual.",
                "bbox": [10, 40, 100, 70],
                "page_idx": 0,
            },
            {
                "type": "image",
                "img_path": "images/figure.jpg",
                "image_caption": ["Charge port"],
                "image_footnote": [],
                "bbox": [10, 80, 100, 150],
                "page_idx": 0,
            },
            {
                "type": "table",
                "img_path": "images/table.jpg",
                "table_caption": ["Specifications"],
                "table_footnote": [],
                "bbox": [10, 160, 100, 220],
                "page_idx": 0,
            },
            {
                "type": "footer",
                "text": "Repeated footer",
                "bbox": [10, 230, 100, 250],
                "page_idx": 0,
            },
        ]

        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            for document in documents:
                write_fixture(project_root, document, source_elements)
            importer = MinerUImporter(
                project_root=project_root,
                parsed_root=project_root / "data" / "parsed",
            )

            all_ids: set[str] = set()
            for document in documents:
                with self.subTest(doc_id=document.doc_id):
                    result = importer.import_document(document)
                    self.assertEqual(
                        result.summary["element_counts"],
                        {"text": 2, "image": 1, "table": 1},
                    )
                    self.assertEqual(result.summary["source_page_count"], 1)
                    self.assertEqual(
                        result.summary["pages_with_source_elements"], 1
                    )
                    self.assertEqual(
                        result.summary["pages_with_output_elements"], 1
                    )
                    self.assertEqual(result.summary["anomaly_count"], 0)
                    self.assertTrue(
                        all(
                            element.doc_id == document.doc_id
                            and element.model == document.model
                            for element in result.elements
                        )
                    )
                    self.assertEqual(
                        result.elements[-1].section_path,
                        ("Charging Your Vehicle", "CHARGING"),
                    )
                    visual_elements = [
                        element
                        for element in result.elements
                        if element.element_type in {"image", "table"}
                    ]
                    self.assertTrue(visual_elements)
                    for element in visual_elements:
                        self.assertIsNotNone(element.asset_path)
                        self.assertTrue(
                            (project_root / str(element.asset_path)).is_file()
                        )
                    self.assertIsNone(result.elements[0].previous_element_id)
                    self.assertIsNone(result.elements[-1].next_element_id)
                    document_ids = {
                        element.element_id for element in result.elements
                    }
                    self.assertTrue(all_ids.isdisjoint(document_ids))
                    all_ids.update(document_ids)

    def test_malformed_fields_are_counted_without_crashing(self) -> None:
        document = make_document("ford_bad_2026_na_en", "Bad Fixture")
        source_elements = [
            {"text": "missing type", "page_idx": 0},
            {"type": "text", "text": "", "page_idx": 0},
            {
                "type": "text",
                "text": "Body with missing page.",
                "page_idx": None,
                "bbox": [10, 20, 5, 40],
            },
            {"type": "image", "page_idx": 0, "bbox": [1, 2, 3, 4]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            write_fixture(project_root, document, source_elements)
            importer = MinerUImporter(
                project_root=project_root,
                parsed_root=project_root / "data" / "parsed",
            )
            result = importer.import_document(document)

        self.assertEqual(result.summary["missing_page_no"], 1)
        self.assertEqual(result.summary["invalid_asset_paths"], 1)
        self.assertEqual(result.summary["skipped_empty_text"], 1)
        self.assertEqual(result.summary["skipped_unsupported_elements"], 1)
        self.assertGreaterEqual(result.summary["anomaly_count"], 4)
        self.assertIsNone(result.elements[0].page_no)
        self.assertIsNone(result.elements[0].bbox)


@unittest.skipUnless(
    any(
        (LOCAL_DATA_ROOT / "parsed").glob(
            "*/*/txt/*_content_list.json"
        )
    ),
    "local MinerU corpus is not available",
)
class LocalCorpusIntegrationTests(unittest.TestCase):
    def test_real_four_document_metadata_and_asset_integrity(self) -> None:
        documents = load_manifest(
            LOCAL_DATA_ROOT / "manifests" / "corpus.csv"
        )
        self.assertEqual(len(documents), 4)
        importer = MinerUImporter(
            project_root=LOCAL_DATA_ROOT.parent,
            parsed_root=LOCAL_DATA_ROOT / "parsed",
        )
        global_ids: set[str] = set()
        for document in documents:
            with self.subTest(doc_id=document.doc_id):
                result = importer.import_document(document)
                self.assertEqual(result.summary["missing_page_no"], 0)
                self.assertEqual(result.summary["invalid_asset_paths"], 0)
                self.assertEqual(result.summary["anomaly_count"], 0)
                self.assertTrue(
                    all(
                        element.doc_id == document.doc_id
                        and element.model == document.model
                        and element.year == document.year
                        and element.region == document.region
                        for element in result.elements
                    )
                )
                visual_elements = [
                    element
                    for element in result.elements
                    if element.element_type in {"image", "table"}
                ]
                self.assertTrue(visual_elements)
                self.assertTrue(
                    all(
                        element.asset_path
                        and (
                            LOCAL_DATA_ROOT.parent / element.asset_path
                        ).is_file()
                        for element in visual_elements
                    )
                )
                element_ids = {
                    element.element_id for element in result.elements
                }
                self.assertEqual(len(element_ids), len(result.elements))
                self.assertTrue(global_ids.isdisjoint(element_ids))
                global_ids.update(element_ids)


if __name__ == "__main__":
    unittest.main()
