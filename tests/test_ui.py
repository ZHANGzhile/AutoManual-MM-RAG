from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from automanual_rag.ui.app import DemoService


class DemoServiceTests(unittest.TestCase):
    def test_verified_table_crop_is_shown_before_topic_matches(self) -> None:
        service = DemoService.__new__(DemoService)
        service.project_root = Path("C:/project")
        service.documents = {
            "Ford Bronco 2026": SimpleNamespace(
                doc_id="ford_bronco_2026_na_en",
                brand="Ford",
                model="Bronco",
                year="2026",
                region="North America",
                language="en",
                manual_type="owner_manual",
            )
        }
        service.table_rows = object()
        service.tables = SimpleNamespace(
            search=lambda *args, **kwargs: [
                {
                    "rank": 1,
                    "page_no": 250,
                    "section_path": ["Unrelated"],
                    "asset_path": "data/unrelated.jpg",
                }
            ]
        )
        answer_result = {
            "status": "answered",
            "answer": "110 lb (50 kg)",
            "evidence": [
                {
                    "citation_id": 1,
                    "brand": "Ford",
                    "model": "Bronco",
                    "year": "2026",
                    "page_no": 294,
                    "section_path": [
                        "Load Carrying",
                        "ROOF RACK LOAD CAPACITIES",
                    ],
                    "cells": {
                        "Description": "When in motion",
                        "Maximum Recommended Load": "110 lb (50 kg)",
                    },
                    "score": 9.0,
                    "transcription_method": "manual_visual_verification",
                    "asset_path": "data/verified.jpg",
                }
            ],
        }

        with patch(
            "automanual_rag.ui.app.answer_table_question",
            return_value=answer_result,
        ):
            _, gallery, _, _ = service.search_table(
                "Ford Bronco 2026",
                "roof load",
            )

        self.assertEqual(
            gallery[0][0],
            str((service.project_root / "data/verified.jpg").resolve()),
        )
        self.assertIn("Verified source [1]", gallery[0][1])
        self.assertIn("ROOF RACK LOAD CAPACITIES", gallery[0][1])


if __name__ == "__main__":
    unittest.main()
