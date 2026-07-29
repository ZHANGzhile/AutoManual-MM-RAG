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

from automanual_rag.chunking import TextChunk
from automanual_rag.evaluation.retrieval import evaluate_bm25
from automanual_rag.retrieval.bm25 import BM25Index, build_bm25_index


def chunk(
    chunk_id: str,
    doc_id: str,
    model: str,
    content: str,
    page_no: int,
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
        section_path=("Steering Wheel",),
        chunk_type="text",
        content=content,
        indexed_text=f"Ford {model} 2026 Steering Wheel {content}",
        element_ids=(f"{chunk_id}:element",),
        previous_chunk_id=None,
        next_chunk_id=None,
    )


class BM25Tests(unittest.TestCase):
    def test_search_uses_metadata_hard_filter(self) -> None:
        chunks = [
            chunk(
                "bronco:chunk:1",
                "ford_bronco_2026_na_en",
                "Bronco",
                "Unlock the steering column and adjust the steering wheel.",
                89,
            ),
            chunk(
                "mache:chunk:1",
                "ford_mache_2026_na_en",
                "Mustang Mach-E",
                "Mustang Mach-E charging at home uses AC charging.",
                175,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk_path = root / "chunks.jsonl"
            chunk_path.write_text(
                "".join(
                    json.dumps(value.to_dict()) + "\n" for value in chunks
                ),
                encoding="utf-8",
            )
            index_path = root / "bm25.sqlite3"
            summary = build_bm25_index(
                index_path=index_path,
                chunk_paths=[chunk_path],
            )
            index = BM25Index(index_path)
            results = index.search(
                "How do I adjust the steering wheel?",
                filters={"model": "Bronco", "year": "2026"},
            )

        self.assertEqual(summary["chunk_count"], 2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "ford_bronco_2026_na_en")
        self.assertEqual(results[0]["page_nos"], [89])

    def test_unknown_filter_is_rejected(self) -> None:
        chunks = [
            chunk(
                "bronco:chunk:1",
                "ford_bronco_2026_na_en",
                "Bronco",
                "Adjust the steering wheel.",
                89,
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk_path = root / "chunks.jsonl"
            chunk_path.write_text(
                json.dumps(chunks[0].to_dict()) + "\n",
                encoding="utf-8",
            )
            index_path = root / "bm25.sqlite3"
            build_bm25_index(index_path=index_path, chunk_paths=[chunk_path])
            index = BM25Index(index_path)
            with self.assertRaises(ValueError):
                index.search("steering", filters={"unsupported": "value"})

    def test_evaluation_uses_gold_element_membership(self) -> None:
        chunks = [
            chunk(
                "bronco:chunk:1",
                "ford_bronco_2026_na_en",
                "Bronco",
                "Unlock the steering column and adjust the steering wheel.",
                89,
            )
        ]
        questions = [
            {
                "question_id": "answerable",
                "category": "procedure",
                "question": "adjust steering wheel",
                "filters": {"model": "Bronco", "year": "2026"},
                "gold_evidence": [
                    {
                        "doc_id": "ford_bronco_2026_na_en",
                        "page_no": 89,
                        "element_id": "bronco:chunk:1:element",
                    }
                ],
                "reference_answer": "Unlock and adjust the column.",
                "answerable": True,
            },
            {
                "question_id": "no_answer",
                "category": "no_answer",
                "question": "diesel exhaust fluid",
                "filters": {"model": "Bronco", "year": "2026"},
                "gold_evidence": [],
                "reference_answer": "Insufficient evidence.",
                "answerable": False,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chunk_path = root / "chunks.jsonl"
            chunk_path.write_text(
                json.dumps(chunks[0].to_dict()) + "\n",
                encoding="utf-8",
            )
            index_path = root / "bm25.sqlite3"
            build_bm25_index(index_path=index_path, chunk_paths=[chunk_path])
            result = evaluate_bm25(
                index=BM25Index(index_path),
                questions=questions,
            )

        self.assertEqual(result["metrics"]["recall_at_5"], 1.0)
        self.assertEqual(result["metrics"]["metadata_filter_violations"], 0)


if __name__ == "__main__":
    unittest.main()
