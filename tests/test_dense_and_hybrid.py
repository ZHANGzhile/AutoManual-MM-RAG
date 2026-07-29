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
from automanual_rag.evaluation.retrieval import evaluate_dense
from automanual_rag.retrieval.bm25 import BM25Index, build_bm25_index
from automanual_rag.retrieval.dense import DenseIndex, build_dense_index
from automanual_rag.retrieval.hybrid import HybridIndex


NUMPY_AVAILABLE = find_spec("numpy") is not None


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
        section_path=("Vehicle operation",),
        chunk_type="text",
        content=content,
        indexed_text=f"Ford {model} 2026 Vehicle operation {content}",
        element_ids=(f"{chunk_id}:element",),
        previous_chunk_id=None,
        next_chunk_id=None,
    )


@unittest.skipUnless(NUMPY_AVAILABLE, "NumPy is required for Dense tests")
class DenseAndHybridTests(unittest.TestCase):
    def _build_indexes(
        self,
        root: Path,
    ) -> tuple[DenseIndex, BM25Index]:
        chunks = [
            chunk(
                "bronco:steering",
                "ford_bronco_2026_na_en",
                "Bronco",
                "Unlock the steering column and adjust the steering wheel.",
                89,
            ),
            chunk(
                "bronco:tire",
                "ford_bronco_2026_na_en",
                "Bronco",
                "Inflate the tires to the pressure on the door label.",
                410,
            ),
            chunk(
                "mache:charge",
                "ford_mache_2026_na_en",
                "Mustang Mach-E",
                "Connect the charging coupler to the charge port.",
                175,
            ),
            chunk(
                "mache:steering",
                "ford_mache_2026_na_en",
                "Mustang Mach-E",
                "Adjust the steering wheel after stopping the vehicle.",
                102,
            ),
        ]
        chunk_path = root / "chunks.jsonl"
        chunk_path.write_text(
            "".join(json.dumps(value.to_dict()) + "\n" for value in chunks),
            encoding="utf-8",
        )
        dense_path = root / "dense.npz"
        bm25_path = root / "bm25.sqlite3"
        build_dense_index(
            index_path=dense_path,
            chunk_paths=[chunk_path],
            feature_count=128,
            dimensions=4,
            oversamples=0,
            seed=7,
        )
        build_bm25_index(index_path=bm25_path, chunk_paths=[chunk_path])
        return DenseIndex(dense_path), BM25Index(bm25_path)

    def test_dense_hard_filter_and_stable_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense, _ = self._build_indexes(root)
            first = dense.search(
                "How do I adjust the steering wheel?",
                filters={"model": "Bronco", "year": "2026"},
                limit=2,
            )
            second_path = root / "dense_second.npz"
            build_dense_index(
                index_path=second_path,
                chunk_paths=[root / "chunks.jsonl"],
                feature_count=128,
                dimensions=4,
                oversamples=0,
                seed=7,
            )
            second = DenseIndex(second_path).search(
                "How do I adjust the steering wheel?",
                filters={"model": "Bronco", "year": "2026"},
                limit=2,
            )

        self.assertTrue(first)
        self.assertTrue(
            all(result["model"] == "Bronco" for result in first)
        )
        self.assertEqual(
            [result["chunk_id"] for result in first],
            [result["chunk_id"] for result in second],
        )
        for left, right in zip(first, second):
            self.assertAlmostEqual(left["score"], right["score"], places=5)

    def test_dense_unknown_filter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dense, _ = self._build_indexes(Path(temporary))
            with self.assertRaises(ValueError):
                dense.search("steering", filters={"unsupported": "value"})

    def test_hybrid_keeps_backend_ranks_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dense, bm25 = self._build_indexes(Path(temporary))
            hybrid = HybridIndex(
                bm25=bm25,
                dense=dense,
                candidate_limit=4,
            )
            results = hybrid.search(
                "adjust steering wheel",
                filters={"model": "Bronco", "year": "2026"},
                limit=2,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(
            all(result["model"] == "Bronco" for result in results)
        )
        self.assertTrue(
            all(
                result["bm25_rank"] is not None
                or result["dense_rank"] is not None
                for result in results
            )
        )
        self.assertEqual(results[0]["rrf_score"], results[0]["score"])

    def test_dense_evaluation_uses_gold_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dense, _ = self._build_indexes(Path(temporary))
            questions = [
                {
                    "question_id": "steering",
                    "category": "procedure",
                    "question": "adjust steering wheel",
                    "filters": {"model": "Bronco", "year": "2026"},
                    "gold_evidence": [
                        {
                            "doc_id": "ford_bronco_2026_na_en",
                            "page_no": 89,
                            "element_id": "bronco:steering:element",
                        }
                    ],
                    "reference_answer": "Unlock and adjust the column.",
                    "answerable": True,
                }
            ]
            result = evaluate_dense(
                index=dense,
                questions=questions,
                limit=10,
            )

        self.assertEqual(result["metrics"]["recall_at_5"], 1.0)
        self.assertEqual(result["metrics"]["metadata_filter_violations"], 0)


if __name__ == "__main__":
    unittest.main()
