from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from automanual_rag.answering import answer_from_results, build_evidence_pack


FILTERS = {
    "brand": "Ford",
    "model": "Bronco",
    "year": "2026",
    "region": "North America",
    "language": "en",
    "manual_type": "owner_manual",
}


def result(
    *,
    chunk_id: str = "chunk-1",
    rank: int = 1,
    score: float = 24.0,
    content: str = (
        "Unlock the steering column. Move the steering wheel to the preferred "
        "position, then lock the steering column."
    ),
    chunk_type: str = "steps",
    section_path: list[str] | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "element_ids": [f"{chunk_id}:element"],
        "doc_id": "ford_bronco_2026_na_en",
        "brand": "Ford",
        "model": "Bronco",
        "year": "2026",
        "region": "North America",
        "language": "en",
        "manual_type": "owner_manual",
        "page_nos": [89],
        "section_path": section_path
        or ["Steering Wheel", "Adjusting the Steering Wheel"],
        "chunk_type": chunk_type,
        "content": content,
        "rank": rank,
        "score": score,
    }


class AnsweringTests(unittest.TestCase):
    def test_answer_contains_evidence_and_source(self) -> None:
        value = answer_from_results(
            "How do I adjust the steering wheel?",
            FILTERS,
            [result()],
        )

        self.assertEqual(value["status"], "answered")
        self.assertIn("[1] Unlock the steering column", value["answer"])
        self.assertIn("physical PDF p.89", value["answer"])
        self.assertEqual(value["evidence"][0]["retrieval_rank"], 1)

    def test_missing_vehicle_context_requests_it_before_answering(self) -> None:
        value = answer_from_results(
            "How do I adjust the steering wheel?",
            {"brand": "Ford"},
            [result()],
        )

        self.assertEqual(value["status"], "needs_context")
        self.assertEqual(value["reason"], "model_and_year_required")
        self.assertEqual(value["evidence"], [])

    def test_weak_evidence_is_refused_but_retained_for_audit(self) -> None:
        weak = result(
            score=6.0,
            content="Emission laws prohibit removing control components.",
            chunk_type="text",
        )
        value = answer_from_results(
            "How do I refill the diesel exhaust fluid tank?",
            FILTERS,
            [weak],
        )

        self.assertEqual(value["status"], "insufficient_evidence")
        self.assertEqual(value["reason"], "evidence_below_threshold")
        self.assertEqual(len(value["evidence"]), 1)

    def test_keyword_match_without_requested_procedure_is_refused(self) -> None:
        mention_only = result(
            score=20.0,
            content=(
                "Tampering with the Diesel Exhaust Fluid system can result "
                "in reduced engine power."
            ),
            chunk_type="text",
            section_path=["Customer Information", "Emission Law"],
        )
        value = answer_from_results(
            "How do I refill the diesel exhaust fluid tank?",
            FILTERS,
            [mention_only],
        )

        self.assertEqual(value["status"], "insufficient_evidence")
        self.assertEqual(value["reason"], "procedure_not_supported")

    def test_reranker_can_promote_more_complete_evidence(self) -> None:
        incomplete = result(
            chunk_id="chunk-1",
            rank=1,
            score=15.0,
            content="The steering wheel is adjustable.",
            chunk_type="text",
            section_path=["General Information"],
        )
        complete = result(
            chunk_id="chunk-2",
            rank=2,
            score=14.5,
        )

        evidence = build_evidence_pack(
            "How do I adjust the steering wheel?",
            [incomplete, complete],
        )

        self.assertEqual(evidence[0]["chunk_id"], "chunk-2")
        self.assertEqual(evidence[0]["retrieval_rank"], 2)

    def test_warning_adds_safety_note(self) -> None:
        warning = result(
            content=(
                "WARNING: Do not adjust the steering wheel when your vehicle "
                "is moving."
            ),
            chunk_type="warning",
        )
        value = answer_from_results(
            "When should I adjust the steering wheel?",
            FILTERS,
            [warning],
        )

        self.assertEqual(value["status"], "answered")
        self.assertIn("Safety note:", value["answer"])


if __name__ == "__main__":
    unittest.main()
