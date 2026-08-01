import unittest

from automanual_rag.agentic import AgenticWorkflow


FILTERS = {
    "doc_id": "ford_alpha_2026_na_en",
    "brand": "Ford",
    "model": "Alpha",
    "year": "2026",
    "region": "North America",
    "language": "en",
    "manual_type": "owner_manual",
}


def text_result():
    return {
        **FILTERS,
        "chunk_id": "chunk-1",
        "chunk_type": "steps",
        "content": "1. Unlock the column.\n2. Adjust the wheel.\n3. Lock it.",
        "element_ids": ["element-1"],
        "page_nos": [10],
        "section_path": ["Steering Wheel", "Adjusting"],
        "rank": 1,
        "score": 20.0,
    }


def graph_path(doc_id=FILTERS["doc_id"]):
    return {
        **FILTERS,
        "doc_id": doc_id,
        "path_id": "path-1",
        "rank": 1,
        "score": 2.0,
        "hops": 2,
        "page_nos": [10],
        "section_path": ["Steering Wheel", "Adjusting"],
        "node_ids": ["procedure", "step", "warning"],
        "node_types": ["Procedure", "Step", "Warning"],
        "node_labels": ["Adjusting", "Step 1", "Warning"],
        "edge_ids": ["requires", "has-warning"],
        "relations": ["REQUIRES_STEP", "HAS_WARNING"],
        "nodes": [],
        "edges": [],
        "evidence_ids": ["element-1", "element-2"],
        "content": "Procedure requires a step and has a warning.",
        "source": "Ford Alpha manual, physical PDF p.10",
    }


class FakeText:
    def __init__(self, results=None):
        self.results = [text_result()] if results is None else results

    def search(self, query, *, filters, limit):
        return list(self.results)


class FakeGraph:
    def __init__(self, results=None):
        self.results = [graph_path()] if results is None else results

    def search(self, query, *, filters, limit, max_hops):
        return list(self.results)


class AgenticWorkflowTests(unittest.TestCase):
    def test_routes_parallel_retrieval_and_returns_cited_trace(self) -> None:
        result = AgenticWorkflow(
            text_index=FakeText(),
            graph_retriever=FakeGraph(),
        ).run(
            query="How do I adjust the steering wheel and what warning applies?",
            filters=FILTERS,
        )
        self.assertEqual(result["status"], "answered")
        self.assertTrue(result["guard"]["passed"])
        self.assertEqual(result["retry_count"], 0)
        self.assertTrue(result["route"]["graph"])
        self.assertIn("[1]", result["answer"])
        nodes = [event["node"] for event in result["trace"]]
        self.assertIn("Planner/Router", nodes)
        self.assertIn("Text Retrieval", nodes)
        self.assertIn("Graph Retrieval", nodes)
        self.assertIn("Evidence Critic", nodes)
        self.assertIn("Answer/Synthesis", nodes)
        self.assertIn("Citation/Metadata Guard", nodes)

    def test_allows_only_one_replan_then_refuses(self) -> None:
        result = AgenticWorkflow(
            text_index=FakeText(),
            graph_retriever=FakeGraph([]),
        ).run(
            query="How do I adjust the steering wheel and what warning applies?",
            filters=FILTERS,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["retry_count"], 1)
        retries = [
            event
            for event in result["trace"]
            if event["node"] == "Conditional Replan"
        ]
        self.assertEqual(len(retries), 1)

    def test_critic_refuses_cross_vehicle_graph_evidence(self) -> None:
        result = AgenticWorkflow(
            text_index=FakeText(),
            graph_retriever=FakeGraph(
                [graph_path("ford_beta_2026_na_en")]
            ),
        ).run(
            query="How do I adjust the steering wheel and what warning applies?",
            filters=FILTERS,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["critic"]["reason"], "metadata_violation")
        self.assertGreater(result["critic"]["metadata_violations"], 0)


if __name__ == "__main__":
    unittest.main()
