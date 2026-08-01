import unittest

from fastapi.testclient import TestClient

from automanual_rag.agentic_api import create_agentic_app


class FakeAgenticService:
    def health(self):
        return {
            "status": "ok",
            "backend": "agentic_graphrag_state_graph_v1",
            "manuals": 4,
            "graph": {"nodes": 10, "edges": 20},
        }

    def answer(self, payload):
        return {
            "status": "answered",
            "query": payload.query,
            "model": payload.vehicle.model,
            "trace": [{"node": "Planner/Router"}],
        }


class AgenticApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TestClient(
            create_agentic_app(service=FakeAgenticService())
        )
        self.client = self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    def test_health_and_agentic_endpoint(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(
            health.json()["backend"],
            "agentic_graphrag_state_graph_v1",
        )
        answer = self.client.post(
            "/v1/agentic",
            json={
                "query": "How do I adjust the steering wheel?",
                "vehicle": {"model": "Bronco", "year": "2026"},
            },
        )
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(answer.json()["model"], "Bronco")
        self.assertEqual(answer.json()["trace"][0]["node"], "Planner/Router")


if __name__ == "__main__":
    unittest.main()
