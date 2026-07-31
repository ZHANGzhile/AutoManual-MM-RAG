import base64
import unittest

from fastapi.testclient import TestClient

from automanual_rag.api import create_app


class FakeService:
    def health(self):
        return {"status": "ok", "manuals": 4}

    def answer_text(self, payload):
        return {
            "status": "answered",
            "query": payload.query,
            "model": payload.vehicle.model,
        }

    def answer_table(self, payload):
        return {
            "status": "answered",
            "query": payload.query,
            "model": payload.vehicle.model,
        }

    def answer_image(self, payload):
        return {
            "status": "answered",
            "bytes": len(base64.b64decode(payload.image_base64)),
            "model": payload.vehicle.model,
        }


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client_context = TestClient(
            create_app(service=FakeService())
        )
        self.client = self.client_context.__enter__()
        self.vehicle = {"model": "Bronco", "year": "2026"}

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["manuals"], 4)

    def test_text_endpoint_validates_and_routes(self) -> None:
        response = self.client.post(
            "/v1/text",
            json={
                "query": "How do I adjust the steering wheel?",
                "vehicle": self.vehicle,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "Bronco")

    def test_table_endpoint_validates_and_routes(self) -> None:
        response = self.client.post(
            "/v1/table",
            json={
                "query": "roof rack load capacity",
                "vehicle": self.vehicle,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")

    def test_image_endpoint_accepts_json_base64(self) -> None:
        response = self.client.post(
            "/v1/image",
            json={
                "image_base64": base64.b64encode(b"image").decode(),
                "filename": "query.jpg",
                "question": "What is this?",
                "vehicle": self.vehicle,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["bytes"], 5)

    def test_request_validation_rejects_missing_vehicle(self) -> None:
        response = self.client.post(
            "/v1/text",
            json={"query": "question"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
