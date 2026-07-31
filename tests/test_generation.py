import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from automanual_rag.generation import (
    GroundedGenerationError,
    LabeledImage,
    REFUSAL_TEXT,
    ResponsesGenerationBackend,
    generate_or_fallback,
    validate_grounded_text,
)


class _Response:
    def __init__(self, value: dict) -> None:
        self.value = value

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


class _FailingBackend:
    name = "failing"

    def generate(self, **kwargs: object) -> str:
        raise GroundedGenerationError("failure")


class GenerationTests(unittest.TestCase):
    def test_citations_must_reference_the_evidence_pack(self) -> None:
        validate_grounded_text("Use this procedure [1].", 2)
        validate_grounded_text(REFUSAL_TEXT, 0)
        with self.assertRaisesRegex(
            GroundedGenerationError,
            "unknown evidence",
        ):
            validate_grounded_text("Unsupported [3].", 2)
        with self.assertRaisesRegex(
            GroundedGenerationError,
            "no evidence citations",
        ):
            validate_grounded_text("Unsupported answer.", 2)

    def test_responses_payload_supports_labeled_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "query.jpg"
            image.write_bytes(b"\xff\xd8\xff\xd9")
            backend = ResponsesGenerationBackend(
                model="test-model",
                endpoint="http://127.0.0.1:8000/v1/responses",
            )
            payload = backend._payload(
                question="What is shown?",
                evidence=[{"citation_id": 1, "content": "cluster"}],
                images=[LabeledImage("Query image", image)],
            )
            content = payload["input"][0]["content"]
            self.assertEqual(content[-2]["text"], "Query image")
            self.assertTrue(
                content[-1]["image_url"].startswith("data:image/jpeg;base64,")
            )

    def test_responses_output_is_parsed_and_validated(self) -> None:
        backend = ResponsesGenerationBackend(
            model="test-model",
            endpoint="http://127.0.0.1:8000/v1/responses",
        )
        response = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The warning is shown here [1].",
                        }
                    ]
                }
            ]
        }
        with patch(
            "automanual_rag.generation.urlopen",
            return_value=_Response(response),
        ):
            answer = backend.generate(
                question="What does it mean?",
                evidence=[{"citation_id": 1, "content": "warning"}],
            )
        self.assertEqual(answer, "The warning is shown here [1].")

    def test_remote_failure_keeps_the_extractive_answer(self) -> None:
        base = {
            "status": "answered",
            "answer": "Safe extractive answer [1].",
            "evidence": [{"citation_id": 1}],
        }
        result = generate_or_fallback(
            base,
            question="Question",
            backend=_FailingBackend(),
        )
        self.assertEqual(result["answer"], base["answer"])
        self.assertEqual(result["generation"]["status"], "fallback")


if __name__ == "__main__":
    unittest.main()
