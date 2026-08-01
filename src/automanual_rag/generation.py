"""Optional LLM/VLM generation constrained to retrieved evidence."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


CITATION_RE = re.compile(r"\[(\d+)\]")
REFUSAL_TEXT = "The current manual evidence is insufficient to answer safely."
SYSTEM_PROMPT = """You are an automotive owner-manual assistant.
Use only the supplied Evidence Pack and evidence images.
Never combine vehicle models, years, or regions.
Preserve ordered procedures and reproduce Warning/Caution meaning.
Cite factual claims with [n], where n is an Evidence Pack citation.
If the evidence does not directly support the answer, reply exactly:
The current manual evidence is insufficient to answer safely.
Do not diagnose faults or invent values."""


def load_environment(path: Path | None = None) -> bool:
    """Load a local ignored .env without overriding process variables."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    env_path = (
        path.resolve()
        if path is not None
        else Path(__file__).resolve().parents[2] / ".env"
    )
    if not env_path.is_file():
        return False
    return bool(load_dotenv(env_path, override=False))


class GenerationBackend(Protocol):
    name: str

    def generate(
        self,
        *,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
        images: Sequence["LabeledImage"] = (),
    ) -> str: ...


class GroundedGenerationError(RuntimeError):
    """Raised when a remote answer is unavailable or fails grounding checks."""


@dataclass(frozen=True)
class LabeledImage:
    label: str
    path: Path


def _safe_endpoint(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.netloc:
        return value
    if (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    ):
        return value
    raise ValueError(
        "Generation endpoint must use HTTPS or loopback HTTP"
    )


def _image_data_url(path: Path) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evidence image not found: {path}")
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if not media_type.startswith("image/"):
        raise ValueError(f"Unsupported evidence image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _evidence_text(
    evidence: Sequence[Mapping[str, Any]],
) -> str:
    records: list[dict[str, Any]] = []
    for position, item in enumerate(evidence, start=1):
        citation_id = int(item.get("citation_id", position))
        records.append(
            {
                "citation_id": citation_id,
                "vehicle": " ".join(
                    str(item.get(field, "")).strip()
                    for field in ("brand", "model", "year")
                    if str(item.get(field, "")).strip()
                ),
                "page": item.get("page_no", item.get("page_nos")),
                "section": item.get("section_path", []),
                "type": item.get(
                    "chunk_type",
                    item.get("element_type", "evidence"),
                ),
                "content": item.get(
                    "content",
                    item.get("cells", ""),
                ),
                "source": item.get("source", ""),
            }
        )
    return json.dumps(records, ensure_ascii=False, indent=2)


def validate_grounded_text(
    answer: str,
    evidence_count: int,
) -> None:
    text = answer.strip()
    if not text:
        raise GroundedGenerationError("Generation returned empty text")
    if text == REFUSAL_TEXT:
        return
    citations = [int(value) for value in CITATION_RE.findall(text)]
    if not citations:
        raise GroundedGenerationError(
            "Generated answer contains no evidence citations"
        )
    invalid = sorted(
        {value for value in citations if value < 1 or value > evidence_count}
    )
    if invalid:
        raise GroundedGenerationError(
            "Generated answer cites unknown evidence: "
            + ", ".join(str(value) for value in invalid)
        )


class ResponsesGenerationBackend:
    """Small standard-library client for Responses-compatible endpoints."""

    name = "responses_api_v1"

    def __init__(
        self,
        *,
        model: str,
        endpoint: str = "https://api.openai.com/v1/responses",
        api_key: str | None = None,
        timeout: float = 90.0,
        max_output_tokens: int = 700,
    ) -> None:
        if not model.strip():
            raise ValueError("Generation model must not be empty")
        if timeout <= 0:
            raise ValueError("Generation timeout must be positive")
        if not 64 <= max_output_tokens <= 8192:
            raise ValueError("max_output_tokens must be from 64 to 8192")
        self.model = model.strip()
        self.endpoint = _safe_endpoint(endpoint.strip())
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        hostname = urlparse(self.endpoint).hostname
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            if not self.api_key:
                raise ValueError(
                    "An API key is required for a remote Responses endpoint"
                )

    @classmethod
    def from_environment(cls) -> "ResponsesGenerationBackend":
        return cls(
            model=os.environ.get(
                "AUTOMANUAL_GENERATION_MODEL",
                "",
            ),
            endpoint=os.environ.get(
                "AUTOMANUAL_RESPONSES_URL",
                "https://api.openai.com/v1/responses",
            ),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def _payload(
        self,
        *,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
        images: Sequence[LabeledImage],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Question:\n{question.strip()}\n\n"
                    f"Evidence Pack:\n{_evidence_text(evidence)}"
                ),
            }
        ]
        for image in images:
            content.append(
                {
                    "type": "input_text",
                    "text": image.label,
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_url(image.path),
                    "detail": "high",
                }
            )
        return {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": self.max_output_tokens,
        }

    @staticmethod
    def _output_text(response: Mapping[str, Any]) -> str:
        direct = response.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        pieces: list[str] = []
        for item in response.get("output", []):
            if not isinstance(item, Mapping):
                continue
            for content in item.get("content", []):
                if not isinstance(content, Mapping):
                    continue
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
        value = "\n".join(pieces).strip()
        if not value:
            raise GroundedGenerationError(
                "Responses endpoint returned no output text"
            )
        return value

    def generate(
        self,
        *,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
        images: Sequence[LabeledImage] = (),
    ) -> str:
        if not question.strip():
            raise ValueError("Generation question must not be empty")
        if not evidence:
            return REFUSAL_TEXT
        payload = self._payload(
            question=question,
            evidence=evidence,
            images=images,
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise GroundedGenerationError(
                f"Responses endpoint returned HTTP {exc.code}"
            ) from exc
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise GroundedGenerationError(
                "Responses endpoint request failed"
            ) from exc
        if not isinstance(value, Mapping):
            raise GroundedGenerationError(
                "Responses endpoint returned an invalid object"
            )
        answer = self._output_text(value)
        validate_grounded_text(answer, len(evidence))
        return answer


class QwenGenerationBackend:
    """DashScope OpenAI-compatible Chat Completions VLM client."""

    name = "qwen_chat_completions_v1"

    def __init__(
        self,
        *,
        model: str = "qwen3-vl-flash",
        endpoint: str = (
            "https://dashscope-intl.aliyuncs.com/"
            "compatible-mode/v1/chat/completions"
        ),
        api_key: str | None = None,
        timeout: float = 90.0,
        max_output_tokens: int = 700,
    ) -> None:
        if not model.strip():
            raise ValueError("Qwen model must not be empty")
        if timeout <= 0:
            raise ValueError("Generation timeout must be positive")
        if not 64 <= max_output_tokens <= 8192:
            raise ValueError("max_output_tokens must be from 64 to 8192")
        self.model = model.strip()
        self.endpoint = _safe_endpoint(endpoint.strip())
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required for Qwen generation"
            )

    @classmethod
    def from_environment(cls) -> "QwenGenerationBackend":
        return cls(
            model=os.environ.get("QWEN_MODEL", "qwen3-vl-flash"),
            endpoint=os.environ.get(
                "DASHSCOPE_CHAT_COMPLETIONS_URL",
                (
                    "https://dashscope-intl.aliyuncs.com/"
                    "compatible-mode/v1/chat/completions"
                ),
            ),
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
        )

    def _payload(
        self,
        *,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
        images: Sequence[LabeledImage],
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Question:\n{question.strip()}\n\n"
                    f"Evidence Pack:\n{_evidence_text(evidence)}"
                ),
            }
        ]
        for image in images:
            content.append({"type": "text", "text": image.label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_data_url(image.path),
                    },
                }
            )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.max_output_tokens,
            "enable_thinking": False,
        }

    @staticmethod
    def _output_text(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise GroundedGenerationError(
                "Qwen endpoint returned no choices"
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise GroundedGenerationError(
                "Qwen endpoint returned an invalid choice"
            )
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise GroundedGenerationError(
                "Qwen endpoint returned no message"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise GroundedGenerationError(
                "Qwen endpoint returned no output text"
            )
        return content.strip()

    def generate(
        self,
        *,
        question: str,
        evidence: Sequence[Mapping[str, Any]],
        images: Sequence[LabeledImage] = (),
    ) -> str:
        if not question.strip():
            raise ValueError("Generation question must not be empty")
        if not evidence:
            return REFUSAL_TEXT
        payload = self._payload(
            question=question,
            evidence=evidence,
            images=images,
        )
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise GroundedGenerationError(
                f"Qwen endpoint returned HTTP {exc.code}"
            ) from exc
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise GroundedGenerationError(
                "Qwen endpoint request failed"
            ) from exc
        if not isinstance(value, Mapping):
            raise GroundedGenerationError(
                "Qwen endpoint returned an invalid object"
            )
        answer = self._output_text(value)
        validate_grounded_text(answer, len(evidence))
        return answer


def configured_backend(
) -> ResponsesGenerationBackend | QwenGenerationBackend | None:
    load_environment()
    name = os.environ.get(
        "AUTOMANUAL_GENERATION_BACKEND",
        "extractive",
    ).strip().casefold()
    if name in {"", "extractive", "offline"}:
        return None
    if name in {"responses", "responses_api"}:
        return ResponsesGenerationBackend.from_environment()
    if name in {"qwen", "dashscope", "qwen_chat"}:
        return QwenGenerationBackend.from_environment()
    raise ValueError(f"Unsupported generation backend: {name}")


def generate_or_fallback(
    result: Mapping[str, Any],
    *,
    question: str,
    backend: GenerationBackend | None,
    images: Sequence[LabeledImage] = (),
) -> dict[str, Any]:
    value = dict(result)
    if backend is None or value.get("status") != "answered":
        value["generation"] = {
            "status": "not_used",
            "backend": "extractive_evidence_v1",
        }
        return value
    try:
        generated = backend.generate(
            question=question,
            evidence=value.get("evidence", []),
            images=images,
        )
        if generated == REFUSAL_TEXT:
            value["status"] = "insufficient_evidence"
            value["reason"] = "generation_refused"
        value["answer"] = generated
        value["generation"] = {
            "status": "generated",
            "backend": backend.name,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        value["generation"] = {
            "status": "fallback",
            "backend": backend.name,
            "error_type": type(exc).__name__,
        }
    return value
