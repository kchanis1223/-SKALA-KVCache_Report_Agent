"""Ollama native API. 모델 다운로드나 서버 시작은 이 adapter가 수행하지 않습니다."""

from threading import Lock

from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.integrations.http import post_json


class OllamaChat:
    def __init__(
        self, model, *, base_url="http://localhost:11434", timeout=120, transport=None, lock=None
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self._lock = lock if lock is not None else Lock()

    def invoke(self, messages):
        return self._invoke(messages)

    def invoke_structured(self, messages, schema):
        return self._invoke(messages, schema)

    def _invoke(self, messages, schema=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": 0,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
        }
        if schema is not None:
            payload["format"] = schema
        with self._lock:
            result = post_json(
                f"{self.base_url}/api/chat", payload, timeout=self.timeout, transport=self.transport
            )
        try:
            content = result["message"]["content"]
            if (
                result.get("done") is not True
                or result.get("done_reason") == "length"
                or not isinstance(content, str)
                or not content.strip()
            ):
                raise ValueError("incomplete")
            return content
        except (KeyError, TypeError, AttributeError, ValueError):
            raise ModelOutputError(
                "Ollama 응답이 비어 있거나 생성이 완료되지 않았습니다."
            ) from None
