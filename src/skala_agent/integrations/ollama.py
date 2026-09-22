"""Ollama native API. 모델 다운로드나 서버 시작은 이 adapter가 수행하지 않습니다."""

from threading import Lock

from skala_agent.integrations.contracts import IncompleteModelOutputError, ModelOutputError
from skala_agent.integrations.http import post_json

# 호출이 끝나도 모델을 메모리에 얼마나 남겨둘지. 0이면 매 호출마다 모델을
# 내렸다가 다시 올려, 관점 평가처럼 같은 모델을 수십 번 부르는 경로에서 적재
# 비용을 반복해서 냅니다. 메모리가 부족한 환경은 0을 주입해 되돌립니다.
DEFAULT_KEEP_ALIVE = "5m"


class OllamaChat:
    def __init__(
        self,
        model,
        *,
        base_url="http://localhost:11434",
        timeout=120,
        transport=None,
        lock=None,
        keep_alive=DEFAULT_KEEP_ALIVE,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        self.keep_alive = keep_alive
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
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": 4096 if schema is not None else 2048,
            },
        }
        if schema is not None:
            payload["format"] = schema
        with self._lock:
            result = post_json(
                f"{self.base_url}/api/chat", payload, timeout=self.timeout, transport=self.transport
            )
        if isinstance(result, dict) and result.get("done_reason") == "length":
            raise IncompleteModelOutputError("Ollama 구조화 응답이 생성 한도에서 잘렸습니다.")
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
