"""OpenAI Responses API adapter for role-specific GPT calls."""

from contextlib import nullcontext

from skala_agent.integrations.contracts import IncompleteModelOutputError, ModelOutputError
from skala_agent.integrations.http import post_json


def _strict_schema(schema):
    if isinstance(schema, dict):
        schema = {key: _strict_schema(value) for key, value in schema.items()}
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        if "prefixItems" in schema:
            schema["items"] = {"type": "string"}
            del schema["prefixItems"]
    elif isinstance(schema, list):
        schema = [_strict_schema(value) for value in schema]
    return schema


class OpenAIResponses:
    def __init__(
        self,
        model,
        *,
        api_key,
        reasoning_effort,
        base_url="https://api.openai.com/v1",
        timeout=120,
        transport=None,
        lock=None,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 필요합니다.")
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport
        # 원격 API 호출은 로컬 GPU 메모리를 공유하지 않으므로 기본값은 직렬화
        # 없음입니다. Lock을 넣으면 관점 fan-out이 순차 실행됩니다. 요청 수를
        # 제한해야 하는 배포에서는 호출자가 lock을 주입할 수 있습니다.
        self._lock = lock if lock is not None else nullcontext()

    def invoke(self, messages):
        return self._invoke(messages)

    def invoke_structured(self, messages, schema):
        return self._invoke(messages, schema)

    def _invoke(self, messages, schema=None):
        payload = {
            "model": self.model,
            "input": messages,
            "reasoning": {"effort": self.reasoning_effort},
        }
        if schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "agent_output",
                    "strict": True,
                    "schema": _strict_schema(schema),
                }
            }
        with self._lock:
            result = post_json(
                f"{self.base_url}/responses",
                payload,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"},
                transport=self.transport,
            )
        if result.get("status") == "incomplete":
            raise IncompleteModelOutputError("OpenAI Responses 출력이 생성 한도에서 잘렸습니다.")
        try:
            text = next(
                content["text"]
                for item in result["output"]
                if item.get("type") == "message"
                for content in item["content"]
                if content.get("type") == "output_text"
            )
            if result.get("status") != "completed" or not isinstance(text, str) or not text.strip():
                raise ValueError("incomplete")
            return text
        except (KeyError, TypeError, ValueError, StopIteration):
            raise ModelOutputError(
                "OpenAI Responses 응답이 비어 있거나 생성이 완료되지 않았습니다."
            ) from None
