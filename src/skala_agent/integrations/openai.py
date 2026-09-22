"""OpenAI Responses API adapter for role-specific GPT calls."""

from contextlib import nullcontext

from skala_agent.integrations.contracts import IncompleteModelOutputError, ModelOutputError
from skala_agent.integrations.http import post_json


def _usable_literal(value):
    """strict 모드가 받아들일 수 있는 문자열 리터럴인지 판단한다.

    실측(gpt-5.4-mini): enum 값에 이중따옴표가 들어가면 400으로 거부됩니다
    ('" is not allowed in string literals for structured outputs').
    아포스트로피·탭·백슬래시·퍼센트·em dash는 모두 통과합니다.

    이중따옴표는 지울 수 없습니다. 지우면 원문의 부분문자열이 아니게 되어
    citations.source_quote가 원문 구간을 찾지 못하고 인용이 버려집니다.
    그래서 해당 후보만 제외합니다.
    """
    return not (isinstance(value, str) and '"' in value)


def _single_line(value):
    """strict 모드가 거부하는 줄바꿈을 문자열 리터럴에서 제거한다.

    structured outputs는 enum·const 문자열에 줄바꿈을 허용하지 않습니다
    ("\n is not allowed in string literals for structured outputs").
    원문 인용 후보는 원문의 연속 부분문자열이라 줄바꿈을 포함할 수 있어,
    그대로 보내면 HTTP 400으로 거부됩니다.

    공백만 한 칸으로 줄이므로 의미는 바뀌지 않고, agents.citations.source_quote가
    공백 차이를 \\s+ 패턴으로 다시 맞춰 원문 구간을 찾아 반환합니다. 따라서
    보고서에 실리는 인용은 원문 그대로 유지됩니다.
    """
    return " ".join(value.split()) if isinstance(value, str) else value


def _strict_schema(schema):
    if isinstance(schema, dict):
        schema = {key: _strict_schema(value) for key, value in schema.items()}
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            # strict 모드는 required가 properties의 모든 키를 포함하기를 요구합니다.
            # Pydantic은 기본값이 있는 필드를 required에서 빼기 때문에, 그대로
            # 보내면 HTTP 400("Missing 'measurements'")으로 거부됩니다.
            # 기본값이 있는 필드도 모델이 값을 채워 보내면 Pydantic이 그대로
            # 받으므로, 여기서 전체 키로 채웁니다.
            if "properties" in schema:
                schema["required"] = list(schema["properties"])
        if "enum" in schema and isinstance(schema["enum"], list):
            # 줄바꿈 제거 후 같아진 후보는 중복이 되므로 함께 정리합니다.
            values = list(dict.fromkeys(_single_line(v) for v in schema["enum"]))
            values = [value for value in values if _usable_literal(value)]
            if values:
                schema["enum"] = values
            else:
                # 후보가 남지 않으면 enum 제약을 떼고 자유 문자열로 둡니다.
                # 빈 enum은 스키마 자체가 무효라 요청이 통째로 거부됩니다.
                # 값의 정합성은 citations.source_quote가 원문과 대조해 확인합니다.
                del schema["enum"]
        if "const" in schema:
            schema["const"] = _single_line(schema["const"])
        if "prefixItems" in schema:
            # strict 모드는 튜플(prefixItems)을 지원하지 않습니다. 그대로 버리면
            # 자리별 제약이 사라져 모델이 순서를 바꿔 넣습니다. 실측: 종합 모델이
            # assessment_refs에 (기술, 관점) 순으로 넣어 Pydantic 검증이 실패하고
            # 실행 전체가 중단됐습니다. 허용 값과 순서를 description으로 남깁니다.
            prefixes = schema.pop("prefixItems")
            schema["items"] = {"type": "string"}
            hints = [
                f"[{index}]={'|'.join(prefix['enum'])}"
                if isinstance(prefix, dict) and prefix.get("enum")
                else f"[{index}]=자유 문자열"
                for index, prefix in enumerate(prefixes)
            ]
            order = "고정 길이 배열이며 자리 순서가 정해져 있습니다: " + ", ".join(hints)
            existing = schema.get("description", "")
            schema["description"] = f"{existing} {order}".strip() if existing else order
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
