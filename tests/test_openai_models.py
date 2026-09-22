import json

import httpx

from skala_agent.integrations.openai import OpenAIResponses
from skala_agent.model_config import ModelRouter, ModelSettings


def test_openai_responses_uses_structured_output_and_reasoning_effort():
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        assert str(request.url) == "https://api.openai.com/v1/responses"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.6-terra"
        assert payload["reasoning"] == {"effort": "low"}
        assert payload["text"]["format"]["type"] == "json_schema"
        assert payload["text"]["format"]["schema"]["additionalProperties"] is False
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"supports_claim":true}'}],
                    }
                ],
            },
        )

    model = OpenAIResponses(
        "gpt-5.6-terra",
        api_key="test-key",
        reasoning_effort="low",
        transport=httpx.MockTransport(handler),
    )

    assert (
        model.invoke_structured([{"role": "user", "content": "fixture"}], {"type": "object"})
        == '{"supports_claim":true}'
    )


def test_model_router_assigns_one_openai_model_to_every_agent():
    """아홉 Agent 모두 같은 모델을 쓰고 추론 강도로만 구분한다."""
    from skala_agent.model_config import AGENTS, DEFAULT_OPENAI_MODEL

    router = ModelRouter(ModelSettings(openai_api_key="test"))

    assert {router.for_agent(agent).model for agent in AGENTS} == {DEFAULT_OPENAI_MODEL}
    assert router.for_agent("synthesis").reasoning_effort == "high"
    assert router.for_agent("validation").reasoning_effort == "low"
    assert router.for_agent("domain").reasoning_effort == "medium"


def test_openai_model_id_is_overridable_by_environment():
    """조직 배포명이 다르면 코드를 고치지 않고 OPENAI_MODEL로 바꿀 수 있다."""
    settings = ModelSettings.from_environment(
        {"OPENAI_MODEL": "gpt-5.4-mini-2026-04-01", "OPENAI_API_KEY": "test"}
    )
    router = ModelRouter(settings)
    assert router.for_agent("trl").model == "gpt-5.4-mini-2026-04-01"


def test_openai_models_are_not_serialized_by_a_shared_lock():
    """원격 호출에 lock을 걸면 관점 fan-out이 순차 실행된다.

    실측: 공통 Lock 때문에 관점 하나가 285.9초를 쓰는 동안 나머지 세 관점이
    대기하다 상한에서 끊겼습니다. OpenAI 경로에는 lock을 두지 않습니다.
    """
    from threading import Lock

    from skala_agent.model_config import AGENTS

    router = ModelRouter(ModelSettings(openai_api_key="test"))
    for agent in AGENTS:
        assert not isinstance(router.for_agent(agent)._lock, type(Lock()))


def test_strict_schema_requires_every_property_including_defaults():
    """strict 모드는 required가 properties의 모든 키를 포함해야 한다.

    Pydantic은 기본값이 있는 필드를 required에서 빼므로, 변환 없이 보내면
    OpenAI가 HTTP 400으로 거부합니다.
    """
    from skala_agent.integrations.openai import _strict_schema
    from skala_agent.integrations.structured import EvaluationDraft

    raw = EvaluationDraft.model_json_schema()
    strict = _strict_schema(raw)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert set(node["required"]) == set(node["properties"]), node["properties"].keys()
                assert node["additionalProperties"] is False
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(strict)
    # 기본값이 있어 Pydantic이 빼뒀던 필드가 required에 들어왔는지 확인합니다.
    finding = strict["$defs"]["Finding"]
    assert "measurements" in finding["required"]
    assert "measurements" not in raw["$defs"]["Finding"].get("required", [])


def test_strict_schema_removes_newlines_from_string_literals():
    """strict 모드는 enum·const 문자열 리터럴에 줄바꿈을 허용하지 않는다.

    원문 인용 후보(agents.citations.quote_options)는 원문의 연속 부분문자열이라
    줄바꿈을 포함할 수 있어, 그대로 보내면 HTTP 400으로 거부됩니다.
    """
    from skala_agent.integrations.openai import _strict_schema

    schema = _strict_schema(
        {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string", "const": "doc\na"},
                        "quote": {
                            "type": "string",
                            "enum": ["Accuracy\npreserved.", "Accuracy preserved.", "Cost fell."],
                        },
                    },
                    "required": ["source_id", "quote"],
                }
            ]
        }
    )
    variant = schema["anyOf"][0]

    assert variant["properties"]["source_id"]["const"] == "doc a"
    # 줄바꿈을 지운 뒤 같아진 후보는 중복이므로 하나만 남습니다.
    assert variant["properties"]["quote"]["enum"] == ["Accuracy preserved.", "Cost fell."]
    assert all("\n" not in value for value in variant["properties"]["quote"]["enum"])


def test_quote_options_still_return_exact_substrings_of_the_source():
    """공백 정규화는 OpenAI 어댑터에서만 한다. 공유 인용 계약은 그대로다."""
    from skala_agent.agents.citations import quote_options, source_quote

    content = "Accuracy preserved.\nCost fell 20% on GPU A."
    options = quote_options(content)

    assert all(option in content for option in options)
    # 어댑터가 한 줄로 만든 인용도 source_quote가 원문 구간으로 되돌립니다.
    flattened = " ".join(content.split())
    assert source_quote(content, flattened) == content


def test_strict_schema_leaves_non_object_nodes_alone():
    from skala_agent.integrations.openai import _strict_schema

    schema = {"type": "array", "items": {"type": "string"}}
    assert _strict_schema(schema) == schema
