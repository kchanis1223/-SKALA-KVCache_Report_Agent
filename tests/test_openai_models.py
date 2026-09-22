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
