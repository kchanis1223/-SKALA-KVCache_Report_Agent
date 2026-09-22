import json

import httpx

from skala_agent.integrations.openai import OpenAIResponses
from skala_agent.model_config import ModelRouter, ModelSettings


def test_openai_responses_uses_structured_output_and_reasoning_effort():
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        assert str(request.url) == "https://api.openai.com/v1/responses"
        payload = json.loads(request.content)
        assert payload["model"] == "gpt-5.4mini"
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
        "gpt-5.4mini",
        api_key="test-key",
        reasoning_effort="low",
        transport=httpx.MockTransport(handler),
    )

    assert (
        model.invoke_structured([{"role": "user", "content": "fixture"}], {"type": "object"})
        == '{"supports_claim":true}'
    )


def test_model_router_assigns_gpt_models_to_only_the_three_final_agents():
    router = ModelRouter(ModelSettings(openai_api_key="test"))

    assert router.for_agent("synthesis").model == "gpt-5.4mini"
    assert router.for_agent("validation").model == "gpt-5.4mini"
    assert router.for_agent("report").model == "gpt-5.4mini"
    assert router.for_agent("domain").model == "qwen3:4b"
