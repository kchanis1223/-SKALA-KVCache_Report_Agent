import json

import httpx
import pytest

from skala_agent.integrations.contracts import ModelOutputError, ServiceConfigurationError
from skala_agent.integrations.qwen import TransformersQwen
from skala_agent.integrations.structured import StructuredExtractor
from skala_agent.integrations.tavily import TavilySearch


def test_model_object_can_be_replaced_and_invalid_json_has_bounded_retry():
    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            assert "JSON Schema" in messages[0]["content"]
            return "not json" if self.calls == 1 else '{"findings": []}'

    model = Model()
    assert StructuredExtractor(model).extract("system", {}).findings == []
    assert model.calls == 2

    class Invalid:
        def invoke(self, messages):
            return "not json"

    with pytest.raises(ModelOutputError):
        StructuredExtractor(Invalid()).extract("system", {})


def test_tavily_request_and_explicit_source_classification():
    def handler(request):
        payload = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer test-key"
        assert payload["include_answer"] is False and payload["max_results"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Official", "url": "https://vendor.example/test", "content": "Fact"},
                    {
                        "title": "Untrusted",
                        "url": "https://vendor.example.evil.test/test",
                        "content": "Ad",
                    },
                    {"title": "Malformed", "url": "not a URL", "content": "Broken"},
                ]
            },
        )

    search = TavilySearch(
        "test-key", official_domains=["vendor.example"], transport=httpx.MockTransport(handler)
    )
    result = search.search("query")
    assert [d.source_type for d in result] == ["official", "news"]
    assert all("test-key" not in d.model_dump_json() for d in result)


@pytest.mark.parametrize(
    "status, error", [(401, ServiceConfigurationError), (429, ConnectionError)]
)
def test_search_errors_are_normalized_without_exposing_service_body(status, error):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, json={"error": "private"}))
    with pytest.raises(error) as caught:
        TavilySearch("test-key", transport=transport).search("query")
    assert "private" not in str(caught.value) and "test-key" not in str(caught.value)


def test_qwen_is_lazy_and_uses_chat_template_without_thinking():
    class Tokens(list):
        @property
        def shape(self):
            return (1, 3)

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["add_generation_prompt"] is True
            return Inputs(input_ids=Tokens([1, 2, 3]))

        def decode(self, tokens, **kwargs):
            assert tokens == [4, 99]
            return '{"findings": []}'

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 10
            return [[1, 2, 3, 4, 99]]

    from contextlib import nullcontext
    from types import SimpleNamespace

    adapter = TransformersQwen(max_new_tokens=10, max_context_tokens=20)
    assert adapter._model is None
    adapter._tokenizer, adapter._model = Tokenizer(), Model()
    adapter._torch = SimpleNamespace(inference_mode=nullcontext)
    assert adapter.invoke([{"role": "user", "content": "test"}]) == '{"findings": []}'
    adapter.max_context_tokens = 12
    with pytest.raises(ModelOutputError, match="context"):
        adapter.invoke([])
