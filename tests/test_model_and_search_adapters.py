import json

import httpx
import pytest

from skala_agent.integrations.contracts import ModelOutputError, ServiceConfigurationError
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
