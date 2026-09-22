import json

import httpx
import pytest
from pydantic import ValidationError

from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.contracts import ModelOutputError, SearchDocument
from skala_agent.integrations.ollama import OllamaChat
from skala_agent.integrations.structured import StructuredExtractor
from skala_agent.model_config import AGENTS, ModelRouter, ModelSettings, read_environment
from skala_agent.schemas import Technology
from skala_agent.workflow.graph import build_graph, initial_state


def test_all_nine_agents_follow_two_tier_assignment():
    settings = ModelSettings()
    assert len(settings.assignment()) == 9
    assert settings.model_for("research") == settings.model_for("additional_search") == "qwen3:4b"
    assert all(
        settings.model_for(a) == "qwen3:8b"
        for a in AGENTS
        if a not in ("research", "additional_search")
    )
    with pytest.raises(ValueError):
        settings.model_for("typo")


def test_single_model_reuses_same_object_for_every_agent():
    router = ModelRouter(ModelSettings(use_single_model=True))
    models = [router.for_agent(a) for a in AGENTS]
    assert all(m is models[0] for m in models)
    assert models[0].model == "qwen3:4b"


def test_dotenv_boolean_and_environment_precedence(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("USE_SINGLE_MODEL=false\nMAIN_MODEL=qwen3:8b\n")
    monkeypatch.delenv("USE_SINGLE_MODEL", raising=False)
    assert ModelSettings.from_environment(read_environment(path)).use_single_model is False
    monkeypatch.setenv("USE_SINGLE_MODEL", "true")
    assert ModelSettings.from_environment(read_environment(path)).model_for("trl") == "qwen3:4b"
    with pytest.raises(ValidationError):
        ModelSettings.from_environment({"USE_SINGLE_MODEL": "perhaps"})
    with pytest.raises(ValidationError):
        ModelSettings(main_model="qwen3:32b")


def test_native_json_schema_is_sent_to_ollama():
    def handler(request):
        body = json.loads(request.content)
        assert str(request.url) == "http://localhost:11434/api/chat"
        assert body["model"] == "qwen3:8b" and body["think"] is False
        assert body["format"]["properties"]["findings"]
        assert body["stream"] is False and body["keep_alive"] == 0
        return httpx.Response(200, json={"done": True, "message": {"content": '{"findings":[]}'}})

    model = OllamaChat("qwen3:8b", transport=httpx.MockTransport(handler))
    assert StructuredExtractor(model).extract("system", {}).findings == []


@pytest.mark.parametrize(
    "response",
    [
        {"done": False, "message": {"content": "{}"}},
        {"done": True, "done_reason": "length", "message": {"content": "{}"}},
        {"done": True, "message": {"content": ""}},
    ],
)
def test_incomplete_ollama_outputs_are_rejected(response):
    model = OllamaChat(
        "qwen3:4b", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    )
    with pytest.raises(ModelOutputError):
        model.invoke([])


@pytest.mark.parametrize("single, expected", [(False, "qwen3:8b"), (True, "qwen3:4b")])
def test_actual_evaluation_requests_use_selected_model(single, expected):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        payload = json.loads(body["messages"][1]["content"])
        if "claim" in payload:
            return httpx.Response(
                200, json={"done": True, "message": {"content": '{"supports_claim": false}'}}
            )
        output = {
            "findings": [
                {
                    "question_id": q["id"],
                    "answer": "unknown",
                    "rationale": "fixture",
                    "citations": [],
                }
                for q in payload["questions"]
            ]
        }
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps(output)}})

    class Search:
        def search(self, query):
            return [
                SearchDocument(
                    id="fixture",
                    title="fixture",
                    url="https://example.org",
                    content="Synthetic TurboQuant and ITME test document.",
                )
            ]

    router = ModelRouter(
        ModelSettings(use_single_model=single), transport=httpx.MockTransport(handler)
    )
    provider = EvaluationProvider(models=router, search=Search())
    tech = [Technology(id="itme", name="ITME", camp="hw")]
    for perspective in ("trl", "market", "stakeholder", "domain"):
        results, _ = provider.assess(perspective, tech, "datacenter", {}, [])
        assert results[0].status == "pending"
    assert calls == [expected] * 4
    # 전체 graph에서도 8B를 실수로 요청하지 않습니다.
    if single:
        result = build_graph(provider).invoke(initial_state())
        assert result["retry_count"] == 2 and result["report"]
        assert set(calls) == {"qwen3:4b"}


def test_local_environment_overrides_base_and_process_overrides_local(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("TAVILY_API_KEY=base-fixture\nUSE_SINGLE_MODEL=false\n")
    (tmp_path / ".env.local").write_text("TAVILY_API_KEY=local-fixture\n")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    values = read_environment(path)
    assert values["TAVILY_API_KEY"] == "local-fixture"
    assert values["USE_SINGLE_MODEL"] == "false"
    monkeypatch.setenv("TAVILY_API_KEY", "process-fixture")
    assert read_environment(path)["TAVILY_API_KEY"] == "process-fixture"


def test_explicit_environment_file_does_not_load_local(monkeypatch, tmp_path):
    path = tmp_path / "custom.env"
    path.write_text("TAVILY_API_KEY=custom-fixture\n")
    (tmp_path / ".env.local").write_text("TAVILY_API_KEY=local-fixture\n")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert read_environment(path)["TAVILY_API_KEY"] == "custom-fixture"
