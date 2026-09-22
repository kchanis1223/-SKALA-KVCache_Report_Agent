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


def test_all_nine_agents_follow_the_design_assignment():
    settings = ModelSettings()
    assert len(settings.assignment()) == 9
    assert all(
        settings.model_for(agent) == "qwen3:4b"
        for agent in ("research", "additional_search", "trl", "market", "stakeholder", "domain")
    )
    assert settings.model_for("synthesis") == "gpt-5.6-sol"
    assert settings.model_for("validation") == settings.model_for("report") == "gpt-5.6-terra"
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
        assert body["stream"] is False and body["keep_alive"] == "5m"
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


@pytest.mark.parametrize("single, expected", [(False, "qwen3:4b"), (True, "qwen3:4b")])
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
        ModelSettings(use_single_model=single, openai_api_key="test"),
        transport=httpx.MockTransport(handler),
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


@pytest.mark.parametrize(
    "configured, expected",
    [
        (None, "5m"),
        ("10m", "10m"),
        ("0", 0),
        ("-1", -1),
        ("300", 300),
        ("1h30m", "1h30m"),
        ("500ms", "500ms"),
        ("1.5s", "1.5s"),
    ],
)
@pytest.mark.parametrize("single", [False, True])
def test_keep_alive_reaches_every_model_and_request(configured, expected, single):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"done": True, "message": {"content": "{}"}})

    env = {} if configured is None else {"OLLAMA_KEEP_ALIVE": configured}
    settings = ModelSettings.from_environment(
        {**env, "USE_SINGLE_MODEL": str(single).lower(), "OPENAI_API_KEY": "test"}
    )
    router = ModelRouter(settings, transport=httpx.MockTransport(handler))
    ollama_agents = [a for a in AGENTS if isinstance(router.for_agent(a), OllamaChat)]
    for agent in ollama_agents:
        router.for_agent(agent).invoke([])
        router.for_agent(agent).invoke_structured([], {"type": "object"})
    assert len(calls) == 2 * (9 if single else 6)
    assert all(call["keep_alive"] == expected for call in calls)
    assert {call["model"] for call in calls} == {"qwen3:4b"}


@pytest.mark.parametrize(
    "value",
    [
        "",
        "forever",
        "5d",
        "-2",
        -2,
        True,
        False,
        None,
        1.5,
        "NaN",
        "inf",
        "5 m",
        "1e3",
        "9223372037",
        "999999999999h",
        {},
        [],
    ],
)
def test_invalid_keep_alive_is_rejected_before_requests(value):
    with pytest.raises(ValidationError):
        ModelSettings(keep_alive=value)
    with pytest.raises(ValueError):
        OllamaChat("qwen3:4b", keep_alive=value)


def test_keep_alive_environment_precedence(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("OLLAMA_KEEP_ALIVE=1m\n")
    (tmp_path / ".env.local").write_text("OLLAMA_KEEP_ALIVE=2m\n")
    monkeypatch.delenv("OLLAMA_KEEP_ALIVE", raising=False)
    assert ModelSettings.from_environment(read_environment(path)).keep_alive == "2m"
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "0")
    assert ModelSettings.from_environment(read_environment(path)).keep_alive == 0


@pytest.mark.parametrize("model_name", ["qwen3:4b", "qwen3:8b"])
@pytest.mark.parametrize("keep_alive", ["5m", 0])
def test_keep_alive_for_each_ollama_model(model_name, keep_alive):
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == model_name
        assert body["keep_alive"] == keep_alive
        return httpx.Response(200, json={"done": True, "message": {"content": "{}"}})

    model = OllamaChat(model_name, keep_alive=keep_alive, transport=httpx.MockTransport(handler))
    model.invoke([])
    model.invoke_structured([], {"type": "object"})
