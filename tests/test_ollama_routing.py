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


def test_all_nine_agents_use_one_openai_model_by_default():
    """기본 경로는 OpenAI 단일 모델이고, 역할 구분은 추론 강도로만 합니다."""
    from skala_agent.model_config import DEFAULT_OPENAI_MODEL

    settings = ModelSettings()
    assert len(settings.assignment()) == 9
    assert set(settings.assignment().values()) == {DEFAULT_OPENAI_MODEL}
    assert settings.effort_for("synthesis") == "high"
    assert settings.effort_for("trl") == "medium"
    assert settings.effort_for("validation") == "low"
    with pytest.raises(ValueError):
        settings.model_for("typo")
    with pytest.raises(ValueError):
        settings.effort_for("typo")


def test_ollama_provider_keeps_the_local_assignment():
    """provider=ollama로 되돌리면 기존 로컬 배정이 그대로 유지됩니다."""
    settings = ModelSettings(provider="ollama")
    assert all(
        settings.model_for(agent) == "qwen3:4b"
        for agent in ("research", "additional_search", "trl", "market", "stakeholder", "domain")
    )
    assert settings.model_for("synthesis") == "qwen3:4b"
    assert ModelSettings(provider="ollama", main_model="qwen3:8b").model_for("synthesis") == (
        "qwen3:8b"
    )


def test_ollama_router_serializes_per_model_not_globally():
    """로컬 추론은 직렬화가 필요하지만 lock은 모델 단위여야 합니다.

    전체에 lock 하나만 두면 서로 다른 모델끼리도 줄을 서서, 관점 fan-out이
    모델 수와 무관하게 완전히 순차 실행됩니다.
    """
    router = ModelRouter(ModelSettings(provider="ollama", main_model="qwen3:8b"))
    light = [router.for_agent(a) for a in ("trl", "market", "stakeholder", "domain")]
    heavy = router.for_agent("synthesis")

    assert {m.model for m in light} == {"qwen3:4b"}
    assert heavy.model == "qwen3:8b"
    # 같은 모델끼리는 lock 공유, 다른 모델과는 분리
    assert all(m._lock is light[0]._lock for m in light)
    assert heavy._lock is not light[0]._lock


def test_dotenv_boolean_and_environment_precedence(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("USE_SINGLE_MODEL=false\nLLM_PROVIDER=ollama\nMAIN_MODEL=qwen3:8b\n")
    monkeypatch.delenv("USE_SINGLE_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MAIN_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    settings = ModelSettings.from_environment(read_environment(path))
    assert settings.use_single_model is False
    assert settings.model_for("synthesis") == "qwen3:8b"
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini-override")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert (
        ModelSettings.from_environment(read_environment(path)).model_for("trl")
        == "gpt-5.4-mini-override"
    )
    with pytest.raises(ValidationError):
        ModelSettings.from_environment({"USE_SINGLE_MODEL": "perhaps"})
    with pytest.raises(ValidationError):
        ModelSettings(main_model="qwen3:32b")
    with pytest.raises(ValidationError):
        ModelSettings(provider="anthropic")


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


@pytest.mark.parametrize(
    "main_model, expected", [("qwen3:4b", "qwen3:4b"), ("qwen3:8b", "qwen3:4b")]
)
def test_actual_evaluation_requests_use_selected_model(main_model, expected):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        payload = json.loads(body["messages"][1]["content"])
        if "evidence" in payload:
            return httpx.Response(
                200,
                json={
                    "done": True,
                    "message": {
                        "content": json.dumps(
                            {
                                "results": [
                                    {"id": e["id"], "supports_claim": False}
                                    for e in payload["evidence"]
                                ]
                            }
                        )
                    },
                },
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
        # use_single_model=True는 종합·보고서에 전용 모델 객체를 두지 않습니다.
        # 이 fixture handler는 관점 평가와 근거 검증 payload만 처리합니다.
        ModelSettings(
            provider="ollama",
            main_model=main_model,
            use_single_model=True,
            openai_api_key="test",
        ),
        transport=httpx.MockTransport(handler),
    )
    provider = EvaluationProvider(models=router, search=Search())
    tech = [Technology(id="itme", name="ITME", camp="hw")]
    for perspective in ("trl", "market", "stakeholder", "domain"):
        results, _ = provider.assess(perspective, tech, "datacenter", {}, [])
        assert results[0].status == "pending"
    assert calls == [expected] * 4
    # 관점 평가는 main_model 설정과 무관하게 light_model만 요청합니다.
    if main_model == "qwen3:4b":
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
@pytest.mark.parametrize("main_model", ["qwen3:4b", "qwen3:8b"])
def test_keep_alive_reaches_every_model_and_request(configured, expected, main_model):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"done": True, "message": {"content": "{}"}})

    env = {} if configured is None else {"OLLAMA_KEEP_ALIVE": configured}
    settings = ModelSettings.from_environment(
        {**env, "LLM_PROVIDER": "ollama", "MAIN_MODEL": main_model}
    )
    router = ModelRouter(settings, transport=httpx.MockTransport(handler))
    for agent in AGENTS:
        assert isinstance(router.for_agent(agent), OllamaChat)
        router.for_agent(agent).invoke([])
        router.for_agent(agent).invoke_structured([], {"type": "object"})
    assert len(calls) == 2 * len(AGENTS)
    assert all(call["keep_alive"] == expected for call in calls)
    assert {call["model"] for call in calls} == {"qwen3:4b", main_model}


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
@pytest.mark.parametrize("keep_alive, expected", [("5m", "5m"), (0, 0), (" 0 ", 0)])
def test_keep_alive_for_each_ollama_model(model_name, keep_alive, expected):
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == model_name
        assert body["keep_alive"] == expected
        return httpx.Response(200, json={"done": True, "message": {"content": "{}"}})

    model = OllamaChat(model_name, keep_alive=keep_alive, transport=httpx.MockTransport(handler))
    model.invoke([])
    model.invoke_structured([], {"type": "object"})


@pytest.mark.parametrize("provider_name", ["ollama", "openai"])
@pytest.mark.parametrize("skip_final_models", [False, True])
def test_provider_final_models_follow_compatibility_setting(provider_name, skip_final_models):
    settings = ModelSettings.from_environment(
        {
            "LLM_PROVIDER": provider_name,
            "OPENAI_API_KEY": "test",
            "USE_SINGLE_MODEL": str(skip_final_models).lower(),
        }
    )
    router = ModelRouter(settings)
    provider = EvaluationProvider(models=router, search=object())
    assert provider.synthesis_model is (
        None if skip_final_models else router.for_agent("synthesis")
    )
    assert provider.report_model is (None if skip_final_models else router.for_agent("report"))
    assert settings.assignment() == ModelSettings(provider=provider_name).assignment()
