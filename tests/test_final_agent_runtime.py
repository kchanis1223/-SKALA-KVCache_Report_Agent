import json
from types import SimpleNamespace

import pytest
from test_report_llm import _make_state as report_state
from test_synthesis_llm import _make_state as synthesis_state

from skala_agent import cli, runtime
from skala_agent.agents import report, synthesis
from skala_agent.agents.grounding import require_grounding
from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.providers import DemoProvider
from skala_agent.schemas import Assessment, Evidence


class Judge:
    def __init__(self, supported=True):
        self.supported = supported
        self.calls = []

    def invoke_structured(self, messages, schema):
        self.calls.append(messages)
        return json.dumps({"supported": self.supported})


@pytest.mark.parametrize("retries", [0, 2])
def test_real_cli_calls_final_models_once_after_validation(monkeypatch, tmp_path, retries):
    events = []

    class SynthesisModel:
        def invoke_structured(self, messages, schema):
            events.append("synthesis")
            payload = json.loads(messages[1]["content"])
            assert all(e["supports_claim"] for e in payload["evidence"])
            return json.dumps({"findings": []})

    class ReportModel:
        def invoke_structured(self, messages, schema):
            events.append("report")
            return json.dumps(
                {
                    "report": messages[1]["content"].replace(
                        "KV cache 최적화의 SW·HW 접근 비교.", "KV cache의 SW·HW 최적화 접근 비교."
                    )
                }
            )

    class Provider(DemoProvider):
        synthesis_model = SynthesisModel()
        report_model = ReportModel()
        validation_model = Judge()

        def assess(self, perspective, technologies, domain, tech_analysis, evidence):
            sources, assessments = [], []
            for tech in technologies:
                eid = f"{perspective}-{tech.id}"
                sources.append(
                    Evidence(
                        id=eid,
                        technology_id=tech.id,
                        claim="fixture",
                        excerpt="fixture",
                        url=f"https://example.org/{eid}",
                        title="fixture",
                        source_type="paper",
                    )
                )
                assessments.append(
                    Assessment(
                        technology_id=tech.id,
                        perspective=perspective,
                        verdict="fixture",
                        rationale="fixture",
                        status="assessed" if events.count("validate") >= retries else "pending",
                        evidence_ids=[eid],
                    )
                )
            return assessments, sources

        def validate_evidence(self, items):
            events.append("validate")
            return [e.model_copy(update={"supports_claim": True}) for e in items]

    provider = Provider()
    monkeypatch.setattr(runtime, "_load_real_provider", lambda: provider)
    output = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", ["skala-agent", "--mode", "real", "--output", str(output)])
    cli.main()
    assert events == ["validate"] * (retries + 1) + ["synthesis", "report"]
    assert "KV cache의 SW·HW 최적화 접근 비교." in output.read_text()
    assert len(provider.validation_model.calls) == 1


@pytest.mark.parametrize("role", ["synthesis_model", "report_model", "validation_model"])
def test_timeout_wrapper_enforces_final_model_timeout(role):
    from threading import Event

    release = Event()

    class Model:
        def invoke_structured(self, messages, schema):
            release.wait(2)
            return "{}"

    provider = runtime.TimeoutProvider(SimpleNamespace(**{role: Model()}), 0.01)
    try:
        with pytest.raises(TimeoutError):
            getattr(provider, role).invoke_structured([], {})
    finally:
        release.set()


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("판단 보류", "도입 확정"),
        lambda text: text + "\n99% 비용 절감\n",
        lambda text: text.replace("TRL 5 수준", "TRL 9 수준"),
    ],
)
def test_report_rejects_changed_verdict_numbers_and_pending(change):
    class Model:
        def invoke_structured(self, messages, schema):
            return json.dumps({"report": change(messages[1]["content"])})

    with pytest.raises(ModelOutputError):
        report.run(report_state(), SimpleNamespace(report_model=Model(), validation_model=Judge()))


def test_report_rejects_unsupported_prose_even_with_unchanged_structure():
    class Model:
        def invoke_structured(self, messages, schema):
            return json.dumps(
                {"report": messages[1]["content"] + "\n모든 환경에서 비용이 감소합니다.\n"}
            )

    judge = Judge(False)
    with pytest.raises(ModelOutputError, match="범위를 벗어"):
        report.run(report_state(), SimpleNamespace(report_model=Model(), validation_model=judge))
    assert len(judge.calls) == 1


def test_synthesis_drops_summary_unsupported_by_valid_evidence():
    class Model:
        def invoke_structured(self, messages, schema):
            return json.dumps(
                {
                    "findings": [
                        {
                            "technology_id": "turboquant",
                            "question": "resource_cost",
                            "summary": "모든 workload에서 비용 99% 절감",
                            "assessment_refs": [["trl", "turboquant"]],
                            "evidence_ids": ["ev_tq_1"],
                        }
                    ]
                }
            )

    judge = Judge(False)
    result = synthesis.run(
        synthesis_state(), SimpleNamespace(synthesis_model=Model(), validation_model=judge)
    )
    assert not any("99%" in f.summary for f in result["synthesis_findings"])
    reference = json.loads(judge.calls[0][1]["content"])["reference"]
    assert reference[0][0]["excerpt"] == "TurboQuant reduces memory by 20%."


@pytest.mark.parametrize("response", ['{"supported":"true"}', "{}", "{", '{"supported":false}'])
def test_invalid_or_negative_grounding_is_not_accepted(response):
    class Model:
        def invoke_structured(self, messages, schema):
            return response

    with pytest.raises(ModelOutputError):
        require_grounding("source", "candidate", Model())
