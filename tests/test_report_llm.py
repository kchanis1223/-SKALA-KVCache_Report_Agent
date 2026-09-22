import json
import os

import pytest

from skala_agent.agents import report
from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.model_config import ModelRouter, ModelSettings, read_environment
from skala_agent.schemas import Assessment, Evidence, MissingEvidence, Technology


class DummyReportModel:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def invoke_structured(self, messages, schema):
        self.calls.append((messages, schema))
        return self.handler(messages)


def _make_state():
    tech1 = Technology(id="turboquant", name="TurboQuant", camp="sw")
    ev1 = Evidence(
        id="ev_tq_1",
        technology_id="turboquant",
        claim="TurboQuant 메모리 절감",
        url="https://example.com/tq",
        title="TQ Title",
        source_type="paper",
        excerpt="TurboQuant excerpt text",
        supports_claim=True,
    )
    analysis_tq = Assessment(
        technology_id="turboquant",
        perspective="trl",
        verdict="TRL 5 수준",
        rationale="실험실 검증 완료",
        status="assessed",
        evidence_ids=["ev_tq_1"],
    )
    return {
        "selected_technologies": [tech1],
        "domain": "datacenter",
        "retry_count": 0,
        "tech_analysis": {},
        "evidence": [ev1],
        "analyses": {"trl": [analysis_tq]},
        "synthesis": [analysis_tq],
        "synthesis_findings": [],
        "missing_evidence": [],
    }


def test_report_llm_refine_valid():
    state = _make_state()

    def handler(messages):
        base_text = messages[1]["content"]
        # 원본 markdown 구조와 인용/목차를 그대로 보존하며 다듬은 문장 생성
        refined = base_text.replace("자동 생성 템플릿입니다.", "검증된 최신 평가 보고서입니다.")
        return json.dumps({"report": refined})

    mock_model = DummyReportModel(handler)

    class Provider:
        report_model = mock_model

    res = report.run(state, Provider())
    assert "report" in res
    assert "검증된 최신 평가 보고서입니다." in res["report"]


def test_report_llm_rejects_altered_headings_or_citations():
    state = _make_state()

    def handler(messages):
        # 목차 ## SUMMARY 제거
        base_text = messages[1]["content"]
        bad_report = base_text.replace("## SUMMARY", "")
        return json.dumps({"report": bad_report})

    mock_model = DummyReportModel(handler)

    class Provider:
        report_model = mock_model

    with pytest.raises(ModelOutputError, match="검증된 목차·인용·참고문헌"):
        report.run(state, Provider())


def test_report_llm_pending_state_preservation():
    state = _make_state()
    # missing evidence 추가
    state["missing_evidence"] = [
        MissingEvidence(
            technology_id="turboquant",
            perspective="market",
            reason="근거 부족",
            kind="missing_source",
            claim="TurboQuant 시장성 근거 부족",
            queries=["query1"],
        )
    ]

    res = report.run(state, provider=None)
    assert "판단 보류" in res["report"]
    assert "turboquant/market: 근거 부족" in res["report"]


def _get_openai_key():
    return os.getenv("OPENAI_API_KEY") or read_environment().get("OPENAI_API_KEY") or ""


@pytest.mark.skipif(not _get_openai_key(), reason="OPENAI_API_KEY가 설정되어 있어야 합니다.")
def test_report_llm_openai_real_api_call():
    state = _make_state()
    settings = ModelSettings(openai_api_key=_get_openai_key())
    router = ModelRouter(settings)

    class RealProvider:
        report_model = router.for_agent("report")

    res = report.run(state, RealProvider())
    assert "report" in res
    assert "## SUMMARY" in res["report"]
    assert "## REFERENCE" in res["report"]
