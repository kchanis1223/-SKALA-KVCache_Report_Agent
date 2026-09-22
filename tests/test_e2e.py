from collections import Counter

import pytest

from skala_agent.providers import DemoProvider
from skala_agent.schemas import Assessment, Evidence
from skala_agent.workflow.graph import (
    AdditionalSearchFailedError,
    ResearchUnavailableError,
    build_graph,
    initial_state,
)


class ScenarioProvider(DemoProvider):
    """네트워크 없이 START부터 report까지 검증하는 fixture provider."""

    def __init__(self, *, missing_market=False, recover=True, wrong_technology=False):
        self.calls = Counter()
        self.searches = 0
        self.missing_market = missing_market
        self.recover = recover
        self.wrong_technology = wrong_technology

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        self.calls[perspective] += 1
        assessments, sources = [], []
        for technology in technologies:
            evidence_id = f"{perspective}-{technology.id}"
            missing = (
                perspective == "market"
                and self.missing_market
                and not (self.searches and self.recover)
            )
            assessments.append(
                Assessment(
                    technology_id=technology.id,
                    perspective=perspective,
                    verdict="fixture verdict",
                    rationale="synthetic fixture only",
                    status="assessed",
                    evidence_ids=[] if missing else [evidence_id],
                )
            )
            if not missing:
                sources.append(
                    Evidence(
                        id=evidence_id,
                        technology_id="other" if self.wrong_technology else technology.id,
                        claim="synthetic fixture only",
                        url=f"https://example.test/{evidence_id}",
                        title="Synthetic fixture",
                        excerpt="Synthetic evidence for an offline test.",
                        source_type="official",
                        supports_claim=True,
                    )
                )
        return assessments, sources

    def search_missing(self, missing):
        self.searches += 1
        return []


def test_e2e_fixture_run_reaches_report_without_retry():
    result = build_graph(ScenarioProvider()).invoke(initial_state())

    assert result["retry_count"] == 0
    assert not result["missing_evidence"]
    assert "https://example.test/trl-turboquant" in result["report"]


def test_e2e_retries_only_missing_perspective_then_recovers():
    provider = ScenarioProvider(missing_market=True)

    result = build_graph(provider).invoke(initial_state())

    assert provider.calls == {"trl": 1, "market": 2, "stakeholder": 1, "domain": 1}
    assert result["retry_count"] == provider.searches == 1
    assert not result["missing_evidence"]


def test_e2e_retry_cap_withholds_unsupported_verdict():
    result = build_graph(ScenarioProvider(missing_market=True, recover=False)).invoke(
        initial_state()
    )

    assert result["retry_count"] == 2
    assert {item.perspective for item in result["missing_evidence"]} == {"market"}
    assert "turboquant / market: 판단 보류" in result["report"]


def test_e2e_wrong_technology_evidence_is_not_cited():
    result = build_graph(ScenarioProvider(wrong_technology=True)).invoke(initial_state())

    assert result["retry_count"] == 2
    assert "https://example.test/trl-turboquant" not in result["report"]
    assert "검증된 인용 출처 없음" in result["report"]


def test_e2e_provider_errors_stop_with_context(monkeypatch):
    class SearchErrorProvider(ScenarioProvider):
        def search_missing(self, missing):
            raise ConnectionError("fixture search unavailable")

    with pytest.raises(AdditionalSearchFailedError, match="fixture search unavailable"):
        build_graph(SearchErrorProvider(missing_market=True)).invoke(initial_state())

    class ResearchErrorProvider(ScenarioProvider):
        def research(self, technologies):
            raise ConnectionError("fixture research unavailable")

    monkeypatch.setattr("skala_agent.workflow.graph.RESEARCH_BACKOFF_SECONDS", 0)
    with pytest.raises(ResearchUnavailableError, match="근거 없는 평가를 만들지 않기 위해"):
        build_graph(ResearchErrorProvider()).invoke(initial_state())
