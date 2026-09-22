import pytest

from skala_agent.providers import DemoProvider
from skala_agent.schemas import Assessment, Evidence
from skala_agent.workflow.graph import AdditionalSearchFailedError, build_graph, initial_state


class SearchFails(DemoProvider):
    """market 관점만 근거가 없어 재검색으로 넘어가고, 그 재검색이 실패하는 provider."""

    def __init__(self, error=ConnectionError("웹검색 연결 실패")):
        self.error = error

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        assessments, sources = [], []
        for tech in technologies:
            eid = f"{perspective}-{tech.id}"
            supported = perspective != "market"
            assessments.append(
                Assessment(
                    technology_id=tech.id,
                    perspective=perspective,
                    verdict="fixture verdict",
                    rationale="test only",
                    status="assessed",
                    evidence_ids=[eid] if supported else [],
                )
            )
            if supported:
                sources.append(
                    Evidence(
                        id=eid,
                        technology_id=tech.id,
                        claim="test only",
                        url=f"https://example.org/{eid}",
                        title="Test fixture",
                        excerpt="Synthetic evidence for testing",
                        source_type="official",
                        supports_claim=True,
                    )
                )
        return assessments, sources

    def search_missing(self, missing):
        raise self.error


def test_failure_stops_the_run_and_names_the_cause():
    with pytest.raises(AdditionalSearchFailedError) as caught:
        build_graph(SearchFails()).invoke(initial_state())

    message = str(caught.value)
    assert "재검색 1회차" in message
    assert "ConnectionError — 웹검색 연결 실패" in message
    assert "대상 관점: market" in message
    assert "네 관점 평가는 완료된 상태" in message
    assert "다시 실행하세요" in message


def test_original_exception_is_preserved_for_debugging():
    with pytest.raises(AdditionalSearchFailedError) as caught:
        build_graph(SearchFails(TimeoutError("검색 API 타임아웃"))).invoke(initial_state())
    assert isinstance(caught.value.__cause__, TimeoutError)


def test_successful_search_is_untouched():
    result = build_graph(DemoProvider()).invoke(initial_state())
    assert result["retry_count"] == 2
    assert result["report"]
