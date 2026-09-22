import pytest

from skala_agent.providers import DemoProvider
from skala_agent.workflow import graph as graph_module
from skala_agent.workflow.graph import ResearchUnavailableError, build_graph, initial_state


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(graph_module, "RESEARCH_BACKOFF_SECONDS", 0)


class FlakyResearch(DemoProvider):
    """`failures`번 실패한 뒤 성공하는 논문 조사."""

    def __init__(self, failures, error=TimeoutError):
        self.failures = failures
        self.error = error
        self.attempts = 0

    def research(self, technologies):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.error("논문 RAG 일시 오류")
        return super().research(technologies)


def test_transient_failure_recovers_and_run_completes():
    provider = FlakyResearch(failures=2)
    result = build_graph(provider).invoke(initial_state())
    assert provider.attempts == 3
    assert result["report"]
    assert set(result["tech_analysis"]) == {"turboquant", "itme"}


def test_persistent_failure_stops_the_run_with_a_clear_reason():
    provider = FlakyResearch(failures=99)
    with pytest.raises(ResearchUnavailableError, match="근거 없는 평가를 만들지 않기 위해"):
        build_graph(provider).invoke(initial_state())
    assert provider.attempts == graph_module.RESEARCH_ATTEMPTS


def test_connection_error_is_retried_too():
    provider = FlakyResearch(failures=1, error=ConnectionError)
    build_graph(provider).invoke(initial_state())
    assert provider.attempts == 2


def test_contract_violation_is_not_retried():
    class BrokenResearch(DemoProvider):
        def __init__(self):
            self.attempts = 0

        def research(self, technologies):
            self.attempts += 1
            return {}, []  # 선택한 기술의 TechAnalysis를 반환하지 않음

    provider = BrokenResearch()
    with pytest.raises(ValueError, match="TechAnalysis를 반환해야"):
        build_graph(provider).invoke(initial_state())
    assert provider.attempts == 1
