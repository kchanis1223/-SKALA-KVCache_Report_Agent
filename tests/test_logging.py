import logging

from skala_agent.providers import DemoProvider
from skala_agent.workflow.graph import build_graph, initial_state

LOGGER = "skala_agent.workflow.graph"


def test_run_logs_fan_out_retry_and_report(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        build_graph(DemoProvider()).invoke(initial_state())
    messages = [r.getMessage() for r in caplog.records]

    assert any("논문 조사 완료" in m for m in messages)
    assert any("관점 fan-out 실행: trl, market, stakeholder, domain" in m for m in messages)
    assert sum("평가 완료" in m for m in messages) == 12  # 4관점 × (최초 1회 + 재평가 2회)
    assert any("관점 재평가(1회차) 실행" in m for m in messages)
    assert any("관점 재평가(2회차) 실행" in m for m in messages)
    assert any("근거 검증 완료 (근거 부족 8건" in m for m in messages)
    assert any("보고서 생성 완료" in m for m in messages)


def test_perspective_failure_is_logged_as_warning(caplog):
    class Failing(DemoProvider):
        def assess(self, perspective, technologies, domain, tech_analysis, evidence):
            if perspective == "domain":
                raise TimeoutError("웹검색 타임아웃")
            return super().assess(perspective, technologies, domain, tech_analysis, evidence)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        build_graph(Failing()).invoke(initial_state())

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("[domain] 평가 실패" in m and "TimeoutError" in m for m in warnings)


def test_default_level_keeps_the_run_quiet(caplog):
    # 기본 실행은 WARNING 이상만 남겨 CI 로그를 채우지 않습니다.
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        build_graph(DemoProvider()).invoke(initial_state())
    assert not caplog.records
