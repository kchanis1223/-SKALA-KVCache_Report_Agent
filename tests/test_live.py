"""실제 OpenAI·Tavily API를 호출하는 테스트. `uv run pytest -m live`로 실행합니다.

키가 없으면 실패합니다(건너뛰지 않음). 키 없이 규칙만 확인하려면
`uv run pytest -m "not live"`를 쓰세요.
"""

import os
import sqlite3
from collections import Counter

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from skala_agent.agents import validator
from skala_agent.checks import BODY_HEADINGS, REFERENCE_HEADING, quote_in_source, validate_plan
from skala_agent.cli import build_resources
from skala_agent.llm import load_env
from skala_agent.workflow.graph import build_graph, checkpoint_serde, run_config
from skala_agent.workflow.state import MAX_RETRIES, initial_state

pytestmark = pytest.mark.live
QUESTION = "데이터센터 LLM 서빙 도입 관점에서 TurboQuant와 ITME의 비용과 SLA 위험을 비교해줘"


@pytest.fixture(scope="module")
def resources():
    load_env()
    for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
        assert os.environ.get(key), f"{key}가 필요합니다"
    return build_resources(os.environ)


@pytest.fixture(scope="module")
def finished(resources, tmp_path_factory):
    """전체 실행 1회. 여러 테스트가 같은 결과를 검사해 API 비용을 아낍니다."""
    path = tmp_path_factory.mktemp("live") / "c.sqlite"
    with sqlite3.connect(path, check_same_thread=False) as conn:
        app = build_graph(resources, checkpointer=SqliteSaver(conn, serde=checkpoint_serde()))
        config = run_config("live-full")
        state = app.invoke(initial_state(QUESTION, "live-full"), config)
        history = list(app.get_state_history(config))
    return state, history


def test_workers_actually_collect_evidence(finished):
    """worker가 모두 실패해도 템플릿 보고서는 나오므로, 근거 수집 자체를 확인합니다."""
    state, _ = finished
    results = state["worker_results"].values()
    errors = [r.error for r in results if not r.success]
    assert any(r.success and r.findings for r in results), f"성공한 worker 없음: {errors}"
    assert state["sources"], "수집된 출처 없음"
    reference = state["report"].split(REFERENCE_HEADING, 1)[1]
    assert "http" in reference, "보고서가 인용한 출처 없음"


def test_report_has_all_sections_and_only_registered_citations(finished):
    state, _ = finished
    report = state["report"]
    positions = [report.find(h) for h in (*BODY_HEADINGS, REFERENCE_HEADING)]
    assert -1 not in positions and positions == sorted(positions)
    assert "[S:" not in report  # 모든 인용이 번호로 바뀜
    reference = report.split(REFERENCE_HEADING, 1)[1]
    for source in state["sources"].values():
        if f"]({source.url})" in reference:
            assert source.url.startswith("http")


def test_plan_follows_rules(finished):
    state, _ = finished
    assert validate_plan(state["plan"]) == []


def test_every_finding_quote_is_in_its_collected_source(finished):
    state, _ = finished
    for result in state["worker_results"].values():
        for finding in result.findings:
            assert quote_in_source(finding.quote, state["sources"].get(finding.source_id))


def test_retry_runs_at_most_once_and_only_for_insufficient_tasks(finished):
    state, history = finished
    assert state["retry_count"] <= MAX_RETRIES
    # 체크포인트 이력의 예약 작업에서 worker 실행 횟수를 셉니다.
    runs = Counter(task.name for snapshot in history for task in snapshot.tasks)["worker"]
    planned = len(state["plan"])
    if state["retry_count"] == 0:
        assert runs == planned
    else:
        assert planned < runs <= planned * 2  # 재시도는 부족한 과제만


def test_run_resumes_after_a_crash(resources, tmp_path, monkeypatch):
    """validate에서 강제로 실패시킨 뒤 같은 run_id로 이어서 완료한다."""
    original = validator.run
    calls = {"n": 0}

    def crash_once(state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("강제 중단")
        return original(state)

    monkeypatch.setattr(validator, "run", crash_once)
    with sqlite3.connect(tmp_path / "c.sqlite", check_same_thread=False) as conn:
        app = build_graph(resources, checkpointer=SqliteSaver(conn, serde=checkpoint_serde()))
        config = run_config("live-resume")
        with pytest.raises(RuntimeError, match="강제 중단"):
            app.invoke(initial_state(QUESTION, "live-resume"), config)
        assert app.get_state(config).next == ("validate",)
        # 재개 후 실행되는 노드를 순서대로 기록합니다.
        order = [
            node
            for chunk in app.stream(None, config, stream_mode="updates")
            for node in chunk
            if not node.startswith("__")
        ]
        state = app.get_state(config).values
    assert state["report"]
    # 끝난 worker를 다시 돌리지 않고 멈춘 지점(validate)부터 이어서 실행해야 합니다.
    # 그 뒤 재시도로 worker가 다시 도는 것은 정상 동작입니다.
    assert order[0] == "validate", order
    assert order[-1] == "report", order
    if state["retry_count"] == 0:
        assert "worker" not in order, order
