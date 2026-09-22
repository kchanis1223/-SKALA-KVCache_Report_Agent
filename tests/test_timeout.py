import time

import pytest

from skala_agent.providers import DemoProvider
from skala_agent.runtime import TimeoutProvider, call_with_timeout, load_provider
from skala_agent.workflow.graph import build_graph, initial_state


class SlowProvider(DemoProvider):
    """domain 관점만 상한을 넘기는 provider."""

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        if perspective == "domain":
            time.sleep(5)
        return super().assess(perspective, technologies, domain, tech_analysis, evidence)


def test_call_with_timeout_raises_on_slow_call():
    with pytest.raises(TimeoutError, match="0.05초를 초과"):
        call_with_timeout(time.sleep, 0.05, 5)


def test_call_with_timeout_passes_through_result_and_errors():
    assert call_with_timeout(lambda x: x * 2, 5, 21) == 42

    def boom():
        raise ConnectionError("웹검색 연결 실패")

    # 상한과 무관한 예외는 TimeoutError로 바꾸지 않고 그대로 전달합니다.
    with pytest.raises(ConnectionError, match="웹검색 연결 실패"):
        call_with_timeout(boom, 5)


def test_zero_timeout_leaves_provider_unwrapped():
    assert isinstance(load_provider("demo", timeout=0), DemoProvider)
    assert isinstance(load_provider("demo", timeout=None), DemoProvider)
    assert isinstance(load_provider("demo", timeout=30), TimeoutProvider)


def test_slow_perspective_fails_alone_and_report_is_still_produced():
    result = build_graph(TimeoutProvider(SlowProvider(), 0.2)).invoke(initial_state())

    domain = [a for a in result["synthesis"] if a.perspective == "domain"]
    assert domain and all(a.status == "failed" for a in domain)
    assert all(a.error.code == "TimeoutError" for a in domain)
    assert all(a.verdict == "판단 보류" for a in domain)

    # 나머지 관점은 상한 안에서 끝나므로 실패로 번지지 않습니다.
    others = [a for a in result["synthesis"] if a.perspective != "domain"]
    assert others and all(a.status == "pending" for a in others)

    assert "TimeoutError" in result["report"]


def test_stuck_perspective_is_skipped_after_repeated_timeouts():
    """응답하지 않는 관점에 상한 시간을 반복해서 버리지 않습니다."""
    provider = TimeoutProvider(SlowProvider(), 0.3, max_abandoned=2)
    start = time.monotonic()
    result = build_graph(provider).invoke(initial_state())
    elapsed = time.monotonic() - start

    # domain 은 0.3초 × 2회만 기다리고, 3회차는 즉시 실패합니다.
    assert provider.abandoned["domain"] == 2
    assert elapsed < 0.3 * 3

    # 나머지 관점은 차단되지 않습니다.
    assert all(provider.abandoned[key] == 0 for key in ("trl", "market", "stakeholder"))
    assert result["report"]
