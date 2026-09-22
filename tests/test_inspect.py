"""그래프 구조·비용 추정이 코드와 어긋나지 않는지 확인."""

import sys

import pytest

from skala_agent import cli
from skala_agent.estimate import estimate_calls
from skala_agent.workflow.graph import MAX_RETRIES, build_graph

EXPECTED_NODES = {
    "__start__",
    "research",
    "evaluate",
    "synthesize",
    "validate",
    "additional_search",
    "report",
    "__end__",
}

EXPECTED_EDGES = {
    ("__start__", "research", False),
    ("research", "evaluate", True),
    ("evaluate", "synthesize", False),
    ("synthesize", "validate", False),
    ("validate", "additional_search", True),
    ("validate", "report", True),
    ("additional_search", "evaluate", True),
    ("report", "__end__", False),
}


def test_compiled_graph_matches_the_documented_flow():
    """README의 Workflow 그림이 실제 그래프와 어긋나면 여기서 잡힙니다."""
    graph = build_graph().get_graph()
    assert set(graph.nodes) == EXPECTED_NODES
    assert {(e.source, e.target, e.conditional) for e in graph.edges} == EXPECTED_EDGES


def test_graph_flag_prints_mermaid(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["skala-agent", "--graph"])
    cli.main()
    out = capsys.readouterr().out
    assert "graph TD;" in out
    assert "research" in out and "additional_search" in out


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # 최선: 논문 조사 1회 + 네 관점 1회씩, 재검색 없음.
        ("best", {"research": 1, "assess": 4, "search_missing": 0}),
        # 최악: 네 관점이 재시도 상한까지 재평가되고 재검색도 그만큼 발생.
        ("worst", {"research": 1, "assess": 4 * (1 + MAX_RETRIES), "search_missing": MAX_RETRIES}),
    ],
)
def test_call_estimate_follows_the_retry_limit(path, expected):
    counts = estimate_calls()[path]
    assert dict(counts) == {k: v for k, v in expected.items() if v}


def test_dry_run_makes_no_external_calls(monkeypatch, capsys, tmp_path):
    output = tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", ["skala-agent", "--dry-run", "--output", str(output)])
    cli.main()
    out = capsys.readouterr().out
    assert "실제 호출 없음" in out
    assert "합계" in out
    assert not output.exists()  # 추정만 하고 보고서는 만들지 않습니다.
