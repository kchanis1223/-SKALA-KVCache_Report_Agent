"""LLM 없이 검사하는 결정론적 규칙."""

import pytest
from langgraph.graph import END, START, StateGraph

from skala_agent.checks import (
    BODY_HEADINGS,
    REFERENCE_HEADING,
    add_limitations,
    attach_references,
    check_citations,
    check_report,
    cited_order,
    default_plan,
    judge_action,
    merge_plan,
    missing_perspectives,
    route_after_judge,
    route_after_validate,
    validate_plan,
)
from skala_agent.workflow.graph import dispatch
from skala_agent.workflow.state import (
    Finding,
    Quality,
    Source,
    State,
    SubTask,
    Verdict,
    WorkerResult,
)


def task(id, agent="market"):
    return SubTask(id=id, agent=agent, instruction="조사")


def source(id, text="TurboQuant compresses the KV cache to 3 bits with no accuracy loss."):
    return Source(id=id, kind="web", title="t", url="https://example.org/" + id, text=text)


# ── 계획 ──
def test_plan_accepts_one_to_six_tasks_with_at_most_two_per_agent():
    plan = [task("m1"), task("m2"), task("d1", "domain"), task("t1", "tech")]
    assert validate_plan(plan) == []


@pytest.mark.parametrize(
    "plan, message",
    [
        ([], "1~6"),
        (
            [
                task(f"x{i}", a)
                for i, a in enumerate(["market", "domain", "tech"] * 2 + ["stakeholder"])
            ],
            "1~6",
        ),
        ([task("m1"), task("m2"), task("m3")], "최대 2번"),
        ([task("m1"), task("m1", "domain")], "중복"),
    ],
)
def test_plan_rule_violations(plan, message):
    assert any(message in error for error in validate_plan(plan))


def test_task_id_must_be_citation_safe():
    with pytest.raises(ValueError):
        SubTask(id="market 1", agent="market", instruction="x")


def test_retry_plan_may_only_redo_existing_tasks_with_same_agent():
    previous = [task("m1"), task("d1", "domain")]
    assert validate_plan([task("m1")], previous) == []
    assert any("기존 과제" in e for e in validate_plan([task("new")], previous))
    assert any("worker를 바꿀" in e for e in validate_plan([task("m1", "tech")], previous))
    assert any("비어" in e for e in validate_plan([], previous))


def test_research_plan_may_add_tasks_only_for_missing_perspectives():
    previous = [task("m1"), task("m2"), task("d1", "domain")]
    ok = [task("m1"), task("s-r1", "stakeholder")]
    assert validate_plan(ok, previous, allow_new_for_missing=True) == []
    assert validate_plan(ok, previous) != []  # validate 재시도에서는 새 과제 금지
    covered = validate_plan([task("m3")], previous, allow_new_for_missing=True)
    assert any("빠진 관점만" in e for e in covered)
    full = [task(f"{a}{i}", a) for a in ("market", "domain", "tech") for i in (1, 2)]
    over = validate_plan([task("s-r1", "stakeholder")], full, allow_new_for_missing=True)
    assert any("1~6" in e for e in over)


def test_merge_plan_replaces_same_id_and_appends_new():
    previous = [task("m1"), task("d1", "domain")]
    redo = SubTask(id="m1", agent="market", instruction="다른 출처")
    merged = merge_plan(previous, [redo, task("s-r1", "stakeholder")])
    assert [(t.id, t.instruction) for t in merged] == [
        ("m1", "다른 출처"),
        ("d1", "조사"),
        ("s-r1", "조사"),
    ]


def test_default_plan_covers_all_perspectives_and_passes_rules():
    plan = default_plan("질문")
    assert validate_plan(plan) == [] and missing_perspectives(plan) == []
    assert missing_perspectives([task("m1")]) == ["domain", "stakeholder", "tech"]


# ── 인용 ──
def test_quotes_must_be_verbatim_substrings_of_a_collected_source():
    sources = {"s1": source("s1")}
    ok = Finding(claim="3비트 압축", source_id="s1", quote="compresses the KV cache to 3 bits")
    spaced = Finding(claim="공백 차이", source_id="s1", quote="compresses  the KV\ncache")
    paraphrase = Finding(claim="의역", source_id="s1", quote="compresses KV caches to 3-bit")
    unknown = Finding(claim="없는 출처", source_id="s9", quote="compresses")
    valid, dropped = check_citations([ok, spaced, paraphrase, unknown], sources)
    assert valid == [ok, spaced] and dropped == [paraphrase, unknown]


def test_long_quote_is_clipped_not_rejected_and_stays_verbatim():
    text = "KV cache " * 200
    finding = Finding(claim="긴 인용", source_id="s1", quote=text)
    assert len(finding.quote) == 500
    valid, _ = check_citations([finding], {"s1": source("s1", text)})
    assert valid == [finding]


# ── 분기 ──
@pytest.mark.parametrize(
    "verdict, retries, expected",
    [
        (Verdict(sufficient=True), 0, "report"),
        (Verdict(sufficient=False, feedback={"m1": "보완"}), 0, "orchestrator"),
        (Verdict(sufficient=False, feedback={"m1": "보완"}), 1, "report"),  # 재시도 1회 소진
        (Verdict(sufficient=False, feedback={}), 0, "report"),  # 다시 할 과제가 없음
        (None, 0, "report"),
    ],
)
def test_route_after_validate(verdict, retries, expected):
    assert route_after_validate({"verdict": verdict, "retry_count": retries}) == expected


def test_validate_does_not_retry_after_judge_research():
    verdict = Verdict(sufficient=False, feedback={"m1": "보완"})
    state = {"verdict": verdict, "retry_count": 0, "research_count": 1}
    assert route_after_validate(state) == "report"


def test_dispatch_sends_only_tasks_in_to_run_with_feedback():
    plan = [task("m1"), task("d1", "domain")]
    first = dispatch({"plan": plan, "question": "q", "to_run": {"m1": "", "d1": ""}})
    assert [(s.arg["task"].id, s.arg["feedback"]) for s in first] == [("m1", None), ("d1", None)]
    retry = dispatch({"plan": plan, "question": "q", "to_run": {"d1": "TTFT 근거 없음"}})
    assert [(s.arg["task"].id, s.arg["feedback"]) for s in retry] == [("d1", "TTFT 근거 없음")]


# ── 품질 평가 ──
@pytest.mark.parametrize(
    "failed, rewrites, research, expected",
    [
        ([], 0, 0, "done"),
        (["neutrality"], 0, 0, "rewrite"),
        (["groundedness"], 1, 0, "done"),  # 재작성 1회 소진
        (["coverage"], 0, 0, "research"),
        (["bias", "neutrality"], 0, 0, "research"),  # 둘 다 미달이면 재조사 우선
        (["bias", "neutrality"], 0, 1, "rewrite"),  # 재조사 소진 → 재작성
        (["bias"], 0, 1, "rewrite"),  # 재조사 못 하면 서술로라도 보완
        (["bias"], 1, 1, "done"),
    ],
)
def test_judge_action(failed, rewrites, research, expected):
    assert judge_action(failed, rewrites, research) == expected


@pytest.mark.parametrize(
    "action, expected",
    [("done", END), ("rewrite", "report"), ("research", "orchestrator")],
)
def test_route_after_judge(action, expected):
    assert route_after_judge({"quality": Quality(action=action)}) == expected
    assert route_after_judge({"quality": None}) == END


# ── 보고서 ──
def body(extra=""):
    return "# 제목\n\n" + "\n\n".join(f"{h}\n내용 [S:s1]" for h in BODY_HEADINGS) + extra


def test_report_check_accepts_valid_body():
    assert check_report(body(), {"s1"}) == []


@pytest.mark.parametrize(
    "text, message",
    [
        (body().replace(BODY_HEADINGS[3], ""), "목차 누락"),
        (body().replace(BODY_HEADINGS[1], "X").replace(BODY_HEADINGS[5], BODY_HEADINGS[1]), "목차"),
        (body(" [S:s9]"), "검증되지 않은 출처"),
        (body(" https://evil.example"), "URL"),
        (body(f"\n{REFERENCE_HEADING}"), "REFERENCE"),
    ],
)
def test_report_check_violations(text, message):
    assert any(message in error for error in check_report(text, {"s1"}))


def test_references_are_numbered_in_order_of_first_citation():
    sources = {"a": source("a"), "b": source("b")}
    report = attach_references("x [S:b] y [S:a] z [S:b]", sources)
    assert report.startswith("x [1] y [2] z [1]")
    assert (
        "- [1] [t](https://example.org/b)" in report
        and "- [2] [t](https://example.org/a)" in report
    )


def test_cited_order_matches_reference_numbering():
    assert cited_order("x [S:b] y [S:a] z [S:b]") == ["b", "a"]


def test_limitations_are_added_before_reference():
    report = attach_references(body(), {"s1": source("s1")})
    amended = add_limitations(report, ["품질 평가 미달: 중립성 — 추천 표현"])
    limits, reference = amended.split(REFERENCE_HEADING)
    assert limits.rstrip().endswith("- 품질 평가 미달: 중립성 — 추천 표현")
    assert reference == report.split(REFERENCE_HEADING)[1]
    assert add_limitations(report, []) == report


# ── 병렬 병합 ──
def test_parallel_workers_merge_without_losing_results():
    """같은 superstep에서 여러 worker가 worker_results에 써도 모두 남는다."""
    graph = StateGraph(State)

    def make(task_id):
        def node(_):
            result = WorkerResult(task_id=task_id, **{"from": "market"}, success=True)
            return {"worker_results": {task_id: result}, "node_status": {task_id: "done"}}

        return node

    for name in ("a", "b", "c"):
        graph.add_node(name, make(name))
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    state = graph.compile().invoke({"worker_results": {}, "node_status": {"orchestrator": "done"}})
    assert set(state["worker_results"]) == {"a", "b", "c"}
    assert state["node_status"] == {"orchestrator": "done", "a": "done", "b": "done", "c": "done"}
