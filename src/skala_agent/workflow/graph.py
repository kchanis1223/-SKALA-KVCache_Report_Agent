"""orchestrator → worker(병렬) → synthesize → validate → report → judge.

되돌아가는 길은 세 가지이고 모두 1회로 제한됩니다.
- validate 부족 → orchestrator (부족한 과제 재실행)
- judge 편향·커버리지 미달 → orchestrator (재조사)
- judge 근거 연결·중립성 미달 → report (재작성)
판정은 LLM이 하고, 경로와 상한은 코드가 정합니다. 무한 루프는
recursion_limit(실행 설정)로 한 번 더 막습니다.
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from skala_agent.agents import judge, orchestrator, reporter, synthesizer, validator, worker
from skala_agent.checks import route_after_judge, route_after_validate
from skala_agent.workflow.state import State

logger = logging.getLogger(__name__)
RECURSION_LIMIT = 25  # 최악 경로(재시도 + 재조사 + 재작성)는 18 superstep


def dispatch(state):
    """orchestrator가 to_run에 적은 과제만 worker로 보냅니다."""
    to_run = state.get("to_run", {})
    tasks = [task for task in state["plan"] if task.id in to_run]
    feedback = {k: v for k, v in to_run.items() if v}
    logger.info("worker 실행: %s", ", ".join(t.id for t in tasks))
    return [
        Send(
            "worker",
            {
                "task": task,
                "question": state["question"],
                "tech_brief": state.get("tech_brief", ""),
                "feedback": feedback.get(task.id),
            },
        )
        for task in tasks
    ]


def build_graph(resources, checkpointer=None):
    graph = StateGraph(State)
    graph.add_node("orchestrator", lambda state: orchestrator.run(state, resources))
    graph.add_node("worker", lambda payload: worker.run(payload, resources))
    graph.add_node("synthesize", synthesizer.run)
    graph.add_node("validate", validator.run)
    graph.add_node("report", reporter.run)
    graph.add_node("judge", judge.run)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", dispatch, ["worker"])
    graph.add_edge("worker", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validate, {"orchestrator": "orchestrator", "report": "report"}
    )
    graph.add_edge("report", "judge")
    graph.add_conditional_edges(
        "judge", route_after_judge, {"orchestrator": "orchestrator", "report": "report", END: END}
    )
    return graph.compile(checkpointer=checkpointer)


def checkpoint_serde():
    """체크포인트에서 되살릴 수 있는 타입을 명시합니다. 목록 밖 타입은 거부됩니다."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    names = (
        "SubTask",
        "Source",
        "Finding",
        "WorkerResult",
        "Verdict",
        "Quality",
        "CriterionCheck",
    )
    return JsonPlusSerializer(
        allowed_msgpack_modules=[("skala_agent.workflow.state", name) for name in names]
    )


def run_config(run_id: str) -> dict:
    """thread_id로 재개 지점을 찾고, metadata로 LangSmith 추적과 보고서를 연결합니다."""
    return {
        "configurable": {"thread_id": run_id},
        "recursion_limit": RECURSION_LIMIT,
        "run_name": "kvcache-report",
        "metadata": {"run_id": run_id},
        "tags": [f"run:{run_id}"],
    }
