"""오케스트레이터 → worker(병렬) → synthesize → validate → (재시도 1회) → report.

판정은 LLM이 하고, 경로와 상한은 코드가 정합니다. 무한 루프는
recursion_limit(실행 설정)로 한 번 더 막습니다.
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from skala_agent.agents import orchestrator, reporter, synthesizer, validator, worker
from skala_agent.checks import route_after_validate
from skala_agent.workflow.state import State

logger = logging.getLogger(__name__)
RECURSION_LIMIT = 15


def dispatch(state):
    """첫 회차는 계획 전체를, 재시도는 부족 판정을 받은 과제만 worker로 보냅니다."""
    verdict = state.get("verdict")
    if state.get("retry_count", 0) > 0 and verdict is not None:
        tasks = [task for task in state["plan"] if task.id in verdict.feedback]
        feedback = verdict.feedback
    else:
        tasks, feedback = state["plan"], {}
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

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", dispatch, ["worker"])
    graph.add_edge("worker", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validate, {"orchestrator": "orchestrator", "report": "report"}
    )
    graph.add_edge("report", END)
    return graph.compile(checkpointer=checkpointer)


def checkpoint_serde():
    """체크포인트에서 되살릴 수 있는 타입을 명시합니다. 목록 밖 타입은 거부됩니다."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    names = ("SubTask", "Source", "Finding", "WorkerResult", "Verdict")
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
