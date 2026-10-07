"""validate: LLM이 결과가 질문에 답하기에 충분한지 판정합니다.

다음 경로(재시도/보고서)는 checks.route_after_validate가 정합니다.
"""

import logging

from pydantic import BaseModel, ConfigDict

from skala_agent.agents.common import as_json, load_prompt
from skala_agent.llm import get_model
from skala_agent.workflow.state import MAX_RETRIES, Verdict

logger = logging.getLogger(__name__)


class TaskFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    feedback: str


class VerdictDraft(BaseModel):
    """모델 출력용. dict 대신 목록을 써서 구조화 출력 스키마를 단순하게 둡니다."""

    model_config = ConfigDict(extra="forbid")
    sufficient: bool
    insufficient_tasks: list[TaskFeedback]


def run(state):
    plan_ids = {task.id for task in state["plan"]}
    results = state.get("worker_results", {})
    payload = {
        "question": state["question"],
        # judge 재조사 뒤에는 다시 돌리지 않으므로 남은 재시도는 0입니다.
        "retries_left": 0
        if state.get("research_count", 0)
        else MAX_RETRIES - state.get("retry_count", 0),
        "tasks": [
            {
                "id": task.id,
                "agent": task.agent,
                "instruction": task.instruction,
                "success": results[task.id].success if task.id in results else False,
                "error": results[task.id].error if task.id in results else "미실행",
                "verdict": results[task.id].verdict if task.id in results else "",
                "claims": [f.claim for f in results[task.id].findings]
                if task.id in results
                else [],
            }
            for task in state["plan"]
        ],
    }
    draft = (
        get_model("validator")
        .with_structured_output(VerdictDraft)
        .invoke(
            [
                {"role": "system", "content": load_prompt("validator")},
                {"role": "user", "content": as_json(payload)},
            ]
        )
    )
    # 존재하지 않는 과제 id에 대한 피드백은 버립니다.
    feedback = {i.task_id: i.feedback for i in draft.insufficient_tasks if i.task_id in plan_ids}
    verdict = Verdict(sufficient=draft.sufficient, feedback=feedback)
    logger.info(
        "검증: %s%s",
        "충분" if verdict.sufficient else "부족",
        f" (보완 대상: {', '.join(feedback)})" if feedback else "",
    )
    return {"verdict": verdict, "node_status": {"validate": "done"}}
