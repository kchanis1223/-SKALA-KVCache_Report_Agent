"""오케스트레이터: 논문으로 기술을 파악하고, 필요한 worker와 지시를 정합니다.

첫 호출은 전체 계획을, 재시도 호출은 validate가 부족하다고 한 과제만 다시
계획합니다. 계획 규칙은 LLM이 아니라 checks.validate_plan이 강제합니다.
"""

import logging

from pydantic import BaseModel, ConfigDict, Field

from skala_agent.agents.common import DOMAIN, TECHNOLOGIES, as_json, load_prompt
from skala_agent.checks import default_plan, validate_plan
from skala_agent.llm import get_model
from skala_agent.workflow.state import SubTask

logger = logging.getLogger(__name__)
BRIEF_QUERIES = ("핵심 원리와 접근 방식", "실험 결과와 측정 조건", "한계와 적용 전제")


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tech_brief: str = Field(min_length=1, max_length=4000)
    tasks: list[SubTask]


class RetryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[SubTask]


def paper_context(resources) -> str:
    """기술별 primary 논문 발췌. 색인이 없으면 그 사실을 그대로 전달합니다."""
    if resources.retriever is None:
        return f"논문 색인 없음 ({resources.retriever_error or '색인 미생성'})."
    blocks = []
    for tech_id in TECHNOLOGIES:
        for query in BRIEF_QUERIES:
            try:
                results = resources.retriever.retrieve(
                    f"{tech_id} {query}", top_k=1, role="primary", paper_id=tech_id
                )
            except Exception as exc:  # noqa: BLE001 - 임베딩 모델 미설치 등
                logger.warning("논문 검색 실패, 논문 없이 계획: %s", exc)
                return f"논문 검색 실패 ({type(exc).__name__})."
            for result in results:
                chunk = result.chunk
                blocks.append(
                    f"[{chunk.paper_id} p.{chunk.page} {chunk.section}]\n{chunk.text[:1200]}"
                )
    return "\n\n".join(dict.fromkeys(blocks)) or "논문 검색 결과 없음."


def _attempt(structured, messages, check):
    try:
        output = structured.invoke(messages)
    except ValueError as exc:  # 스키마 검증 실패(OutputParserException 포함)
        return None, [f"출력 형식 오류: {type(exc).__name__}"]
    return output, check(output)


def _ask(model, schema, messages, check):
    """구조화 출력을 받고 규칙 위반이면 위반 내용을 알려 1회 다시 요청합니다."""
    structured = model.with_structured_output(schema)
    output, errors = _attempt(structured, messages, check)
    if not errors:
        return output, []
    logger.warning("계획 규칙 위반, 1회 재요청: %s", errors)
    retry = messages + [
        {"role": "user", "content": "다음 규칙 위반을 고쳐 다시 출력하라: " + " / ".join(errors)}
    ]
    output, errors = _attempt(structured, retry, check)
    return output, errors


def plan(state, resources):
    messages = [
        {"role": "system", "content": load_prompt("orchestrator")},
        {
            "role": "user",
            "content": as_json(
                {
                    "question": state["question"],
                    "technologies": TECHNOLOGIES,
                    "domain": DOMAIN,
                    "paper_excerpts": paper_context(resources),
                }
            ),
        },
    ]
    output, errors = _ask(
        get_model("orchestrator"), Plan, messages, lambda o: validate_plan(o.tasks)
    )
    tasks = output.tasks if output else []
    if errors:
        logger.warning("계획 규칙을 계속 어겨 기본 계획을 씁니다: %s", errors)
        tasks = default_plan(state["question"])
    brief = (
        output.tech_brief if output else "기술 개요 생성 실패. " + " / ".join(TECHNOLOGIES.values())
    )
    logger.info("계획 %d개: %s", len(tasks), ", ".join(f"{t.id}({t.agent})" for t in tasks))
    return {
        "tech_brief": brief,
        "plan": tasks,
        "node_status": {"orchestrator": "done", **{t.id: "pending" for t in tasks}},
    }


def replan(state):
    previous = state["plan"]
    feedback = state["verdict"].feedback
    targets = [task for task in previous if task.id in feedback]
    messages = [
        {"role": "system", "content": load_prompt("orchestrator_retry")},
        {
            "role": "user",
            "content": as_json(
                {
                    "question": state["question"],
                    "tasks_to_redo": [
                        {**task.model_dump(), "feedback": feedback[task.id]} for task in targets
                    ],
                }
            ),
        },
    ]

    def check(output):
        errors = validate_plan(output.tasks, previous)
        extra = {t.id for t in output.tasks} - set(feedback)
        if extra:
            errors.append(f"부족 판정을 받지 않은 과제는 다시 하지 않는다: {', '.join(extra)}")
        return errors

    output, errors = _ask(get_model("orchestrator"), RetryPlan, messages, check)
    redo = output.tasks if output else []
    if errors:
        # 지시만 보완해서 같은 worker에게 다시 맡깁니다.
        redo = [
            task.model_copy(
                update={"instruction": f"{task.instruction}\n보완 요청: {feedback[task.id]}"}
            )
            for task in targets
        ]
    updated = {task.id: task for task in redo}
    logger.info("재계획 (%d개): %s", len(redo), ", ".join(updated))
    return {
        "plan": [updated.get(task.id, task) for task in previous],
        "retry_count": state["retry_count"] + 1,
        "node_status": {"orchestrator": "done", **{t.id: "pending" for t in redo}},
    }


def run(state, resources):
    if state.get("verdict") is not None and state["verdict"].feedback:
        return replan(state)
    return plan(state, resources)
