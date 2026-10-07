"""worker: 지시받은 관점을 검색 도구로 조사하고 finding + source를 돌려줍니다.

순서: ReAct 조사(도구 최대 3회) → 코드 인용 검사 → LLM 자기검토 1회 → 코드 인용 재검사.
실패는 이 과제만 success=False로 기록하고 실행 전체는 계속됩니다.
"""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy

from skala_agent.agents.common import DOMAIN, TECHNOLOGIES, as_json, load_prompt
from skala_agent.checks import check_citations
from skala_agent.llm import get_model
from skala_agent.tools import make_tools
from skala_agent.workflow.state import AGENT_LABELS, WorkerDraft, WorkerResult

logger = logging.getLogger(__name__)
MAX_TOOL_CALLS = 3


def _research(task, payload, registry, resources) -> WorkerDraft:
    agent = create_agent(
        get_model("worker"),
        make_tools(task.id, registry, resources),
        system_prompt=load_prompt("worker") + "\n\n" + load_prompt(f"worker_{task.agent}"),
        middleware=[ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS, exit_behavior="continue")],
        response_format=ProviderStrategy(WorkerDraft),
        name=f"worker-{task.id}",
    )
    message = as_json(
        {
            "instruction": task.instruction,
            "question": payload["question"],
            "technologies": TECHNOLOGIES,
            "domain": DOMAIN,
            "tech_brief": payload["tech_brief"],
            "previous_feedback": payload.get("feedback"),
        }
    )
    output = agent.invoke({"messages": [{"role": "user", "content": message}]})
    return output["structured_response"]


def _self_review(task, draft: WorkerDraft, registry) -> WorkerDraft:
    cited = {f.source_id for f in draft.findings}
    review = get_model("self_review").with_structured_output(WorkerDraft)
    return review.invoke(
        [
            {"role": "system", "content": load_prompt("self_review")},
            {
                "role": "user",
                "content": as_json(
                    {
                        "instruction": task.instruction,
                        "draft": draft.model_dump(),
                        "sources": [
                            s.model_dump(mode="json")
                            for s in registry.values()
                            if s.id in cited or len(registry) <= 9
                        ],
                    }
                ),
            },
        ]
    )


def run(payload, resources):
    task = payload["task"]
    registry = {}
    label = AGENT_LABELS[task.agent]
    try:
        draft = _research(task, payload, registry, resources)
        valid, dropped = check_citations(draft.findings, registry)
        if dropped:
            logger.info("[%s] 원문에 없는 인용 %d건 제거", task.id, len(dropped))
        draft = draft.model_copy(update={"findings": valid})
        try:
            reviewed = _self_review(task, draft, registry)
            checked, _ = check_citations(reviewed.findings, registry)
            # 자기검토가 근거를 잃게 만들었으면 검토 전 결과를 씁니다.
            if checked or not valid:
                draft = reviewed.model_copy(update={"findings": checked})
        except Exception as exc:  # noqa: BLE001 - 자기검토 실패는 검토 전 결과로 진행
            logger.warning("[%s] 자기검토 실패, 검토 전 결과 사용: %s", task.id, exc)
        result = WorkerResult(
            task_id=task.id,
            **{"from": task.agent},
            success=bool(draft.findings),
            findings=draft.findings,
            verdict=draft.verdict,
            error=None if draft.findings else "원문으로 확인된 근거 없음",
        )
    except Exception as exc:  # noqa: BLE001 - 한 과제의 실패가 전체를 멈추지 않게 합니다
        logger.warning("[%s] %s worker 실패: %s", task.id, label, exc)
        result = WorkerResult(
            task_id=task.id,
            **{"from": task.agent},
            success=False,
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
    cited = {f.source_id for f in result.findings}
    logger.info(
        "[%s] %s 완료: success=%s, 근거 %d건", task.id, label, result.success, len(result.findings)
    )
    return {
        "worker_results": {task.id: result},
        "sources": {sid: registry[sid] for sid in cited},
        "node_status": {task.id: "done" if result.success else "failed"},
    }
