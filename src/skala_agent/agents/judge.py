"""judge: 완성된 보고서를 4개 품질 항목으로 판정합니다.

판정(통과/미달 + 사유)은 LLM이 하고, 다음 작업(재조사/재작성/종료)과 상한은
checks.judge_action이 코드로 정합니다. 더 할 수 있는 작업이 없는데 미달이면
보고서 한계점에 기록하고 끝냅니다. judge 호출이 실패하면 통과로 처리하고
그 사실을 한계점에 남깁니다.
"""

import logging

from pydantic import BaseModel, ConfigDict

from skala_agent.agents.common import as_json, load_prompt
from skala_agent.checks import add_limitations, judge_action
from skala_agent.llm import get_model
from skala_agent.workflow.state import (
    AGENT_LABELS,
    CRITERION_LABELS,
    CriterionCheck,
    Quality,
)

logger = logging.getLogger(__name__)


class JudgeDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    groundedness: CriterionCheck
    neutrality: CriterionCheck
    bias: CriterionCheck
    coverage: CriterionCheck


def _sources(state) -> list[dict]:
    """보고서의 [번호] 순서대로 출처와, 그 출처를 쓴 finding을 붙입니다."""
    sources = state.get("sources", {})
    findings = [
        f for r in state.get("worker_results", {}).values() if r.success for f in r.findings
    ]
    rows = []
    for number, source_id in enumerate(state.get("cited_ids", []), start=1):
        source = sources.get(source_id)
        if source is None:
            continue
        rows.append(
            {
                "number": number,
                "title": source.title,
                "url": source.url,
                "findings": [
                    {"claim": f.claim, "quote": f.quote}
                    for f in findings
                    if f.source_id == source_id
                ],
            }
        )
    return rows


def _judge(state) -> JudgeDraft:
    messages = [
        {"role": "system", "content": load_prompt("judge")},
        {
            "role": "user",
            "content": as_json(
                {
                    "question": state["question"],
                    "perspectives": AGENT_LABELS,
                    "report": state["report"],
                    "sources": _sources(state),
                }
            ),
        },
    ]
    structured = get_model("judge").with_structured_output(JudgeDraft)
    try:
        return structured.invoke(messages)
    except ValueError as exc:  # 형식 오류는 1회 재요청
        logger.warning("품질 평가 형식 오류, 1회 재요청: %s", type(exc).__name__)
        return structured.invoke(messages)


def run(state):
    rewrites, research = state.get("rewrite_count", 0), state.get("research_count", 0)
    try:
        draft = _judge(state)
    except Exception as exc:  # noqa: BLE001 - 품질 평가 실패는 통과로 보고 기록만 남김
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
        logger.warning("품질 평가 실패, 통과로 처리: %s", error)
        return {
            "quality": Quality(action="done", error=error),
            "report": add_limitations(state["report"], [f"품질 평가를 수행하지 못함 ({error})"]),
            "node_status": {"judge": "fallback"},
        }

    quality = Quality(checks=draft.model_dump())
    failed = quality.failed()
    action = judge_action(failed, rewrites, research)
    quality = quality.model_copy(update={"action": action})
    logger.info(
        "품질 평가: %s → %s",
        "통과" if not failed else "미달(" + ", ".join(CRITERION_LABELS[c] for c in failed) + ")",
        {"done": "종료", "rewrite": "보고서 재작성", "research": "재조사"}[action],
    )
    update = {"quality": quality, "node_status": {"judge": "done"}}
    if action == "rewrite":
        update["rewrite_count"] = rewrites + 1
    elif action == "research":
        update["research_count"] = research + 1
    elif failed:
        update["report"] = add_limitations(
            state["report"],
            [f"품질 평가 미달: {CRITERION_LABELS[c]} — {quality.checks[c].reason}" for c in failed],
        )
    return update
