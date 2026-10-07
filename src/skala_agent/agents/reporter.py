"""report: LLM이 8개 목차로 보고서를 쓰고, 인용과 목차는 코드가 검사합니다.

검사를 두 번 통과하지 못하면 템플릿 보고서로 대체합니다. 어느 쪽이든
REFERENCE는 실제로 인용된 출처만으로 코드가 만듭니다. judge가 미달 판정을
남겼으면(재작성 또는 재조사 후) 그 지적을 반영해 다시 씁니다.
"""

import logging

from pydantic import BaseModel

from skala_agent.agents.common import DOMAIN, TECHNOLOGIES, as_json, load_prompt
from skala_agent.checks import (
    BODY_HEADINGS,
    attach_references,
    check_report,
    cited_order,
    missing_perspectives,
)
from skala_agent.llm import get_model
from skala_agent.workflow.state import AGENT_LABELS, CRITERION_LABELS

logger = logging.getLogger(__name__)


class ReportDraft(BaseModel):
    markdown: str


def _evidence(state):
    """보고서가 쓸 수 있는 근거: 성공한 worker의 finding과 그 출처."""
    rows = []
    for task in state["plan"]:
        result = state.get("worker_results", {}).get(task.id)
        if result is None:
            continue
        rows.append(
            {
                "task_id": task.id,
                "perspective": AGENT_LABELS[task.agent],
                "success": result.success,
                "verdict": result.verdict,
                "error": result.error,
                "findings": [f.model_dump() for f in result.findings],
            }
        )
    return rows


def _limitations(state) -> list[str]:
    notes = []
    missing = [AGENT_LABELS[a] for a in missing_perspectives(state["plan"])]
    if missing:
        notes.append(f"미조사 관점: {', '.join(missing)} (오케스트레이터가 계획에서 제외)")
    failed = [r.task_id for r in state.get("worker_results", {}).values() if not r.success]
    if failed:
        notes.append(f"근거를 확보하지 못한 과제: {', '.join(failed)}")
    verdict = state.get("verdict")
    if verdict is not None and not verdict.sufficient:
        notes.append(f"재시도 {state.get('retry_count', 0)}회 후에도 검증 단계가 부족하다고 판정함")
    return notes


def template_report(state, reason: str) -> str:
    lines = [
        "# KV cache 기술 비교 보고서",
        "",
        f"> run_id: {state['run_id']} · LLM 보고서 검사 실패로 템플릿 보고서를 생성했습니다"
        f" ({reason}).",
        "",
        BODY_HEADINGS[0],
        "",
        f"질문: {state['question']}",
        "",
        BODY_HEADINGS[1],
        "",
        f"평가 도메인: {DOMAIN}",
        "",
        BODY_HEADINGS[2],
        "",
        *[f"- {text}" for text in TECHNOLOGIES.values()],
        "",
        BODY_HEADINGS[3],
        "",
        state.get("tech_brief", ""),
        "",
        BODY_HEADINGS[4],
        "",
    ]
    for row in _evidence(state):
        lines.append(f"### {row['perspective']} ({row['task_id']})")
        lines.append("")
        if not row["success"]:
            lines.append(f"판단 보류: {row['error']}")
        else:
            lines.append(row["verdict"])
            lines.extend(f"- {f['claim']} [S:{f['source_id']}]" for f in row["findings"])
        lines.append("")
    lines += [BODY_HEADINGS[5], "", "관점별 결과를 종합한 서술은 생성하지 못했습니다.", ""]
    lines += [BODY_HEADINGS[6], ""] + [f"- {n}" for n in _limitations(state) or ["없음"]]
    return "\n".join(lines)


def _quality_feedback(state) -> dict[str, str]:
    quality = state.get("quality")
    if quality is None:
        return {}
    return {CRITERION_LABELS[c]: quality.checks[c].reason for c in quality.failed()}


def run(state):
    sources = state.get("sources", {})
    allowed = {
        f.source_id
        for r in state.get("worker_results", {}).values()
        if r.success
        for f in r.findings
        if f.source_id in sources
    }
    messages = [
        {"role": "system", "content": load_prompt("reporter")},
        {
            "role": "user",
            "content": as_json(
                {
                    "run_id": state["run_id"],
                    "question": state["question"],
                    "technologies": TECHNOLOGIES,
                    "domain": DOMAIN,
                    "tech_brief": state.get("tech_brief", ""),
                    "evidence": _evidence(state),
                    "limitations": _limitations(state),
                    "headings": list(BODY_HEADINGS),
                    "previous_review": _quality_feedback(state),
                }
            ),
        },
    ]
    writer = get_model("reporter").with_structured_output(ReportDraft)
    body, errors = None, ["생성 전"]
    for attempt in range(2):
        try:
            body = writer.invoke(messages).markdown
            errors = check_report(body, allowed)
        except ValueError as exc:
            errors = [f"출력 형식 오류: {type(exc).__name__}"]
        if not errors:
            break
        logger.warning("보고서 검사 실패 (%d/2): %s", attempt + 1, errors)
        messages = messages + [
            {"role": "user", "content": "다음 문제를 고쳐 다시 작성하라: " + " / ".join(errors)}
        ]
    if errors:
        body = template_report(state, errors[0])
    report = attach_references(body, sources)
    return {
        "report": report,
        "cited_ids": cited_order(body),
        "node_status": {"report": "done" if not errors else "fallback"},
    }
