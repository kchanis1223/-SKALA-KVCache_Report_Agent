"""LLM을 부르지 않는 결정론적 규칙. 판정은 LLM이 하더라도 경계는 코드가 지킵니다."""

import re
from collections import Counter

from skala_agent.workflow.state import (
    AGENTS,
    MAX_RETRIES,
    Finding,
    Source,
    SubTask,
)

MAX_TASKS = 6
MAX_PER_AGENT = 2

# 보고서 본문 목차. REFERENCE는 코드가 인용 목록으로 만들어 붙입니다.
BODY_HEADINGS = (
    "## SUMMARY",
    "## 1. 분석 배경",
    "## 2. 비교 기술 선정",
    "## 3. 기술 개요",
    "## 4. 관점별 평가",
    "## 5. 종합 비교 및 시사점",
    "## 6. 한계점",
)
REFERENCE_HEADING = "## REFERENCE"
CITATION = re.compile(r"\[S:([A-Za-z0-9_\-]+)\]")
URL = re.compile(r"https?://\S+")


# ── 계획 ──────────────────────────────────────────────────────────────
def validate_plan(plan: list[SubTask], previous: list[SubTask] | None = None) -> list[str]:
    """계획 규칙 위반 목록을 돌려줍니다. 비어 있으면 통과입니다.

    previous가 있으면 재시도 계획입니다. 재시도는 기존 과제 id만 다시 지시할 수
    있고, 맡은 worker를 바꿀 수 없습니다.
    """
    errors = []
    ids = [task.id for task in plan]
    if len(ids) != len(set(ids)):
        errors.append("과제 id가 중복되었습니다.")
    if previous is None:
        if not 1 <= len(plan) <= MAX_TASKS:
            errors.append(f"과제는 1~{MAX_TASKS}개여야 합니다 (현재 {len(plan)}개).")
        over = [a for a, n in Counter(t.agent for t in plan).items() if n > MAX_PER_AGENT]
        if over:
            errors.append(f"같은 worker는 최대 {MAX_PER_AGENT}번입니다: {', '.join(over)}")
        return errors
    original = {task.id: task for task in previous}
    if not plan:
        errors.append("재시도 계획이 비어 있습니다.")
    for task in plan:
        if task.id not in original:
            errors.append(f"재시도는 기존 과제만 가능합니다: {task.id}")
        elif task.agent != original[task.id].agent:
            errors.append(f"재시도에서 worker를 바꿀 수 없습니다: {task.id}")
    return errors


def default_plan(question: str) -> list[SubTask]:
    """LLM 계획이 규칙을 계속 어길 때 쓰는 기본 계획: 4관점 각 1개."""
    return [
        SubTask(
            id=f"{agent}-1",
            agent=agent,
            instruction=f"다음 질문을 이 관점에서 조사하라: {question}",
        )
        for agent in AGENTS
    ]


def missing_perspectives(plan: list[SubTask]) -> list[str]:
    planned = {task.agent for task in plan}
    return [agent for agent in AGENTS if agent not in planned]


# ── 인용 ──────────────────────────────────────────────────────────────
def _squash(text: str) -> str:
    return " ".join(text.split())


def quote_in_source(quote: str, source: Source | None) -> bool:
    """quote가 원문의 연속된 부분인지 확인합니다. 공백 차이만 허용합니다."""
    if source is None or not quote.strip():
        return False
    return quote in source.text or _squash(quote) in _squash(source.text)


def check_citations(
    findings: list[Finding], sources: dict[str, Source]
) -> tuple[list[Finding], list[Finding]]:
    """(유효한 finding, 버린 finding). 원문에 없는 인용은 쓰지 않습니다."""
    valid, dropped = [], []
    for finding in findings:
        target = (
            valid if quote_in_source(finding.quote, sources.get(finding.source_id)) else dropped
        )
        target.append(finding)
    return valid, dropped


# ── 분기 ──────────────────────────────────────────────────────────────
def route_after_validate(state) -> str:
    """판정은 LLM이 하고, 재시도 상한은 코드가 강제합니다."""
    verdict = state.get("verdict")
    if verdict is None or verdict.sufficient or state.get("retry_count", 0) >= MAX_RETRIES:
        return "report"
    if not verdict.feedback:
        return "report"
    return "orchestrator"


# ── 보고서 ────────────────────────────────────────────────────────────
def check_report(text: str, allowed_ids: set[str]) -> list[str]:
    """LLM 보고서 본문의 위반 목록. 비어 있으면 통과입니다."""
    errors = []
    positions = [text.find(heading) for heading in BODY_HEADINGS]
    if -1 in positions:
        missing = [h for h, p in zip(BODY_HEADINGS, positions, strict=True) if p == -1]
        errors.append(f"목차 누락: {', '.join(missing)}")
    elif positions != sorted(positions):
        errors.append("목차 순서가 다릅니다.")
    unknown = sorted(set(CITATION.findall(text)) - allowed_ids)
    if unknown:
        errors.append(f"검증되지 않은 출처 인용: {', '.join(unknown)}")
    if URL.search(text):
        errors.append("본문에 URL을 직접 쓸 수 없습니다. [S:id]로만 인용하세요.")
    if REFERENCE_HEADING in text:
        errors.append("REFERENCE는 코드가 생성합니다.")
    return errors


def attach_references(text: str, sources: dict[str, Source]) -> str:
    """[S:id]를 등장 순서대로 [1], [2]…로 바꾸고 REFERENCE를 붙입니다."""
    order: list[str] = []

    def number(match: re.Match) -> str:
        source_id = match.group(1)
        if source_id not in order:
            order.append(source_id)
        return f"[{order.index(source_id) + 1}]"

    body = CITATION.sub(number, text).rstrip()
    lines = ["", REFERENCE_HEADING, ""]
    for index, source_id in enumerate(order, start=1):
        source = sources[source_id]
        page = f", p.{source.page}" if source.page else ""
        lines.append(f"- [{index}] [{source.title}]({source.url}){page}")
    if not order:
        lines.append("인용된 출처 없음.")
    return body + "\n" + "\n".join(lines) + "\n"
