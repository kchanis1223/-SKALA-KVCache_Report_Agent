import re

from pydantic import BaseModel, ValidationError

from skala_agent.agents.validation import valid_sources
from skala_agent.integrations.contracts import ModelOutputError


class ReportDraft(BaseModel):
    report: str


HEADINGS = (
    "## SUMMARY",
    "## 1. 분석 배경",
    "## 2. 비교 기술 선정",
    "## 3. 기술 개요",
    "## 4. 관점별 평가",
    "## 5. 종합 비교 및 시사점",
    "## 6. 한계점",
    "## REFERENCE",
)


def _sources(technology_id, evidence_ids, evidence):
    return dict(
        sorted(
            {
                str(item.url): item
                for item in evidence
                if item.id in evidence_ids
                and item.technology_id == technology_id
                and item.supports_claim
            }.items()
        )
    )


def _citations(sources, used):
    for url, evidence in sources.items():
        used.setdefault(url, evidence)
    return " ".join(f"[{list(used).index(url) + 1}]" for url in sources)


def _value(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _refine(base, model):
    messages = [
        {
            "role": "developer",
            "content": (
                "검증된 Markdown 보고서를 간결하고 자연스럽게 다듬으세요. 새로운 기술 주장, URL, "
                "인용 번호, 제목 또는 섹션을 추가·삭제·변경하지 마세요. JSON만 반환하세요."
            ),
        },
        {"role": "user", "content": base},
    ]
    try:
        report = ReportDraft.model_validate_json(
            model.invoke_structured(messages, ReportDraft.model_json_schema())
        ).report
    except (ValidationError, TypeError, ValueError) as exc:
        raise ModelOutputError("보고서 모델의 구조화 출력이 유효하지 않습니다.") from exc
    try:
        headings_changed = [report.index(heading) for heading in HEADINGS] != sorted(
            report.index(heading) for heading in HEADINGS
        )
    except ValueError:
        headings_changed = True
    if (
        headings_changed
        or re.findall(r"https?://[^\s)]+", report) != re.findall(r"https?://[^\s)]+", base)
        or re.findall(r"\[\d+\]", report) != re.findall(r"\[\d+\]", base)
    ):
        raise ModelOutputError("보고서 모델이 검증된 목차·인용·참고문헌을 변경했습니다.")
    return report


def _header(run_mode):
    """실행 모드에 따라 제목과 서두 안내를 고른다.

    이전에는 모드와 무관하게 "개발용 뼈대"와 "실제 기술 평가 보고서가
    아닙니다"를 고정 출력했습니다. real provider는 주장·발췌 지지 여부를
    실제로 검증하므로 출력이 동작과 어긋났습니다. 모드는 State가 명시적으로
    전달하며 근거 수나 문구로 추측하지 않습니다.
    """
    if run_mode == "real":
        return [
            "# KV cache 기술 비교 보고서",
            "",
            "> 실제 평가 실행 결과입니다. 판정은 지지 여부가 검증된 근거에만 기반하며, "
            "근거가 부족한 항목은 판단 보류로 남습니다.",
        ]
    return [
        "# KV cache 기술 비교 보고서 — 개발용 뼈대",
        "",
        "> 합성 fixture로 만든 개발용 출력입니다. 실제 기술 평가 결과가 아닙니다.",
    ]


def _limitations(state):
    """실제 검증 수행 여부와 미해결 항목을 사실대로 적는다.

    "의미적 근거 검증과 중립성 검사는 추가 구현 필요"를 고정 출력하고 있었지만
    real provider는 validate_evidence로 이미 지지 여부를 판정합니다. 반대로
    demo는 검증을 수행하지 않습니다. 양쪽 모두 과장 없이 적습니다.
    """
    evidence = state["evidence"]
    verified = [item for item in evidence if item.supports_claim]
    # analyses는 보고서 본문이 쓰지 않는 값이라 없을 수도 있습니다.
    failed = [
        item
        for values in state.get("analyses", {}).values()
        for item in values
        if item.status == "failed"
    ]
    lines = ["## 6. 한계점", ""]
    if state["run_mode"] == "real":
        lines += [
            f"- 근거 검증: 수집한 근거 {len(evidence)}건 중 {len(verified)}건이 "
            "주장 지지 판정을 통과했습니다.",
            "- 지지 판정과 중립성 판단은 같은 모델 호출에서 함께 이뤄지며, "
            "별도 검사로 분리되어 있지 않습니다.",
        ]
    else:
        lines += [
            "- 근거 검증을 수행하지 않았습니다. demo provider는 지지 여부를 "
            "판정하지 않으므로 아래 항목은 모두 미검증입니다.",
        ]
    if failed:
        lines.append(f"- 평가 실패로 판단 보류한 항목 {len(failed)}건이 있습니다.")
    if state["missing_evidence"]:
        lines.append(
            f"- 근거가 부족해 해결되지 않은 항목 {len(state['missing_evidence'])}건 "
            f"(재검색 {state['retry_count']}회 수행):"
        )
    else:
        lines.append("- 근거가 부족해 남은 항목은 없습니다.")
    return lines


def run(state, provider=None):
    lines = _header(state["run_mode"]) + [
        "",
        "## SUMMARY",
        "",
        f"평가 도메인: {state['domain']}",
        f"추가 검색 횟수: {state['retry_count']}",
        "",
        "## 1. 분석 배경",
        "",
        "KV cache 최적화의 SW·HW 접근 비교.",
        "",
        "## 2. 비교 기술 선정",
        "",
        ", ".join(t.name for t in state["selected_technologies"]),
        "",
        "## 3. 기술 개요",
        "",
    ]
    used = {}
    for index, technology in enumerate(state["selected_technologies"], start=1):
        analysis = state["tech_analysis"].get(technology.id)
        sources = _sources(
            technology.id,
            analysis.evidence_ids if analysis else [],
            state["evidence"],
        )
        lines += [f"### 3.{index} {technology.name}", ""]
        verified_ids = {
            item.id
            for item in state["evidence"]
            if item.technology_id == technology.id and item.supports_claim
        }
        if (
            not analysis
            or analysis.status != "assessed"
            or not sources
            or not set(analysis.evidence_ids) <= verified_ids
        ):
            lines += ["판단 보류 (검증된 기술 개요 근거 부족).", ""]
            # 조사 전체를 확정하지 못해도 개별 지지가 확인된 논문 관측은 구분해 게시합니다.
            for url, item in sources.items():
                if item.source_type == "paper":
                    citation = _citations({url: item}, used)
                    lines += [f"- 검증된 논문 관측: {item.claim} {citation}", ""]
            continue
        citations = _citations(sources, used)
        lines += [f"{analysis.overview} {citations}", ""]
        if analysis.scope:
            lines += [f"적용 범위: {', '.join(analysis.scope)} {citations}", ""]
        if analysis.limitations:
            lines += [f"한계: {', '.join(analysis.limitations)} {citations}", ""]
        if analysis.experiments:
            lines += [f"실험 및 조건: {', '.join(analysis.experiments)} {citations}", ""]
    lines += ["## 4. 관점별 평가", ""]
    missing_by_assessment = {
        (item.technology_id, item.perspective): item for item in state["missing_evidence"]
    }
    for item in state["synthesis"]:
        sources = dict(sorted(valid_sources(item, state["evidence"]).items()))
        label = f"{item.technology_id} / {item.perspective}"
        missing = missing_by_assessment.get((item.technology_id, item.perspective))
        if item.status != "assessed" or not sources or missing:
            reason = (
                missing.reason
                if missing
                else f"평가 실패: {item.error.code}"
                if item.error
                else "검증된 근거 부족"
            )
            lines.append(f"- {label}: 판단 보류 ({reason})")
            continue
        refs = _citations(sources, used)
        lines.append(f"- {label}: {item.verdict} (confidence: {item.confidence}) {refs}")
    lines += [
        "",
        "## 5. 종합 비교 및 시사점",
        "",
        "### 5.1 관점 간 일치 지점",
        "",
    ]
    agreements = {}
    for item in state["synthesis"]:
        sources = valid_sources(item, state["evidence"])
        if (
            item.status == "assessed"
            and sources
            and (item.technology_id, item.perspective) not in missing_by_assessment
        ):
            agreements.setdefault((item.perspective, item.verdict), []).append((item, sources))
    common_points = [
        (key, items)
        for key, items in agreements.items()
        if len({item.technology_id for item, _ in items}) > 1
    ]
    for (perspective, verdict), items in common_points:
        sources = {}
        for _, item_sources in items:
            sources.update(item_sources)
        technologies = ", ".join(item.technology_id for item, _ in items)
        citations = _citations(sources, used)
        lines.append(f"- {perspective}: {verdict} (기술: {technologies}) {citations}")
    if not common_points:
        lines.append("검증된 공통 판정 없음.")
    lines += ["", "### 5.2 관점 간 상충 지점", ""]
    findings = state.get("synthesis_findings", [])
    conflicts = [item for item in findings if _value(item, "question") == "maturity_adoption"]
    tradeoffs = [item for item in findings if _value(item, "question") != "maturity_adoption"]
    for item in conflicts:
        sources = _sources(
            _value(item, "technology_id"), _value(item, "evidence_ids", []), state["evidence"]
        )
        if sources:
            lines.append(f"- {_value(item, 'summary')} {_citations(sources, used)}")
    if not conflicts:
        lines.append("검증된 관점 간 상충 없음.")
    lines += [
        "",
        "### 5.3 주요 trade-off",
        "",
    ]
    for item in tradeoffs:
        sources = _sources(
            _value(item, "technology_id"), _value(item, "evidence_ids", []), state["evidence"]
        )
        if sources:
            lines.append(f"- {_value(item, 'summary')} {_citations(sources, used)}")
    if not tradeoffs:
        lines.append("검증된 trade-off 없음.")
    lines += [
        "",
        "### 5.4 데이터센터 도입 의사결정 시 고려사항",
        "",
        "검증된 4장 판정과 위 trade-off를 함께 검토하며, 판단 보류 항목은 "
        "추가 근거를 확보한 뒤 결정한다.",
        "",
    ]
    lines += _limitations(state)
    lines += [f"- {m.technology_id}/{m.perspective}: {m.reason}" for m in state["missing_evidence"]]
    lines += ["", "## REFERENCE", ""]
    for index, (url, item) in enumerate(used.items(), start=1):
        lines.append(f"- [{index}] [{item.title}]({url})")
        if item.chunk_id:
            lines.append(f"  - page: {item.page}; chunk_id: {item.chunk_id}")
            lines.extend(f"  > {line}" for line in item.excerpt.splitlines())
    if not used:
        lines.append("검증된 인용 출처 없음.")
    report = "\n".join(lines) + "\n"
    model = getattr(provider, "report_model", None)
    return {"report": _refine(report, model) if model is not None else report}
