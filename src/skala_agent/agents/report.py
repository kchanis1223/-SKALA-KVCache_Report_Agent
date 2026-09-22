import re

from pydantic import BaseModel, ValidationError

from skala_agent.agents.validation import valid_evidence, valid_sources
from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.schemas import PERSPECTIVES


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


def _question_label(text: str) -> str:
    """본문에 보일 질문 문구만 남긴다.

    Signal.question에는 재현용으로 "[trl_1] ... (응답: yes)"처럼 내부 질문 ID와
    응답값이 함께 들어 있습니다. 보고서 본문에 그대로 내보내면 읽는 사람에게
    의미 없는 식별자가 노출됩니다. 원본 값은 State에 그대로 남습니다.
    """
    label = re.sub(r"^\s*\[[^\]]+\]\s*", "", text)
    return re.sub(r"\s*\(응답:[^)]*\)\s*$", "", label).strip()


def _signal_support(item, evidence):
    """질문(Signal)별로 검증된 출처가 있는지 나눈다."""
    verified = valid_evidence(item, evidence)
    supported, unsupported = [], []
    for signal in item.signals:
        if any(eid in verified for eid in signal.evidence_ids):
            supported.append(signal)
        else:
            unsupported.append(signal)
    return supported, unsupported


def _split_gaps(item, gaps):
    """부족 항목을 Assessment 전체 부족과 질문 단위 부족으로 나눈다.

    validation은 Assessment 전체가 유효하지 않을 때 claim에 verdict(또는 실패
    시 rationale)를 넣고, 개별 Signal이 부족할 때는 claim에 그 질문 텍스트를
    넣습니다. 그 차이로 두 종류를 구분합니다.
    """
    anchors = {item.verdict, item.rationale}
    blocking = [gap for gap in gaps if gap.claim in anchors]
    signal_level = [gap for gap in gaps if gap.claim not in anchors]
    return blocking, signal_level


def _conclusive(item, evidence, gaps=()):
    """전체 판정을 확정해도 되는지 판단한다. 4장과 5장이 같은 규칙을 쓴다.

    보류를 유지해야 하는 경우를 넷으로 구분합니다.

    1. Assessment 자체가 유효하지 않음: status가 assessed가 아님(pending·failed)
    2. 필수 판단 축이 비었음: 이 기술의 검증된 출처가 하나도 없음
    3. 질문 단위 근거가 전무함: Signal이 있는데 지지되는 Signal이 하나도 없음
    4. Assessment 전체에 대한 부족 항목이 남아 있음

    assessed 상태만으로 근거가 검증됐다고 보지 않습니다. 반대로 **일부 질문만**
    부족한 경우는 판정을 지우지 않고 부족 질문을 함께 표시합니다. 이전에는
    missing_evidence가 하나라도 있으면 Assessment 전체를 판단 보류로 덮어,
    검증된 부분 판정까지 사라졌습니다.
    """
    if item.status != "assessed":
        return False
    if not valid_sources(item, evidence):
        return False
    blocking, _ = _split_gaps(item, gaps)
    if blocking:
        return False
    supported, _ = _signal_support(item, evidence)
    return bool(supported) or not item.signals


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
                "인용 번호, 제목 또는 섹션을 추가·삭제·변경하지 마세요. JSON만 반환하세요.\n"
                "본문이 영어로 된 문장은 한국어로 옮겨 적으세요. 번역할 때 내용을 더하거나 "
                "빼지 말고, 기술명·제품명·지표명(TTFT, TPOT, vLLM, CXL 등)과 수치는 원문 "
                "표기를 유지하세요.\n"
                "REFERENCE의 출처 제목과 원문 발췌(> 로 시작하는 줄)는 원문 그대로 두세요. "
                "인용의 정확성을 확인할 수 있어야 합니다."
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


def _summary_line(state) -> str:
    """판정 집계를 요약 첫 줄로 둔다.

    이전 SUMMARY는 평가 도메인과 재검색 횟수만 있어, 읽는 사람이 결론을
    파악하려면 4장까지 내려가야 했습니다. 집계는 State에서 세는 값이고
    없는 판정을 만들지 않습니다.
    """
    items = state["synthesis"]
    concluded = [item for item in items if _conclusive(item, state["evidence"])]
    verdicts = ", ".join(
        f"{item.technology_id}/{item.perspective}={item.verdict}" for item in concluded
    )
    head = f"기술 {len({i.technology_id for i in items})}건 × 관점 {len(PERSPECTIVES)}개 중 "
    head += f"판정 {len(concluded)}건, 판단 보류 {len(items) - len(concluded)}건."
    return f"{head} 확정 판정: {verdicts}." if concluded else f"{head} 확정 판정 없음."


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
        _summary_line(state),
        "",
        f"평가 도메인: {state['domain']} · 추가 검색 {state['retry_count']}회",
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
    # 같은 (기술, 관점)에 부족 사유가 여러 개면 전부 보존합니다. 하나만 남기면
    # 다른 사유가 조용히 사라집니다.
    missing_by_assessment: dict[tuple[str, str], list] = {}
    for gap in state["missing_evidence"]:
        missing_by_assessment.setdefault((gap.technology_id, gap.perspective), []).append(gap)
    for item in state["synthesis"]:
        sources = dict(sorted(valid_sources(item, state["evidence"]).items()))
        label = f"{item.technology_id} / {item.perspective}"
        gaps = missing_by_assessment.get((item.technology_id, item.perspective), [])
        blocking, signal_gaps = _split_gaps(item, gaps)
        if not _conclusive(item, state["evidence"], gaps):
            reason = (
                f"평가 실패: {item.error.code}"
                if item.error
                else (blocking or gaps)[0].reason
                if gaps
                else "검증된 근거 부족"
            )
            lines.append(f"- {label}: 판단 보류 ({_question_label(reason)})")
            # 같은 관점의 나머지 부족 사유도 남기되 중복은 접습니다.
            seen = {reason}
            others = []
            for gap in gaps:
                if gap.reason not in seen:
                    seen.add(gap.reason)
                    others.append(gap.reason)
            if others:
                lines.append(f"  - 그 외 부족 사유: {', '.join(others[:2])}")
            continue
        refs = _citations(sources, used)
        lines.append(f"- {label}: {item.verdict} (confidence: {item.confidence}) {refs}")
        # 판정은 검증됐지만 일부 질문의 근거가 없으면 함께 드러냅니다.
        #
        # 미지지 Signal과 Signal 단위 부족 항목은 사실상 같은 집합입니다. 둘 다
        # 나열하면 판정 한 줄 밑에 같은 내용이 두 번씩 쌓여 본문이 묻힙니다.
        # 본문에는 건수와 대표 항목만 남기고 전체 목록은 6장으로 보냅니다.
        _, unsupported = _signal_support(item, state["evidence"])
        questions = list(
            dict.fromkeys(
                [_question_label(signal.question) for signal in unsupported]
                + [_question_label(gap.claim) for gap in signal_gaps]
            )
        )
        if questions:
            shown = ", ".join(questions[:2])
            more = f" 외 {len(questions) - 2}건" if len(questions) > 2 else ""
            lines.append(
                f"  - 근거 부족 질문 {len(questions)}/{len(item.signals)}건: {shown}{more}"
            )
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
        # 4장과 같은 규칙을 씁니다. 일부 질문만 부족한 판정은 공통 판정 후보로
        # 남기고, 확정할 수 없는 판정만 제외합니다.
        if _conclusive(
            item,
            state["evidence"],
            missing_by_assessment.get((item.technology_id, item.perspective), []),
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
    # 같은 (기술, 유형)에 대해 findings가 여러 개 나오면 5.3에 같은 문장이
    # 반복됩니다. 인용은 합치고 문장은 한 번만 싣습니다.
    merged: dict[tuple[str, str], dict] = {}
    for item in tradeoffs:
        sources = _sources(
            _value(item, "technology_id"), _value(item, "evidence_ids", []), state["evidence"]
        )
        if not sources:
            continue
        key = (_value(item, "technology_id"), _value(item, "question"))
        entry = merged.setdefault(key, {"summary": _value(item, "summary"), "sources": {}})
        entry["sources"].update(sources)
    for entry in merged.values():
        lines.append(f"- {entry['summary']} {_citations(entry['sources'], used)}")
    if not merged:
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
    # 부족 항목을 그대로 나열하면 같은 줄이 수십 번 반복됩니다(실측 33줄 중
    # 서로 다른 사유는 몇 개뿐). (기술/관점, 사유)로 묶어 건수로 적습니다.
    grouped: dict[tuple[str, str, str], int] = {}
    for item in state["missing_evidence"]:
        key = (item.technology_id, item.perspective, _question_label(item.reason))
        grouped[key] = grouped.get(key, 0) + 1
    for (technology_id, perspective, reason), count in grouped.items():
        suffix = f" ({count}건)" if count > 1 else ""
        lines.append(f"- {technology_id}/{perspective}: {reason}{suffix}")
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
