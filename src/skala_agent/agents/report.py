from skala_agent.agents.validation import valid_sources


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


def run(state):
    lines = [
        "# KV cache 기술 비교 보고서 — 개발용 뼈대",
        "",
        "> 자동 생성 템플릿입니다. 실제 기술 평가 보고서가 아닙니다.",
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
        if not analysis or analysis.status != "assessed" or not sources:
            lines += ["판단 보류 (검증된 기술 개요 근거 부족).", ""]
            continue
        citations = _citations(sources, used)
        lines += [f"{analysis.overview} {citations}", ""]
        if analysis.scope:
            lines += [f"적용 범위: {', '.join(analysis.scope)} {citations}", ""]
        if analysis.limitations:
            lines += [f"한계: {', '.join(analysis.limitations)} {citations}", ""]
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
        "## 6. 한계점",
        "",
        "의미적 근거 검증과 중립성 검사는 추가 구현 필요.",
    ]
    lines += [f"- {m.technology_id}/{m.perspective}: {m.reason}" for m in state["missing_evidence"]]
    lines += ["", "## REFERENCE", ""]
    lines += [
        f"- [{index}] [{e.title}]({url})" for index, (url, e) in enumerate(used.items(), start=1)
    ]
    if not used:
        lines.append("검증된 인용 출처 없음.")
    return {"report": "\n".join(lines) + "\n"}
