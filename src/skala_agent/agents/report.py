from skala_agent.agents.validation import valid_sources


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
        "논문 기반 기술 요약 구현 대기.",
        "",
        "## 4. 관점별 평가",
        "",
    ]
    used = {}
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
        for url, evidence in sources.items():
            used.setdefault(url, evidence)
        refs = " ".join(f"[{list(used).index(url) + 1}]" for url in sources)
        lines.append(f"- {label}: {item.verdict} (confidence: {item.confidence}) {refs}")
    lines += [
        "",
        "## 5. 종합 비교 및 시사점",
        "",
        "상충 탐지·종합 서술 구현 대기.",
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
