from skala_agent.schemas import PERSPECTIVES, SynthesisFinding

QUESTION_KEYWORDS = {
    "quality_stability": (
        ("메모리", "throughput", "처리량", "성능"),
        ("정확도", "품질", "안정성", "sla"),
    ),
    "resource_cost": (("메모리", "hbm", "gpu"), ("cpu", "bandwidth", "storage", "비용")),
    "operational_complexity": (
        ("성능", "개선", "처리량"),
        ("배포", "통합", "운영", "유지보수", "인프라"),
    ),
    "condition_limited": (("조건", "context length", "batch size", "bandwidth", "gpu"),),
}


def _text(assessment):
    return " ".join(
        [assessment.verdict, assessment.rationale]
        + [signal.question for signal in assessment.signals]
    ).lower()


def _linked_evidence_ids(assessment, evidence):
    return [
        item.id
        for item in evidence
        if item.id in assessment.evidence_ids
        and item.technology_id == assessment.technology_id
        and item.supports_claim
    ]


def _matches(text, groups):
    return all(any(keyword in text for keyword in group) for group in groups)


def _maturity_adoption_finding(assessments, evidence):
    by_technology = {}
    for assessment in assessments:
        by_technology.setdefault(assessment.technology_id, {})[assessment.perspective] = assessment
    findings = []
    for technology_id, items in by_technology.items():
        trl, market = items.get("trl"), items.get("market")
        if not trl or not market or not trl.details or not market.details:
            continue
        level = getattr(trl.details, "level", None)
        adoption = getattr(market.details, "adoption", None)
        mismatch = (level is not None and level <= 6 and adoption == "상용 채택") or (
            level is not None and level >= 7 and adoption == "연구 단계"
        )
        evidence_ids = _linked_evidence_ids(trl, evidence) + _linked_evidence_ids(market, evidence)
        if mismatch and evidence_ids:
            findings.append(
                SynthesisFinding(
                    technology_id=technology_id,
                    question="maturity_adoption",
                    summary="기술 성숙도와 시장 채택 수준 사이의 차이를 추가 확인해야 한다.",
                    assessment_refs=[("trl", technology_id), ("market", technology_id)],
                    evidence_ids=list(dict.fromkeys(evidence_ids)),
                )
            )
    return findings


def run(state):
    assessments = [item for key in PERSPECTIVES for item in state["analyses"].get(key, [])]
    findings = _maturity_adoption_finding(assessments, state["evidence"])
    for assessment in assessments:
        evidence_ids = _linked_evidence_ids(assessment, state["evidence"])
        if not evidence_ids:
            continue
        text = _text(assessment)
        for question, groups in QUESTION_KEYWORDS.items():
            if _matches(text, groups):
                findings.append(
                    SynthesisFinding(
                        technology_id=assessment.technology_id,
                        question=question,
                        summary=f"{question} 관련 trade-off 또는 적용 조건이 확인됐다.",
                        assessment_refs=[(assessment.perspective, assessment.technology_id)],
                        evidence_ids=evidence_ids,
                    )
                )
    return {"synthesis": assessments, "synthesis_findings": findings}
