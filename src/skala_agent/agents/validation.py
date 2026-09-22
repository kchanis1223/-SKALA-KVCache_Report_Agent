from skala_agent.schemas import MissingEvidence


def valid_sources(assessment, evidence):
    return {
        str(e.url): e
        for e in evidence
        if e.id in assessment.evidence_ids
        and e.technology_id == assessment.technology_id
        and e.supports_claim
    }


def run(state):
    missing = []
    normalized = []
    for item in state["synthesis"]:
        sources = valid_sources(item, state["evidence"])
        if len(sources) < 2:
            item = item.model_copy(update={"confidence": "low"})
        normalized.append(item)
        if item.status != "assessed" or not sources:
            missing.append(
                MissingEvidence(
                    technology_id=item.technology_id,
                    perspective=item.perspective,
                    reason="평가 미완료 또는 주장을 지지하는 출처 없음",
                )
            )
    # TODO(이준형): 문장별 entailment·중립성·질문별 등급 검증 추가.
    return {"missing_evidence": missing, "synthesis": normalized}
