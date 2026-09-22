from skala_agent.schemas import MissingEvidence


def valid_evidence(assessment, evidence):
    return {
        e.id: e
        for e in evidence
        if e.id in assessment.evidence_ids
        and e.technology_id == assessment.technology_id
        and e.supports_claim
    }


def valid_sources(assessment, evidence):
    return {str(e.url): e for e in valid_evidence(assessment, evidence).values()}


def run(state, provider=None):
    updates = provider.validate_evidence(state["evidence"]) if provider else []
    evidence = {item.id: item for item in state["evidence"]}
    evidence.update({item.id: item for item in updates})
    evidence = list(evidence.values())
    missing = []
    normalized = []
    for item in state["synthesis"]:
        sources = valid_sources(item, evidence)
        if len(sources) < 2:
            item = item.model_copy(update={"confidence": "low"})
        normalized.append(item)
        if item.status != "assessed" or not sources:
            if item.status == "failed":
                kind, reason = (
                    "agent_failed",
                    f"평가 실패: {item.error.code} — {item.error.message}",
                )
            elif item.status == "pending":
                kind, reason = "not_evaluated", item.rationale
            elif not item.evidence_ids:
                kind, reason = "missing_source", "판정을 뒷받침할 출처 없음"
            else:
                kind, reason = "unsupported_claim", "참조 근거가 없거나 주장을 지지하지 않음"
            claim = item.verdict if item.status == "assessed" else item.rationale
            topics = [signal.question for signal in item.signals] or [claim]
            missing.append(
                MissingEvidence(
                    technology_id=item.technology_id,
                    perspective=item.perspective,
                    reason=reason,
                    kind=kind,
                    claim=claim,
                    queries=[
                        f"{item.technology_id} {item.perspective} {topic}" for topic in topics
                    ],
                    evidence_ids=item.evidence_ids,
                    retryable=item.error.retryable if item.error else True,
                )
            )
            continue
        for signal in item.signals:
            signal_sources = {
                e.id: e
                for e in valid_evidence(item, evidence).values()
                if e.id in signal.evidence_ids
            }
            if not signal_sources:
                missing.append(
                    MissingEvidence(
                        technology_id=item.technology_id,
                        perspective=item.perspective,
                        reason="질문별 판정을 지지하는 검증된 출처 없음",
                        kind="unsupported_claim",
                        claim=signal.question,
                        queries=[f"{item.technology_id} {item.perspective} {signal.question}"],
                        evidence_ids=signal.evidence_ids,
                    )
                )
    # 원문 entailment·중립성 판정은 provider가 Evidence.supports_claim에 반영한다.
    result = {"missing_evidence": missing, "synthesis": normalized}
    if updates:
        result["evidence"] = updates
    return result
