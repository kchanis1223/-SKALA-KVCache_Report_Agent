import json

from pydantic import BaseModel, ValidationError

from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.schemas import PERSPECTIVES, SynthesisFinding


class SynthesisDraft(BaseModel):
    findings: list[SynthesisFinding]


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
    return " ".join([assessment.verdict, assessment.rationale]).lower()


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


def _generated_findings(state, model):
    evidence = [item for item in state["evidence"] if item.supports_claim]
    payload = {
        "assessments": [
            item.model_dump(mode="json") for values in state["analyses"].values() for item in values
        ],
        "evidence": [item.model_dump(mode="json") for item in evidence],
    }
    messages = [
        {
            "role": "developer",
            "content": (
                "상충 또는 trade-off만 findings로 반환하세요. "
                "assessment_refs는 [perspective, technology_id] 튜플 리스트 형태"
                "(예: [[\"trl\", \"turboquant\"]])여야 합니다. "
                "제공한 assessments의 (perspective, technology_id) 조합과 "
                "supports_claim=true Evidence ID만 사용하고, 근거 없는 항목은 생략하세요."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        draft = SynthesisDraft.model_validate_json(
            model.invoke_structured(messages, SynthesisDraft.model_json_schema())
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise ModelOutputError("종합 모델의 구조화 출력이 유효하지 않습니다.") from exc
    assessments = {
        (item.perspective, item.technology_id): item
        for values in state["analyses"].values()
        for item in values
    }
    valid_evidence = {item.id: item for item in evidence}
    for finding in draft.findings:
        if not all(ref in assessments for ref in finding.assessment_refs) or not all(
            item.technology_id == finding.technology_id
            for ref in finding.assessment_refs
            for item in [assessments[ref]]
        ):
            raise ModelOutputError(
                "종합 모델이 존재하지 않거나 다른 기술의 assessment를 참조했습니다."
            )
        if not all(
            evidence_id in valid_evidence
            and valid_evidence[evidence_id].technology_id == finding.technology_id
            for evidence_id in finding.evidence_ids
        ):
            raise ModelOutputError(
                "종합 모델이 검증되지 않았거나 다른 기술의 Evidence를 참조했습니다."
            )
    return draft.findings


def run(state, provider=None):
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
    model = getattr(provider, "synthesis_model", None)
    if model is not None:
        generated = _generated_findings(state, model)
        generated_keys = {
            (item.question, item.technology_id, tuple(item.evidence_ids)) for item in generated
        }
        findings = [
            item
            for item in findings
            if (item.question, item.technology_id, tuple(item.evidence_ids)) not in generated_keys
        ] + generated
    return {"synthesis": assessments, "synthesis_findings": findings}
