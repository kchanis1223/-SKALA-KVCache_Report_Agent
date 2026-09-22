import json
import logging

from pydantic import BaseModel, ValidationError

from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.schemas import PERSPECTIVES, SynthesisFinding

logger = logging.getLogger(__name__)


class SynthesisDraft(BaseModel):
    findings: list[SynthesisFinding]


# 규칙 기반 findings의 본문 문구. 이전에는 "{키} 관련 trade-off 또는 적용
# 조건이 확인됐다."를 그대로 써서 보고서 5.3에 같은 문장이 여러 번 실렸습니다.
# 키마다 무엇을 뜻하는지 한국어 서술로 적습니다.
TRADEOFF_SUMMARIES = {
    "quality_stability": "처리량·메모리 개선이 정확도·안정성과 맞바꿔지는 구간이 있다",
    "resource_cost": "메모리 절감이 CPU·대역폭·스토리지 비용으로 옮겨가는 구간이 있다",
    "operational_complexity": "성능 개선이 배포·통합·운영 부담을 늘리는 구간이 있다",
    "condition_limited": "효과가 context length·batch size·대역폭 등 특정 조건에서만 나타난다",
}

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
    assessments = [item for values in state["analyses"].values() for item in values]
    # 허용 ref를 payload에 그대로 싣습니다. 프롬프트가 "제공한 assessment_refs"를
    # 가리키는데 실제로는 Assessment 전체만 넘겨서, 모델이 쌍을 직접 유추하다
    # 존재하지 않는 조합을 참조했습니다.
    allowed_refs = [[item.perspective, item.technology_id] for item in assessments]
    payload = {
        "allowed_assessment_refs": allowed_refs,
        "assessments": [item.model_dump(mode="json") for item in assessments],
        "evidence": [item.model_dump(mode="json") for item in evidence],
    }
    messages = [
        {
            "role": "developer",
            "content": (
                "상충 또는 trade-off만 findings로 반환하세요. assessment_refs의 각 항목은 "
                "allowed_assessment_refs에 그대로 있는 [perspective, technology_id] "
                "쌍이어야 하며 순서를 바꾸지 마세요. 한 finding의 technology_id는 그 "
                "finding이 참조하는 모든 ref의 technology_id와 같아야 합니다. "
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
                        summary=(
                            f"{assessment.technology_id}: "
                            f"{TRADEOFF_SUMMARIES[question]} "
                            f"({assessment.perspective} 관점 판정 근거에서 확인)"
                        ),
                        assessment_refs=[(assessment.perspective, assessment.technology_id)],
                        evidence_ids=evidence_ids,
                    )
                )
    model = getattr(provider, "synthesis_model", None)
    if model is not None:
        # 종합 모델 출력이 계약을 어기면 그 출력만 버립니다. 규칙 기반 findings는
        # 이미 계산돼 있으므로, 실행 전체를 중단시키는 대신 검증을 통과한 부분만
        # 남깁니다. 버린 사실은 로그에 남기고 없던 판정을 만들지는 않습니다.
        try:
            generated = _generated_findings(state, model)
        except ModelOutputError as exc:
            logger.warning("종합 모델 출력을 버립니다 (규칙 기반 findings만 사용): %s", exc)
            generated = []
        generated_keys = {
            (item.question, item.technology_id, tuple(item.evidence_ids)) for item in generated
        }
        findings = [
            item
            for item in findings
            if (item.question, item.technology_id, tuple(item.evidence_ids)) not in generated_keys
        ] + generated
    return {"synthesis": assessments, "synthesis_findings": findings}
