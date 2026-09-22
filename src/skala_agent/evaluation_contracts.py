"""평가 provider 경계의 구조·참조 검증. 의미적 근거 검증(#11)과 구분합니다."""

from pydantic import Field, model_validator

from skala_agent.schemas import (
    Assessment,
    DomainDetails,
    EvaluationRecord,
    Evidence,
    MarketDetails,
    NonBlank,
    Perspective,
    StakeholderDetails,
    TRLDetails,
)

STAKEHOLDER_KEYS = {"gpu_vendor", "memory_vendor", "cloud_operator", "open_source", "investor"}


def unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f"{label} 중복은 허용하지 않습니다.")


def check_details(assessment):
    details = assessment.details
    if assessment.status == "failed":
        return
    if details is None:
        raise ValueError("평가 출력에는 관점별 details가 필요합니다.")
    if isinstance(details, StakeholderDetails):
        keys = [p.stakeholder for p in details.positions]
        if len(keys) != 5 or set(keys) != STAKEHOLDER_KEYS:
            raise ValueError("이해관계자 5개 주체를 각각 한 번 반환해야 합니다.")
        observed = [p for p in details.positions if p.stance != "자료 없음"]
        expected = None
        if observed:
            support = sum(p.stance == "지지" for p in observed)
            skeptical = sum(p.stance == "회의적" for p in observed)
            expected = (
                "우호적"
                if support > len(observed) / 2
                else "부정적"
                if skeptical > len(observed) / 2
                else "혼재"
            )
        if details.overall != expected:
            raise ValueError("이해관계자 종합 판정이 자료 없음 제외 과반 규칙과 다릅니다.")
    if assessment.status != "assessed":
        return
    complete = (
        details.level is not None
        if isinstance(details, TRLDetails)
        else all((details.demand, details.adoption, details.ecosystem))
        if isinstance(details, MarketDetails)
        else details.overall is not None
        if isinstance(details, StakeholderDetails)
        else bool(
            details.cost and details.operations and details.sla_risk not in (None, "판단 불가")
        )
        if isinstance(details, DomainDetails)
        else False
    )
    if not complete:
        raise ValueError("미확인 평가 축이 있는 결과는 assessed일 수 없습니다.")


class EvaluationOutput(EvaluationRecord):
    """한 관점의 선택 기술별 결과와 그 결과가 참조하는 근거를 함께 검증합니다."""

    perspective: Perspective
    technology_ids: list[NonBlank] = Field(min_length=1)
    assessments: list[Assessment] = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_output(self):
        unique(self.technology_ids, "선택 기술")
        result_ids = [a.technology_id for a in self.assessments]
        unique(result_ids, "평가 기술")
        if set(result_ids) != set(self.technology_ids):
            raise ValueError("선택 기술마다 정확히 하나의 평가가 필요합니다.")
        unique([e.id for e in self.evidence], "Evidence ID")
        sources = {e.id: e for e in self.evidence}
        if any(e.technology_id not in self.technology_ids for e in self.evidence):
            raise ValueError("선택하지 않은 기술의 근거가 포함되었습니다.")
        normalized = []
        for assessment in self.assessments:
            if assessment.perspective != self.perspective:
                raise ValueError("요청 관점과 다른 평가입니다.")
            check_details(assessment)
            ids = assessment.evidence_ids
            unique(ids, "평가 근거 참조")
            for eid in ids:
                if eid not in sources or sources[eid].technology_id != assessment.technology_id:
                    raise ValueError("근거 참조가 없거나 다른 기술의 근거입니다.")
            unique([s.question for s in assessment.signals], "조사 질문")
            groups = [s.evidence_ids for s in assessment.signals]
            for signal in assessment.signals:
                if signal.grade != "하" and not signal.evidence_ids:
                    raise ValueError("상·중 signals에는 근거 참조가 필요합니다.")
            if isinstance(assessment.details, StakeholderDetails):
                groups += [p.evidence_ids for p in assessment.details.positions]
                for position in assessment.details.positions:
                    if position.stance != "자료 없음" and not position.evidence_ids:
                        raise ValueError("입장이 있는 이해관계자에는 근거가 필요합니다.")
            for references in groups:
                unique(references, "세부 근거 참조")
                if not set(references).issubset(ids):
                    raise ValueError(
                        "세부 근거 참조는 Assessment.evidence_ids에 포함되어야 합니다."
                    )
            if assessment.status == "assessed" and (not ids or not assessment.signals):
                raise ValueError("assessed 결과에는 근거와 질문별 signals가 필요합니다.")
            url_count = len({str(sources[eid].url) for eid in ids})
            low = assessment.status != "assessed" or url_count < 2
            normalized.append(
                assessment.model_copy(update={"confidence": "low"}) if low else assessment
            )
        # 전달받은 Assessment나 공유 State를 변경하지 않습니다.
        self.assessments = normalized
        return self


def validate_evaluation_output(perspective, technologies, assessments, evidence):
    """기존 provider tuple 반환 계약을 유지하는 공통 경계 함수."""
    output = EvaluationOutput(
        perspective=perspective,
        technology_ids=[t.id for t in technologies],
        assessments=assessments,
        evidence=evidence,
    )
    return output.assessments, output.evidence
