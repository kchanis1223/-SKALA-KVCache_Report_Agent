"""공유 계약. 변경 시 retrieval / agents / workflow 담당자와 함께 검토."""

from typing import Annotated, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    model_validator,
)

Perspective = Literal["trl", "market", "stakeholder", "domain"]
PERSPECTIVES: tuple[Perspective, ...] = ("trl", "market", "stakeholder", "domain")


class Technology(BaseModel):
    id: str
    name: str
    camp: Literal["sw", "hw"]


NonBlank = Annotated[str, StringConstraints(pattern=r"\S")]
Confidence = Literal["low", "medium", "high"]


class EvaluationRecord(BaseModel):
    """평가 출력의 오타 필드는 묵인하지 않습니다. 원문 공백은 보존합니다."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="always")


class AgentError(EvaluationRecord):
    code: NonBlank
    message: NonBlank
    retryable: bool = True


class Evidence(EvaluationRecord):
    id: NonBlank
    technology_id: NonBlank
    claim: NonBlank
    url: HttpUrl
    title: NonBlank
    source_type: Literal["paper", "official", "market_report", "news", "community"]
    excerpt: NonBlank = Field(
        min_length=1, validation_alias=AliasChoices("excerpt", "evidence_text")
    )
    section_or_page: str | None = None
    chunk_id: str | None = None
    confidence: Confidence = "low"
    page: int | None = Field(default=None, ge=1)
    # 검색된 출처의 존재와 주장 지지는 별개. 실제 검증자가 확인한 경우만 True.
    supports_claim: bool = False


class Signal(EvaluationRecord):
    question: NonBlank
    grade: Literal["상", "중", "하"]
    evidence_ids: list[str] = Field(default_factory=list)


class TechAnalysis(BaseModel):
    technology_id: str
    overview: str
    scope: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    experiments: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "assessed"] = "pending"


class TRLDetails(EvaluationRecord):
    perspective: Literal["trl"] = "trl"
    level: int | None = Field(default=None, ge=1, le=9)


class MarketDetails(EvaluationRecord):
    perspective: Literal["market"] = "market"
    demand: Literal["수요 불명확", "수요 존재", "수요 확실"] | None = None
    adoption: Literal["연구 단계", "시범 적용", "상용 채택"] | None = None
    ecosystem: Literal["생태계 부재", "형성 중", "확보"] | None = None
    dependency_risks: list[str] = Field(default_factory=list)


class StakeholderPosition(EvaluationRecord):
    stakeholder: Literal["gpu_vendor", "memory_vendor", "cloud_operator", "open_source", "investor"]
    stance: Literal["지지", "유보", "회의적", "자료 없음"]
    rationale: NonBlank
    evidence_ids: list[str] = Field(default_factory=list)


class StakeholderDetails(EvaluationRecord):
    perspective: Literal["stakeholder"] = "stakeholder"
    positions: list[StakeholderPosition] = Field(default_factory=list)
    overall: Literal["우호적", "부정적", "혼재"] | None = None


class DomainDetails(EvaluationRecord):
    perspective: Literal["domain"] = "domain"
    cost: Literal["원가 개선 명확", "조건부 개선", "개선 불명확"] | None = None
    sla_risk: Literal["낮음", "중간", "높음", "판단 불가"] | None = None
    operations: Literal["즉시 도입 가능", "인프라 전환 전제"] | None = None
    operational_risks: list[str] = Field(default_factory=list)


class Assessment(EvaluationRecord):
    technology_id: NonBlank
    perspective: Perspective
    verdict: NonBlank
    rationale: NonBlank = Field(validation_alias=AliasChoices("rationale", "reason"))
    confidence: Confidence = "low"
    signals: list[Signal] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "assessed", "failed"] = "pending"
    error: AgentError | None = None
    details: TRLDetails | MarketDetails | StakeholderDetails | DomainDetails | None = Field(
        default=None, discriminator="perspective"
    )

    @model_validator(mode="after")
    def validate_result(self):
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("failed 상태에만 error를 반드시 지정해야 합니다.")
        if self.details is not None and self.details.perspective != self.perspective:
            raise ValueError("details의 관점은 Assessment 관점과 일치해야 합니다.")
        if self.status == "failed":
            self.verdict = "판단 보류"
            self.confidence = "low"
        return self


class MissingEvidence(BaseModel):
    technology_id: str
    perspective: Perspective
    reason: str
    kind: Literal["not_evaluated", "missing_source", "unsupported_claim", "agent_failed"]
    claim: str
    queries: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    retryable: bool = True


class Chunk(BaseModel):
    id: str
    text: str
    paper_id: str
    camp: Literal["sw", "hw"]
    role: Literal["primary", "reference"]
    section: str
    page: int = Field(ge=1)
    source_url: HttpUrl


class RetrievalResult(BaseModel):
    """높을수록 관련성이 높은 점수. 서로 다른 score_type끼리 비교하지 않습니다."""

    chunk: Chunk
    score: float = Field(allow_inf_nan=False)
    score_type: Literal["dense", "sparse", "hybrid", "rrf"] = "dense"
    dense_score: float | None = Field(default=None, allow_inf_nan=False)
    sparse_score: float | None = Field(default=None, allow_inf_nan=False)
