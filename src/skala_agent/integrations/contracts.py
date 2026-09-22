from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class SearchDocument(BaseModel):
    id: str
    title: str
    url: HttpUrl
    content: str = Field(min_length=1)
    source_type: Literal["paper", "official", "market_report", "news", "community"] = "news"
    paper_role: Literal["primary", "reference"] | None = None
    chunk_id: str | None = None
    page: int | None = Field(default=None, ge=1)
    section_or_page: str | None = None


class CitationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    quote: str = Field(min_length=1, max_length=300)


class MeasurementDraft(BaseModel):
    """수치와 조건은 연결된 citation.quote의 원문 부분문자열이어야 합니다."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    metric: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=80)
    conditions: str = Field(min_length=1, max_length=160)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    answer: Literal["yes", "no", "unknown"]
    rationale: str = Field(min_length=1, max_length=300)
    citations: list[CitationDraft] = Field(max_length=2)
    measurements: list[MeasurementDraft] = Field(default_factory=list, max_length=3)


class EvaluationDraft(BaseModel):
    """소형 LLM은 질문별 근거만 출력. 등급·ID·URL은 애플리케이션이 결정."""

    model_config = ConfigDict(extra="forbid")
    findings: list[Finding] = Field(max_length=15)


class ModelOutputError(ValueError):
    """JSON/schema/출처 참조 오류. 원문 응답을 오류 메시지에 노출하지 않습니다."""


class IncompleteModelOutputError(ModelOutputError):
    """생성 한도 초과 등 미완료 응답. 질문을 나누어 다시 추출할 수 있습니다."""


class ServiceConfigurationError(ValueError):
    """인증 또는 모델 설치 등 재시도로 해결되지 않는 설정 오류."""


class WebSearch(Protocol):
    def search(self, query: str) -> list[SearchDocument]: ...


class StructuredModel(Protocol):
    def extract(self, system: str, payload: dict) -> EvaluationDraft: ...
