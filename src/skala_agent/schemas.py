"""공유 계약. 변경 시 retrieval / agents / workflow 담당자와 함께 검토."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

Perspective = Literal["trl", "market", "stakeholder", "domain"]
PERSPECTIVES: tuple[Perspective, ...] = ("trl", "market", "stakeholder", "domain")


class Technology(BaseModel):
    id: str
    name: str
    camp: Literal["sw", "hw"]


class Evidence(BaseModel):
    id: str
    technology_id: str
    claim: str
    url: HttpUrl
    title: str
    source_type: Literal["paper", "official", "market_report", "news", "community"]
    excerpt: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    # 검색된 출처의 존재와 주장 지지는 별개. 실제 검증자가 확인한 경우만 True.
    supports_claim: bool = False


class Signal(BaseModel):
    question: str
    grade: Literal["상", "중", "하"]
    evidence_ids: list[str] = Field(default_factory=list)


class Assessment(BaseModel):
    technology_id: str
    perspective: Perspective
    verdict: str
    rationale: str
    confidence: Literal["low", "medium", "high"] = "low"
    signals: list[Signal] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "assessed"] = "pending"


class MissingEvidence(BaseModel):
    technology_id: str
    perspective: Perspective
    reason: str


class Chunk(BaseModel):
    id: str
    text: str
    paper_id: str
    camp: Literal["sw", "hw"]
    role: Literal["primary", "reference"]
    section: str
    page: int = Field(ge=1)
    source_url: HttpUrl
