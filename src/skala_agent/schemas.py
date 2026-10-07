"""검색 계층이 공유하는 데이터 모양. 에이전트 State는 workflow/state.py에 있습니다."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


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
