"""평가셋 라벨링 전 고정하는 청킹 계약. 구현은 retrieval 담당 모듈에서 수행."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: str = "v1"
    tokenizer: str = "BAAI/bge-m3"
    chunk_size_tokens: int = Field(default=1500, gt=0)
    overlap_tokens: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("overlap은 chunk_size보다 작아야 합니다.")
        return self


DEFAULT_CHUNKING = ChunkingConfig()


def chunk_id(paper_id: str, page: int, sequence: int) -> str:
    """페이지 내 순번은 1부터 시작. 청크는 페이지 경계를 넘기지 않습니다."""
    if not paper_id or page < 1 or sequence < 1:
        raise ValueError("paper_id와 1 이상의 page/sequence가 필요합니다.")
    return f"{paper_id}-p{page}-{sequence}"
