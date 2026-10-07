from typing import Literal, Protocol

from pydantic import BaseModel, Field, HttpUrl


class SearchDocument(BaseModel):
    id: str
    title: str
    url: HttpUrl
    content: str = Field(min_length=1)
    source_type: Literal["paper", "official", "market_report", "news", "community"] = "news"


class ServiceConfigurationError(ValueError):
    """인증 등 재시도로 해결되지 않는 설정 오류."""


class WebSearch(Protocol):
    def search(self, query: str) -> list[SearchDocument]: ...
