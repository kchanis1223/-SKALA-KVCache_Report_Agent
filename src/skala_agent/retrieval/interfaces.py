"""김동찬 담당: 구현체는 이 계약을 만족하도록 추가합니다."""

from pathlib import Path
from typing import Literal, Protocol

from skala_agent.schemas import Chunk


class PDFParser(Protocol):
    def parse(self, path: Path) -> list[tuple[int, str]]:
        """(1부터 시작하는 페이지 번호, 텍스트) 반환. 표/OCR은 별도 처리."""
        ...


class Embedder(Protocol):
    model_name: str  # 기본 선택: BAAI/bge-m3

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 3,
        role: Literal["primary", "reference"] | None = None,
        paper_id: str | None = None,
    ) -> list[Chunk]: ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        role: Literal["primary", "reference"] | None = None,
    ) -> list[Chunk]: ...
