"""김동찬 담당: 임베딩 + VectorStore를 묶은 Retriever."""

from __future__ import annotations

from typing import Literal

from skala_agent.retrieval.embedding import Embedder
from skala_agent.retrieval.vector_store import NumpyVectorStore
from skala_agent.schemas import Chunk, RetrievalResult


class DenseRetriever:
    """`Retriever` 프로토콜 구현. 색인과 질의에 같은 embedder를 씁니다."""

    def __init__(self, embedder: Embedder, store: NumpyVectorStore | None = None) -> None:
        self.embedder = embedder
        self.store = store or NumpyVectorStore()

    def index(self, chunks: list[Chunk], *, batch_size: int = 16) -> int:
        """청크를 임베딩해 색인에 넣고 색인된 개수를 반환합니다."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.store.upsert(batch, self.embedder.encode([chunk.text for chunk in batch]))
        return len(self.store)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        role: Literal["primary", "reference"] | None = None,
        paper_id: str | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("빈 질의는 검색할 수 없습니다.")
        vector = self.embedder.encode([query])[0]
        return self.store.search(vector, top_k=top_k, role=role, paper_id=paper_id)
