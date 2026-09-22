"""김동찬 담당: numpy 브루트포스 VectorStore.

문서 풀이 61청크 규모라 전수 비교로도 즉시 끝나며, 근사 색인의 오차가 없어
Retrieval 성능 측정(#4)이 색인 구조가 아닌 임베딩 품질만 반영합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from skala_agent.schemas import Chunk, RetrievalResult


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class NumpyVectorStore:
    """`VectorStore` 프로토콜 구현. 코사인 유사도로 전수 검색합니다."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """같은 id는 교체하고 새 id는 추가합니다. 재색인해도 중복이 생기지 않습니다."""
        if len(chunks) != len(vectors):
            raise ValueError("chunk 수와 vector 수가 같아야 합니다.")
        if not chunks:
            return
        incoming = np.asarray(vectors, dtype=np.float32)
        if incoming.ndim != 2:
            raise ValueError("vectors는 2차원이어야 합니다.")
        if self._vectors is not None and incoming.shape[1] != self._vectors.shape[1]:
            raise ValueError("기존 색인과 임베딩 차원이 다릅니다. 색인을 다시 만드세요.")
        index = {chunk.id: position for position, chunk in enumerate(self._chunks)}
        appended_chunks: list[Chunk] = []
        appended_vectors: list[np.ndarray] = []
        for chunk, vector in zip(chunks, incoming, strict=True):
            position = index.get(chunk.id)
            if position is None:
                index[chunk.id] = len(self._chunks) + len(appended_chunks)
                appended_chunks.append(chunk)
                appended_vectors.append(vector)
            else:
                self._chunks[position] = chunk
                assert self._vectors is not None
                self._vectors[position] = vector
        if appended_chunks:
            stacked = np.vstack(appended_vectors)
            self._chunks.extend(appended_chunks)
            if self._vectors is None:
                self._vectors = stacked
            else:
                self._vectors = np.vstack([self._vectors, stacked])

    def search(
        self,
        vector: list[float],
        *,
        top_k: int = 3,
        role: Literal["primary", "reference"] | None = None,
        paper_id: str | None = None,
        section: str | None = None,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        if self._vectors is None or not self._chunks:
            return []
        positions = [
            position
            for position, chunk in enumerate(self._chunks)
            if (role is None or chunk.role == role)
            and (paper_id is None or chunk.paper_id == paper_id)
            and (section is None or chunk.section == section)
        ]
        if not positions:
            return []
        query = np.asarray([vector], dtype=np.float32)
        if query.shape[1] != self._vectors.shape[1]:
            raise ValueError("질의 벡터 차원이 색인과 다릅니다.")
        scores = (_normalize(self._vectors[positions]) @ _normalize(query).T).ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            RetrievalResult(
                chunk=self._chunks[positions[i]],
                score=float(scores[i]),
                score_type="dense",
                dense_score=float(scores[i]),
            )
            for i in order
        ]

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = [chunk.model_dump(mode="json") for chunk in self._chunks]
        (path / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        vectors = self._vectors if self._vectors is not None else np.empty((0, 0))
        np.save(path / "vectors.npy", vectors)

    @classmethod
    def load(cls, directory: str | Path) -> NumpyVectorStore:
        path = Path(directory)
        store = cls()
        payload = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(path / "vectors.npy")
        store._chunks = [Chunk.model_validate(item) for item in payload]
        store._vectors = vectors if vectors.size else None
        return store
