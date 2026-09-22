"""김동찬 담당: 저장된 색인을 Retriever로 되살리는 진입점.

기술 조사·도메인 평가에서 논문 RAG를 붙일 때 이 함수 하나만 호출하면 됩니다.
색인을 만든 모델을 manifest에서 읽어 질의 임베딩에 같은 모델을 씁니다.
"""

from __future__ import annotations

from pathlib import Path

from skala_agent.retrieval.embedding import BgeM3Embedder
from skala_agent.retrieval.indexing import DEFAULT_INDEX_DIR, load_manifest
from skala_agent.retrieval.interfaces import Embedder
from skala_agent.retrieval.retriever import DenseRetriever
from skala_agent.retrieval.vector_store import NumpyVectorStore


class IndexNotBuiltError(RuntimeError):
    """색인이 없을 때. 워크플로는 이 예외를 잡아 RAG 없이 진행할 수 있습니다."""


def load_retriever(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    embedder: Embedder | None = None,
) -> DenseRetriever:
    """저장된 색인을 읽어 검색 가능한 Retriever를 돌려줍니다.

    색인은 커밋하지 않으므로 먼저 `uv run skala-index`로 만들어야 합니다.
    """
    path = Path(index_dir)
    if not (path / "manifest.json").is_file():
        raise IndexNotBuiltError(
            f"색인이 없습니다: {path}. `uv run skala-index`로 먼저 생성하세요."
        )
    manifest = load_manifest(path)
    store = NumpyVectorStore.load(path)
    return DenseRetriever(embedder or BgeM3Embedder(manifest.embedding_model), store)


def try_load_retriever(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    embedder: Embedder | None = None,
) -> DenseRetriever | None:
    """색인이 없으면 None. RAG를 선택적으로 붙이는 호출부에서 사용합니다."""
    try:
        return load_retriever(index_dir, embedder=embedder)
    except IndexNotBuiltError:
        return None
