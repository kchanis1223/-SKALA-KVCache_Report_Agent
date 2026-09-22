"""김동찬 담당: 청크 -> 임베딩 -> 색인 저장과 재현용 manifest."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from skala_agent.retrieval.config import DEFAULT_CHUNKING, ChunkingConfig
from skala_agent.retrieval.interfaces import Embedder
from skala_agent.retrieval.pipeline import DOCUMENTS_CONFIG, build_all_chunks, load_documents
from skala_agent.retrieval.vector_store import NumpyVectorStore
from skala_agent.schemas import Chunk

DEFAULT_INDEX_DIR = Path("index/bge-m3")


class IndexManifest(BaseModel):
    """색인을 다시 만들 때 필요한 설정 기록. 평가셋 재현성의 근거가 됩니다."""

    created_at: str
    embedding_model: str
    chunking: ChunkingConfig
    tokenizer: str
    chunk_count: int
    dimension: int
    documents: list[dict] = Field(default_factory=list)


def build_index_from_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    *,
    tokenizer_name: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    config: ChunkingConfig = DEFAULT_CHUNKING,
    documents: list[dict] | None = None,
    batch_size: int = 16,
) -> IndexManifest:
    """청크를 임베딩해 색인과 manifest를 저장합니다."""
    if not chunks:
        raise ValueError("색인할 청크가 없습니다.")
    store = NumpyVectorStore()
    dimension = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.encode([chunk.text for chunk in batch])
        if not vectors or not vectors[0]:
            raise ValueError("임베더가 빈 벡터를 반환했습니다.")
        dimension = len(vectors[0])
        store.upsert(batch, vectors)
    path = Path(index_dir)
    store.save(path)
    manifest = IndexManifest(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        embedding_model=getattr(embedder, "model_name", "unknown"),
        chunking=config,
        tokenizer=tokenizer_name,
        chunk_count=len(store),
        dimension=dimension,
        documents=documents or [],
    )
    (path / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def build_index(
    embedder: Embedder,
    tokenizer,
    *,
    documents_path: str | Path = DOCUMENTS_CONFIG,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    config: ChunkingConfig = DEFAULT_CHUNKING,
    root: str | Path = ".",
    batch_size: int = 16,
) -> IndexManifest:
    """documents.json의 모든 문서를 파싱·청킹·임베딩해 색인을 만듭니다."""
    chunks = build_all_chunks(tokenizer, path=documents_path, config=config, root=root)
    documents = [
        {"paper_id": d.paper_id, "role": d.role, "camp": d.camp, "pages": d.pages}
        for d in load_documents(documents_path)
    ]
    return build_index_from_chunks(
        chunks,
        embedder,
        tokenizer_name=getattr(tokenizer, "name", config.tokenizer),
        index_dir=index_dir,
        config=config,
        documents=documents,
        batch_size=batch_size,
    )


def load_manifest(index_dir: str | Path = DEFAULT_INDEX_DIR) -> IndexManifest:
    payload = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
    return IndexManifest.model_validate(payload)
