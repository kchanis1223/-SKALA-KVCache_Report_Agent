"""김동찬 담당: configs/documents.json -> 파싱 -> 섹션 -> 청크 연결."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, HttpUrl

from skala_agent.retrieval.chunker import Tokenizer, chunk_pages
from skala_agent.retrieval.config import DEFAULT_CHUNKING, ChunkingConfig
from skala_agent.retrieval.pdf_parser import PyPDFParser
from skala_agent.retrieval.sections import section_resolver
from skala_agent.schemas import Chunk

DOCUMENTS_CONFIG = Path("configs/documents.json")


class DocumentSource(BaseModel):
    """`configs/documents.json`의 문서 한 건. PDF 원문은 커밋하지 않습니다."""

    paper_id: str
    camp: Literal["sw", "hw"]
    role: Literal["primary", "reference"]
    url: HttpUrl
    title: str | None = None
    pages: int | None = None
    local_path: str | None = None

    @property
    def pdf_path(self) -> Path:
        return Path(self.local_path or f"data/raw/{self.paper_id}.pdf")


def load_documents(path: str | Path = DOCUMENTS_CONFIG) -> list[DocumentSource]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DocumentSource.model_validate(item) for item in payload["documents"]]


def build_chunks(
    document: DocumentSource,
    tokenizer: Tokenizer,
    *,
    config: ChunkingConfig = DEFAULT_CHUNKING,
    parser: PyPDFParser | None = None,
    root: str | Path = ".",
) -> list[Chunk]:
    """문서 1건을 Chunk 목록으로 만듭니다. PDF가 없으면 FileNotFoundError."""
    pdf_path = Path(root) / document.pdf_path
    pages = (parser or PyPDFParser()).parse(pdf_path)
    return chunk_pages(
        pages,
        paper_id=document.paper_id,
        camp=document.camp,
        role=document.role,
        source_url=str(document.url),
        tokenizer=tokenizer,
        config=config,
        section_of=section_resolver(pdf_path, len(pages)),
    )


def build_all_chunks(
    tokenizer: Tokenizer,
    *,
    path: str | Path = DOCUMENTS_CONFIG,
    config: ChunkingConfig = DEFAULT_CHUNKING,
    root: str | Path = ".",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_documents(path):
        chunks.extend(build_chunks(document, tokenizer, config=config, root=root))
    return chunks
