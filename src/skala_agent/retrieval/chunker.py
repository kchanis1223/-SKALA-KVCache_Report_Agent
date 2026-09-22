"""김동찬 담당: 토큰 기반 청킹.

`ChunkingConfig`를 그대로 따르며 청크는 페이지 경계를 넘지 않습니다.
토크나이저는 Protocol로 주입받아, 모델 다운로드 없이 청킹 로직을 테스트할 수 있습니다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Protocol

from skala_agent.retrieval.config import DEFAULT_CHUNKING, ChunkingConfig, chunk_id
from skala_agent.schemas import Chunk

UNKNOWN_SECTION = "unknown"


class Tokenizer(Protocol):
    """토큰 경계를 원문 문자 위치로 되돌려 줄 수 있어야 합니다."""

    name: str

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """토큰별 (시작, 끝) 문자 오프셋을 순서대로 반환합니다."""
        ...


def split_page_text(
    text: str,
    tokenizer: Tokenizer,
    config: ChunkingConfig = DEFAULT_CHUNKING,
) -> list[str]:
    """한 페이지 텍스트를 config 크기로 자릅니다. 페이지 경계는 넘지 않습니다."""
    if not text.strip():
        return []
    offsets = tokenizer.offsets(text)
    if not offsets:
        return []
    size = config.chunk_size_tokens
    step = size - config.overlap_tokens
    pieces: list[str] = []
    start_index = 0
    while start_index < len(offsets):
        window = offsets[start_index : start_index + size]
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            pieces.append(piece)
        if start_index + size >= len(offsets):
            break
        start_index += step
    return pieces


def chunk_pages(
    pages: Sequence[tuple[int, str]],
    *,
    paper_id: str,
    camp: Literal["sw", "hw"],
    role: Literal["primary", "reference"],
    source_url: str,
    tokenizer: Tokenizer,
    config: ChunkingConfig = DEFAULT_CHUNKING,
    section_of: Callable[[int], str] | None = None,
) -> list[Chunk]:
    """페이지 목록을 Chunk 목록으로 변환합니다.

    빈 페이지는 청크를 만들지 않지만 페이지 번호는 건너뛰지 않으므로
    chunk ID의 page 값은 원문 페이지와 항상 일치합니다.
    """
    resolve_section = section_of or (lambda _page: UNKNOWN_SECTION)
    chunks: list[Chunk] = []
    for page, text in pages:
        section = resolve_section(page) or UNKNOWN_SECTION
        for sequence, piece in enumerate(split_page_text(text, tokenizer, config), 1):
            chunks.append(
                Chunk(
                    id=chunk_id(paper_id, page, sequence),
                    text=piece,
                    paper_id=paper_id,
                    camp=camp,
                    role=role,
                    section=section,
                    page=page,
                    source_url=source_url,
                )
            )
    return chunks


class HFTokenizer:
    """HuggingFace fast tokenizer 어댑터. 기본 대상은 BAAI/bge-m3입니다.

    모델 파일은 네트워크가 막힌 환경에서 받을 수 없으므로,
    내려받은 `tokenizer.json` 경로로도 생성할 수 있게 두 경로를 모두 제공합니다.
    """

    def __init__(self, tokenizer, name: str) -> None:
        self._tokenizer = tokenizer
        self.name = name

    @classmethod
    def from_file(cls, path: str | Path) -> HFTokenizer:
        tokenizer = cls._backend().from_file(str(path))
        return cls(tokenizer, Path(path).stem)

    @classmethod
    def from_pretrained(cls, name: str = DEFAULT_CHUNKING.tokenizer) -> HFTokenizer:
        return cls(cls._backend().from_pretrained(name), name)

    @staticmethod
    def _backend():
        try:
            from tokenizers import Tokenizer as Backend
        except ImportError as exc:  # pragma: no cover - 설치 환경에 따름
            message = "토큰 청킹에는 tokenizers가 필요합니다. `uv sync`로 설치하세요."
            raise RuntimeError(message) from exc
        return Backend

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return [tuple(pair) for pair in encoding.offsets]
