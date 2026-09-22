import re

import pytest

from skala_agent.retrieval.chunker import UNKNOWN_SECTION, chunk_pages, split_page_text
from skala_agent.retrieval.config import ChunkingConfig


class WordTokenizer:
    """공백 단위 토크나이저. 청킹 로직 검증 전용이며 실제 인덱싱에는 쓰지 않습니다."""

    name = "test-word"

    def offsets(self, text):
        return [m.span() for m in re.finditer(r"\S+", text)]


TOKENIZER = WordTokenizer()
SMALL = ChunkingConfig(version="test", tokenizer="test-word", chunk_size_tokens=5, overlap_tokens=2)


def test_split_returns_empty_for_blank_page():
    assert split_page_text("   \n ", TOKENIZER, SMALL) == []


def test_split_keeps_short_page_as_single_chunk():
    assert split_page_text("a b c", TOKENIZER, SMALL) == ["a b c"]


def test_split_applies_overlap_between_chunks():
    text = " ".join(str(i) for i in range(1, 12))
    pieces = split_page_text(text, TOKENIZER, SMALL)
    assert pieces[0] == "1 2 3 4 5"
    assert pieces[1] == "4 5 6 7 8"  # overlap 2 -> step 3
    assert pieces[-1].endswith("11")


def test_split_covers_every_token():
    text = " ".join(str(i) for i in range(1, 20))
    pieces = split_page_text(text, TOKENIZER, SMALL)
    covered = {token for piece in pieces for token in piece.split()}
    assert covered == {str(i) for i in range(1, 20)}


def test_chunk_pages_builds_stable_ids_and_metadata():
    pages = [(1, "a b c"), (2, "d e f")]
    chunks = chunk_pages(
        pages,
        paper_id="itme",
        camp="hw",
        role="primary",
        source_url="https://arxiv.org/abs/2606.12556",
        tokenizer=TOKENIZER,
        config=SMALL,
    )
    assert [c.id for c in chunks] == ["itme-p1-1", "itme-p2-1"]
    assert all(c.paper_id == "itme" and c.camp == "hw" for c in chunks)
    assert all(c.section == UNKNOWN_SECTION for c in chunks)


def test_chunk_pages_skips_empty_page_without_shifting_page_numbers():
    pages = [(1, "a b"), (2, ""), (3, "c d")]
    chunks = chunk_pages(
        pages,
        paper_id="tq",
        camp="sw",
        role="primary",
        source_url="https://example.com/tq",
        tokenizer=TOKENIZER,
        config=SMALL,
    )
    assert [c.id for c in chunks] == ["tq-p1-1", "tq-p3-1"]
    assert [c.page for c in chunks] == [1, 3]


def test_chunk_pages_numbers_sequence_within_page():
    pages = [(7, " ".join(str(i) for i in range(1, 12)))]
    chunks = chunk_pages(
        pages,
        paper_id="tq",
        camp="sw",
        role="primary",
        source_url="https://example.com/tq",
        tokenizer=TOKENIZER,
        config=SMALL,
    )
    assert [c.id for c in chunks] == ["tq-p7-1", "tq-p7-2", "tq-p7-3"]


def test_chunk_pages_uses_section_resolver():
    chunks = chunk_pages(
        [(4, "a b")],
        paper_id="tq",
        camp="sw",
        role="primary",
        source_url="https://example.com/tq",
        tokenizer=TOKENIZER,
        config=SMALL,
        section_of=lambda page: f"Related Work (p{page})",
    )
    assert chunks[0].section == "Related Work (p4)"


def test_config_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size_tokens=100, overlap_tokens=100)


def test_reference_chunk_overrides_inherited_section():
    """북마크가 없는 논문은 참고문헌이 직전 본문 섹션을 물려받습니다."""
    refs = "References " + " ".join(
        f"[{i}] Author, A. Title of the work. Venue {2000 + i}." for i in range(1, 12)
    )
    chunks = chunk_pages(
        [(24, refs)],
        paper_id="tq",
        camp="sw",
        role="primary",
        source_url="https://example.com/tq",
        tokenizer=TOKENIZER,
        config=ChunkingConfig(version="t", tokenizer="test-word", chunk_size_tokens=500),
        section_of=lambda _page: "Near Neighbour Search Experiments",
    )
    assert chunks[0].section == "References"


def test_body_chunk_keeps_bookmark_section():
    chunks = chunk_pages(
        [(4, "본문입니다 [1] 인용 하나만 있습니다.")],
        paper_id="tq",
        camp="sw",
        role="primary",
        source_url="https://example.com/tq",
        tokenizer=TOKENIZER,
        config=SMALL,
        section_of=lambda _page: "Related Work",
    )
    assert chunks[0].section == "Related Work"
