"""김동찬 담당: PDF 북마크로 페이지별 section 결정.

정규식 헤딩 탐지는 논문마다 번호 체계가 달라 실패율이 높아, 북마크를 1순위로 씁니다.
매핑 로직은 순수 함수라 PDF 없이 테스트할 수 있습니다.
"""

from __future__ import annotations

import re
from pathlib import Path

UNKNOWN_SECTION = "unknown"
REFERENCES_SECTION = "References"

# 참고문헌 판정 기준. 실제 논문 3건(69청크)으로 맞춘 값입니다.
# 인용 밀도만 보면 Related Work와 구분되지 않아 연도 표기 수를 함께 봅니다.
#   참고문헌 목록: 글자/마커 183~269, 연도 8~30
#   Related Work: 글자/마커 290, 연도 1
_CITATION = re.compile(r"\[\d+(?:,\s*\d+)*\]")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
MIN_CITATIONS = 4
MAX_CHARS_PER_CITATION = 320
MIN_YEARS = 8


def looks_like_references(text: str) -> bool:
    """참고문헌 목록으로 보이면 True.

    북마크가 없는 논문은 참고문헌 페이지가 직전 본문 섹션 이름을 물려받습니다
    (turboquant p21~25). 본문 근거로 쓸 수 없는 청크이므로 따로 표시합니다.
    """
    if not text:
        return False
    citations = len(_CITATION.findall(text))
    if citations < MIN_CITATIONS:
        return False
    if len(text) // citations > MAX_CHARS_PER_CITATION:
        return False
    return len(_YEAR.findall(text)) >= MIN_YEARS


def sections_by_page(
    outline: list[tuple[int, str]],
    page_count: int,
    *,
    default: str = UNKNOWN_SECTION,
) -> dict[int, str]:
    """(페이지, 제목) 북마크 목록을 페이지별 section 표로 펼칩니다.

    한 페이지에 북마크가 여러 개면 가장 마지막 것을 씁니다. 하위 절이 뒤에 오므로
    그 페이지에서 가장 구체적인 제목이 선택됩니다. 첫 북마크 이전 페이지는 default입니다.
    """
    if page_count < 0:
        raise ValueError("page_count는 0 이상이어야 합니다.")
    titles: dict[int, str] = {}
    for page, title in sorted(outline, key=lambda item: item[0]):
        cleaned = " ".join(title.split())
        if cleaned and 1 <= page <= page_count:
            titles[page] = cleaned
    table: dict[int, str] = {}
    current = default
    for page in range(1, page_count + 1):
        current = titles.get(page, current)
        table[page] = current
    return table


def read_outline(path: str | Path) -> list[tuple[int, str]]:
    """PDF 북마크를 (페이지, 제목) 목록으로 읽습니다. 북마크가 없으면 빈 목록."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 설치 환경에 따름
        message = "북마크 읽기에는 pypdf가 필요합니다. `uv sync`로 설치하세요."
        raise RuntimeError(message) from exc
    reader = PdfReader(str(path))
    found: list[tuple[int, str]] = []

    def walk(entries) -> None:
        for entry in entries:
            if isinstance(entry, list):
                walk(entry)
                continue
            try:
                page = reader.get_destination_page_number(entry) + 1
            except Exception:  # noqa: BLE001 - 깨진 목적지는 건너뜁니다.
                continue
            found.append((page, str(entry.title)))

    try:
        walk(reader.outline)
    except Exception:  # noqa: BLE001 - 북마크가 없거나 손상된 PDF
        return []
    return found


def section_resolver(path: str | Path, page_count: int):
    """`chunk_pages(section_of=...)`에 바로 넘길 수 있는 조회 함수를 만듭니다."""
    table = sections_by_page(read_outline(path), page_count)
    return lambda page: table.get(page, UNKNOWN_SECTION)
