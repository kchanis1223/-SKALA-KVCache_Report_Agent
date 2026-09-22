"""김동찬 담당: PDF 페이지별 텍스트 추출과 정규화.

`PDFParser` 프로토콜 구현. 정규화 함수는 PDF 없이 단위 테스트할 수 있도록
순수 함수로 분리했습니다.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# 줄바꿈으로 끊긴 단어: "data-\ndependent" -> "data-dependent"
_HYPHEN_BREAK = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
# 문단 내부 줄바꿈: 앞줄이 문장 종결이 아니면 공백으로 이음
_SOFT_BREAK = re.compile(r"(?<=[^\s.!?:;])\n(?=[^\s\n])")
_BLANK_RUN = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]{2,}")
# 페이지 번호만 있는 줄
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:[-–—]\s*)?\d{1,4}\s*(?:[-–—])?\s*$")


def normalize_page_text(text: str, *, keep_hyphen: bool = False) -> str:
    """추출 원문을 검색용 텍스트로 정리합니다.

    줄 끝에서 끊긴 단어를 잇고, 문단 내부 줄바꿈을 공백으로 바꾸고,
    페이지 번호만 있는 줄과 중복 공백을 제거합니다.

    keep_hyphen=False는 조판 하이픈("quan-\ntization" -> "quantization")을 기준으로
    합니다. 실제 복합어("data-\ndependent")는 하이픈을 잃지만, 논문 2단 조판에서는
    조판 하이픈이 복합어보다 훨씬 잦아 이쪽을 기본값으로 두었습니다.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _HYPHEN_BREAK.sub("-" if keep_hyphen else "", text)
    # 페이지 번호 줄은 삭제하지 않고 빈 줄로 바꿔 앞뒤 문단이 붙지 않게 합니다.
    kept = ["" if _PAGE_NUMBER_LINE.match(line) else line for line in text.split("\n")]
    text = "\n".join(kept)
    text = _SOFT_BREAK.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip()


def find_running_lines(pages: list[str], *, min_ratio: float = 0.6) -> set[str]:
    """머리말·꼬리말처럼 여러 페이지에 반복되는 짧은 줄을 찾습니다.

    3페이지 미만 문서는 판단 근거가 부족하므로 빈 집합을 돌려줍니다.
    """
    if len(pages) < 3:
        return set()
    counter: Counter[str] = Counter()
    for page in pages:
        lines = {line.strip() for line in page.split("\n") if 0 < len(line.strip()) <= 120}
        counter.update(lines)
    threshold = max(2, int(len(pages) * min_ratio))
    return {line for line, count in counter.items() if count >= threshold}


def strip_running_lines(page: str, running: set[str]) -> str:
    if not running:
        return page
    kept = [line for line in page.split("\n") if line.strip() not in running]
    return "\n".join(kept).strip()


class PyPDFParser:
    """pypdf 기반 `PDFParser` 구현. 페이지 번호는 1부터 시작합니다.

    min_chars 미만인 페이지는 그림·표만 있는 페이지로 보고 빈 문자열로 남깁니다.
    페이지 번호를 유지해야 chunk ID가 안정적이므로 페이지 자체를 버리지 않습니다.
    """

    def __init__(self, *, min_chars: int = 40, drop_running_lines: bool = True) -> None:
        if min_chars < 0:
            raise ValueError("min_chars는 0 이상이어야 합니다.")
        self.min_chars = min_chars
        self.drop_running_lines = drop_running_lines

    def parse(self, path: Path) -> list[tuple[int, str]]:
        raw = self._extract(path)
        normalized = [normalize_page_text(text) for text in raw]
        running = find_running_lines(normalized) if self.drop_running_lines else set()
        pages: list[tuple[int, str]] = []
        for number, text in enumerate(normalized, 1):
            cleaned = strip_running_lines(text, running)
            if len(cleaned.replace(" ", "")) < self.min_chars:
                cleaned = ""
            pages.append((number, cleaned))
        return pages

    @staticmethod
    def _extract(path: Path) -> list[str]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - 설치 환경에 따름
            raise RuntimeError("PDF 파싱에는 pypdf가 필요합니다. `uv sync`로 설치하세요.") from exc
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {path}")
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
