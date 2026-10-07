"""최종 보고서 PDF 변환."""

from pypdf import PdfReader
from reportlab.platypus import ListFlowable, Paragraph, Table

from skala_agent.pdf import _styles, inline, markdown_flowables, markdown_to_pdf, register_font

REPORT = """# KV cache 기술 비교 보고서

## SUMMARY
TurboQuant는 **저비트** 압축 [1]. A & B <c>

## 5. 종합 비교 및 시사점
| 항목 | TurboQuant | ITME |
|---|---|---|
| 방식 | SW | HW |

## REFERENCE

- [1] [Title | Site](https://example.org/a?x=1&y=2)
"""


def test_inline_escapes_and_converts_markup():
    out = inline("**굵게** `code` [t](https://e.org/a?x=1&y=2) <x> & y")
    assert "<b>굵게</b>" in out and "&lt;x&gt; &amp; y" in out
    assert '<link href="https://e.org/a?x=1&amp;y=2"' in out


def test_markdown_maps_to_headings_tables_and_lists():
    styles = _styles(register_font())
    story = markdown_flowables(REPORT, styles, 400)
    assert sum(isinstance(f, Table) for f in story) == 1
    assert sum(isinstance(f, ListFlowable) for f in story) == 1
    headings = [f.text for f in story if isinstance(f, Paragraph) and f.style.name == "h2"]
    assert headings == ["SUMMARY", "5. 종합 비교 및 시사점", "REFERENCE"]


def test_markdown_to_pdf_writes_a_readable_pdf(tmp_path):
    path = markdown_to_pdf(REPORT, tmp_path / "r.pdf")
    reader = PdfReader(path)
    assert len(reader.pages) == 1
    assert reader.metadata.title == "KV cache 기술 비교 보고서"
