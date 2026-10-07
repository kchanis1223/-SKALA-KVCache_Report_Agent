"""최종 보고서(Markdown)를 PDF로 내보냅니다.

보고서 생성 파이프라인은 그대로 Markdown을 다루고, 마지막 저장 단계에서만 PDF로
바꿉니다. 보고서가 쓰는 Markdown 일부(제목, 목록, 인용문, 표, 굵게, 코드, 링크)만
해석합니다.

한글 글꼴은 TrueType 파일을 찾아 PDF에 포함합니다. 찾지 못하면 reportlab에 내장된
한글 CID 글꼴(HYGothic)을 쓰고, 이때는 글꼴이 포함되지 않아 PDF 뷰어의 한글
글꼴을 사용합니다. `REPORT_FONT` 환경변수로 .ttf 경로를 직접 지정할 수 있습니다.
"""

import logging
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
    "/Library/Fonts/NanumGothic.ttf",
    str(Path.home() / "Library/Fonts/NanumGothic.ttf"),
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Linux (fonts-nanum)
    "C:/Windows/Fonts/malgun.ttf",  # Windows
)
CID_FALLBACK = "HYGothic-Medium"
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
logger = logging.getLogger(__name__)
BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")


def register_font() -> str:
    """사용할 한글 글꼴 이름을 돌려줍니다. TrueType이 있으면 포함(embed)합니다."""
    candidates = [os.environ.get("REPORT_FONT", ""), *FONT_CANDIDATES]
    for path in filter(None, candidates):
        if Path(path).is_file():
            try:
                pdfmetrics.registerFont(TTFont("ReportFont", path))
            except Exception:  # noqa: BLE001 - CFF 기반 글꼴 등은 다음 후보로
                continue
            name = "ReportFont"
            break
    else:
        logger.warning(
            "한글 TrueType 글꼴을 찾지 못해 내장 CID 글꼴을 씁니다. 일부 뷰어에서 한글이 "
            "보이지 않을 수 있으니 REPORT_FONT에 .ttf 경로를 지정하세요."
        )
        pdfmetrics.registerFont(UnicodeCIDFont(CID_FALLBACK))
        name = CID_FALLBACK
    # 굵은 글꼴 파일이 없어도 <b>가 오류 없이 처리되도록 같은 글꼴로 묶습니다.
    pdfmetrics.registerFontFamily(name, normal=name, bold=name, italic=name, boldItalic=name)
    return name


def _styles(font: str) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "body", fontName=font, fontSize=10, leading=15, alignment=TA_LEFT, wordWrap="CJK"
    )
    return {
        "body": base,
        "title": ParagraphStyle("title", parent=base, fontSize=18, leading=24, spaceAfter=10),
        "h2": ParagraphStyle(
            "h2", parent=base, fontSize=14, leading=19, spaceBefore=12, spaceAfter=5
        ),
        "h3": ParagraphStyle(
            "h3", parent=base, fontSize=11.5, leading=16, spaceBefore=8, spaceAfter=3
        ),
        "quote": ParagraphStyle(
            "quote", parent=base, textColor=colors.HexColor("#555555"), leftIndent=8
        ),
        "cell": ParagraphStyle("cell", parent=base, fontSize=9, leading=12.5),
    }


def inline(text: str) -> str:
    """Markdown 인라인 문법을 reportlab Paragraph 마크업으로 바꿉니다."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font color="#444444">\1</font>', text)
    return re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<link href="\2" color="#1a56b0">\1</link>', text
    )


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table(rows: list[str], styles, width: float) -> Table:
    body = [_cells(row) for row in rows if not TABLE_SEPARATOR.match(row)]
    columns = max(len(row) for row in body)
    data = [
        [Paragraph(inline(cell), styles["cell"]) for cell in row + [""] * (columns - len(row))]
        for row in body
    ]
    table = Table(data, colWidths=[width / columns] * columns, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def markdown_flowables(markdown: str, styles, width: float) -> list:
    story, paragraph, bullets, table = [], [], [], []

    def flush():
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
            story.append(Spacer(1, 4))
            paragraph.clear()
        if bullets:
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(inline(b), styles["body"])) for b in bullets],
                    bulletType="bullet",
                    bulletFontName=styles["body"].fontName,
                    leftIndent=12,
                )
            )
            story.append(Spacer(1, 4))
            bullets.clear()
        if table:
            story.append(_table(table, styles, width))
            story.append(Spacer(1, 6))
            table.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if TABLE_ROW.match(line):
            if not table:
                flush()
            table.append(line)
            continue
        if table:
            flush()
        bullet = BULLET.match(line)
        if not stripped:
            flush()
        elif stripped.startswith("# "):
            flush()
            story.append(Paragraph(inline(stripped[2:]), styles["title"]))
            story.append(HRFlowable(width="100%", color=colors.HexColor("#333333")))
            story.append(Spacer(1, 6))
        elif stripped.startswith("## "):
            flush()
            story.append(Paragraph(inline(stripped[3:]), styles["h2"]))
        elif stripped.startswith("### "):
            flush()
            story.append(Paragraph(inline(stripped[4:]), styles["h3"]))
        elif stripped.startswith(">"):
            flush()
            story.append(Paragraph(inline(stripped.lstrip("> ")), styles["quote"]))
            story.append(Spacer(1, 4))
        elif bullet:
            if paragraph:
                flush()
            bullets.append(bullet.group(2))
        elif bullets:
            bullets[-1] += " " + stripped  # 목록 항목이 다음 줄로 이어짐
        else:
            paragraph.append(stripped)
    flush()
    return story


def markdown_to_pdf(markdown: str, path: str | Path, title: str = "KV cache 기술 비교 보고서"):
    """Markdown 보고서를 A4 PDF로 저장합니다."""
    font = register_font()
    styles = _styles(font)
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, str(document.page))
        canvas.restoreState()

    doc.build(
        markdown_flowables(markdown, styles, doc.width), onFirstPage=footer, onLaterPages=footer
    )
    return Path(path)
