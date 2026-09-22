from skala_agent.retrieval.pdf_parser import (
    find_running_lines,
    normalize_page_text,
    strip_running_lines,
)


def test_normalize_joins_hyphenated_line_break():
    assert "quantization" in normalize_page_text("online quan-\ntization methods")


def test_normalize_can_keep_hyphen_for_compound_words():
    text = "offline (data-\ndependent) methods"
    assert "data-dependent" in normalize_page_text(text, keep_hyphen=True)


def test_normalize_joins_soft_line_break_but_keeps_sentence_end():
    text = "vector quantization\ntheory started here. \nNext sentence."
    result = normalize_page_text(text)
    assert "quantization theory" in result
    assert "here.\nNext sentence." in result


def test_normalize_drops_page_number_only_lines():
    assert normalize_page_text("Method\n12\nEvaluation") == "Method\n\nEvaluation"


def test_normalize_handles_empty_input():
    assert normalize_page_text("") == ""


def test_find_running_lines_detects_repeated_header():
    pages = [f"ITME: Tiered Memory\nbody {i}" for i in range(5)]
    assert "ITME: Tiered Memory" in find_running_lines(pages)


def test_find_running_lines_needs_at_least_three_pages():
    assert find_running_lines(["header\na", "header\nb"]) == set()


def test_strip_running_lines_removes_only_matched_lines():
    page = "ITME: Tiered Memory\n실제 본문"
    assert strip_running_lines(page, {"ITME: Tiered Memory"}) == "실제 본문"
