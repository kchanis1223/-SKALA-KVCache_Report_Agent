import pytest

from skala_agent.retrieval.sections import UNKNOWN_SECTION, sections_by_page


def test_pages_before_first_bookmark_use_default():
    table = sections_by_page([(3, "Method")], 4)
    assert table[1] == UNKNOWN_SECTION
    assert table[2] == UNKNOWN_SECTION
    assert table[3] == "Method"


def test_section_carries_forward_until_next_bookmark():
    table = sections_by_page([(1, "Introduction"), (4, "Evaluation")], 5)
    assert [table[p] for p in range(1, 6)] == [
        "Introduction",
        "Introduction",
        "Introduction",
        "Evaluation",
        "Evaluation",
    ]


def test_last_bookmark_on_a_page_wins():
    outline = [(2, "2 Background"), (2, "2.1 Memory Hierarchy")]
    assert sections_by_page(outline, 2)[2] == "2.1 Memory Hierarchy"


def test_titles_are_whitespace_normalized():
    table = sections_by_page([(1, "  3.1   Architecture\n Design ")], 1)
    assert table[1] == "3.1 Architecture Design"


def test_bookmarks_outside_page_range_are_ignored():
    assert sections_by_page([(9, "Appendix")], 2)[2] == UNKNOWN_SECTION


def test_empty_outline_gives_default_for_every_page():
    assert sections_by_page([], 3) == dict.fromkeys([1, 2, 3], UNKNOWN_SECTION)


def test_negative_page_count_rejected():
    with pytest.raises(ValueError):
        sections_by_page([], -1)
