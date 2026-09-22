"""재검색 질의 예산: (기술, 관점) 그룹당 최대 2질의."""

import pytest

from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.contracts import SearchDocument
from skala_agent.schemas import MissingEvidence


class RecordingSearch:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [
            SearchDocument(
                id=f"doc-{len(self.queries)}",
                title=f"Document {len(self.queries)}",
                url=f"https://example.org/{len(self.queries)}",
                content="Synthetic retry document.",
                source_type="official",
            )
        ]


class StubModel:
    def invoke_structured(self, messages, schema):  # pragma: no cover - 호출되지 않음
        raise AssertionError("search_missing은 모델을 호출하지 않습니다.")


def make_provider():
    search = RecordingSearch()
    return EvaluationProvider(StubModel(), search), search


def missing(technology_id, perspective, claim, queries, retryable=True):
    return MissingEvidence(
        technology_id=technology_id,
        perspective=perspective,
        reason="missing",
        kind="missing_source",
        claim=claim,
        queries=queries,
        retryable=retryable,
    )


def test_one_group_with_many_missing_items_spends_at_most_two_queries():
    """부족 항목이 여러 개여도 한 그룹의 한 회차 검색은 최대 2회다."""
    provider, search = make_provider()
    items = [
        missing("turboquant", "trl", f"질문 {n}", [f"turboquant trl 질문 {n}"]) for n in range(9)
    ]

    provider.search_missing(items)

    assert len(search.queries) == EvaluationProvider.QUERIES_PER_GROUP
    assert search.queries == ["turboquant trl 질문 0", "turboquant trl 질문 1"]


def test_each_group_has_an_independent_budget():
    """서로 다른 (기술, 관점) 그룹은 예산을 나눠 쓰지 않는다."""
    provider, search = make_provider()
    items = [
        missing("turboquant", "trl", "a", ["q1", "q2", "q3"]),
        missing("itme", "trl", "b", ["q4", "q5", "q6"]),
        missing("turboquant", "market", "c", ["q7", "q8", "q9"]),
    ]

    provider.search_missing(items)

    assert search.queries == ["q1", "q2", "q4", "q5", "q7", "q8"]


def test_non_retryable_items_are_not_searched():
    provider, search = make_provider()
    items = [
        missing("turboquant", "trl", "a", ["q1"], retryable=False),
        missing("itme", "trl", "b", ["q2"]),
    ]

    provider.search_missing(items)

    assert search.queries == ["q2"]


def test_duplicate_and_whitespace_variant_queries_are_deduplicated():
    """공백만 다른 질의는 같은 질의로 보고, 먼저 나온 것이 예산을 차지한다."""
    provider, search = make_provider()
    items = [
        missing("turboquant", "trl", "a", ["  turboquant  trl  a  ", "turboquant trl a"]),
        missing("turboquant", "trl", "b", ["turboquant trl a", "turboquant trl b"]),
    ]

    provider.search_missing(items)

    assert search.queries == ["turboquant trl a", "turboquant trl b"]


def test_empty_target_list_performs_no_search():
    provider, search = make_provider()
    assert provider.search_missing([]) == []
    assert search.queries == []


def test_retry_evidence_claim_carries_the_missing_claim_not_the_document_title():
    """claim에 문서 제목이 아니라 부족 항목의 주장이 들어간다.

    validate_evidence는 "발췌가 주장을 직접 지지하는가"를 판정합니다. claim이
    문서 제목이면 판정할 주장 자체가 없어 재검색 근거가 전부 기각됩니다.
    "추가 검색 자료" 표식은 web_evaluation이 재검색 근거를 식별하는 데 쓰므로
    유지합니다.
    """
    provider, search = make_provider()
    item = missing("turboquant", "trl", "TRL 6 수준의 시연 사례가 있는가", ["q1"])

    evidence = provider.search_missing([item])

    assert len(evidence) == 1
    claim = evidence[0].claim
    assert claim.startswith("turboquant: 추가 검색 자료")
    assert "TRL 6 수준의 시연 사례가 있는가" in claim
    assert evidence[0].title == "Document 1"
    assert evidence[0].title not in claim
    assert not evidence[0].supports_claim


def test_same_document_from_different_groups_keeps_separate_evidence():
    """관점이 다르면 같은 문서라도 별도 근거로 남는다(소유 관점이 다름)."""
    provider, _ = make_provider()
    items = [
        missing("turboquant", "trl", "a", ["q1"]),
        missing("turboquant", "market", "a", ["q1"]),
    ]

    evidence = provider.search_missing(items)

    assert len({e.id for e in evidence}) == len(evidence)
    assert len(evidence) == 2


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_budget_constant_is_the_single_source_of_truth(monkeypatch, budget):
    """주석이 아니라 상수가 실제 제한을 결정한다."""
    provider, search = make_provider()
    monkeypatch.setattr(EvaluationProvider, "QUERIES_PER_GROUP", budget)
    provider.search_missing([missing("turboquant", "trl", "a", [f"q{n}" for n in range(5)])])
    assert len(search.queries) == budget
