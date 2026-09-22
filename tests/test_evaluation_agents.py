import json

import pytest

from skala_agent.agents.evaluation_rubrics import market_details
from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.contracts import SearchDocument
from skala_agent.schemas import Evidence, Technology
from skala_agent.workflow.graph import build_graph, initial_state

TECHS = [
    Technology(id="turboquant", name="TurboQuant", camp="sw"),
    Technology(id="itme", name="ITME", camp="hw"),
]


class SearchFixture:
    def __init__(self, kind="official", empty=False):
        self.kind, self.empty, self.calls = kind, empty, []

    def search(self, query):
        self.calls.append(query)
        if self.empty:
            return []
        return [
            SearchDocument(
                id=f"source-{i}",
                title="Synthetic fixture",
                source_type=self.kind,
                url=f"https://example.org/{i}",
                content=f"Fixture passage {i}. TurboQuant and ITME.",
            )
            for i in range(2)
        ]


class ModelFixture:
    def __init__(self, answers=None, alter=None):
        self.answers, self.alter, self.calls = answers, alter, []

    def invoke(self, messages):
        self.calls.append(messages)
        payload = json.loads(messages[1]["content"])
        if "evidence" in payload:
            return json.dumps(
                {"results": [{"id": e["id"], "supports_claim": False} for e in payload["evidence"]]}
            )
        findings = []
        questions = payload.get("questions", []) if isinstance(payload, dict) else []
        for q in questions:
            answer = (
                self.answers.get(q["id"], "unknown")
                if self.answers is not None
                else "no"
                if q["id"] == "dependency"
                else "yes"
            )
            findings.append(
                {
                    "question_id": q["id"],
                    "answer": answer,
                    "rationale": "test only",
                    "citations": []
                    if answer == "unknown"
                    else [
                        {"source_id": d["id"], "quote": d["content"]}
                        for d in payload["sources"][:2]
                    ],
                }
            )
        if self.alter:
            self.alter(findings)
        return json.dumps({"findings": findings})


def assess(perspective, model=None, search=None):
    provider = EvaluationProvider(model or ModelFixture(), search or SearchFixture())
    return provider.assess(perspective, TECHS, "데이터센터", {}, [])


def test_two_technologies_have_primary_cited_trl_and_unverified_evidence():
    results, evidence = assess("trl", ModelFixture({"trl_3": "yes", "trl_6": "yes"}))
    assert [a.technology_id for a in results] == ["turboquant", "itme"]
    assert all(a.details.level == 6 and a.status == "assessed" for a in results)
    assert all(len(a.signals) == 9 and a.confidence == "medium" for a in results)
    assert all(not e.supports_claim for e in evidence)
    for a in results:
        assert set(a.evidence_ids) <= {e.id for e in evidence if e.technology_id == a.technology_id}


def test_news_cannot_establish_commercial_trl():
    results, _ = assess("trl", ModelFixture({"trl_9": "yes"}), SearchFixture(kind="news"))
    assert all(a.status == "pending" and a.details.level is None for a in results)
    assert all(a.signals[-1].grade == "중" for a in results)


def test_market_uses_three_axes_and_excludes_reverse_indicator_from_positive_scores():
    results, _ = assess("market")
    for result in results:
        assert result.details.demand == "수요 확실"
        assert result.details.adoption == "상용 채택"
        assert result.details.ecosystem == "확보"
        assert len(result.signals) == 11
    answers = {f"ecosystem_{i}": "yes" for i in range(1, 5)} | {"dependency": "yes"}
    results, _ = assess("market", ModelFixture(answers))
    assert all(a.details.ecosystem == "생태계 부재" for a in results)
    assert all(a.details.dependency_risks and a.status == "pending" for a in results)


def test_empty_search_does_not_call_llm_or_invent_negative_market_verdict():
    model = ModelFixture()
    results, evidence = assess("market", model, SearchFixture(empty=True))
    assert not model.calls and not evidence
    assert all(a.status == "pending" and a.confidence == "low" for a in results)
    assert all(a.details.adoption is None for a in results)
    assert all(s.grade == "하" for a in results for s in a.signals)


@pytest.mark.parametrize(
    "alter",
    [
        lambda findings: findings.pop(),
        lambda findings: findings.append(findings[0]),
        lambda findings: findings[0]["citations"][0].update(source_id="invented"),
        lambda findings: findings[0]["citations"][0].update(quote="not found in source"),
    ],
)
def test_bad_question_or_source_references_fail_closed(alter):
    results, evidence = assess("trl", ModelFixture(alter=alter))
    assert not evidence
    assert all(a.status == "failed" and a.error.code == "ModelOutputError" for a in results)
    assert all(not a.error.retryable for a in results)


def test_uncited_yes_is_downgraded_and_same_source_counts_once():
    def remove_citations(findings):
        for f in findings:
            f["citations"] = []

    results, evidence = assess("trl", ModelFixture(alter=remove_citations))
    assert all(a.status == "pending" for a in results) and not evidence

    class OneSource(SearchFixture):
        def search(self, query):
            return super().search(query)[:1]

    results, _ = assess("trl", search=OneSource())
    assert all(a.confidence == "low" for a in results)


def test_failure_of_one_technology_keeps_the_other_result():
    class FailingSearch(SearchFixture):
        def search(self, query):
            if '"ITME"' in query:
                raise TimeoutError("private details")
            return super().search(query)

    results, _ = assess("trl", search=FailingSearch())
    assert results[0].status == "assessed"
    assert results[1].status == "failed" and results[1].error.retryable
    assert "private details" not in results[1].model_dump_json()


def test_all_perspectives_and_graph_do_not_publish_unverified_verdicts():
    provider = EvaluationProvider(ModelFixture(), SearchFixture())
    result = build_graph(provider).invoke(initial_state())
    assert all(a.details.perspective == "domain" for a in result["analyses"]["domain"])
    assert all(a.status == "assessed" for a in result["analyses"]["stakeholder"])
    assert result["retry_count"] == 2
    assert "turboquant / trl: 판단 보류" in result["report"]
    ids = [e.id for e in result["evidence"]]
    assert len(ids) == len(set(ids))


def test_market_absence_is_different_from_explicit_negative_evidence():
    grades = {f"adoption_{i}": "하" for i in range(1, 4)}
    assert market_details(grades, "unknown").adoption is None
    assert market_details(grades, "unknown", {"adoption_1": "no"}).adoption == "연구 단계"
    positive = {f"ecosystem_{i}": "상" for i in range(1, 5)}
    assert market_details(positive, "unknown").ecosystem == "형성 중"
    assert market_details(positive, "no").ecosystem == "확보"


def test_retry_search_candidates_reach_model_and_ids_remain_stable():
    from skala_agent.schemas import MissingEvidence

    class ExpandedSearch(SearchFixture):
        def search(self, query):
            if query == "targeted retry":
                return [
                    SearchDocument(
                        id="new",
                        title="New evidence",
                        source_type="official",
                        url="https://new.example/paper",
                        content="New TurboQuant test evidence.",
                    )
                ]
            return super().search(query)

    model = ModelFixture()
    provider = EvaluationProvider(model, ExpandedSearch())
    missing = MissingEvidence(
        technology_id="turboquant",
        perspective="trl",
        reason="missing",
        kind="missing_source",
        claim="test",
        queries=["targeted retry"],
    )
    extra = provider.search_missing([missing])
    assert len(extra) == 1 and not extra[0].supports_claim
    first, sources = provider.assess("trl", TECHS[:1], "datacenter", {}, extra)
    payload = json.loads(model.calls[0][1]["content"])
    assert payload["sources"][0]["url"] == "https://new.example/paper"
    second, _ = provider.assess("trl", TECHS[:1], "datacenter", {}, extra + sources)
    assert first[0].evidence_ids == second[0].evidence_ids


def test_provider_updates_supports_claim_without_changing_evidence_id():
    class ClaimModel:
        def invoke_structured(self, messages, schema):
            payload = json.loads(messages[1]["content"])
            return json.dumps(
                {"results": [{"id": e["id"], "supports_claim": True} for e in payload["evidence"]]}
            )

    provider = EvaluationProvider(ClaimModel(), SearchFixture())
    evidence = Evidence(
        id="claim-1",
        technology_id="turboquant",
        claim="TurboQuant: 메모리 사용량 감소",
        url="https://example.org/claim",
        title="Claim fixture",
        excerpt="Memory usage decreased in the measured workload.",
        source_type="paper",
    )

    updated = provider.validate_evidence([evidence])

    assert updated == [evidence.model_copy(update={"supports_claim": True})]
