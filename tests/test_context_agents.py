"""합성 근거로 #6 판정·인용·RAG 계약을 검증합니다. 실제 품질 평가는 아닙니다."""

import json

import pytest
from test_evaluation_agents import TECHS, ModelFixture, SearchFixture, assess

from skala_agent.agents.context_rubrics import domain_details
from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.schemas import Chunk, RetrievalResult


def test_stakeholders_separate_absence_awareness_support_and_conflict():
    answers = {
        "gpu_vendor_support": "yes",
        "memory_vendor_aware": "yes",
        "cloud_operator_skeptical": "yes",
        "open_source_support": "yes",
        "open_source_skeptical": "yes",
    }
    results, evidence = assess("stakeholder", ModelFixture(answers))
    for result in results:
        assert [p.stance for p in result.details.positions] == [
            "지지",
            "유보",
            "회의적",
            "유보",
            "자료 없음",
        ]
        assert result.details.overall == "혼재"
        assert len(result.signals) == 15
        assert not result.details.positions[-1].evidence_ids
        assert result.details.positions[0].evidence_ids
    assert all(not e.supports_claim for e in evidence)


@pytest.mark.parametrize("stance, expected", [("support", "우호적"), ("skeptical", "부정적")])
def test_stakeholder_majority_excludes_absence(stance, expected):
    results, _ = assess("stakeholder", ModelFixture({f"gpu_vendor_{stance}": "yes"}))
    assert all(a.details.overall == expected for a in results)
    assert all(
        len([p for p in a.details.positions if p.stance == "자료 없음"]) == 4 for a in results
    )


@pytest.mark.parametrize("perspective", ["stakeholder", "domain"])
def test_no_documents_returns_pending_without_model_call(perspective):
    model = ModelFixture()
    results, evidence = assess(perspective, model, SearchFixture(empty=True))
    assert not model.calls and not evidence
    assert all(a.status == "pending" for a in results)
    if perspective == "stakeholder":
        assert all(p.stance == "자료 없음" for a in results for p in a.details.positions)
        assert all(a.details.overall is None for a in results)
    else:
        assert all(a.details.cost is None and a.details.sla_risk == "판단 불가" for a in results)


@pytest.mark.parametrize(
    "latency,accuracy,expected",
    [
        ("no", "no", "낮음"),
        ("yes", "no", "중간"),
        ("no", "yes", "중간"),
        ("yes", "yes", "높음"),
        ("unknown", "no", "판단 불가"),
        ("no", "unknown", "판단 불가"),
    ],
)
def test_sla_requires_both_latency_and_accuracy(latency, accuracy, expected):
    assert (
        domain_details({"latency_degraded": latency, "accuracy_loss": accuracy}).sla_risk
        == expected
    )


def test_operations_conflict_and_qualitative_cost_are_conservative():
    answers = {
        "cost_qualitative": "yes",
        "sw_only": "yes",
        "hw_required": "yes",
        "latency_degraded": "no",
        "accuracy_loss": "no",
        "isolation_risk": "yes",
    }
    results, _ = assess("domain", ModelFixture(answers))
    for a in results:
        assert a.details.cost == "개선 불명확" and a.confidence == "low"
        assert a.details.operations == "인프라 전환 전제"
        assert len(a.details.operational_risks) == 2
        assert "Retriever 미연결" in a.rationale


class RetrieverFixture:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            RetrievalResult(
                chunk=Chunk(
                    id=f"review-p{i}-1",
                    text=f"Cost decreased 20% on GPU A batch 8. Section {i}.",
                    paper_id="review",
                    camp="sw",
                    role=role,
                    section=f"Section {i}",
                    page=i,
                    source_url="https://example.org/review",
                ),
                score=1.0,
            )
            for i, role in ((1, "primary"), (2, "reference"))
        ]


class MeasurementModel(ModelFixture):
    def __init__(self, *, tamper=False):
        super().__init__(
            {
                "cost_measured": "yes",
                "cost_conditional": "yes",
                "latency_degraded": "no",
                "accuracy_loss": "no",
                "sw_only": "yes",
            }
        )
        self.tamper = tamper

    def invoke(self, messages):
        output = json.loads(super().invoke(messages))
        payload = json.loads(messages[1]["content"])
        finding = next(f for f in output["findings"] if f["question_id"] == "cost_measured")
        finding["measurements"] = [
            {
                "source_id": payload["sources"][0]["id"],
                "metric": "Cost",
                "value": "99%" if self.tamper else "20%",
                "conditions": "GPU A batch 8",
            }
        ]
        return json.dumps(output)


def test_domain_combines_rag_web_and_preserves_same_url_distinct_chunks():
    retriever, model = RetrieverFixture(), MeasurementModel()
    provider = EvaluationProvider(model, SearchFixture(), retriever=retriever)
    results, evidence = provider.assess("domain", TECHS, "datacenter", {}, [])
    assert len(retriever.calls) == 6
    assert all(args == {"top_k": 3, "role": None, "paper_id": None} for _, args in retriever.calls)
    for a in results:
        assert a.status == "assessed" and a.details.cost == "조건부 개선"
        assert "20%" in a.rationale and "GPU A batch 8" in a.rationale
    assert {e.chunk_id for e in evidence} == {"review-p1-1", "review-p2-1"}
    assert {e.page for e in evidence} == {1, 2}
    assert all(e.section_or_page and not e.supports_claim for e in evidence)
    payload = json.loads(model.calls[0][1]["content"])
    assert {d["paper_role"] for d in payload["sources"] if d["chunk_id"]} == {
        "primary",
        "reference",
    }
    assert any(d["source_type"] == "official" for d in payload["sources"])
    # 같은 URL의 다른 chunk_id 근거는 덮어쓰지 않습니다.
    assert len({e.id for e in evidence}) == len(evidence)


def test_fabricated_measurement_fails_without_publishing_evidence():
    provider = EvaluationProvider(
        MeasurementModel(tamper=True), SearchFixture(), retriever=RetrieverFixture()
    )
    results, evidence = provider.assess("domain", TECHS, "datacenter", {}, [])
    assert not evidence and all(a.status == "failed" for a in results)
    assert all(a.error.code == "ModelOutputError" for a in results)


def test_rag_is_not_called_for_stakeholders_and_rag_failure_is_isolated():
    class BrokenRetriever:
        def retrieve(self, *args, **kwargs):
            raise TimeoutError("private")

    provider = EvaluationProvider(
        ModelFixture({"gpu_vendor_support": "yes"}), SearchFixture(), retriever=BrokenRetriever()
    )
    results, _ = provider.assess("stakeholder", TECHS, "datacenter", {}, [])
    assert all(a.status == "assessed" for a in results)
    results, _ = provider.assess("domain", TECHS, "datacenter", {}, [])
    assert all(a.status == "failed" and a.error.retryable for a in results)


def test_numeric_claim_without_conditions_is_not_accepted():
    results, evidence = assess("domain", ModelFixture({"cost_measured": "yes"}))
    assert not evidence
    assert all(a.details.cost is None and a.status == "pending" for a in results)


@pytest.mark.parametrize(
    "conditional,expected",
    [("no", "원가 개선 명확"), ("yes", "조건부 개선"), ("unknown", "조건부 개선")],
)
def test_quantitative_cost_requires_explicit_scope_for_generalization(conditional, expected):
    assert (
        domain_details({"cost_measured": "yes", "cost_conditional": conditional}).cost == expected
    )


def test_explicit_unawareness_is_not_treated_as_a_reserved_position():
    results, _ = assess("stakeholder", ModelFixture({"gpu_vendor_aware": "no"}))
    assert all(a.details.positions[0].stance == "자료 없음" for a in results)
    assert all(a.details.overall is None for a in results)
