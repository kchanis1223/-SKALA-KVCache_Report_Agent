import json

from skala_agent.agents.synthesis import run
from skala_agent.schemas import (
    Assessment,
    DomainDetails,
    Evidence,
    MarketDetails,
    Signal,
    TRLDetails,
)


def test_synthesis_records_a_condition_limited_tradeoff_with_traceable_evidence():
    assessment = Assessment(
        technology_id="turboquant",
        perspective="domain",
        verdict="조건부 개선",
        rationale="긴 context length에서 메모리 절감이 보고됐다.",
        status="assessed",
        details=DomainDetails(cost="조건부 개선", sla_risk="낮음"),
        signals=[
            Signal(
                question="어떤 context length에서 효과가 나타나는가?",
                grade="상",
                evidence_ids=["domain-1"],
            )
        ],
        evidence_ids=["domain-1"],
    )
    evidence = Evidence(
        id="domain-1",
        technology_id="turboquant",
        claim="긴 context length에서 메모리 절감",
        url="https://example.org/turboquant",
        title="fixture",
        excerpt="The result applies at a long context length.",
        source_type="paper",
        supports_claim=True,
    )

    result = run({"analyses": {"domain": [assessment]}, "evidence": [evidence]})

    assert result["synthesis"] == [assessment]
    assert result["synthesis_findings"][0].question == "condition_limited"
    assert result["synthesis_findings"][0].technology_id == "turboquant"
    assert result["synthesis_findings"][0].assessment_refs == [("domain", "turboquant")]
    assert result["synthesis_findings"][0].evidence_ids == ["domain-1"]


def test_synthesis_covers_all_five_design_questions_when_the_assessments_state_them():
    def assessment(perspective, rationale, details, evidence_id):
        return Assessment(
            technology_id="turboquant",
            perspective=perspective,
            verdict="fixture verdict",
            rationale=rationale,
            status="assessed",
            details=details,
            evidence_ids=[evidence_id],
        )

    assessments = [
        assessment("domain", "메모리 절감과 정확도 저하", DomainDetails(), "quality"),
        assessment("domain", "GPU 메모리 절감이 CPU bandwidth 비용 증가", DomainDetails(), "cost"),
        assessment("domain", "성능 개선은 배포와 운영 복잡도 증가", DomainDetails(), "operations"),
        assessment("domain", "GPU와 context length 조건에서만 성립", DomainDetails(), "condition"),
        assessment("trl", "TRL 근거", TRLDetails(level=5), "trl"),
        assessment("market", "시장 근거", MarketDetails(adoption="상용 채택"), "market"),
    ]
    evidence = [
        Evidence(
            id=evidence_id,
            technology_id="turboquant",
            claim="fixture claim",
            url=f"https://example.org/{evidence_id}",
            title="fixture",
            excerpt="fixture evidence",
            source_type="official",
            supports_claim=True,
        )
        for evidence_id in ("quality", "cost", "operations", "condition", "trl", "market")
    ]

    result = run(
        {
            "analyses": {
                "trl": [assessments[4]],
                "market": [assessments[5]],
                "domain": assessments[:4],
            },
            "evidence": evidence,
        }
    )

    assert {finding.question for finding in result["synthesis_findings"]} == {
        "quality_stability",
        "resource_cost",
        "operational_complexity",
        "maturity_adoption",
        "condition_limited",
    }


def test_synthesis_does_not_treat_a_question_prompt_as_a_tradeoff():
    assessment = Assessment(
        technology_id="turboquant",
        perspective="domain",
        verdict="판단 보류",
        rationale="비교할 근거가 아직 부족하다.",
        status="assessed",
        evidence_ids=["evidence-1"],
        signals=[
            Signal(
                question="메모리 절감이 정확도와 안정성에 미치는 영향은 무엇인가?",
                grade="하",
                evidence_ids=["evidence-1"],
            )
        ],
    )
    evidence = Evidence(
        id="evidence-1",
        technology_id="turboquant",
        claim="검토 질문",
        url="https://example.org/question",
        title="fixture",
        excerpt="The question needs further evidence.",
        source_type="paper",
        supports_claim=True,
    )

    result = run({"analyses": {"domain": [assessment]}, "evidence": [evidence]})

    assert result["synthesis_findings"] == []


def test_synthesis_uses_the_configured_model_for_traceable_findings():
    assessment = Assessment(
        technology_id="turboquant",
        perspective="domain",
        verdict="조건부 개선",
        rationale="fixture",
        status="assessed",
        evidence_ids=["evidence-1"],
    )
    evidence = Evidence(
        id="evidence-1",
        technology_id="turboquant",
        claim="fixture",
        url="https://example.org/source",
        title="fixture",
        excerpt="fixture",
        source_type="paper",
        supports_claim=True,
    )

    class Model:
        def invoke_structured(self, messages, schema):
            assert messages[0]["role"] == "developer" and schema["type"] == "object"
            return json.dumps(
                {
                    "findings": [
                        {
                            "technology_id": "turboquant",
                            "question": "condition_limited",
                            "summary": "근거가 특정 조건에 한정된다.",
                            "assessment_refs": [["domain", "turboquant"]],
                            "evidence_ids": ["evidence-1"],
                        }
                    ]
                }
            )

    provider = type("Provider", (), {"synthesis_model": Model()})()
    result = run({"analyses": {"domain": [assessment]}, "evidence": [evidence]}, provider)

    assert result["synthesis_findings"][0].summary == "근거가 특정 조건에 한정된다."
