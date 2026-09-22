from skala_agent.agents.validation import run
from skala_agent.schemas import Assessment, Evidence, Signal


def test_unverified_or_wrong_technology_evidence_does_not_validate_claim():
    assessment = Assessment(
        technology_id="itme",
        perspective="trl",
        verdict="TRL 9",
        rationale="unverified",
        status="assessed",
        evidence_ids=["e"],
    )
    for tech, supports in [("turboquant", True), ("itme", False)]:
        evidence = Evidence(
            id="e",
            technology_id=tech,
            claim="test",
            url="https://example.org",
            title="fixture",
            excerpt="test",
            source_type="paper",
            supports_claim=supports,
        )
        assert run({"synthesis": [assessment], "evidence": [evidence]})["missing_evidence"]


def test_signal_without_a_valid_linked_source_requests_that_question_again():
    assessment = Assessment(
        technology_id="itme",
        perspective="domain",
        verdict="조건부 개선",
        rationale="비용 개선과 SLA 영향을 함께 확인해야 한다.",
        status="assessed",
        evidence_ids=["valid"],
        signals=[Signal(question="SLA 영향은 무엇인가?", grade="하", evidence_ids=["wrong"])],
    )
    evidence = Evidence(
        id="valid",
        technology_id="itme",
        claim="비용 개선",
        url="https://example.org/cost",
        title="cost",
        excerpt="cost evidence",
        source_type="official",
        supports_claim=True,
    )

    missing = run({"synthesis": [assessment], "evidence": [evidence]})["missing_evidence"]

    assert len(missing) == 1
    assert missing[0].kind == "unsupported_claim"
    assert missing[0].claim == "SLA 영향은 무엇인가?"
    assert missing[0].queries == ["itme domain SLA 영향은 무엇인가?"]
