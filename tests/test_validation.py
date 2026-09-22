from skala_agent.agents.validation import run
from skala_agent.schemas import Assessment, Evidence


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
