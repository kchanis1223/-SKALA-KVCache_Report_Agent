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


def test_signals_keep_distinct_evidence_ids_when_they_share_a_url():
    assessment = Assessment(
        technology_id="itme",
        perspective="domain",
        verdict="조건부 개선",
        rationale="비용과 SLA를 함께 확인했다.",
        status="assessed",
        evidence_ids=["cost", "sla"],
        signals=[
            Signal(question="비용 영향", grade="중", evidence_ids=["cost"]),
            Signal(question="SLA 영향", grade="중", evidence_ids=["sla"]),
        ],
    )
    evidence = [
        Evidence(
            id=evidence_id,
            technology_id="itme",
            claim=claim,
            url="https://example.org/shared",
            title="shared",
            excerpt=claim,
            source_type="official",
            supports_claim=True,
        )
        for evidence_id, claim in (("cost", "비용 근거"), ("sla", "SLA 근거"))
    ]

    result = run({"synthesis": [assessment], "evidence": evidence})

    assert result["missing_evidence"] == []
    assert result["synthesis"][0].confidence == "low"


def test_provider_evidence_verdict_is_used_in_the_same_validation_pass():
    assessment = Assessment(
        technology_id="itme",
        perspective="trl",
        verdict="TRL 6",
        rationale="실증 근거가 있다.",
        status="assessed",
        evidence_ids=["evidence-1"],
    )
    evidence = Evidence(
        id="evidence-1",
        technology_id="itme",
        claim="ITME: TRL 6 실증 근거",
        url="https://example.org/trl",
        title="TRL fixture",
        excerpt="A system prototype was evaluated.",
        source_type="paper",
    )

    class Provider:
        def validate_evidence(self, items):
            return [item.model_copy(update={"supports_claim": True}) for item in items]

    result = run({"synthesis": [assessment], "evidence": [evidence]}, Provider())

    assert result["missing_evidence"] == []
    assert result["evidence"] == [evidence.model_copy(update={"supports_claim": True})]


def test_only_referenced_evidence_is_sent_and_unused_candidates_are_preserved():
    from types import SimpleNamespace

    items = [
        Evidence(
            id=key,
            technology_id="itme",
            claim=key,
            url=f"https://example.org/{key}",
            title="fixture",
            excerpt="fixture",
            source_type="paper",
        )
        for key in ("assessment", "signal", "research", "finding", "unused", "historical")
    ]
    assessment = Assessment(
        technology_id="itme",
        perspective="trl",
        verdict="TRL 6",
        rationale="fixture",
        status="assessed",
        evidence_ids=["assessment"],
        signals=[Signal(question="fixture?", grade="중", evidence_ids=["signal"])],
    )
    seen = []

    class Provider:
        def validate_evidence(self, targets):
            seen.extend(item.id for item in targets)
            return [item.model_copy(update={"supports_claim": True}) for item in targets]

    state = {
        "synthesis": [assessment],
        "evidence": items,
        "tech_analysis": {"itme": SimpleNamespace(evidence_ids=["research"])},
        "synthesis_findings": [SimpleNamespace(evidence_ids=["finding"])],
    }
    result = run(state, Provider())
    assert seen == ["assessment", "signal", "research", "finding"]
    assert len(result["evidence"]) == 4
    assert len(state["evidence"]) == 6
    assert not any(item.supports_claim for item in state["evidence"])


def test_no_referenced_evidence_does_not_call_provider():
    class Provider:
        def validate_evidence(self, targets):
            raise AssertionError("no targets")

    assert run({"synthesis": [], "evidence": []}, Provider()) == {
        "missing_evidence": [],
        "synthesis": [],
    }
