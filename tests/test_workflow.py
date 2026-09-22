from collections import Counter

from skala_agent.providers import DemoProvider
from skala_agent.schemas import Assessment, Evidence
from skala_agent.workflow.graph import build_graph, initial_state


class FixtureProvider(DemoProvider):
    def __init__(self, missing_market=False, recover=True):
        self.calls = Counter()
        self.searches = 0
        self.missing_market = missing_market
        self.recover = recover

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        self.calls[perspective] += 1
        assessments, sources = [], []
        for tech in technologies:
            eid = f"{perspective}-{tech.id}"
            missing = (
                perspective == "market"
                and self.missing_market
                and (not self.searches or not self.recover)
            )
            assessments.append(
                Assessment(
                    technology_id=tech.id,
                    perspective=perspective,
                    verdict="fixture verdict",
                    rationale="test only",
                    confidence="high",
                    status="assessed",
                    evidence_ids=[] if missing else [eid],
                )
            )
            if not missing:
                sources.append(
                    Evidence(
                        id=eid,
                        technology_id=tech.id,
                        claim="test only",
                        url=f"https://example.org/{eid}",
                        title="Test fixture",
                        excerpt="Synthetic evidence for testing",
                        source_type="official",
                        supports_claim=True,
                    )
                )
        return assessments, sources

    def search_missing(self, missing):
        assert {m.perspective for m in missing} == {"market"}
        self.searches += 1
        return []


def test_demo_stops_after_two_retries_without_fabricated_verdicts():
    result = build_graph().invoke(initial_state())
    assert result["retry_count"] == 2
    assert len(result["synthesis"]) == 8
    assert len(result["missing_evidence"]) == 8
    assert "판단 보류" in result["report"]
    assert "검증된 인용 출처 없음" in result["report"]


def test_fan_in_collects_all_perspectives_and_downgrades_single_source():
    provider = FixtureProvider()
    result = build_graph(provider).invoke(initial_state())
    assert result["retry_count"] == 0
    assert provider.calls == dict.fromkeys(("trl", "market", "stakeholder", "domain"), 1)
    assert len(result["evidence"]) == 8
    assert len(result["synthesis"]) == 8
    assert all(a.confidence == "low" for a in result["synthesis"])


def test_only_missing_perspective_retries_and_replaces_old_result():
    provider = FixtureProvider(missing_market=True)
    result = build_graph(provider).invoke(initial_state())
    assert provider.calls == {"trl": 1, "market": 2, "stakeholder": 1, "domain": 1}
    assert result["retry_count"] == 1
    assert not result["missing_evidence"]
    assert len(result["synthesis"]) == 8
    # 첫 실행의 정상 세 관점 6건 + 시장성 재실행의 2건.
    assert len(result["evidence"]) == 8


def test_retry_cap_preserves_valid_results_and_withholds_unsupported_claims():
    result = build_graph(FixtureProvider(missing_market=True, recover=False)).invoke(
        initial_state()
    )
    assert result["retry_count"] == 2
    assert len(result["missing_evidence"]) == 2
    assert "turboquant / market: 판단 보류" in result["report"]
    assert "turboquant / trl: fixture verdict" in result["report"]


class UpdatedEvidenceProvider(FixtureProvider):
    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        assessments, sources = super().assess(
            perspective, technologies, domain, tech_analysis, evidence
        )
        if perspective == "market" and not self.searches:
            sources = [e.model_copy(update={"supports_claim": False}) for e in sources]
        return assessments, sources


def test_retry_updates_same_evidence_ids_without_retaining_unsupported_versions():
    provider = UpdatedEvidenceProvider()
    result = build_graph(provider).invoke(initial_state())
    assert provider.calls == {"trl": 1, "market": 2, "stakeholder": 1, "domain": 1}
    assert len(result["evidence"]) == len({e.id for e in result["evidence"]}) == 8
    assert all(e.supports_claim for e in result["evidence"])
    assert not result["missing_evidence"]


class FailingDomainProvider(FixtureProvider):
    def __init__(self, recover=False):
        super().__init__()
        self.recover_domain = recover

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        if perspective == "domain" and not (self.recover_domain and self.searches):
            self.calls[perspective] += 1
            raise TimeoutError("sensitive provider detail should not enter the report")
        return super().assess(perspective, technologies, domain, tech_analysis, evidence)

    def search_missing(self, missing):
        assert {m.perspective for m in missing} == {"domain"}
        assert all(m.kind == "agent_failed" and m.queries for m in missing)
        self.searches += 1
        return []


def test_failed_perspective_preserves_successful_results_and_reports_failure():
    provider = FailingDomainProvider()
    result = build_graph(provider).invoke(initial_state())
    assert provider.calls == {"trl": 1, "market": 1, "stakeholder": 1, "domain": 3}
    assert result["retry_count"] == 2
    assert all(a.status == "failed" for a in result["analyses"]["domain"])
    assert "평가 실패: TimeoutError" in result["report"]
    assert "sensitive provider detail" not in result["report"]
    assert "turboquant / trl: fixture verdict" in result["report"]


def test_failed_perspective_can_recover_on_retry():
    provider = FailingDomainProvider(recover=True)
    result = build_graph(provider).invoke(initial_state())
    assert result["retry_count"] == 1
    assert not result["missing_evidence"]
    assert all(a.status == "assessed" for a in result["analyses"]["domain"])


def test_non_retryable_failure_does_not_trigger_search():
    from skala_agent.schemas import AgentError

    class PermanentFailureProvider(FixtureProvider):
        def assess(self, perspective, technologies, domain, tech_analysis, evidence):
            results, sources = super().assess(
                perspective, technologies, domain, tech_analysis, evidence
            )
            if perspective == "domain":
                return [
                    Assessment(
                        technology_id=t.id,
                        perspective=perspective,
                        verdict="판단 보류",
                        rationale="서비스 설정 누락",
                        status="failed",
                        error=AgentError(
                            code="ConfigurationError", message="설정 누락", retryable=False
                        ),
                    )
                    for t in technologies
                ], []
            return results, sources

    provider = PermanentFailureProvider()
    result = build_graph(provider).invoke(initial_state())
    assert result["retry_count"] == provider.searches == 0
    assert len(result["missing_evidence"]) == 2
    assert all(not m.retryable for m in result["missing_evidence"])
    assert "평가 실패: ConfigurationError" in result["report"]


def test_malformed_provider_output_is_not_silently_converted_to_failure():
    import pytest

    class InvalidProvider(FixtureProvider):
        def assess(self, perspective, technologies, domain, tech_analysis, evidence):
            return [], []

    with pytest.raises(ValueError, match="정확히 하나"):
        build_graph(InvalidProvider()).invoke(initial_state())


class VerifyingTradeoffProvider(FixtureProvider):
    """종합 키워드 규칙에 걸리는 판정을 내고, 검증 단계에서 근거를 승인합니다.

    FixtureProvider는 근거를 supports_claim=True로 만들어 반환하지만, 실제
    provider는 검증 단계에서만 True가 됩니다. 그 순서를 재현하려고 평가 시점에는
    False로 두고 validate_evidence에서 승인합니다.
    """

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        assessments, sources = super().assess(
            perspective, technologies, domain, tech_analysis, evidence
        )
        assessments = [
            item.model_copy(
                update={
                    "verdict": "조건부 개선",
                    "rationale": "긴 context length 조건에서 GPU 메모리 절감이 보고됐다.",
                }
            )
            for item in assessments
        ]
        return assessments, [e.model_copy(update={"supports_claim": False}) for e in sources]

    def validate_evidence(self, evidence):
        return [e.model_copy(update={"supports_claim": True}) for e in evidence]


def test_final_synthesis_reflects_evidence_verified_in_the_same_run():
    """첫 검증에서 False→True가 된 근거가 같은 실행의 최종 종합에 반영된다.

    synthesize가 validate보다 먼저 실행되므로 첫 종합은 supports_claim=False
    상태에서 계산됩니다. 최종 검증 이후 종합을 다시 하지 않으면 승인된 근거가
    보고서 5장에 반영되지 않았습니다.
    """
    result = build_graph(VerifyingTradeoffProvider()).invoke(initial_state())

    assert all(e.supports_claim for e in result["evidence"])
    assert result["synthesis_findings"], "검증된 근거가 최종 findings에 반영되어야 합니다"
    verified = {e.id for e in result["evidence"] if e.supports_claim}
    for finding in result["synthesis_findings"]:
        assert set(finding.evidence_ids) <= verified
    assert "검증된 trade-off 없음" not in result["report"]


def test_final_synthesis_drops_evidence_withdrawn_by_verification():
    """True→False로 철회된 근거는 최종 findings에서 빠진다."""

    class WithdrawingProvider(VerifyingTradeoffProvider):
        def assess(self, perspective, technologies, domain, tech_analysis, evidence):
            assessments, sources = super().assess(
                perspective, technologies, domain, tech_analysis, evidence
            )
            return assessments, [e.model_copy(update={"supports_claim": True}) for e in sources]

        def validate_evidence(self, evidence):
            return [e.model_copy(update={"supports_claim": False}) for e in evidence]

        def search_missing(self, missing):
            # 근거가 전부 철회되면 네 관점이 모두 부족해집니다.
            self.searches += 1
            return []

    result = build_graph(WithdrawingProvider()).invoke(initial_state())

    assert not any(e.supports_claim for e in result["evidence"])
    assert result["synthesis_findings"] == []
    assert "검증된 trade-off 없음" in result["report"]


def test_final_synthesis_runs_once_on_every_termination_path(caplog):
    """재검색 없이 종료·2회 소진·재시도 불가 경로 모두 최종 종합을 한 번 지난다."""
    import logging

    from skala_agent.schemas import AgentError

    class NonRetryable(FixtureProvider):
        def assess(self, perspective, technologies, domain, tech_analysis, evidence):
            results, sources = super().assess(
                perspective, technologies, domain, tech_analysis, evidence
            )
            if perspective != "domain":
                return results, sources
            return [
                Assessment(
                    technology_id=item.technology_id,
                    perspective=perspective,
                    verdict="판단 보류",
                    rationale="영구 실패",
                    status="failed",
                    error=AgentError(code="ValueError", message="영구 실패", retryable=False),
                )
                for item in results
            ], sources

    cases = {
        "재검색 없이 종료": (VerifyingTradeoffProvider(), None),
        "재검색 2회 소진": (DemoProvider(), 2),
        "재시도 불가": (NonRetryable(), None),
    }
    for label, (provider, expected_retries) in cases.items():
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="skala_agent.workflow.graph"):
            result = build_graph(provider).invoke(initial_state())
        messages = [r.getMessage() for r in caplog.records]
        assert sum("최종 종합 재계산" in m for m in messages) == 1, f"{label}: {messages}"
        assert result["report"], label
        if expected_retries is not None:
            assert result["retry_count"] == expected_retries, label
