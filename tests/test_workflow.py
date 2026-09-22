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
