"""재검색 결과가 부족 관점에 전달되고 정상 관점 결과가 보존되는지 확인."""

from skala_agent.providers import DemoProvider
from skala_agent.schemas import Assessment, Evidence
from skala_agent.workflow.graph import build_graph, initial_state


def make_evidence(eid, technology_id):
    return Evidence(
        id=eid,
        technology_id=technology_id,
        claim="test only",
        url=f"https://example.org/{eid}",
        title="Test fixture",
        excerpt="Synthetic evidence for testing",
        source_type="official",
        supports_claim=True,
    )


class RecordingProvider(DemoProvider):
    """market만 1회차에 근거가 없고, 재검색이 새 근거를 제공합니다."""

    def __init__(self):
        self.calls = {}
        self.seen_evidence = {}

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        self.calls[perspective] = self.calls.get(perspective, 0) + 1
        self.seen_evidence[(perspective, self.calls[perspective])] = {e.id for e in evidence}

        assessments, sources = [], []
        supported = perspective != "market" or self.calls[perspective] > 1
        for tech in technologies:
            ids = [f"{perspective}-{tech.id}", f"extra-{perspective}-{tech.id}"]
            assessments.append(
                Assessment(
                    technology_id=tech.id,
                    perspective=perspective,
                    verdict="fixture verdict",
                    rationale="test only",
                    status="assessed",
                    evidence_ids=ids if supported else [],
                )
            )
            if supported:
                sources += [make_evidence(eid, tech.id) for eid in ids]
        return assessments, sources

    def search_missing(self, missing):
        return [make_evidence(f"found-{m.technology_id}", m.technology_id) for m in missing]


def test_retry_passes_new_evidence_and_preserves_healthy_perspectives():
    provider = RecordingProvider()
    result = build_graph(provider).invoke(initial_state())

    # 부족한 관점만 재실행합니다.
    assert provider.calls == {"trl": 1, "market": 2, "stakeholder": 1, "domain": 1}

    # 재평가되는 market이 추가 검색 산출물을 입력으로 받습니다.
    first_pass = provider.seen_evidence[("market", 1)]
    retry_pass = provider.seen_evidence[("market", 2)]
    assert not {e for e in first_pass if e.startswith("found-")}
    assert {"found-turboquant", "found-itme"} <= retry_pass

    # 정상 관점의 결과는 재실행 없이 보존됩니다.
    for key in ("trl", "stakeholder", "domain"):
        assert [a.status for a in result["analyses"][key]] == ["assessed", "assessed"]

    assert result["retry_count"] == 1
    assert not result["missing_evidence"]
