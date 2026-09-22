"""재검색 결과가 부족 관점에 전달되고 정상 관점 결과가 보존되는지 확인."""

from skala_agent.providers import DemoProvider
from skala_agent.schemas import AgentError, Assessment, Evidence
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


def test_timeout_and_connection_failures_get_distinct_reasons():
    """상한 초과와 연결 실패가 같은 문구로 뭉쳐지지 않는다.

    TimeoutError는 외부 장애가 아니라 로컬 호출 상한(runtime.TimeoutProvider)
    초과일 수 있습니다. 두 경우를 모두 "외부 서비스 요청 실패"로 적으면 보고서에
    사실과 다른 사유가 남습니다. provider 예외 원문은 보고서로 내보내지 않으므로
    (tests/test_workflow.py가 금지) 예외 종류별 고정 문구로 구분합니다.
    """

    def reason_for(exc):
        class Failing(DemoProvider):
            def assess(self, perspective, technologies, domain, tech_analysis, evidence):
                if perspective == "domain":
                    raise exc
                return super().assess(perspective, technologies, domain, tech_analysis, evidence)

        state = build_graph(Failing()).invoke(initial_state())
        failed = [a for a in state["analyses"]["domain"] if a.status == "failed"]
        assert failed, "domain 관점이 실패로 기록되어야 합니다"
        return {a.error.message for a in failed if a.error}

    timed_out = reason_for(TimeoutError("provider.assess 호출이 1200.0초를 초과했습니다."))
    disconnected = reason_for(ConnectionError("검색 서비스 응답에 results 목록이 없습니다."))

    assert timed_out == {"호출 상한 시간 초과"}
    assert disconnected == {"외부 서비스 연결 실패"}


def test_failure_reason_does_not_leak_provider_exception_text():
    """예외 원문이 판정에 섞이지 않는다. 내부 경로·URL이 보고서로 나갈 수 있다."""

    class Leaky(DemoProvider):
        def assess(self, perspective, technologies, domain, tech_analysis, evidence):
            if perspective == "domain":
                raise TimeoutError("http://localhost:11434 내부 경로 노출 금지")
            return super().assess(perspective, technologies, domain, tech_analysis, evidence)

    state = build_graph(Leaky()).invoke(initial_state())

    for assessment in state["analyses"]["domain"]:
        assert "localhost" not in assessment.rationale
        assert assessment.error is None or "localhost" not in assessment.error.message
    assert "localhost" not in state["report"]


class PartialMissProvider(DemoProvider):
    """turboquant만 근거가 부족하고 itme은 충분한 상태를 만듭니다."""

    def __init__(self):
        self.assessed = []

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        assessments, sources = [], []
        for tech in technologies:
            self.assessed.append((perspective, tech.id))
            supported = tech.id == "itme"
            eid = f"{perspective}-{tech.id}"
            assessments.append(
                Assessment(
                    technology_id=tech.id,
                    perspective=perspective,
                    verdict="fixture verdict",
                    rationale="test only",
                    confidence="high",
                    status="assessed",
                    evidence_ids=[eid],
                )
            )
            sources.append(
                Evidence(
                    id=eid,
                    technology_id=tech.id,
                    claim="test only",
                    url=f"https://example.org/{eid}",
                    title="Test fixture",
                    excerpt="Synthetic evidence for testing",
                    source_type="official",
                    supports_claim=supported,
                )
            )
        return assessments, sources

    def search_missing(self, missing):
        # 부족 항목은 turboquant만이어야 합니다.
        assert {m.technology_id for m in missing} == {"turboquant"}
        return []


def test_retry_only_reassesses_the_technology_that_is_missing_evidence():
    """한 기술만 부족하면 같은 관점의 다른 기술은 다시 평가하지 않는다.

    dispatch가 perspective만 추출하면 한 기술이 부족해도 두 기술을 모두 다시
    평가해 모델 호출이 두 배가 됩니다.
    """
    provider = PartialMissProvider()
    state = build_graph(provider).invoke(initial_state())

    first_round = provider.assessed[:8]
    retries = provider.assessed[8:]

    assert sorted(first_round) == sorted(
        (p, t) for p in ("trl", "market", "stakeholder", "domain") for t in ("turboquant", "itme")
    )
    assert retries, "재평가가 한 번은 실행되어야 합니다"
    assert {t for _, t in retries} == {"turboquant"}, provider.assessed
    assert state["retry_count"] == 2


def test_partial_retry_preserves_the_other_technology_verdict():
    """재평가하지 않은 기술의 판정과 근거가 그대로 남는다."""
    state = build_graph(PartialMissProvider()).invoke(initial_state())

    for perspective in ("trl", "market", "stakeholder", "domain"):
        results = state["analyses"][perspective]
        assert [a.technology_id for a in results] == ["turboquant", "itme"], perspective
        assert all(a.status == "assessed" for a in results), perspective
    assert {e.id for e in state["evidence"] if e.supports_claim} == {
        f"{p}-itme" for p in ("trl", "market", "stakeholder", "domain")
    }


def test_non_retryable_technology_is_not_reassessed():
    """non-retryable 부족 항목은 재평가 대상에서 빠진다."""

    class PermanentFailure(PartialMissProvider):
        def assess(self, perspective, technologies, domain, tech_analysis, evidence):
            assessments, sources = super().assess(
                perspective, technologies, domain, tech_analysis, evidence
            )
            if perspective != "domain":
                return assessments, sources
            return [
                item.model_copy(
                    update={
                        "status": "failed",
                        "verdict": "판단 보류",
                        "error": AgentError(
                            code="ValueError", message="영구 실패", retryable=False
                        ),
                    }
                )
                for item in assessments
            ], sources

        def search_missing(self, missing):
            assert "domain" not in {m.perspective for m in missing if m.retryable}
            return []

    provider = PermanentFailure()
    build_graph(provider).invoke(initial_state())

    assert [entry for entry in provider.assessed[8:] if entry[0] == "domain"] == []
