import json

import pytest

from skala_agent.agents.report import run
from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.schemas import Assessment, Evidence, MissingEvidence, TechAnalysis, Technology


def _state(assessment, evidence, missing=None, run_mode="demo"):
    return {
        "run_mode": run_mode,
        "domain": "data center serving",
        "retry_count": 0,
        "selected_technologies": [Technology(id="turboquant", name="TurboQuant", camp="sw")],
        "tech_analysis": {},
        "synthesis": [assessment],
        "evidence": evidence,
        "missing_evidence": missing or [],
    }


def _evidence(evidence_id, url, *, supports_claim=True):
    return Evidence(
        id=evidence_id,
        technology_id="turboquant",
        claim="assessment claim",
        url=url,
        title=f"Source {evidence_id}",
        excerpt="A test excerpt supporting the assessment claim.",
        source_type="official",
        supports_claim=supports_claim,
    )


def _assessment(evidence_ids, *, perspective="trl", verdict="TRL 5"):
    return Assessment(
        technology_id="turboquant",
        perspective=perspective,
        verdict=verdict,
        rationale="verified fixture",
        status="assessed",
        evidence_ids=evidence_ids,
    )


def test_report_uses_design_outline_and_cites_only_verified_assessment_sources():
    assessment = Assessment(
        technology_id="turboquant",
        perspective="trl",
        verdict="TRL 5",
        rationale="verified fixture",
        status="assessed",
        evidence_ids=["valid", "unsupported", "other-technology"],
    )
    other_technology = _evidence("other-technology", "https://example.org/other")
    other_technology = other_technology.model_copy(update={"technology_id": "itme"})

    report = run(
        _state(
            assessment,
            [
                _evidence("valid", "https://example.org/valid"),
                _evidence("unsupported", "https://example.org/unsupported", supports_claim=False),
                other_technology,
            ],
        )
    )["report"]

    headings = [
        "## SUMMARY",
        "## 1. 분석 배경",
        "## 2. 비교 기술 선정",
        "## 3. 기술 개요",
        "## 4. 관점별 평가",
        "## 5. 종합 비교 및 시사점",
        "## 6. 한계점",
        "## REFERENCE",
    ]
    assert [report.index(heading) for heading in headings] == sorted(
        report.index(heading) for heading in headings
    )
    assert "turboquant / trl: TRL 5 (confidence: low) [1]" in report
    assert "https://example.org/valid" in report
    assert "https://example.org/unsupported" not in report
    assert "https://example.org/other" not in report
    assert report.count("[1]") == 2


def test_report_deduplicates_reference_urls_and_withholds_unverified_assessments():
    assessment = Assessment(
        technology_id="turboquant",
        perspective="market",
        verdict="상용 채택",
        rationale="fixture",
        status="assessed",
        evidence_ids=["one", "duplicate"],
    )
    missing = MissingEvidence(
        technology_id="turboquant",
        perspective="market",
        reason="검증된 근거 부족",
        kind="unsupported_claim",
        claim="상용 채택",
        queries=["turboquant market adoption"],
    )

    report = run(
        _state(
            assessment,
            [
                _evidence("one", "https://example.org/shared"),
                _evidence("duplicate", "https://example.org/shared"),
            ],
            [missing],
        )
    )["report"]

    assert "turboquant / market: 판단 보류 (검증된 근거 부족)" in report
    assert "https://example.org/shared" not in report
    assert "turboquant/market: 검증된 근거 부족" in report
    assert "검증된 인용 출처 없음." in report


def test_report_renders_verified_technical_overview_and_tradeoff_findings():
    state = _state(
        Assessment(
            technology_id="turboquant",
            perspective="trl",
            verdict="TRL 5",
            rationale="fixture",
            status="assessed",
            evidence_ids=[],
        ),
        [
            _evidence("overview", "https://example.org/overview"),
            _evidence("tradeoff", "https://example.org/tradeoff"),
            _evidence("unverified", "https://example.org/unverified", supports_claim=False),
        ],
    )
    state["tech_analysis"] = {
        "turboquant": TechAnalysis(
            technology_id="turboquant",
            overview="검증된 기술 개요",
            scope=["검증된 적용 범위"],
            limitations=["검증된 한계"],
            evidence_ids=["overview"],
            status="assessed",
        )
    }
    state["synthesis_findings"] = [
        {
            "technology_id": "turboquant",
            "question": "condition_limited",
            "summary": "특정 조건에서만 성능 차이가 확인됨",
            "evidence_ids": ["tradeoff"],
        }
    ]

    report = run(state)["report"]

    assert "논문 기반 기술 요약 구현 대기." not in report
    assert "상충 탐지·종합 서술 구현 대기." not in report
    assert "### 3.1 TurboQuant" in report
    assert "검증된 기술 개요 [1]" in report
    assert "검증된 적용 범위 [1]" in report
    assert "검증된 한계 [1]" in report
    assert "https://example.org/unverified" not in report
    assert "### 5.3 주요 trade-off" in report
    assert "특정 조건에서만 성능 차이가 확인됨 [2]" in report
    assert report.count("https://example.org/overview") == 1
    assert report.count("https://example.org/tradeoff") == 1


def test_report_lists_shared_verified_verdicts_as_common_points():
    turboquant = Assessment(
        technology_id="turboquant",
        perspective="trl",
        verdict="공통 판정",
        rationale="fixture",
        status="assessed",
        evidence_ids=["turboquant-source"],
    )
    itme = turboquant.model_copy(update={"technology_id": "itme", "evidence_ids": ["itme-source"]})
    itme_source = _evidence("itme-source", "https://example.org/itme")
    itme_source = itme_source.model_copy(update={"technology_id": "itme"})
    state = _state(
        turboquant,
        [
            _evidence("turboquant-source", "https://example.org/turboquant"),
            itme_source,
        ],
    )
    state["selected_technologies"].append(Technology(id="itme", name="ITME", camp="hw"))
    state["synthesis"].append(itme)

    report = run(state)["report"]

    assert "trl: 공통 판정 (기술: turboquant, itme) [1] [2]" in report


def test_report_withholds_research_when_only_some_claims_are_verified():
    state = _state(
        Assessment(
            technology_id="turboquant", perspective="trl", verdict="판단 보류", rationale="fixture"
        ),
        [
            _evidence("overview", "https://example.org/overview"),
            _evidence("unverified", "https://example.org/unverified", supports_claim=False),
        ],
    )
    state["tech_analysis"] = {
        "turboquant": TechAnalysis(
            technology_id="turboquant",
            overview="Supported overview",
            scope=["Unsupported scope"],
            evidence_ids=["overview", "unverified"],
            status="assessed",
        )
    }
    report = run(state)["report"]
    assert "Supported overview" not in report and "Unsupported scope" not in report
    assert "검증된 기술 개요 근거 부족" in report


def test_partial_research_shows_only_verified_paper_observations():
    good = _evidence("good", "https://example.org/paper").model_copy(
        update={
            "source_type": "paper",
            "claim": "Supported observation",
            "chunk_id": "paper-p2-1",
            "page": 2,
        }
    )
    bad = _evidence("bad", "https://example.org/paper", supports_claim=False).model_copy(
        update={"claim": "Unsupported claim"}
    )
    state = _state(
        Assessment(
            technology_id="turboquant", perspective="trl", verdict="판단 보류", rationale="fixture"
        ),
        [good, bad],
    )
    state["tech_analysis"] = {
        "turboquant": TechAnalysis(
            technology_id="turboquant",
            overview="Unverified composite overview",
            evidence_ids=["good", "bad"],
            status="assessed",
        )
    }
    report = run(state)["report"]
    assert "판단 보류 (검증된 기술 개요 근거 부족)" in report
    assert "검증된 논문 관측: Supported observation" in report
    assert "page: 2; chunk_id: paper-p2-1" in report
    assert "Unverified composite overview" not in report and "Unsupported claim" not in report


def test_report_uses_the_configured_model_without_allowing_citation_changes():
    assessment = Assessment(
        technology_id="turboquant",
        perspective="trl",
        verdict="TRL 5",
        rationale="fixture",
        status="assessed",
        evidence_ids=["valid"],
    )

    class Model:
        def invoke_structured(self, messages, schema):
            assert messages[0]["role"] == "developer" and schema["type"] == "object"
            return json.dumps({"report": messages[1]["content"]})

    provider = type("Provider", (), {"report_model": Model()})()
    result = run(_state(assessment, [_evidence("valid", "https://example.org/valid")]), provider)

    assert "https://example.org/valid" in result["report"]


def test_report_rejects_a_model_that_changes_verified_citations():
    assessment = Assessment(
        technology_id="turboquant",
        perspective="trl",
        verdict="TRL 5",
        rationale="fixture",
        status="assessed",
        evidence_ids=["valid"],
    )

    class Model:
        def invoke_structured(self, messages, schema):
            return json.dumps({"report": messages[1]["content"].replace("[1]", "[2]")})

    provider = type("Provider", (), {"report_model": Model()})()
    with pytest.raises(ModelOutputError, match="인용"):
        run(_state(assessment, [_evidence("valid", "https://example.org/valid")]), provider)


def test_real_and_demo_headers_are_distinguished_on_the_same_path():
    """같은 보고서 경로에서 실행 모드에 따라 제목과 서두 안내가 달라진다.

    이전에는 모드와 무관하게 "개발용 뼈대"와 "실제 기술 평가 보고서가 아닙니다"를
    고정 출력했습니다. real provider는 지지 여부를 실제로 검증하므로 출력이
    동작과 어긋났습니다.
    """
    assessment = _assessment(["e-1"])
    evidence = [_evidence("e-1", "https://example.org/a")]

    demo = run(_state(assessment, evidence, run_mode="demo"))["report"]
    real = run(_state(assessment, evidence, run_mode="real"))["report"]

    assert demo.startswith("# KV cache 기술 비교 보고서 — 개발용 뼈대")
    assert "실제 기술 평가 결과가 아닙니다" in demo
    assert "합성 fixture" in demo

    assert real.startswith("# KV cache 기술 비교 보고서\n")
    assert "개발용 뼈대" not in real
    assert "실제 평가 실행 결과입니다" in real


def test_demo_limitations_say_verification_did_not_run():
    report = run(_state(_assessment(["e-1"]), [_evidence("e-1", "https://example.org/a")]))[
        "report"
    ]
    assert "근거 검증을 수행하지 않았습니다" in report
    assert "추가 구현 필요" not in report


def test_real_limitations_report_verified_counts_without_overstating():
    """real 한계점은 검증 통과 건수를 사실대로 적고 보장 범위를 과장하지 않는다."""
    evidence = [
        _evidence("e-1", "https://example.org/a"),
        _evidence("e-2", "https://example.org/b"),
        _evidence("e-3", "https://example.org/c", supports_claim=False),
    ]
    report = run(_state(_assessment(["e-1", "e-2"]), evidence, run_mode="real"))["report"]

    assert "수집한 근거 3건 중 2건이 주장 지지 판정을 통과했습니다" in report
    assert "별도 검사로 분리되어 있지 않습니다" in report
    assert "근거 검증을 수행하지 않았습니다" not in report


def test_limitations_report_missing_and_failed_counts():
    """근거 부족과 평가 실패 건수가 정확히 표시된다."""
    from skala_agent.schemas import AgentError, Assessment, MissingEvidence

    failed = Assessment(
        technology_id="turboquant",
        perspective="domain",
        verdict="판단 보류",
        rationale="호출 상한 시간 초과",
        status="failed",
        error=AgentError(code="TimeoutError", message="호출 상한 시간 초과"),
    )
    missing = MissingEvidence(
        technology_id="turboquant",
        perspective="domain",
        reason="질문별 판정을 지지하는 검증된 출처 없음",
        kind="missing_source",
        claim="판단 보류",
        queries=["turboquant domain"],
    )
    state = _state(
        _assessment(["e-1"]), [_evidence("e-1", "https://x.example/a")], [missing], "real"
    )
    state["analyses"] = {"domain": [failed]}

    report = run(state)["report"]

    assert "평가 실패로 판단 보류한 항목 1건" in report
    assert "근거가 부족해 해결되지 않은 항목 1건" in report


def test_limitations_state_no_remaining_gaps_when_everything_is_verified():
    report = run(_state(_assessment(["e-1"]), [_evidence("e-1", "https://example.org/a")]))[
        "report"
    ]
    assert "근거가 부족해 남은 항목은 없습니다" in report
