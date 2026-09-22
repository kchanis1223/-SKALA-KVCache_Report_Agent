from skala_agent.agents.report import run
from skala_agent.schemas import Assessment, Evidence, MissingEvidence, TechAnalysis, Technology


def _state(assessment, evidence, missing=None):
    return {
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
            evidence_ids=["overview", "unverified"],
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
