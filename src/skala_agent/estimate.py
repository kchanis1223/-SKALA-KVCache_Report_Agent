"""윤소영 담당: 실제 호출 없이 외부 서비스 호출 횟수를 그래프로 직접 계산합니다.

손으로 센 숫자가 아니라 실제 컴파일된 그래프를 두 가지 경로로 돌려 얻은 값이라,
분기·재시도 구조가 바뀌면 추정치도 같이 바뀝니다.
"""

import unicodedata
from collections import Counter

from skala_agent.providers import DemoProvider
from skala_agent.schemas import Assessment, Evidence

CALLS = ("research", "assess", "search_missing")


class CountingProvider:
    """호출 횟수만 세고 안쪽 provider에 그대로 넘깁니다."""

    def __init__(self, inner):
        self.inner = inner
        self.counts: Counter[str] = Counter()

    def research(self, technologies):
        self.counts["research"] += 1
        return self.inner.research(technologies)

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        self.counts["assess"] += 1
        return self.inner.assess(perspective, technologies, domain, tech_analysis, evidence)

    def search_missing(self, missing):
        self.counts["search_missing"] += 1
        return self.inner.search_missing(missing)


class _SufficientProvider(DemoProvider):
    """모든 관점이 1회차에 지지 출처 2건씩을 확보하는 최선 경로."""

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        assessments, sources = [], []
        for tech in technologies:
            ids = [f"{perspective}-{tech.id}-{n}" for n in (1, 2)]
            assessments.append(
                Assessment(
                    technology_id=tech.id,
                    perspective=perspective,
                    verdict="충분",
                    rationale="추정용 시뮬레이션",
                    status="assessed",
                    evidence_ids=ids,
                )
            )
            sources += [
                Evidence(
                    id=eid,
                    technology_id=tech.id,
                    claim="추정용 시뮬레이션",
                    url=f"https://example.org/{eid}",
                    title="estimate",
                    excerpt="estimate",
                    source_type="official",
                    supports_claim=True,
                )
                for eid in ids
            ]
        return assessments, sources


def estimate_calls() -> dict[str, Counter[str]]:
    """최선(재시도 없음)과 최악(재시도 2회 소진) 경로의 호출 횟수."""
    from skala_agent.workflow.graph import build_graph, initial_state

    result = {}
    for label, inner in (("best", _SufficientProvider()), ("worst", DemoProvider())):
        provider = CountingProvider(inner)
        build_graph(provider).invoke(initial_state())
        result[label] = provider.counts
    return result


def _width(text: str) -> int:
    """한글은 두 칸을 차지하므로 표 정렬에 실제 표시 폭을 씁니다."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _row(label: str, best: object, worst: object) -> str:
    return f"  {label}{' ' * max(1, 18 - _width(label))}{best:>5}{worst:>7}"


def format_estimate(timeout: float | None = None) -> str:
    counts = estimate_calls()
    best, worst = counts["best"], counts["worst"]
    rule = "  " + "-" * 30
    lines = [
        "외부 서비스 호출 예상 (실제 호출 없음)",
        "",
        _row("단계", "최선", "최악"),
        rule,
    ]
    lines += [_row(name, best[name], worst[name]) for name in CALLS]
    lines += [
        rule,
        _row("합계", sum(best.values()), sum(worst.values())),
        "",
        "최선은 네 관점이 1회차에 지지 출처를 확보한 경우,",
        "최악은 모두 근거를 확보하지 못해 재시도 2회를 소진한 경우입니다.",
    ]
    if timeout:
        worst_minutes = sum(worst.values()) * timeout / 60
        lines.append(
            f"호출 1건의 상한이 {timeout:g}초이므로 최악의 경우 약 {worst_minutes:.0f}분입니다."
        )
    return "\n".join(lines)
