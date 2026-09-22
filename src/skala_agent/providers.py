"""외부 연동 계약. 기본 provider는 비용·네트워크 호출 없는 미구현 표시용."""

from typing import Protocol

from skala_agent.schemas import Assessment, Evidence, MissingEvidence, Perspective, Technology


class Provider(Protocol):
    def research(self, technologies: list[Technology]) -> tuple[dict[str, str], list[Evidence]]:
        """role=primary 논문 RAG에서 기술 개요·한계·실험 조건을 추출."""
        ...

    def assess(
        self,
        perspective: Perspective,
        technologies: list[Technology],
        domain: str,
        tech_analysis: dict[str, str],
        evidence: list[Evidence],
    ) -> tuple[list[Assessment], list[Evidence]]:
        """도메인: 전체 논문 RAG + 웹. 나머지 관점: 웹검색."""
        ...

    def search_missing(self, missing: list[MissingEvidence]) -> list[Evidence]: ...


class DemoProvider:
    def research(self, technologies):
        return {t.id: "논문 RAG 연결 대기" for t in technologies}, []

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        return [
            Assessment(
                technology_id=t.id,
                perspective=perspective,
                verdict="판단 보류",
                rationale="실제 검색 및 평가 provider 연결 대기",
            )
            for t in technologies
        ], []

    def search_missing(self, missing):
        return []
