from typing import Annotated, Literal, TypedDict

from skala_agent.schemas import (
    Assessment,
    Evidence,
    MissingEvidence,
    SynthesisFinding,
    TechAnalysis,
    Technology,
)


def merge_analyses(left: dict, right: dict) -> dict:
    """관점별 분리 쓰기 + retry 시 해당 관점 결과 교체."""
    return {**left, **right}


def merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """같은 ID의 최신 근거를 유지하며 다른 주장의 ID 재사용을 거부합니다."""
    merged = {e.id: e for e in left}
    for item in right:
        previous = merged.get(item.id)
        if previous is not None and (
            previous.technology_id,
            previous.claim,
            previous.url,
            previous.chunk_id,
        ) != (item.technology_id, item.claim, item.url, item.chunk_id):
            raise ValueError(f"Evidence ID 충돌: {item.id}")
        merged[item.id] = item
    return list(merged.values())


class EvaluationState(TypedDict):
    # 보고서가 real/demo를 추측하지 않도록 실행 모드를 명시적으로 전달합니다.
    # 근거 수나 문구로 모드를 유추하면 근거가 적은 real 실행이 demo로 표시됩니다.
    run_mode: Literal["demo", "real"]
    selected_technologies: list[Technology]
    domain: str
    tech_analysis: dict[str, TechAnalysis]
    analyses: Annotated[dict[str, list[Assessment]], merge_analyses]
    evidence: Annotated[list[Evidence], merge_evidence]
    missing_evidence: list[MissingEvidence]
    synthesis: list[Assessment]
    synthesis_findings: list[SynthesisFinding]
    retry_count: int
    report: str
