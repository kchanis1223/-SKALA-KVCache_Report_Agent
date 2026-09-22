import operator
from typing import Annotated, TypedDict

from skala_agent.schemas import Assessment, Evidence, MissingEvidence, Technology


def merge_analyses(left: dict, right: dict) -> dict:
    """관점별 분리 쓰기 + retry 시 해당 관점 결과 교체."""
    return {**left, **right}


class EvaluationState(TypedDict):
    selected_technologies: list[Technology]
    domain: str
    tech_analysis: dict[str, str]
    analyses: Annotated[dict[str, list[Assessment]], merge_analyses]
    evidence: Annotated[list[Evidence], operator.add]
    missing_evidence: list[MissingEvidence]
    synthesis: list[Assessment]
    retry_count: int
    report: str
