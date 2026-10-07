"""오케스트레이터 구조의 공유 State와 데이터 모양.

노드는 State 전체를 반환하지 않고 바뀐 키만 반환합니다. 병렬 worker는
자기 task_id 키에만 쓰므로 dict 병합 Reducer로 충돌 없이 합쳐집니다.
"""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

Agent = Literal["domain", "market", "stakeholder", "tech"]
AGENTS: tuple[Agent, ...] = ("domain", "market", "stakeholder", "tech")
AGENT_LABELS = {
    "domain": "도메인",
    "market": "시장성",
    "stakeholder": "이해관계자",
    "tech": "기술평가",
}
MAX_RETRIES = 1


def merge_dict(left: dict, right: dict) -> dict:
    """같은 키는 새 값으로 교체하고 나머지는 보존합니다."""
    return {**(left or {}), **(right or {})}


def clipped(limit: int):
    """LLM 출력 길이 상한. 넘으면 거부하지 않고 앞부분만 남깁니다.

    인용(quote)은 원문의 앞부분을 잘라도 여전히 원문의 부분문자열이므로
    인용 검사 규칙이 그대로 유지됩니다. 긴 인용 하나 때문에 worker 전체가
    실패하지 않게 하려는 것입니다.
    """

    def clip(value):
        return value.strip()[:limit] if isinstance(value, str) else value

    return BeforeValidator(clip)


class SubTask(BaseModel):
    """오케스트레이터가 worker에게 주는 하위 과제."""

    model_config = ConfigDict(extra="forbid")
    # 출처 id 접두사로도 쓰므로 인용 표기 [S:id]에 들어갈 수 있는 문자만 허용합니다.
    id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    agent: Agent
    instruction: str = Field(min_length=1, max_length=800)


class Source(BaseModel):
    """도구가 가져온 원문. finding의 quote는 반드시 text 안에 있어야 합니다."""

    id: str
    kind: Literal["paper", "web"]
    title: str
    url: str  # 검색 계층에서 검증된 URL. 체크포인트 직렬화를 위해 문자열로 저장
    text: str
    page: int | None = None


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: Annotated[str, clipped(400), Field(min_length=1)]
    source_id: str
    quote: Annotated[str, clipped(500), Field(min_length=1)]


class WorkerDraft(BaseModel):
    """worker LLM의 구조화 출력."""

    model_config = ConfigDict(extra="forbid")
    findings: list[Finding] = Field(default_factory=list, max_length=8)
    verdict: Annotated[str, clipped(600), Field(min_length=1)]


class WorkerResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    task_id: str
    from_: Agent = Field(alias="from")
    success: bool
    findings: list[Finding] = Field(default_factory=list)
    verdict: str = ""
    error: str | None = None


class Verdict(BaseModel):
    """validate 에이전트의 판정. feedback은 부족한 task_id → 다시 할 일."""

    model_config = ConfigDict(extra="forbid")
    sufficient: bool
    feedback: dict[str, str] = Field(default_factory=dict)


class State(TypedDict, total=False):
    run_id: str
    question: str
    tech_brief: str
    plan: list[SubTask]
    worker_results: Annotated[dict[str, WorkerResult], merge_dict]
    sources: Annotated[dict[str, Source], merge_dict]
    result: dict[str, list[WorkerResult]]
    verdict: Verdict | None
    retry_count: int
    node_status: Annotated[dict[str, str], merge_dict]
    last_error: str | None
    report: str


def initial_state(question: str, run_id: str) -> State:
    return {
        "run_id": run_id,
        "question": question,
        "tech_brief": "",
        "plan": [],
        "worker_results": {},
        "sources": {},
        "result": {},
        "verdict": None,
        "retry_count": 0,
        "node_status": {},
        "last_error": None,
        "report": "",
    }
