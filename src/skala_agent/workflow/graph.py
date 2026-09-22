import logging
import time

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from skala_agent.agents import (
    additional_search,
    domain,
    market,
    report,
    research,
    stakeholder,
    synthesis,
    trl,
    validation,
)
from skala_agent.providers import DemoProvider, Provider
from skala_agent.schemas import PERSPECTIVES, AgentError, Assessment, Evidence, Technology
from skala_agent.workflow.state import EvaluationState

logger = logging.getLogger(__name__)

AGENTS = {"trl": trl, "market": market, "stakeholder": stakeholder, "domain": domain}
MAX_RETRIES = 2

# 논문 조사는 이후 모든 관점 평가의 근거이므로 일시적 오류를 재시도합니다.
RESEARCH_ATTEMPTS = 3
RESEARCH_BACKOFF_SECONDS = 1.0


class ResearchUnavailableError(RuntimeError):
    """재시도 후에도 논문 조사를 마치지 못한 경우."""


def research_with_retry(state, provider):
    last: Exception | None = None
    for attempt in range(1, RESEARCH_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            result = research.run(state, provider)
        except (TimeoutError, ConnectionError) as exc:
            # 일시적 외부 오류만 재시도합니다. 계약 위반이나 프로그래밍 오류는 그대로 전달합니다.
            last = exc
            logger.warning(
                "논문 조사 %d/%d 실패 (%.1fs): %s",
                attempt,
                RESEARCH_ATTEMPTS,
                time.monotonic() - started,
                type(exc).__name__,
            )
            if attempt < RESEARCH_ATTEMPTS:
                time.sleep(RESEARCH_BACKOFF_SECONDS * attempt)
        else:
            logger.info(
                "논문 조사 완료 (%.1fs, 시도 %d회, 기술 %d건)",
                time.monotonic() - started,
                attempt,
                len(result["tech_analysis"]),
            )
            return result
    raise ResearchUnavailableError(
        f"논문 조사를 {RESEARCH_ATTEMPTS}회 시도했으나 모두 실패했습니다"
        f" (마지막 오류: {type(last).__name__}). 근거 없는 평가를 만들지 않기 위해"
        " 실행을 중단합니다."
    ) from last


class AdditionalSearchFailedError(RuntimeError):
    """추가 검색 단계에서 외부 서비스 호출이 실패한 경우."""


def additional_search_with_context(state, provider):
    """실패 시 원인·대상·진행 상황을 붙여서 중단합니다.

    네 관점 평가가 끝난 뒤의 근거 보완 단계이므로, 무엇이 실패했고 무엇이
    완료된 상태인지 구분할 수 있어야 원인을 찾을 수 있습니다.
    """
    targets_log = sorted({m.perspective for m in state["missing_evidence"] if m.retryable})
    logger.info(
        "추가 검색 시작 (%d회차, 대상: %s)",
        state["retry_count"] + 1,
        ", ".join(targets_log),
    )
    try:
        result = additional_search.run(state, provider)
    except (TimeoutError, ConnectionError) as exc:
        targets = sorted({m.perspective for m in state["missing_evidence"] if m.retryable})
        raise AdditionalSearchFailedError(
            f"추가 검색(재검색 {state['retry_count'] + 1}회차)이 실패했습니다: "
            f"{type(exc).__name__} — {exc}. "
            f"대상 관점: {', '.join(targets) or '없음'}. "
            "네 관점 평가는 완료된 상태이고 근거 보완만 실패했습니다. "
            "네트워크와 검색 API 설정을 확인한 뒤 다시 실행하세요."
        ) from exc
    logger.info("추가 검색 완료 (근거 %d건 확보)", len(result["evidence"]))
    return result


def initial_state() -> EvaluationState:
    return {
        "selected_technologies": [
            Technology(id="turboquant", name="TurboQuant", camp="sw"),
            Technology(id="itme", name="ITME", camp="hw"),
        ],
        "domain": "데이터센터 · 클라우드 서빙",
        "tech_analysis": {},
        "analyses": {},
        "evidence": [],
        "missing_evidence": [],
        "synthesis": [],
        "retry_count": 0,
        "report": "",
    }


def _logged_synthesize(state):
    result = synthesis.run(state)
    logger.info("종합 완료 (판정 %d건)", len(result["synthesis"]))
    return result


def _logged_validate(state):
    result = validation.run(state)
    missing = result["missing_evidence"]
    logger.info(
        "근거 검증 완료 (근거 부족 %d건%s)",
        len(missing),
        ": " + ", ".join(sorted({m.perspective for m in missing})) if missing else "",
    )
    return result


def _logged_report(state):
    result = report.run(state)
    logger.info("보고서 생성 완료 (%d자, 재검색 %d회)", len(result["report"]), state["retry_count"])
    return result


def build_graph(provider: Provider | None = None):
    provider = provider or DemoProvider()
    graph = StateGraph(EvaluationState)
    graph.add_node("research", lambda state: research_with_retry(state, provider))

    def evaluate(payload):
        key, state = payload["perspective"], payload["state"]
        started = time.monotonic()
        try:
            assessments, evidence = AGENTS[key].run(state, provider)
        except (TimeoutError, ConnectionError) as exc:
            # Adapter는 외부 서비스 오류를 아래 표준 예외로 변환합니다.
            # 구조화 출력 오류나 프로그래밍 오류는 숨기지 않습니다.
            error = AgentError(code=type(exc).__name__, message="외부 서비스 요청 실패")
            assessments = [
                Assessment(
                    technology_id=t.id,
                    perspective=key,
                    verdict="판단 보류",
                    rationale=error.message,
                    status="failed",
                    error=error,
                )
                for t in state["selected_technologies"]
            ]
            evidence = []
            logger.warning(
                "[%s] 평가 실패 (%.1fs): %s", key, time.monotonic() - started, error.code
            )
        assessments = [Assessment.model_validate(a) for a in assessments]
        evidence = [Evidence.model_validate(e) for e in evidence]
        expected = {t.id for t in state["selected_technologies"]}
        if (
            len(assessments) != len(expected)
            or {a.technology_id for a in assessments} != expected
            or any(a.perspective != key for a in assessments)
        ):
            raise ValueError("각 관점은 선택된 기술마다 정확히 하나의 평가를 반환해야 합니다.")
        logger.info(
            "[%s] 평가 완료 (%.1fs, 판정 %d건, 근거 %d건, status=%s)",
            key,
            time.monotonic() - started,
            len(assessments),
            len(evidence),
            "/".join(sorted({a.status for a in assessments})),
        )
        return {"analyses": {key: assessments}, "evidence": evidence}

    def dispatch(state):
        targets = (
            PERSPECTIVES
            if state["retry_count"] == 0
            else sorted({m.perspective for m in state["missing_evidence"] if m.retryable})
        )
        logger.info(
            "관점 %s 실행: %s",
            "병렬" if state["retry_count"] == 0 else f"재평가({state['retry_count']}회차)",
            ", ".join(targets) or "없음",
        )
        return [Send("evaluate", {"perspective": key, "state": state}) for key in targets]

    graph.add_node("evaluate", evaluate)
    graph.add_node("synthesize", _logged_synthesize)
    graph.add_node("validate", _logged_validate)
    graph.add_node(
        "additional_search", lambda state: additional_search_with_context(state, provider)
    )
    graph.add_node("report", _logged_report)
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", dispatch, ["evaluate"])
    graph.add_edge("evaluate", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda state: (
            "additional_search"
            if any(m.retryable for m in state["missing_evidence"])
            and state["retry_count"] < MAX_RETRIES
            else "report"
        ),
        ["additional_search", "report"],
    )
    graph.add_conditional_edges("additional_search", dispatch, ["evaluate"])
    graph.add_edge("report", END)
    return graph.compile()
