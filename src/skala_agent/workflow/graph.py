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
        "synthesis_findings": [],
        "retry_count": 0,
        "report": "",
    }


def _logged_synthesize(state, provider):
    result = synthesis.run(state, provider)
    logger.info("종합 완료 (판정 %d건)", len(result["synthesis"]))
    return result


def _logged_validate(state, provider):
    result = validation.run(state, provider)
    missing = result["missing_evidence"]
    logger.info(
        "근거 검증 완료 (근거 부족 %d건%s)",
        len(missing),
        ": " + ", ".join(sorted({m.perspective for m in missing})) if missing else "",
    )
    return result


def _logged_report(state, provider):
    result = report.run(state, provider)
    logger.info("보고서 생성 완료 (%d자, 재검색 %d회)", len(result["report"]), state["retry_count"])
    return result


def finalize_synthesis(state, provider):
    """최종 근거 검증 결과를 반영해 종합 findings를 다시 계산합니다.

    synthesize가 validate보다 먼저 실행되므로, 종합 시점에는 Evidence의
    supports_claim이 아직 갱신되지 않았습니다(기본값 False). synthesis는
    supports_claim=True인 근거만 쓰기 때문에 첫 바퀴 종합은 근거 0건으로
    계산되고, 마지막 validate 이후에는 종합을 다시 하지 않아 승인된 근거가
    최종 보고서 5장에 반영되지 않았습니다.

    보고서로 나가기 직전에 한 번 더 계산해 이 누락을 없앱니다. 철회된 근거
    (True→False)와 재검색으로 추가된 근거도 같은 경로로 반영됩니다.

    판정(synthesis)은 다시 만들지 않고 validate가 남긴 값을 그대로 둡니다.
    synthesis.run은 analyses에서 판정을 새로 만들기 때문에, 반환값을 그대로
    쓰면 validate가 낮춘 confidence 같은 정규화가 사라집니다.
    """
    result = synthesis.run(state, provider)
    findings = result["synthesis_findings"]
    logger.info("최종 종합 재계산 (findings %d건)", len(findings))
    return {"synthesis_findings": findings}


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
            #
            # 두 예외를 모두 "외부 서비스 요청 실패"로 적었더니 보고서에 사실과
            # 다른 사유가 남았습니다. TimeoutError는 외부 장애가 아니라 로컬 호출
            # 상한(runtime.TimeoutProvider) 초과일 수 있고, 관점이 직렬화된 추론을
            # 기다리다 상한에 걸린 경우가 실제로 그렇게 기록됐습니다.
            #
            # provider 예외 문구 자체는 보고서로 내보내지 않습니다. URL이나 내부
            # 경로가 섞일 수 있어 tests/test_workflow.py가 유출을 금지합니다.
            # 그래서 예외 종류별 고정 문구만 남기고 원문은 로그로 보냅니다.
            error = AgentError(
                code=type(exc).__name__,
                message="호출 상한 시간 초과"
                if isinstance(exc, TimeoutError)
                else "외부 서비스 연결 실패",
            )
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
            # 원문은 보고서가 아니라 로그로만 보냅니다. status와 code만으로는
            # 상한 초과인지 연결 실패인지 재현 때 다시 계측해야 했습니다.
            logger.warning(
                "[%s] 평가 실패 (%.1fs): %s — %s",
                key,
                time.monotonic() - started,
                error.code,
                exc,
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
        # provider가 반환한 실패 사유를 남깁니다. status만 보면 원인을 알 수 없어
        # 재현 시 별도 계측이 필요했습니다.
        for failed in (a for a in assessments if a.status == "failed" and a.error):
            logger.warning(
                "[%s] %s 판단 보류: %s — %s",
                key,
                failed.technology_id,
                failed.error.code,
                failed.error.message,
            )
        return {"analyses": {key: assessments}, "evidence": evidence}

    def dispatch(state):
        targets = (
            PERSPECTIVES
            if state["retry_count"] == 0
            else sorted({m.perspective for m in state["missing_evidence"] if m.retryable})
        )
        # "병렬"이라고 적었더니 실제 추론이 직렬인 사실이 로그에서 가려졌습니다.
        # 그래프가 하는 일은 fan-out(대상 선정과 분기)까지이고, 동시 실행 여부는
        # 모델 계층의 lock이 결정합니다. 로그는 그래프가 보장하는 것만 말합니다.
        logger.info(
            "관점 %s 실행: %s",
            "fan-out" if state["retry_count"] == 0 else f"재평가({state['retry_count']}회차)",
            ", ".join(targets) or "없음",
        )
        return [Send("evaluate", {"perspective": key, "state": state}) for key in targets]

    graph.add_node("evaluate", evaluate)
    graph.add_node("synthesize", lambda state: _logged_synthesize(state, provider))
    graph.add_node("validate", lambda state: _logged_validate(state, provider))
    graph.add_node(
        "additional_search", lambda state: additional_search_with_context(state, provider)
    )
    graph.add_node("report", lambda state: _logged_report(state, provider))
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", dispatch, ["evaluate"])
    graph.add_edge("evaluate", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_node("finalize_synthesis", lambda state: finalize_synthesis(state, provider))
    # 종료 경로는 모두 finalize_synthesis를 지납니다. 재검색 없이 끝나는 경우,
    # 재검색 2회를 소진한 경우, 재시도 불가로 끝나는 경우가 모두 해당합니다.
    graph.add_conditional_edges(
        "validate",
        lambda state: (
            "additional_search"
            if any(m.retryable for m in state["missing_evidence"])
            and state["retry_count"] < MAX_RETRIES
            else "finalize_synthesis"
        ),
        ["additional_search", "finalize_synthesis"],
    )
    graph.add_edge("finalize_synthesis", "report")
    graph.add_conditional_edges("additional_search", dispatch, ["evaluate"])
    graph.add_edge("report", END)
    return graph.compile()
