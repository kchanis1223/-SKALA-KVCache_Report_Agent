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
from skala_agent.schemas import PERSPECTIVES, Assessment, Evidence, Technology
from skala_agent.workflow.state import EvaluationState

AGENTS = {"trl": trl, "market": market, "stakeholder": stakeholder, "domain": domain}
MAX_RETRIES = 2


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


def build_graph(provider: Provider | None = None):
    provider = provider or DemoProvider()
    graph = StateGraph(EvaluationState)
    graph.add_node("research", lambda state: research.run(state, provider))

    def evaluate(payload):
        key, state = payload["perspective"], payload["state"]
        assessments, evidence = AGENTS[key].run(state, provider)
        assessments = [Assessment.model_validate(a) for a in assessments]
        evidence = [Evidence.model_validate(e) for e in evidence]
        expected = {t.id for t in state["selected_technologies"]}
        if (
            len(assessments) != len(expected)
            or {a.technology_id for a in assessments} != expected
            or any(a.perspective != key for a in assessments)
        ):
            raise ValueError("각 관점은 선택된 기술마다 정확히 하나의 평가를 반환해야 합니다.")
        return {"analyses": {key: assessments}, "evidence": evidence}

    def dispatch(state):
        targets = (
            PERSPECTIVES
            if state["retry_count"] == 0
            else sorted({m.perspective for m in state["missing_evidence"]})
        )
        return [Send("evaluate", {"perspective": key, "state": state}) for key in targets]

    graph.add_node("evaluate", evaluate)
    graph.add_node("synthesize", synthesis.run)
    graph.add_node("validate", validation.run)
    graph.add_node("additional_search", lambda state: additional_search.run(state, provider))
    graph.add_node("report", report.run)
    graph.add_edge(START, "research")
    graph.add_conditional_edges("research", dispatch, ["evaluate"])
    graph.add_edge("evaluate", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_conditional_edges(
        "validate",
        lambda state: (
            "additional_search"
            if state["missing_evidence"] and state["retry_count"] < MAX_RETRIES
            else "report"
        ),
        ["additional_search", "report"],
    )
    graph.add_conditional_edges("additional_search", dispatch, ["evaluate"])
    graph.add_edge("report", END)
    return graph.compile()
