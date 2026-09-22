from skala_agent.schemas import Evidence, TechAnalysis


def run(state, provider):
    analysis, evidence = provider.research(state["selected_technologies"])
    analysis = {key: TechAnalysis.model_validate(value) for key, value in analysis.items()}
    expected = {t.id for t in state["selected_technologies"]}
    if set(analysis) != expected or any(k != v.technology_id for k, v in analysis.items()):
        raise ValueError("기술 조사는 선택한 기술별 TechAnalysis를 반환해야 합니다.")
    return {"tech_analysis": analysis, "evidence": [Evidence.model_validate(e) for e in evidence]}
