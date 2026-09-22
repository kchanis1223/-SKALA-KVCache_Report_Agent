def run(state, provider):
    analysis, evidence = provider.research(state["selected_technologies"])
    return {"tech_analysis": analysis, "evidence": evidence}
