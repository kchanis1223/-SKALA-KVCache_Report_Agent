from skala_agent.schemas import Evidence


def run(state, provider):
    missing = [m for m in state["missing_evidence"] if m.retryable]
    return {
        "evidence": [Evidence.model_validate(e) for e in provider.search_missing(missing)],
        "retry_count": state["retry_count"] + 1,
    }
