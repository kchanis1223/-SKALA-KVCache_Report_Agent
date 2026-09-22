"""김강휘 담당: domain 평가와 structured output 구현 지점."""


def run(state, provider):
    return provider.assess(
        "domain",
        state["selected_technologies"],
        state["domain"],
        state["tech_analysis"],
        state["evidence"],
    )
