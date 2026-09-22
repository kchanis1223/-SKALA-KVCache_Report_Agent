"""김강휘 담당: trl 평가와 structured output 구현 지점."""


def run(state, provider):
    return provider.assess(
        "trl",
        state["selected_technologies"],
        state["domain"],
        state["tech_analysis"],
        state["evidence"],
    )
