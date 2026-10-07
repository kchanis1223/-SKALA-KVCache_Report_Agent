"""synthesizer: worker 결과를 관점별로 모읍니다. LLM을 쓰지 않습니다."""

from skala_agent.workflow.state import AGENTS


def run(state):
    results = state.get("worker_results", {})
    grouped = {agent: [] for agent in AGENTS}
    for task in state["plan"]:
        if task.id in results:
            grouped[task.agent].append(results[task.id])
    return {
        "result": {agent: items for agent, items in grouped.items() if items},
        "node_status": {"synthesize": "done"},
    }
