import argparse
from pathlib import Path

from skala_agent.workflow.graph import build_graph, initial_state


def main():
    parser = argparse.ArgumentParser(description="API 키 없이 실행하는 workflow 뼈대")
    parser.add_argument("--output", type=Path, default=Path("outputs/report.md"))
    args = parser.parse_args()
    state = build_graph().invoke(initial_state())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(state["report"], encoding="utf-8")
    print(f"개발용 보고서 생성: {args.output} (재검색 {state['retry_count']}회)")
