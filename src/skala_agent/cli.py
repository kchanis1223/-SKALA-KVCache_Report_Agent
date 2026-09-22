import argparse
from pathlib import Path

from skala_agent.runtime import MODES, ProviderUnavailableError, load_provider
from skala_agent.workflow.graph import build_graph, initial_state


def main():
    parser = argparse.ArgumentParser(description="KV cache 기술 비교 workflow 실행")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="demo",
        help="demo: API 키 없이 실행하는 기본값. real: 실제 provider 연결",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/report.md"))
    args = parser.parse_args()

    try:
        provider = load_provider(args.mode)
    except ProviderUnavailableError as exc:
        raise SystemExit(f"실행 중단: {exc}") from exc

    state = build_graph(provider).invoke(initial_state())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(state["report"], encoding="utf-8")
    print(f"[{args.mode}] 보고서 생성: {args.output} (재검색 {state['retry_count']}회)")
