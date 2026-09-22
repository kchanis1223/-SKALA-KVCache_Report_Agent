import argparse
import logging
from pathlib import Path

from skala_agent.runtime import (
    DEFAULT_TIMEOUT_SECONDS,
    MODES,
    ProviderUnavailableError,
    load_provider,
)
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="외부 서비스 호출 1건의 상한(초). 0이면 상한 없음",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="단계별 진행 상황과 소요 시간을 출력",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        provider = load_provider(args.mode, timeout=args.timeout)
    except ProviderUnavailableError as exc:
        raise SystemExit(f"실행 중단: {exc}") from exc

    state = build_graph(provider).invoke(initial_state())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(state["report"], encoding="utf-8")
    print(f"[{args.mode}] 보고서 생성: {args.output} (재검색 {state['retry_count']}회)")
