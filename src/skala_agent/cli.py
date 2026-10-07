import argparse
import logging
import os
import sqlite3
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from skala_agent.integrations.tavily import TavilySearch
from skala_agent.llm import load_env
from skala_agent.pdf import markdown_to_pdf
from skala_agent.retrieval.factory import try_load_retriever
from skala_agent.retrieval.indexing import DEFAULT_INDEX_DIR
from skala_agent.tools import Resources
from skala_agent.workflow.graph import build_graph, checkpoint_serde, run_config
from skala_agent.workflow.state import CRITERION_LABELS, initial_state

DEFAULT_QUESTION = "데이터센터 LLM 서빙 도입 관점에서 TurboQuant와 ITME를 비교해줘"


def build_resources(env) -> Resources:
    search = TavilySearch(
        env.get("TAVILY_API_KEY", ""),
        official_domains=env.get("OFFICIAL_SOURCE_DOMAINS", "").split(","),
    )
    try:
        retriever = try_load_retriever(env.get("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
        error = None if retriever else "색인 없음 (uv run skala-index로 생성)"
    except Exception as exc:  # noqa: BLE001 - 논문 검색 없이도 웹 근거로 진행
        retriever, error = None, f"{type(exc).__name__}: {exc}"
    return Resources(search=search, retriever=retriever, retriever_error=error)


def main(argv=None):
    parser = argparse.ArgumentParser(description="TurboQuant vs ITME 비교 보고서 생성")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="보고서가 답할 질문")
    parser.add_argument("--resume", metavar="RUN_ID", help="중단된 실행을 이어서 진행")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--verbose", action="store_true", help="단계별 진행 로그 출력")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    load_env()
    missing = [k for k in ("OPENAI_API_KEY", "TAVILY_API_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"실행 중단: {', '.join(missing)}가 필요합니다 (.env 또는 환경변수).")
    args.output.mkdir(parents=True, exist_ok=True)
    run_id = args.resume or uuid.uuid4().hex[:12]
    config = run_config(run_id)

    with sqlite3.connect(args.output / "checkpoints.sqlite", check_same_thread=False) as conn:
        app = build_graph(
            build_resources(os.environ), checkpointer=SqliteSaver(conn, serde=checkpoint_serde())
        )
        if args.resume and not app.get_state(config).values:
            raise SystemExit(f"재개할 실행이 없습니다: {run_id}")
        try:
            state = app.invoke(
                None if args.resume else initial_state(args.question, run_id), config
            )
        except Exception as exc:
            # 진행표(node_status)와 오류 메모를 남기고 재개 방법을 안내합니다.
            try:
                app.update_state(config, {"last_error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            except Exception:  # noqa: BLE001 - 첫 노드 전에 실패하면 기록할 체크포인트가 없음
                pass
            status = app.get_state(config).values.get("node_status", {})
            print(f"[{run_id}] 실행 중단: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"진행 상태: {status}", file=sys.stderr)
            print(f"이어서 실행: uv run skala-agent --resume {run_id}", file=sys.stderr)
            raise SystemExit(1) from exc

    path = markdown_to_pdf(state["report"], args.output / f"report-{run_id}.pdf")
    print(f"[{run_id}] 보고서 생성: {path} (재시도 {state.get('retry_count', 0)}회)")
    quality = state.get("quality")
    if quality is None or quality.error:
        verdict = "평가 못 함" + (f" ({quality.error})" if quality else "")
    elif quality.failed():
        verdict = "미달(" + ", ".join(CRITERION_LABELS[c] for c in quality.failed()) + ")"
    else:
        verdict = "통과"
    print(
        f"품질: {verdict} · 재작성 {state.get('rewrite_count', 0)}회"
        f" · 재조사 {state.get('research_count', 0)}회"
    )
    failed = [k for k, v in state["worker_results"].items() if not v.success]
    if failed:
        names = ", ".join(failed)
        print(f"실패한 과제 {len(failed)}개: {names} (보고서 한계점 참고)", file=sys.stderr)
