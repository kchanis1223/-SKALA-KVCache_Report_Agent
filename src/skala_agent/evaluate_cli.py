"""이슈 #5의 두 관점을 독립 실행합니다. 공통 workflow CLI와 분리."""

import argparse
import json
import os
from pathlib import Path

from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.qwen import TransformersQwen
from skala_agent.integrations.tavily import TavilySearch
from skala_agent.workflow.graph import initial_state


def main():
    parser = argparse.ArgumentParser(description="Qwen3 로컬 TRL·시장성 잠정 평가")
    parser.add_argument("--perspective", choices=("trl", "market", "all"), default="all")
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B"))
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default=os.getenv("QWEN_DEVICE", "auto")
    )
    parser.add_argument("--revision", default=os.getenv("QWEN_REVISION", "main"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluations.json"))
    args = parser.parse_args()
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key.strip():
        parser.error("웹검색에 사용할 TAVILY_API_KEY를 환경변수로 설정하세요.")
    model = TransformersQwen(model_id=args.model, device=args.device, revision=args.revision)
    search = TavilySearch(
        api_key, official_domains=os.getenv("OFFICIAL_SOURCE_DOMAINS", "").split(",")
    )
    provider = EvaluationProvider(model=model, search=search)
    state = initial_state()
    technologies = state["selected_technologies"]
    research, _ = provider.research(technologies)
    assessments, evidence = [], []
    for perspective in ("trl", "market") if args.perspective == "all" else (args.perspective,):
        results, sources = provider.assess(perspective, technologies, state["domain"], research, [])
        assessments.extend(results)
        evidence.extend(sources)
    output = {
        "notice": "이슈 #5의 잠정 평가입니다. 의미적 근거 검증 전이며 최종 보고서가 아닙니다.",
        "model": args.model,
        "revision": args.revision,
        "assessments": [a.model_dump(mode="json") for a in assessments],
        "evidence": [e.model_dump(mode="json") for e in evidence],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failed = sum(a.status == "failed" for a in assessments)
    print(f"잠정 평가 저장: {args.output} (평가 {len(assessments)}건, 실패 {failed}건)")
    if failed:
        raise SystemExit(1)
