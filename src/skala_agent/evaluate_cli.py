"""Ollama 모델 배정 설정을 사용하는 4개 관점 잠정 평가 명령."""

import argparse
import json
from pathlib import Path

from skala_agent.adapters import build_provider
from skala_agent.runtime import ProviderUnavailableError
from skala_agent.workflow.graph import initial_state


def main():
    parser = argparse.ArgumentParser(
        description="Ollama Qwen3 TRL·시장성·이해관계자·도메인 잠정 평가"
    )
    parser.add_argument(
        "--perspective", choices=("trl", "market", "stakeholder", "domain", "all"), default="all"
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluations.json"))
    args = parser.parse_args()
    try:
        provider = build_provider(args.env_file)
    except ProviderUnavailableError as exc:
        parser.error(str(exc))
    # build_provider()로 real provider를 만들어 쓰므로 real 실행입니다.
    state = initial_state("real")
    technologies = state["selected_technologies"]
    research, research_evidence = provider.research(technologies)
    assessments, evidence = [], []
    for perspective in (
        ("trl", "market", "stakeholder", "domain")
        if args.perspective == "all"
        else (args.perspective,)
    ):
        results, sources = provider.assess(
            perspective, technologies, state["domain"], research, research_evidence
        )
        assessments.extend(results)
        evidence.extend(sources)
    output = {
        "notice": "잠정 평가입니다. 의미적 근거 검증 전이며 최종 보고서가 아닙니다.",
        "provider": "ollama",
        "models": provider.models.settings.assignment(),
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
