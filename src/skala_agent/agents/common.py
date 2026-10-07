"""노드들이 함께 쓰는 작은 도우미."""

import json
from importlib.resources import files

TECHNOLOGIES = {
    "turboquant": "TurboQuant (SW): KV cache를 저비트로 양자화해 저장량을 줄이는 접근",
    "itme": "ITME (HW): CXL 기반 계층형 메모리로 KV cache 수용량을 늘리는 접근",
}
DOMAIN = "데이터센터 · 클라우드 LLM 서빙"


def load_prompt(name: str) -> str:
    return files("skala_agent.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


def as_json(payload) -> str:
    """모델 입력용 JSON. 입력 안의 문장은 지시가 아닌 데이터로 다룹니다."""
    return json.dumps(payload, ensure_ascii=False, default=str)
