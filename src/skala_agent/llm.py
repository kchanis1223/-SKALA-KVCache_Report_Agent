"""역할별 LLM. 모델은 하나로 두고 추론 강도(reasoning effort)만 다르게 씁니다."""

import os
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-5.4-mini"
EFFORTS = {
    "orchestrator": "high",
    "worker": "medium",
    "self_review": "low",
    "validator": "medium",
    "reporter": "medium",
    "judge": "medium",
}
TIMEOUT_SECONDS = 180


def load_env(root: str | Path = ".") -> None:
    """.env.local이 .env보다 우선합니다. 이미 설정된 환경변수는 덮지 않습니다."""
    root = Path(root)
    load_dotenv(root / ".env.local")
    load_dotenv(root / ".env")


@cache
def get_model(role: str) -> ChatOpenAI:
    if role not in EFFORTS:
        raise ValueError(f"알 수 없는 역할: {role}")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 필요합니다. .env 또는 환경변수에 설정하세요.")
    # gpt-5 계열은 /v1/chat/completions에서 function tool과 reasoning effort를
    # 함께 쓸 수 없어(400) Responses API(/v1/responses)를 사용합니다.
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        use_responses_api=True,
        reasoning={"effort": EFFORTS[role]},
        timeout=TIMEOUT_SECONDS,
        max_retries=2,
    )
