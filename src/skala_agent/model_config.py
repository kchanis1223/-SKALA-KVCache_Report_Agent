"""설계서의 Agent별 Ollama·OpenAI 모델 배정. 객체 생성은 네트워크를 호출하지 않습니다."""

import os
from pathlib import Path
from threading import Lock
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator

from skala_agent.integrations.ollama import DEFAULT_KEEP_ALIVE, OllamaChat, validate_keep_alive
from skala_agent.integrations.openai import OpenAIResponses

AGENTS = (
    "research",
    "additional_search",
    "trl",
    "market",
    "stakeholder",
    "domain",
    "validation",
    "synthesis",
    "report",
)
# 팀 결정: 모든 Agent를 OpenAI 단일 모델로 통일합니다. 로컬 Ollama 경로는
# integrations/ollama.py에 남겨두어 오프라인 재현과 비교 실행에 쓸 수 있게
# 합니다(OLLAMA_MODEL 지정 + USE_OLLAMA=true).
#
# 모델 ID는 OPENAI_MODEL 환경변수로 덮어쓸 수 있습니다. 조직 배포명이 다르면
# 코드를 고치지 않고 .env.local에서 바꾸세요.
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

# 모델은 하나로 통일하고, 역할별로 추론 강도만 다르게 둡니다. 반복 호출이 많은
# 조사·검색·평가는 낮게, 관점 간 상충을 종합하는 단계는 높게 배치합니다.
OPENAI_EFFORTS = {
    "research": "low",
    "additional_search": "low",
    "trl": "medium",
    "market": "medium",
    "stakeholder": "medium",
    "domain": "medium",
    "synthesis": "high",
    "validation": "low",
    "report": "medium",
}

# 로컬 Ollama로 되돌릴 때 사용하는 Agent 집합. 기본 경로에서는 쓰이지 않습니다.
OLLAMA_AGENTS = {"research", "additional_search", "trl", "market", "stakeholder", "domain"}


def read_environment(env_file: str | Path = ".env") -> dict[str, str]:
    # 환경변수 > .env.local > .env. 명시적인 다른 파일은 단독으로 읽습니다.
    path = Path(env_file)
    values = {k: v for k, v in dotenv_values(path).items() if v is not None}
    if path.name == ".env":
        values.update(
            {k: v for k, v in dotenv_values(path.with_name(".env.local")).items() if v is not None}
        )
    return {**values, **os.environ}


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: Literal["openai", "ollama"] = "openai"
    openai_model: str = DEFAULT_OPENAI_MODEL
    # provider="ollama"로 되돌릴 때만 쓰입니다.
    light_model: Literal["qwen3:4b"] = "qwen3:4b"
    main_model: Literal["qwen3:4b", "qwen3:8b"] = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    timeout: float = Field(default=120, gt=0)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    keep_alive: int | str = DEFAULT_KEEP_ALIVE

    @field_validator("keep_alive", mode="before")
    @classmethod
    def check_keep_alive(cls, value):
        return validate_keep_alive(value)

    @classmethod
    def from_environment(cls, env):
        names = {
            "provider": "LLM_PROVIDER",
            "openai_model": "OPENAI_MODEL",
            "light_model": "LIGHT_MODEL",
            "main_model": "MAIN_MODEL",
            "base_url": "OLLAMA_BASE_URL",
            "timeout": "OLLAMA_TIMEOUT",
            "openai_api_key": "OPENAI_API_KEY",
            "openai_base_url": "OPENAI_BASE_URL",
            "keep_alive": "OLLAMA_KEEP_ALIVE",
        }
        return cls(**{key: env[name] for key, name in names.items() if name in env})

    def model_for(self, agent: str) -> str:
        if agent not in AGENTS:
            raise ValueError(f"알 수 없는 Agent: {agent}")
        if self.provider == "ollama":
            return self.light_model if agent in OLLAMA_AGENTS else self.main_model
        # 아홉 Agent 모두 같은 모델을 쓰고 추론 강도로만 구분합니다.
        return self.openai_model

    def effort_for(self, agent: str) -> str:
        if agent not in AGENTS:
            raise ValueError(f"알 수 없는 Agent: {agent}")
        return OPENAI_EFFORTS[agent]

    def assignment(self) -> dict[str, str]:
        return {agent: self.model_for(agent) for agent in AGENTS}


class ModelRouter:
    """Agent별 모델 객체를 보관합니다. 모델은 같고 추론 강도만 다릅니다."""

    def __init__(self, settings: ModelSettings, *, transport=None):
        self.settings = settings
        self._models = (
            self._build_ollama(settings, transport)
            if settings.provider == "ollama"
            else self._build_openai(settings, transport)
        )

    @staticmethod
    def _build_ollama(settings, transport):
        # 로컬 추론은 같은 GPU 메모리를 쓰므로 직렬화가 필요합니다. 단, lock을
        # 모델 단위로 둡니다. 전체에 하나만 두면 서로 다른 모델끼리도 줄을 섭니다.
        locks: dict[str, Lock] = {}
        models = {}
        for agent in AGENTS:
            name = settings.model_for(agent)
            models[agent] = OllamaChat(
                name,
                base_url=settings.base_url,
                timeout=settings.timeout,
                keep_alive=settings.keep_alive,
                transport=transport,
                lock=locks.setdefault(name, Lock()),
            )
        return models

    @staticmethod
    def _build_openai(settings, transport):
        # OpenAI 호출은 원격이라 로컬 자원을 공유하지 않습니다. lock을 걸면 관점
        # fan-out이 직렬화되어 실행 시간이 관점 수만큼 늘어납니다(실측: 관점 하나가
        # 285.9초를 쓰는 동안 나머지 세 관점이 대기하다 상한에서 끊겼습니다).
        # 그래서 lock을 두지 않습니다.
        shared: dict[tuple[str, str], OpenAIResponses] = {}
        models = {}
        for agent in AGENTS:
            key = (settings.model_for(agent), settings.effort_for(agent))
            if key not in shared:
                shared[key] = OpenAIResponses(
                    key[0],
                    api_key=settings.openai_api_key,
                    reasoning_effort=key[1],
                    base_url=settings.openai_base_url,
                    timeout=settings.timeout,
                    transport=transport,
                )
            models[agent] = shared[key]
        return models

    def for_agent(self, agent):
        if agent not in self._models:
            raise ValueError(f"알 수 없는 Agent: {agent}")
        return self._models[agent]
