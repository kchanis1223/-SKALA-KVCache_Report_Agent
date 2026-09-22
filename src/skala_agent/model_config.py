"""전체 Agent 공통 Ollama 4B/8B 모델 배정. 객체 생성은 네트워크를 호출하지 않습니다."""

import os
from pathlib import Path
from threading import Lock
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field

from skala_agent.integrations.ollama import OllamaChat

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
LIGHT_AGENTS = {"research", "additional_search"}


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
    provider: Literal["ollama"] = "ollama"
    light_model: Literal["qwen3:4b"] = "qwen3:4b"
    main_model: Literal["qwen3:4b", "qwen3:8b"] = "qwen3:8b"
    use_single_model: bool = False
    single_model: Literal["qwen3:4b"] = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    timeout: float = Field(default=120, gt=0)

    @classmethod
    def from_environment(cls, env):
        names = {
            "provider": "LLM_PROVIDER",
            "light_model": "LIGHT_MODEL",
            "main_model": "MAIN_MODEL",
            "use_single_model": "USE_SINGLE_MODEL",
            "single_model": "SINGLE_MODEL",
            "base_url": "OLLAMA_BASE_URL",
            "timeout": "OLLAMA_TIMEOUT",
        }
        return cls(**{key: env[name] for key, name in names.items() if name in env})

    def model_for(self, agent: str) -> str:
        if agent not in AGENTS:
            raise ValueError(f"알 수 없는 Agent: {agent}")
        if self.use_single_model:
            return self.single_model
        return self.light_model if agent in LIGHT_AGENTS else self.main_model

    def assignment(self) -> dict[str, str]:
        return {agent: self.model_for(agent) for agent in AGENTS}


class ModelRouter:
    def __init__(self, settings: ModelSettings, *, transport=None):
        self.settings = settings
        # 두 모델을 동시에 추론하지 않도록 같은 lock을 공유합니다.
        lock = Lock()
        self._models = {
            name: OllamaChat(
                name,
                base_url=settings.base_url,
                timeout=settings.timeout,
                transport=transport,
                lock=lock,
            )
            for name in set(settings.assignment().values())
        }

    def for_agent(self, agent):
        return self._models[self.settings.model_for(agent)]
