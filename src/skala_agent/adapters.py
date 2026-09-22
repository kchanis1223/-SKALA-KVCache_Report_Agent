"""runtime.load_provider('real')의 Ollama + Tavily provider factory."""

from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.tavily import TavilySearch
from skala_agent.model_config import ModelRouter, ModelSettings, read_environment
from skala_agent.runtime import ProviderUnavailableError


def build_provider(env_file=".env"):
    env = read_environment(env_file)
    try:
        settings = ModelSettings.from_environment(env)
        search = TavilySearch(
            env.get("TAVILY_API_KEY", ""),
            official_domains=env.get("OFFICIAL_SOURCE_DOMAINS", "").split(","),
        )
        return EvaluationProvider(models=ModelRouter(settings), search=search)
    except ValueError as exc:
        raise ProviderUnavailableError(
            ".env/.env.local의 Ollama 모델 설정 및 TAVILY_API_KEY를 확인하세요. "
            "모델은 qwen3:4b/8b를 지원합니다."
        ) from exc
