"""runtime.load_provider('real')의 Ollama + Tavily provider factory."""

from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.tavily import TavilySearch
from skala_agent.model_config import ModelRouter, ModelSettings, read_environment
from skala_agent.retrieval.factory import try_load_retriever
from skala_agent.retrieval.indexing import DEFAULT_INDEX_DIR
from skala_agent.runtime import ProviderUnavailableError

_AUTO_RETRIEVER = object()


def build_provider(env_file=".env", *, retriever=_AUTO_RETRIEVER):
    env = read_environment(env_file)
    try:
        settings = ModelSettings.from_environment(env)
        search = TavilySearch(
            env.get("TAVILY_API_KEY", ""),
            official_domains=env.get("OFFICIAL_SOURCE_DOMAINS", "").split(","),
        )
    except ValueError as exc:
        raise ProviderUnavailableError(
            ".env/.env.local의 Ollama 모델 설정, OPENAI_API_KEY, TAVILY_API_KEY를 확인하세요."
        ) from exc
    if retriever is _AUTO_RETRIEVER:
        retriever = try_load_retriever(env.get("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    return EvaluationProvider(models=ModelRouter(settings), search=search, retriever=retriever)
