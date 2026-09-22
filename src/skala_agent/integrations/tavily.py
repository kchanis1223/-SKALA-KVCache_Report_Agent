import hashlib
from urllib.parse import urlparse

from pydantic import ValidationError

from skala_agent.integrations.contracts import SearchDocument, ServiceConfigurationError
from skala_agent.integrations.http import post_json


def document_id(url: str) -> str:
    return "web-" + hashlib.sha256(url.encode()).hexdigest()[:20]


class TavilySearch:
    def __init__(self, api_key: str, *, official_domains=(), timeout=30.0, transport=None):
        if not api_key.strip():
            raise ServiceConfigurationError("TAVILY_API_KEY 환경변수가 필요합니다.")
        self._api_key = api_key
        self.official_domains = {d.strip().lower() for d in official_domains if d.strip()}
        self.timeout = timeout
        self.transport = transport

    def search(self, query: str) -> list[SearchDocument]:
        result = post_json(
            "https://api.tavily.com/search",
            {
                "query": query,
                "search_depth": "basic",
                "max_results": 2,
                "include_answer": False,
                "include_raw_content": False,
            },
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self.timeout,
            transport=self.transport,
        )
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            raise ConnectionError("검색 서비스 응답에 results 목록이 없습니다.")
        documents = []
        for row in result["results"]:
            try:
                url = row["url"]
                host = (urlparse(url).hostname or "").lower()
                kind = (
                    "paper"
                    if host == "arxiv.org"
                    else "official"
                    if host in self.official_domains
                    else "news"
                )
                document = SearchDocument(
                    id=document_id(url),
                    title=row["title"],
                    url=url,
                    content=row["content"],
                    source_type=kind,
                )
            except (KeyError, TypeError, ValidationError, ValueError):
                continue
            documents.append(document)
        return documents
