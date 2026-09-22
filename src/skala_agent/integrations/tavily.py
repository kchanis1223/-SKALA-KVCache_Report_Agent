import hashlib
import re
from urllib.parse import urlparse

from pydantic import ValidationError

from skala_agent.integrations.contracts import SearchDocument, ServiceConfigurationError
from skala_agent.integrations.http import post_json


def document_id(url: str) -> str:
    return "web-" + hashlib.sha256(url.encode()).hexdigest()[:20]


# 보고서는 한국어로 쓰고 근거는 한국어·영어 원문만 인용합니다. 그 외 문자
# 체계의 기사가 섞이면 인용을 읽을 수 없고 검증 모델도 판정하기 어렵습니다.
# 실측: 태국어 기사가 REFERENCE에 인용됐습니다.
_READABLE = re.compile(r"[0-9A-Za-z\u3131-\u318E\uAC00-\uD7A3]")
_MIN_READABLE_RATIO = 0.5


def is_readable(text: str) -> bool:
    """한글·라틴 문자 비중이 절반 이상인지 본다. 공백·기호는 세지 않는다."""
    letters = [ch for ch in text if not ch.isspace() and ch.isalnum()]
    if not letters:
        return False
    readable = sum(1 for ch in letters if _READABLE.match(ch))
    return readable / len(letters) >= _MIN_READABLE_RATIO


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
            if not is_readable(f"{document.title} {document.content}"):
                continue
            documents.append(document)
        return documents
