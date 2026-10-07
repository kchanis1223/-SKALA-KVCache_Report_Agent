"""worker가 쓰는 검색 도구. 호출마다 새 registry에 원문을 기록합니다.

도구가 돌려준 원문만 출처로 인정합니다. 그래서 worker가 만든 quote가
registry의 원문 안에 있는지 코드로 확인할 수 있습니다.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import tool

from skala_agent.workflow.state import Source

logger = logging.getLogger(__name__)
EXCERPT_CHARS = 1200


@dataclass
class Resources:
    """실행 동안 공유하는 외부 자원. retriever는 색인이 없으면 None입니다."""

    search: object
    retriever: object | None = None
    retriever_error: str | None = field(default=None)


def _format(sources: list[Source]) -> str:
    if not sources:
        return "검색 결과 없음."
    blocks = [
        f"[source_id: {s.id}] {s.title}" + (f" (p.{s.page})" if s.page else "") + f"\n{s.text}"
        for s in sources
    ]
    return "\n\n".join(blocks)


def make_tools(task_id: str, registry: dict[str, Source], resources: Resources):
    counter = {"web": 0, "paper": 0}

    def register(kind: str, **fields) -> Source:
        counter[kind] += 1
        source = Source(id=f"{task_id}-{kind[0]}{counter[kind]}", kind=kind, **fields)
        registry[source.id] = source
        return source

    @tool
    def web_search(query: str) -> str:
        """웹에서 기사·공식 자료·보고서를 검색한다. 영어 질의가 결과가 많다."""
        try:
            documents = resources.search.search(query)
        except Exception as exc:  # noqa: BLE001 - 도구 실패는 모델에게 알리고 계속
            logger.warning("[%s] web_search 실패: %s", task_id, exc)
            return f"웹 검색 실패: {type(exc).__name__}"
        found = [
            register(
                "web",
                title=d.title,
                url=str(d.url),
                text=d.content[:EXCERPT_CHARS],
            )
            for d in documents
        ]
        return _format(found)

    @tool
    def paper_search(
        query: str,
        paper_id: Literal["turboquant", "turboquant_analysis", "itme"] | None = None,
    ) -> str:
        """TurboQuant·ITME 논문 원문을 의미 검색한다. paper_id로 논문을 좁힐 수 있다.
        turboquant_analysis는 TurboQuant에 대한 제3자 독립 검토 논문이다."""
        if resources.retriever is None:
            return f"논문 검색 불가: {resources.retriever_error or '논문 색인 없음'}"
        try:
            results = resources.retriever.retrieve(query, top_k=3, paper_id=paper_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] paper_search 실패: %s", task_id, exc)
            return f"논문 검색 실패: {type(exc).__name__}"
        found = [
            register(
                "paper",
                title=f"{r.chunk.paper_id} · {r.chunk.section}",
                url=str(r.chunk.source_url),
                text=r.chunk.text[:EXCERPT_CHARS],
                page=r.chunk.page,
            )
            for r in results
        ]
        return _format(found)

    return [web_search, paper_search]
