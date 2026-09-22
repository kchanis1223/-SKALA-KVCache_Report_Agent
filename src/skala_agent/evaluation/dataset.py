"""김동찬 담당: 검색 평가셋 로딩과 검증.

정답은 chunk ID 기준이라 청킹 설정이 바뀌면 재라벨링이 필요합니다.
평가셋과 색인의 청킹 버전이 다르면 측정 전에 막습니다.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

DEFAULT_EVAL_SET = Path("evals/retrieval_queries.json")


class EvalQuery(BaseModel):
    id: str
    query: str = Field(min_length=1)
    gold_chunk_ids: list[str] = Field(min_length=1)

    @field_validator("gold_chunk_ids")
    @classmethod
    def unique_gold(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("정답 chunk ID는 중복될 수 없습니다.")
        return value


class EvalSet(BaseModel):
    version: str
    chunking_version: str
    queries: list[EvalQuery] = Field(min_length=1)
    index_dir: str | None = None
    note: str | None = None

    @field_validator("queries")
    @classmethod
    def unique_ids(cls, value: list[EvalQuery]) -> list[EvalQuery]:
        if len({query.id for query in value}) != len(value):
            raise ValueError("질의 id는 중복될 수 없습니다.")
        return value

    def gold_ids(self) -> set[str]:
        return {gold for query in self.queries for gold in query.gold_chunk_ids}

    def check_against_index(self, indexed_ids: set[str]) -> list[str]:
        """색인에 없는 정답 chunk ID를 돌려줍니다. 비어 있어야 정상입니다."""
        return sorted(self.gold_ids() - indexed_ids)


def load_eval_set(path: str | Path = DEFAULT_EVAL_SET) -> EvalSet:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalSet.model_validate(payload)
