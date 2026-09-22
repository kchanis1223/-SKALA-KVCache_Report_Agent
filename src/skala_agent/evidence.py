"""Evidence 생성 주체가 호출하는 공통 ID 발급 함수."""

import hashlib
import json

from pydantic import HttpUrl

from skala_agent.schemas import Perspective


def evidence_id(
    *,
    owner: Perspective | str,
    technology_id: str,
    claim: str,
    url: str,
    chunk_id: str | None = None,
) -> str:
    """검색 순위·재시도·검증 결과와 무관한 ID. owner는 연구/관점별 namespace."""
    if owner not in {"research", "trl", "market", "stakeholder", "domain"}:
        raise ValueError("owner는 research 또는 평가 관점이어야 합니다.")
    content = [owner, technology_id, claim, str(HttpUrl(url)), chunk_id]
    digest = hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()[:24]
    return f"{owner}-{technology_id}-{digest}"
