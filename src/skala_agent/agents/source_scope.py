"""기술명 동음이의어·검색 무관 문서를 걸러내는 보수적 입력 필터."""

import re

ANCHORS = {
    "itme": '"ITME" "CXL" memory inference',
    "turboquant": '"TurboQuant" "KV cache"',
}
ALIASES = {
    "itme": ("ITME", "Inference Tiered Memory Expansion"),
    "turboquant": ("TurboQuant",),
}
STAKEHOLDER_SEARCH_TERMS = (
    "GPU accelerator vendor technical statement",
    "memory vendor semiconductor technical response",
    "cloud operator deployment support",
    "open source serving framework integration",
    "investor analyst semiconductor outlook",
)


def mentions_technology(document, technology):
    if document.chunk_id:
        return True  # 논문 소속은 Retriever의 paper_id/metadata로 추적합니다.
    text = " ".join((document.title, str(document.url), document.content))
    key = technology if isinstance(technology, str) else technology.id
    name = technology if isinstance(technology, str) else technology.name
    aliases = ALIASES.get(key, (name,))
    return any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I) for name in aliases)


def explicit_sla_negative(question_id, quotes):
    """미보고를 no로 바꾸지 않도록 최소 요건 검사. 의미 검증의 대체는 아닙니다."""
    text = " ".join(quotes).lower()
    if question_id == "latency_degraded":
        return bool(re.search(r"\bttft\b", text) and re.search(r"\btpot\b", text))
    if question_id == "accuracy_loss":
        return bool(
            re.search(
                r"(?:no|without)\s+(?:accuracy|quality)\s+(?:loss|degradation)|"
                r"(?:accuracy|quality)\s+(?:is\s+)?(?:preserved|unchanged)|"
                r"preserv\w*\s+(?:accuracy|quality)|lossless|정확도\s*보존",
                text,
            )
        )
    return True
