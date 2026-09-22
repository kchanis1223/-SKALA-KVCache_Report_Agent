"""공백·문장 첫 일반어의 대문자화만 원문의 연속된 구간으로 복원합니다. 의역은 허용하지 않습니다."""

import re


def quote_options(content: str) -> list[str]:
    """재추출용 원문 후보. 문장과 겹치는 창 모두 원문의 연속 부분문자열입니다."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", content)]
    windows = [content[start : start + 300].strip() for start in range(0, len(content), 250)]
    return list(dict.fromkeys(q for q in sentences + windows if q and len(q) <= 300))


def source_quote(content: str, quote: str) -> str | None:
    if not quote.strip():
        return None
    if quote in content:
        return quote
    pattern = r"\s+".join(re.escape(word) for word in quote.split())
    match = re.search(pattern, content)
    if match:
        return match.group(0)
    # 모델이 문장 중간의 we/the 등을 인용 첫머리에서 대문자로 바꾸는 경우만 허용합니다.
    # 일반 ignorecase를 쓰면 US/us, 변수 V/v 등을 혼동하므로 허용하지 않습니다.
    words = quote.split()
    if words[0] in {"We", "The", "This", "Our", "In", "It", "A", "An"}:
        words[0] = words[0].lower()
        match = re.search(r"\s+".join(re.escape(word) for word in words), content)
        if match:
            return match.group(0)
    return None


def align_citations(draft, documents):
    sources = {d.id: d.content for d in documents}
    findings = []
    for finding in draft.findings:
        citations = []
        for citation in finding.citations:
            exact = source_quote(sources.get(citation.source_id, ""), citation.quote)
            citations.append(citation.model_copy(update={"quote": exact}) if exact else citation)
        findings.append(finding.model_copy(update={"citations": citations}))
    return draft.model_copy(update={"findings": findings})
