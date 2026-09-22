def retrieval_metrics(rows: list[tuple[list[str], set[str]]]) -> dict[str, float]:
    """(검색 순서 chunk IDs, 정답 chunk IDs). MRR은 제공된 전체 순위 기준."""
    if not rows or any(not expected for _, expected in rows):
        raise ValueError("평가 질의와 질의별 정답이 필요합니다.")
    ranks = []
    for retrieved, expected in rows:
        if len(retrieved) != len(set(retrieved)):
            raise ValueError("검색 결과 chunk ID는 중복될 수 없습니다.")
        ranks.append(next((i for i, item in enumerate(retrieved, 1) if item in expected), 0))
    count = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks) / count,
        "hit@3": sum(0 < r <= 3 for r in ranks) / count,
        "mrr": sum(1 / r for r in ranks if r) / count,
    }
