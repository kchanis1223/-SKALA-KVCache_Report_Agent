"""TDD tests for in-memory caching and batch processing in EvaluationProvider.validate_evidence."""

from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.schemas import Evidence


class DummyModel:
    def __init__(self, responses=None):
        self.call_count = 0
        self.calls = []
        self.responses = responses or []

    def invoke_structured(self, messages, schema):
        self.call_count += 1
        self.calls.append((messages, schema))
        if self.responses:
            res = self.responses.pop(0)
            if isinstance(res, Exception):
                raise res
            return res
        return '{"results": []}'


class DummySearch:
    pass


def make_evidence(item_id, claim="Claim 1", excerpt="Excerpt 1"):
    return Evidence(
        id=item_id,
        technology_id="turboquant",
        claim=claim,
        url="https://example.com/test",
        title="Test Title",
        source_type="official",
        excerpt=excerpt,
        supports_claim=True,
    )


def test_batch_validation_single_llm_call():
    """N개의 미검증 근거가 1회 배치 LLM 호출로 검증되는지 확인."""
    model = DummyModel(
        responses=[
            '{"results": [{"id": "ev_1", "supports_claim": true}, '
            '{"id": "ev_2", "supports_claim": false}]}'
        ]
    )
    provider = EvaluationProvider(model=model, search=DummySearch())

    ev1 = make_evidence("ev_1")
    ev2 = make_evidence("ev_2")

    updated = provider.validate_evidence([ev1, ev2])

    assert len(updated) == 2
    assert model.call_count == 1  # 2개 근거를 1회 배치 호출로 처리
    res_map = {item.id: item.supports_claim for item in updated}
    assert res_map["ev_1"] is True
    assert res_map["ev_2"] is False


def test_caching_prevents_duplicate_llm_calls():
    """동일한 근거 목록으로 다음 라운드 실행 시 LLM 호출이 0회(캐시 히트)가 되는지 확인."""
    model = DummyModel(
        responses=[
            '{"results": [{"id": "ev_1", "supports_claim": true}, '
            '{"id": "ev_2", "supports_claim": false}]}'
        ]
    )
    provider = EvaluationProvider(model=model, search=DummySearch())

    ev1 = make_evidence("ev_1")
    ev2 = make_evidence("ev_2")

    # Round 1: 배치 호출 1회 발생
    updated1 = provider.validate_evidence([ev1, ev2])
    assert model.call_count == 1
    assert updated1[0].supports_claim is True
    assert updated1[1].supports_claim is False

    # Round 2: 완전히 동일한 근거 -> 캐시 히트, LLM 호출 0회 추가 (총 1회)
    updated2 = provider.validate_evidence([ev1, ev2])
    assert model.call_count == 1
    assert updated2[0].supports_claim is True
    assert updated2[1].supports_claim is False


def test_cache_invalidation_on_excerpt_or_claim_change():
    """excerpt나 claim이 변경되면 이전 캐시가 무효화되어 다시 LLM을 호출하는지 확인."""
    model = DummyModel(
        responses=[
            '{"results": [{"id": "ev_1", "supports_claim": true}]}',
            '{"results": [{"id": "ev_1", "supports_claim": false}]}',
        ]
    )
    provider = EvaluationProvider(model=model, search=DummySearch())

    ev1 = make_evidence("ev_1", excerpt="Original Excerpt")
    provider.validate_evidence([ev1])
    assert model.call_count == 1

    # Excerpt 변경 -> Cache Miss -> 2번째 LLM 호출 발생
    ev1_modified = make_evidence("ev_1", excerpt="Modified Excerpt")
    updated = provider.validate_evidence([ev1_modified])
    assert model.call_count == 2
    assert updated[0].supports_claim is False


def test_partial_cache_miss_batches_only_missing_items():
    """일부 근거만 캐시 히트되고 신규 근거만 묶어서 배치 호출하는지 확인."""
    model = DummyModel(
        responses=[
            '{"results": [{"id": "ev_1", "supports_claim": true}]}',
            '{"results": [{"id": "ev_2", "supports_claim": true}]}',
        ]
    )
    provider = EvaluationProvider(model=model, search=DummySearch())

    # ev_1 검증 -> 캐시 저장
    provider.validate_evidence([make_evidence("ev_1")])
    assert model.call_count == 1

    # ev_1 (캐시 히트) + ev_2 (캐시 미스) -> LLM 호출 1회만 추가 (ev_2만 배치)
    updated = provider.validate_evidence([make_evidence("ev_1"), make_evidence("ev_2")])
    assert model.call_count == 2
    assert len(updated) == 2


def test_batch_response_missing_id_safety():
    """배치 응답에서 특정 ID가 누락되면 허위 승인하지 않고
    False로 유지되거나 안전 처리되는지 확인."""
    model = DummyModel(
        responses=[
            '{"results": [{"id": "ev_1", "supports_claim": true}]}'  # ev_2 ID 누락됨
        ]
    )
    provider = EvaluationProvider(model=model, search=DummySearch())

    ev1 = make_evidence("ev_1")
    ev2 = make_evidence("ev_2")
    updated = provider.validate_evidence([ev1, ev2])

    res_map = {item.id: item.supports_claim for item in updated}
    assert res_map["ev_1"] is True
    # 누락된 ev_2는 True로 자동 승인되면 안 되고 False여야 함
    assert res_map["ev_2"] is False
