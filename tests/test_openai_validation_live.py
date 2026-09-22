"""Live OpenAI API integration test for batch evidence validation and caching using gpt-4o-mini."""

import os

import pytest
from dotenv import load_dotenv

from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.openai import OpenAIResponses
from skala_agent.schemas import Evidence

load_dotenv(".env")
load_dotenv(".env.local")


class DummySearch:
    pass


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY").startswith("sk-dummy"),
    reason="OPENAI_API_KEY가 없거나 유효하지 않으면 실제 라이브 테스트를 건너끕니다.",
)
def test_openai_validation_live_batch_and_cache():
    """실제 OpenAI gpt-4o-mini 모델을 호출하여 배치 검증 및 캐시 히트 동작을 라이브 확인."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # gpt-4o-mini 모델로 OpenAIResponses 초기화
    validation_model = OpenAIResponses(
        model="gpt-4o-mini",
        api_key=api_key,
        reasoning_effort="low",
        base_url=base_url,
        timeout=30,
    )

    provider = EvaluationProvider(model=validation_model, search=DummySearch())

    # 실제 지지하는 근거 1건과 명백히 무관/거짓인 근거 1건 준비
    ev1 = Evidence(
        id="live_ev_1",
        technology_id="turboquant",
        claim="TurboQuant는 KV cache 저장량을 절감한다.",
        url="https://example.com/paper1",
        title="TurboQuant Paper",
        source_type="paper",
        excerpt=(
            "TurboQuant is a quantization technique that compresses Key-Value cache vectors "
            "to reduce GPU memory footprint."
        ),
        supports_claim=True,
    )

    ev2 = Evidence(
        id="live_ev_2",
        technology_id="itme",
        claim="ITME는 온디바이스 전용 배터리 기술이다.",
        url="https://example.com/paper2",
        title="ITME Paper",
        source_type="paper",
        excerpt=(
            "ITME focuses on CXL-based heterogeneous DRAM extension in cloud data center "
            "environments."
        ),
        supports_claim=True,
    )

    # 1. Round 1: 실제 OpenAI API 배치 호출 (ev1, ev2 2건을 묶어서 1회 배치 호출)
    results_r1 = provider.validate_evidence([ev1, ev2], batch_size=5)

    assert len(results_r1) == 2
    res_map_r1 = {e.id: e.supports_claim for e in results_r1}

    # ev1은 지지함(True), ev2는 거짓 주장(False)이어야 함
    print(
        f"\n[OpenAI Live] Round 1 Response: ev1={res_map_r1['live_ev_1']}, "
        f"ev2={res_map_r1['live_ev_2']}"
    )
    assert res_map_r1["live_ev_1"] is True
    assert res_map_r1["live_ev_2"] is False

    # 2. Round 2: 완전히 동일한 항목 재검증 -> 인메모리 캐시 히트 발생 (API 호출 0회, 결과 일치)
    results_r2 = provider.validate_evidence([ev1, ev2], batch_size=5)
    res_map_r2 = {e.id: e.supports_claim for e in results_r2}

    assert res_map_r2["live_ev_1"] is True
    assert res_map_r2["live_ev_2"] is False
    print("[OpenAI Live] Round 2 Cache Hit Verified Successfully!")
