# 이슈 #6 — 이해관계자·도메인 평가

`EvaluationProvider`의 stakeholder/domain 경로가 두 기술마다 `Assessment`와 `Evidence`를 반환합니다. 기존 LangGraph 노드가 같은 provider를 호출하므로 State나 공유 schema 변경 없이 연결됩니다. Ollama 기본 모델은 qwen3:8b, `USE_SINGLE_MODEL=true`이면 qwen3:4b입니다. `.env.local`의 Tavily 키를 자동 사용합니다.

```bash
uv run skala-evaluate --perspective stakeholder --output outputs/stakeholder.json
uv run skala-evaluate --perspective domain --output outputs/domain.json
uv run skala-evaluate --perspective all  # 두 기술 × 네 관점 = 8개 결과
```

## 이해관계자

GPU 벤더, 메모리 벤더, 클라우드 운영자, 오픈소스, 투자 업계마다 지지·회의·인지 여부를 조사합니다. 총 15개 질문을 구조화 추출합니다. 웹 질의도 주체별 5개이며, 질의별 결과를 번갈아 배치해 앞쪽 질의만 컨텍스트를 차지하지 않도록 합니다.

- 지지 또는 회의가 인용으로 확인되면 각각 `지지`, `회의적`입니다.
- 인지 등 입장 관련 자료는 있지만 명확한 방향이 없으면 `유보`입니다. 지지와 회의가 함께 있으면 유보로 남기고 rationale에 상충 관측을 보존합니다.
- 인용 가능한 입장 자료가 없으면 `자료 없음`입니다. 5개 주체를 모두 출력하고 자료 없음은 집계에서 제외합니다.
- 나머지 주체 중 지지/회의가 엄격한 과반이면 우호적/부정적, 그 외 혼재입니다. 자료가 전혀 없으면 overall=null, status=pending입니다.

## 데이터센터

비용·효율, SLA, 운영을 9개 질문으로 조사합니다. 원가·효율 개선은 수치와 실험 조건이 원문에 같이 있어야 정량 근거로 사용합니다. 추출용 `measurements`의 metric/value/conditions가 동일한 citation.quote에 포함되는지 검사하며, value에 숫자가 있어야 합니다. 원문에 없는 측정값·조건은 ModelOutputError로 처리합니다.

공유 Assessment schema는 바꾸지 않습니다. 수치·조건·source_id는 rationale에 저장하고, signals/evidence_ids가 인용 원문·URL·페이지·chunk_id로 연결합니다. 이는 원문 일치 검증이며, 측정의 타당성·비교 가능한 실험 조건·주장 지지의 의미적 검증은 #11의 책임입니다.

| 축 | 판정 |
| --- | --- |
| 비용 | 정량 개선 + 일반화 가능 명시: 원가 개선 명확. 제한 조건이 있거나 범위 미확인: 조건부 개선. 정성/부정 관측만 있음: 개선 불명확·low. 근거 없음: null |
| SLA | 지연·정확도 모두 확인되어야 낮음/중간/높음을 계산. 하나라도 미보고면 판단 불가·pending |
| 운영 | SW만 적용: 즉시 도입 가능. HW 변경: 인프라 전환 전제. 상충 시 인프라 전환을 택하고 위험 기록 |
| 별도 위험 | 멀티테넌시 격리, 노드 밖 장애 반경·운영 복잡도 증가 |

모든 축을 확인해야 domain status=assessed입니다. 부분 결과는 유지하되 pending으로 남깁니다. 고유 출처 URL이 2개 미만이면 confidence=low이며, 그 이상도 의미 검증 전에는 최대 medium입니다.

## 논문 RAG 연결

실제 VectorDB/Embedding 구현을 대신하지 않습니다. 김동찬 담당 `Retriever.retrieve(query, top_k=3, role=None, paper_id=None)` 계약을 주입합니다. role 제한 없이 primary와 독립 reference를 함께 검색하고, 반환 결과에 두 종류가 있으면 각 1건을 우선 포함합니다. 기술 이름을 포함한 도메인 질의 3개를 사용합니다. 제공 Retriever는 질의 관련성을 보장해야 합니다.

```python
from skala_agent.adapters import build_provider

# retriever는 retrieval 담당 구현 인스턴스입니다.
provider = build_provider(retriever=retriever)
# build_graph(provider) 또는 provider.assess(...)에 사용합니다.
```

웹과 논문에서 각 2건을 우선 확보하고 전체 최대 6건, 건당 800자를 전달합니다. 추가 검색 후보는 최대 2건을 우선 배치합니다. 같은 URL의 서로 다른 청크는 유지하며 Evidence ID에 chunk_id를 포함합니다. 페이지·섹션도 보존합니다. 이해관계자 평가는 Retriever를 호출하지 않습니다.

기본 CLI factory에는 실제 Retriever가 연결되어 있지 않아 웹/기존 Evidence만 사용하며 결과 rationale에 이를 명시합니다. RAG 주입 경로는 합성 fixture로 검증했습니다. 별도 file 모드는 없습니다.

## 검증과 한계

테스트는 자료 없음/유보/상충/과반, SLA 조합, 수치·조건 원문 검증, 두 기술 결과, 웹+독립 논문 검색, 동일 URL의 서로 다른 청크, 4B/8B HTTP 요청, 재검색과 전체 그래프, CLI 네 관점을 확인합니다. Ollama 실제 모델의 평가 품질과 실제 논문 인덱스 통합은 별도 검증이 필요합니다. 새 Evidence의 supports_claim은 False이며 미검증 판정을 최종 보고서에서 확정하지 않습니다.
