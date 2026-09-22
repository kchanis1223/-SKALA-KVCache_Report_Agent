# 검색 인계 문서 (김동찬 → 김강휘·윤소영)

`development-guide.md`의 "기술 조사 연결은 김강휘가 구현하고 김동찬이 검색을 제공합니다"에 따른
인계 문서입니다. 검색 쪽에서 준비한 것과, 붙이는 방법을 정리합니다.

## 준비된 것

| 항목 | 위치 |
| --- | --- |
| Retriever 구현체 | `retrieval/retriever.py` (`DenseRetriever`) |
| 색인 로딩 진입점 | `retrieval/factory.py` (`load_retriever` / `try_load_retriever`) |
| 색인 생성 명령 | `uv run skala-index` |
| 검색 확인 명령 | `uv run skala-search "질의"` |
| 성능 측정 | `uv run skala-eval-retrieval` — Hit@1 0.750 / Hit@3 0.900 / MRR 0.838 |
| 측정 근거 | `docs/issue-4-retrieval-evaluation.md` |

## 붙이는 방법

색인은 커밋하지 않으므로 먼저 만들어야 합니다.

```bash
uv sync --extra embedding
uv run skala-index          # data/raw/*.pdf -> index/bge-m3
```

실제 모드의 `build_provider()`는 이제 색인을 자동으로 읽어 기술 조사와 도메인 평가에
연결합니다(#35). `RAG_INDEX_DIR`로 경로를 변경할 수 있습니다. 기본 demo는 색인을
읽지 않습니다. 직접 주입하거나 자동 연결을 끄려면 다음처럼 지정합니다.

```python
from skala_agent.adapters import build_provider
from skala_agent.retrieval.factory import try_load_retriever

# 색인이 없으면 None이 돌아오고, 기술 조사는 pending이며 도메인은 웹검색만 사용합니다.
provider = build_provider(".env", retriever=try_load_retriever())
```

## 검색 결과의 형태

`retrieve()`는 `RetrievalResult` 목록을 점수 내림차순으로 돌려줍니다.

```python
retriever = try_load_retriever()
for result in retriever.retrieve("CXL 기반 계층적 메모리 확장", top_k=3, role="primary"):
    chunk = result.chunk
    print(result.score, chunk.id, chunk.page, chunk.section, chunk.source_url)
```

필터는 세 가지입니다. `role="primary"`는 주 분석 대상 논문만(설계서 B-2의 기술 조사 조건),
`paper_id`는 특정 논문만, `section`은 특정 섹션만 봅니다.

## 청크 예제

```json
{
  "id": "turboquant-p10-1",
  "paper_id": "turboquant",
  "camp": "sw",
  "role": "primary",
  "section": "MSE Optimal TurboQuant",
  "page": 10,
  "source_url": "https://arxiv.org/abs/2504.19874",
  "text": "Algorithm 1 TurboQuantmse: optimized for MSE 1: input: dimension d and bit-width b ..."
}
```

```json
{
  "id": "itme-p4-1",
  "paper_id": "itme",
  "camp": "hw",
  "role": "primary",
  "section": "3.1 Architecture and Hardware Design",
  "page": 4,
  "source_url": "https://arxiv.org/abs/2606.12556",
  "text": "CXL IP Cache Controller Meta Data Gen5 x8 Prefetch Buffer .mem .io Memory Controller ..."
}
```

`Evidence`를 만들 때 `chunk_id`에 청크 id를, `page`에 `chunk.page`를, `url`에
`chunk.source_url`을 넣으면 논문 근거가 원문 페이지까지 역추적됩니다.

## 한국어 질의 → 원문 청크 검색 예제

색인 69청크, 실측 결과입니다. 질의는 한국어, 청크는 영문입니다.

| 질의 | 1위 청크 | 점수 | 섹션 |
| --- | --- | --- | --- |
| KV 캐시 양자화가 정확도에 미치는 영향 | `turboquant-p20-1` | 0.571 | Near Neighbour Search Experiments |
| CXL 기반 계층적 메모리 확장 | `itme-p4-1` | 0.648 | 3.1 Architecture and Hardware Design |
| 실험 환경과 측정 조건 | `turboquant_analysis-p11-1` | 0.522 | Setup |

## 알아둘 제약

**참고문헌 청크를 하드 필터로 제외하지 마세요.** `section == "References"`로 거르면
Hit@3가 0.900에서 0.850으로 떨어집니다. 그림과 참고문헌이 섞인 페이지가 있어
라벨이 페이지 전체에 붙기 때문입니다. 근거는 `docs/issue-4-retrieval-evaluation.md`에 있습니다.

**고유 약어 구분은 약합니다.** `KV` / `KQV` / `QKQV`처럼 한두 글자만 다른 약어는
dense 임베딩이 구분하지 못해 검색에 실패합니다. 이런 질의는 웹검색이나 본문 직접 확인으로
보완하세요. 설계서 B-4가 언급한 sparse 신호(하이브리드 검색)는 아직 미구현입니다.

**색인이 없으면 조용히 실패하지 않게 하세요.** `load_retriever`는 색인이 없으면
`IndexNotBuiltError`를 던지고, `try_load_retriever`는 `None`을 돌려줍니다.
후자를 쓸 때는 RAG 없이 돌았다는 사실이 결과에 남도록 해주세요.

## 연결 검증 (#35)

- 기본 factory/runtime 및 `skala-evaluate`에서 자동 로딩하는 경로를 회귀 테스트합니다.
- 합성 색인을 사용하는 전체 workflow 테스트는 기술별 primary 필터, domain RAG,
  검증된 논문 원문·페이지·chunk_id의 보고서 출력을 검사합니다.
- 로컬 원문 PDF로 BGE-M3 색인 69청크/1024차원을 생성하고, Qwen3 4B로 두 기술의
  assessed TechAnalysis와 논문 근거 12건을 추출했습니다. 별도 Qwen3 8B 지지 검증은
  이 중 2건을 통과시켰습니다. 전체 기술 요약은 보류하고 검증된 논문 관측만 보고서에
  원문·페이지·chunk_id와 함께 출력되는 것을 확인했습니다.
- 이 실측은 인용 연결 검증이며 기술 주장 자체의 정확도를 보증하지 않습니다.
  실제 4관점 평가의 모델 출력 안정화는 #34에서 RAG 연결 상태로 이어서 검증합니다.
