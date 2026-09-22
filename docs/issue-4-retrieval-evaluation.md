# 이슈 #4 — Retrieval 성능 평가

## 평가셋

`evals/retrieval_queries.json` — 한국어 질의 20개와 정답 chunk ID.

질의는 한국어, 문서는 영문입니다. 설계서가 bge-m3를 선정한 근거가 교차 언어 검색이므로
평가도 같은 조건으로 둡니다. 정답은 `index/bge-m3/chunks.json`의 본문을 직접 확인해
라벨링했고, 각 질의의 정답은 1개입니다.

| 문서 | 질의 수 | 대상 구간 |
| --- | --- | --- |
| turboquant | 8 | 알고리즘 의사코드, 왜곡 수치, LongBench 표, 재현율 비교, Related Work |
| turboquant_analysis | 6 | 스킴 정의, 분산 비율, 6D 오차 벡터, 실험 설정, 키 분포 |
| itme | 6 | 데이터 계층 표, 하드웨어 구성, 대역폭 표, pagemap 절차, 원격 매니저, FPGA |

정답이 chunk ID 기준이므로 **청킹 설정이 바뀌면 평가셋을 다시 라벨링해야 합니다.**
평가셋과 색인의 `chunking_version`이 다르면 측정이 중단됩니다.

## 실험 설정

| 항목 | 값 |
| --- | --- |
| 임베딩 모델 | BAAI/bge-m3 (dense only) |
| 벡터 차원 | 1024, float32, 정규화 |
| 유사도 | 코사인, 전수 비교 (근사 색인 없음) |
| 청킹 | v1 — bge-m3 tokenizer, 1500 tokens, overlap 200, 페이지 경계 불침범 |
| 색인 크기 | 69청크 (turboquant 25 / turboquant_analysis 23 / itme 21) |
| 순위 깊이 | top_k=10 (MRR 계산 기준) |

색인 생성 당시 설정은 `index/bge-m3/manifest.json`에 기록됩니다.

## 재현 명령

```bash
uv sync --extra embedding      # FlagEmbedding 포함
uv run skala-index             # 논문 PDF -> 청킹 -> 임베딩 -> index/bge-m3
uv run skala-eval-retrieval --output evals/results-bge-m3.json
```

PDF는 커밋하지 않습니다. `configs/documents.json`의 URL에서 `data/raw/{paper_id}.pdf`로
내려받으세요. 같은 판본인지는 `sha256`으로 확인합니다.

참고문헌 청크를 제외한 조건도 함께 측정할 수 있습니다.

```bash
uv run skala-eval-retrieval --exclude-references --output evals/results-no-refs.json
```

## 결과

색인 69청크, 질의 20개, top_k=10 기준입니다.

| 조건 | Hit@1 | Hit@3 | MRR |
| --- | --- | --- | --- |
| bge-m3, 전체 청크 | **0.750** | **0.900** | **0.838** |
| bge-m3, References 제외 | 0.750 | 0.850 | 0.812 |

20문항 중 17문항이 1위로 정답을 찾았습니다. 한국어 질의로 영문 청크를 찾는
교차 언어 검색이 의도대로 동작하며, 설계서 B-4의 bge-m3 선정 근거가 실측으로 확인됩니다.

### References 제외는 오히려 손해다

제외 조건에서 Hit@3가 0.900에서 0.850으로 떨어졌습니다. 원인은 참고문헌 판정의 오탐입니다.

`turboquant-p21-1`은 Recall@1@k 그래프가 있는 그림 페이지인데 같은 페이지 아래쪽에서
참고문헌이 시작됩니다. 인용 표기와 연도가 많아 통째로 References로 분류되었고,
q08("GloVe와 OpenAI 임베딩에서 PQ, RabitQ와 재현율 비교")의 정답이 1위에서 미검출로 바뀌었습니다.

청킹이 페이지 단위라 한 페이지 안에서 그림과 참고문헌을 분리할 수 없습니다.
**References 라벨은 메타데이터로 유지하되 검색 단계의 하드 필터로 쓰지 않습니다.**

### 미검출 질의

q10 "KV, KQV, QKQV 세 가지 양자화 스킴의 정의"는 두 조건 모두에서 top-10에 들지 못했습니다.
정답은 `turboquant_analysis-p5-1`이고 1위로 올라온 것은 같은 논문의 초록 페이지였습니다.

원인은 `KV`, `KQV`, `QKQV`가 서로 한두 글자만 다른 고유 약어라는 점입니다. dense 임베딩은
이런 약어를 비슷한 벡터로 밀어넣어 구분하지 못합니다. 설계서 B-4가 bge-m3를 고른 근거 중
하나가 "Dense + Sparse를 한 모델에서 제공"이었는데, **이 질의가 정확히 sparse 신호가
필요한 경우**입니다. 현재 구현은 dense만 사용합니다.

## 한계

설계서 B-4는 임베딩 후보 3종(bge-m3 / multilingual-e5-large / Qwen3-Embedding-0.6B)의
비교를 계획했습니다. 현재는 bge-m3만 측정했습니다. 후보 비교는 `--model` 인자로
같은 평가셋에 대해 실행할 수 있으나, 모델별 차원이 달라 색인을 따로 만들어야 합니다.

질의 20개는 소규모라 지표 한 건당 0.05씩 움직입니다. 절대값보다 조건 간 상대 비교로 읽으세요.
설계서에 적힌 수치는 이 실측으로 대체하지 않습니다.
