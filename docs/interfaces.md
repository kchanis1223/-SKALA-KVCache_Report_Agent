# State / Agent I/O 계약 v1

[이슈 #1](https://github.com/kchanis1223/-SKALA-KVCache_Report_Agent/issues/1)의 김강휘·윤소영·김동찬 의견을 통합한 구현 계약입니다. 댓글에 선택지로 남아 있던 부분은 아래 기준으로 정리했습니다. 코드의 기준은 `src/skala_agent/schemas.py`이며, 독립 개발용 예제는 `tests/fixtures/contracts.json`입니다. fixture의 판정·출처는 모두 테스트용입니다.

## 댓글 반영 결정

| 안건 | 구현 계약 |
| --- | --- |
| verdict / reason / confidence / evidence 공통 형식 | `Assessment` + 중앙 `Evidence` 목록. `reason`은 `rationale`, `evidence_text`는 `excerpt`의 입력 alias로 허용. 저장·출력은 기존 canonical 이름 사용 |
| 관점별 State 분리 | `analyses["trl" / "market" / "stakeholder" / "domain"]` 유지. 각 값은 두 기술의 `list[Assessment]` |
| 같은 Evidence ID 재수집 | 새 ID를 만들지 않고 같은 ID의 최신 값으로 교체. 반대 주장은 다른 ID |
| 한 관점 실패 | `status="failed"` + `AgentError`. 나머지 관점은 보존하고 실패는 보고서 한계점에 명시 |
| 추가 검색 정보 | `MissingEvidence.kind`, `claim`, `queries`, `evidence_ids`, `retryable` 추가 |
| 기술 조사 출력 | 기술 ID → `TechAnalysis` 구조화 결과 |
| Chunk ID | `{paper_id}-p{page}-{sequence}`, page와 페이지 내 sequence는 1부터 |
| 청킹 설정 | 초기 v1 기준: BGE-M3 tokenizer, 1,500 tokens, overlap 200. 페이지 경계를 넘지 않음 |
| 검색 점수 | `RetrievalResult(chunk, score, score_type, dense_score?, sparse_score?)`. Chunk 자체에는 점수를 저장하지 않음 |
| 논문 인용 역추적 | `Evidence.chunk_id` 추가. 웹 근거는 null |

청킹 수치는 댓글에 명시되지 않아 이번 통합의 초기 기본값으로 정했습니다. `retrieval/config.py`의 `DEFAULT_CHUNKING`을 사용하며, 파서·tokenizer revision·전처리·청킹 설정 변경 시 버전, 인덱스, 정답 평가셋을 함께 갱신합니다. 원문 PDF와 tokenizer revision은 실제 수집 시 인덱스 manifest에 기록합니다.

## State 읽기·쓰기 책임

| key | 타입 | 생성 / 수정 주체 | 갱신 규칙 |
| --- | --- | --- | --- |
| selected_technologies | list[Technology] | 입력 / CLI | 실행 중 고정 |
| domain | str | 입력 / CLI | 실행 중 고정 |
| tech_analysis | dict[str, TechAnalysis] | research | 기술별 조사 결과 |
| analyses | dict[str, list[Assessment]] | evaluate | 관점별 분리 쓰기, 같은 관점의 재실행은 교체 |
| evidence | list[Evidence] | research / evaluate / additional_search; 향후 의미 검증자 | ID 기준 upsert |
| synthesis | list[Assessment] | synthesize / validate | 취합 후 confidence 정규화 |
| synthesis_findings | list[SynthesisFinding] | synthesize | 설계서 4-7의 다섯 질문으로 탐지한 근거 연결 상충·trade-off |
| missing_evidence | list[MissingEvidence] | validate | 검증 회차마다 전체 교체 |
| retry_count | int | additional_search | 실제 추가 검색 회차마다 +1, 최대 2 |
| report | str | report | 최종 Markdown |

모든 노드는 공유 State를 직접 수정하지 않고 변경분만 반환합니다. 병렬 평가 노드는 자신의 관점만 `analyses`에 쓰며 다른 관점의 기존 값은 반환하지 않습니다.

## Agent / Provider I/O

| 단계 | 읽는 입력 | 출력 / 계약 |
| --- | --- | --- |
| 기술 조사 | selected_technologies | `research(technologies) -> (dict[str, TechAnalysis], list[Evidence])` |
| 관점별 평가 | perspective, technologies, domain, tech_analysis, evidence | `assess(...) -> (list[Assessment], list[Evidence])` |
| 종합 | analyses, evidence | `synthesis`와 `synthesis_findings`. 외부 검색 없음 |
| 검증 | synthesis, evidence | provider의 `validate_evidence(evidence) -> list[Evidence]`로 지지 여부를 갱신한 뒤 정규화한 synthesis와 missing_evidence 반환 |
| 추가 검색 | retryable인 missing_evidence | `search_missing(missing) -> list[Evidence]`, graph에서 retry_count 갱신 |
| 보고서 | synthesis, evidence, missing_evidence, 입력 기술·도메인, retry_count | report. 외부 검색 없음 |

각 `assess()` 호출은 선택된 기술마다 정확히 하나의 결과를 반환해야 합니다. 기술 ID와 관점이 다르거나 빠지면 계약 오류입니다. `research()`도 선택된 기술 전체를 키로 반환하며, 키와 `TechAnalysis.technology_id`가 일치해야 합니다. Provider 경계에서 Pydantic으로 dict를 검증할 수 있지만 반환 계약은 해당 모델 기준입니다.

`TechAnalysis`는 overview, scope, limitations, experiments(측정값·실험 조건 서술), evidence_ids, status를 갖습니다. 페이지·URL·원문은 연결된 Evidence에 보관합니다. 문자열 하나만 반환하던 provider는 이 모델로 변경해야 합니다.

## 평가 출력

- `Assessment`: technology_id, perspective, verdict, rationale, confidence, signals, evidence_ids, status, error, details.
- `signals`: 질문별 question, grade(상/중/하), evidence_ids.
- `status`: pending(미구현/근거 부족), assessed(평가 완료), failed(실행 실패).
- `failed`에는 error가 필수이고 다른 상태에는 error를 넣지 않습니다. failed는 verdict를 판단 보류, confidence를 low로 정규화합니다.
- `details`는 perspective로 구분하는 union입니다. pending 및 기존 provider 호환을 위해 null을 허용하며, 실제 평가 provider는 해당 관점의 세부 결과를 채웁니다.

| details 타입 | 필드 |
| --- | --- |
| TRLDetails | level: 1~9 또는 null |
| MarketDetails | demand, adoption, ecosystem: 설계서 축별 등급 또는 null; dependency_risks |
| StakeholderDetails | positions: gpu_vendor / memory_vendor / cloud_operator / open_source / investor별 stance·rationale·evidence_ids; overall |
| DomainDetails | cost, sla_risk, operations: 설계서 축별 등급 또는 null; operational_risks |

미확인 축은 null, 이해관계자 자료 없음은 명시적인 stance로 표현합니다. details의 perspective는 상위 Assessment와 일치해야 합니다. 축별 질문·집계는 #5/#6 평가 구현에, 관점별 완성도·5개 주체·근거 참조 검증은 #7의 EvaluationOutput에 적용되어 있습니다. 검증된 고유 URL이 2개 미만인 결과는 검증 단계에서 confidence=low가 됩니다.

## Evidence ID와 검증 책임

Evidence 발급 주체는 근거를 생성하는 research / 각 평가 provider입니다. 공통 `evidence.evidence_id()`에 owner(research 또는 관점), technology_id, claim, 정규화 URL, chunk_id를 넘깁니다. 동일 입력은 같은 SHA-256 기반 ID를 반환하며 재시도 횟수·점수·supports_claim은 ID에 포함하지 않습니다. 기존 근거를 보완하는 search_missing은 전달받은 evidence_ids를 유지합니다.

- 한 owner만 같은 ID를 갱신합니다. 관점 간 동일 원문을 사용해도 owner namespace가 달라 병렬 쓰기가 섞이지 않습니다.
- ID가 같으면 excerpt·confidence·supports_claim 등의 최신 값으로 교체합니다. 지지가 철회된 False도 이전 True를 대체합니다.
- 같은 ID에서 기술·주장·URL·chunk_id를 바꾸면 충돌 오류입니다. 새로운 주장 또는 출처에는 새 ID를 발급합니다.
- `supports_claim`은 기본 False. 출처 수집 담당자는 검색 성공만으로 True로 설정하지 않습니다. 검증 provider는 원문 발췌와 주장을 비교해 지지·중립성을 판정하고 동일 ID Evidence 업데이트를 반환합니다.
- validation은 provider 업데이트와 Assessment·Signal 참조 관계를 함께 검사합니다. provider가 없거나 demo 모드이면 기존 플래그만 검사합니다.
- 보고서는 최종 Assessment가 참조하고 supports_claim=True인 해당 기술의 근거만 인용합니다. URL dedup은 독립된 출처라는 보장은 아닙니다.

## 종합 결과

`SynthesisFinding`은 `technology_id`, 설계서 4-7 질문(`quality_stability`, `resource_cost`, `operational_complexity`, `maturity_adoption`, `condition_limited`), 요약, `assessment_refs`, `evidence_ids`를 갖습니다. 근거가 연결되지 않은 상충 후보는 결과에 넣지 않습니다. 현재 rule 기반 탐지는 Assessment에 명시된 조건·trade-off와 구조화된 TRL/시장성 세부 결과만 사용하며, 원문 의미 지지 판정은 Evidence를 만든 검증 컴포넌트가 `supports_claim`으로 반영합니다.

## RAG 반환과 점수

`VectorStore.search()`와 `Retriever.retrieve()` 모두 `list[RetrievalResult]`를 반환합니다. 둘 다 top_k, role, paper_id 필터를 지원합니다. 기술 조사는 role=primary, 도메인은 role 제한 없이 독립 검토 자료를 포함합니다.

`score`는 높을수록 관련성이 높고 NaN/무한대는 허용하지 않습니다. score_type은 dense/sparse/hybrid/rrf이며 서로 다른 방식의 점수를 같은 threshold로 비교하지 않습니다. 벡터 DB가 distance를 주면 adapter에서 similarity 방향으로 변환해야 합니다. hybrid 방식의 정규화·가중치는 인덱스/실험 설정에 기록합니다. dense_score와 sparse_score로 융합 실험을 역추적할 수 있습니다.

김동찬의 검색 평가에서는 `result.chunk.id`를 순위대로 추출하여 기존 retrieval_metrics에 전달합니다. Chunk 필드는 id, text, paper_id, camp, role, section, page, source_url입니다. score는 질의마다 달라지므로 Chunk에 넣지 않습니다.

## 부족 근거와 실패 / 재실행

`MissingEvidence` 필수 필드: technology_id, perspective, reason, kind, claim, queries(최소 한 개). 선택 필드: evidence_ids, retryable(기본 True).

| kind | 의미 |
| --- | --- |
| not_evaluated | 아직 평가되지 않음 |
| missing_source | 평가했지만 출처가 없음 |
| unsupported_claim | 참조 ID가 없거나 다른 기술이거나 지지가 확인되지 않음 |
| agent_failed | 평가 서비스 호출 실패 |

validator는 signals의 조사 질문을 우선 사용해 기술·관점이 포함된 검색 질의를 만듭니다. 질문이 없으면 판정 또는 미완료 이유를 사용합니다. 실제 도메인별 질의 정교화는 검증/검색 Agent에서 확장합니다.

평가 provider의 TimeoutError / ConnectionError는 해당 관점의 두 기술을 failed로 변환합니다. 실제 SDK adapter는 이에 해당하는 오류를 표준 예외로 변환하거나 명시적 failed Assessment를 반환해야 합니다. raw exception 메시지는 보고서에 넣지 않습니다. `AgentError.retryable=False`이면 해당 부족 항목은 재검색하지 않습니다. 같은 관점에 retryable 항목이 하나라도 있으면 두 기술을 함께 재평가합니다.

재검색은 최대 2회입니다. 성공한 관점은 보존하고, 미해결·실패 관점은 판단 보류 및 6장 한계점에 표시합니다. malformed output / ValueError 등 계약·프로그래밍 오류는 숨기지 않고 실행을 중단합니다. 공통 기술 조사·추가 검색 자체의 서비스 오류 처리, 네트워크 backoff, checkpoint는 #10의 후속 범위입니다.

## 공동 fixture와 검증

`tests/fixtures/contracts.json`에 chunk, 점수 포함 검색 결과, evidence, tech_analysis, 정상 Assessment, pending Assessment, failed Assessment, MissingEvidence가 있습니다. 모두 실제 논문과 무관한 합성 데이터입니다. `make test`는 JSON round-trip, ID 중복 교체·철회, 관점별 실패·복구, 검색 점수 및 청킹 설정을 검사합니다.

기존 provider 수정 사항: tech_analysis 문자열 → TechAnalysis, 검색 Chunk 목록 → RetrievalResult 목록, MissingEvidence 생성 시 kind/claim/queries 지정. 기존 rationale/excerpt 이름은 유지됩니다. 상세 schema와 설명은 이 문서를 기준으로 개발하고, 과거 설계서의 operator.add 및 별도 *_analysis 키는 참고 이력으로만 봅니다.

## 공통 모델 배정

`ModelRouter.for_agent(name)`을 사용합니다. research/additional_search는 4B, trl/market/stakeholder/domain/validation/synthesis/report는 8B입니다. `USE_SINGLE_MODEL=true`이면 모든 역할이 같은 4B 객체를 사용합니다. 이슈 #5의 `adapters.build_provider()`가 real runtime에 연결되며 다른 Agent는 후속 구현에서 이 배정 API를 사용합니다. 모델 설정·실행 방법은 [실행 안내](issue-5-evaluation.md)를 참고하세요.

## 이해관계자·도메인 구현 (#6)

`EvaluationProvider(..., retriever=None)` 및 `build_provider(retriever=...)`로 도메인 논문 검색을 주입합니다. 도메인은 role/paper_id 제한 없이 검색하며 primary와 독립 reference를 포함합니다. 미연결 시 결과 rationale에 명시합니다. 이해관계자는 웹검색을 사용합니다. 공유 State와 Assessment/Evidence schema는 그대로 유지하며, 수치·실험 조건은 rationale과 연결된 Evidence 원문·페이지·chunk_id로 보존합니다. 자세한 집계 및 검증 범위는 [#6 실행 안내](issue-6-evaluation.md)를 참고하세요.

## 공통 평가 경계 검증 (#7)

`evaluation_contracts.EvaluationOutput`이 평가와 Evidence를 함께 검사합니다. 실제 EvaluationProvider는 선택 기술·관점 일치, 참조 존재·기술 일치, 세부 근거 포함 관계, assessed 축 완성도, 5개 주체와 집계 일관성을 검증하고 고유 참조 URL이 2개 미만이면 confidence를 low로 제한합니다. pending도 low입니다. 평가 관련 모델의 알 수 없는 필드와 공백만 있는 필수 문자열은 거부합니다. 기존 tuple 반환·State·입력 alias는 유지합니다. [상세 계약 및 예제](issue-7-evaluation-schema.md)를 참고하세요.
