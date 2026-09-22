# 담당별 개발 안내

## 먼저 모두 같이 할 일

현재 뼈대는 그래프 실행과 데이터 전달을 확인하는 개발용 코드입니다. 실제 검색·LLM 평가·상충 탐지·보고서 본문은 각 담당자가 구현해야 합니다.

```bash
git clone https://github.com/kchanis1223/-SKALA-KVCache_Report_Agent.git
cd -- -SKALA-KVCache_Report_Agent
git switch main
make setup
make run
make test
```

기본 브랜치인 `main`에서 최신 코드를 받은 뒤 기능 브랜치를 만들어 시작합니다. 이미 clone했다면 작업을 커밋한 뒤 `git switch main`, `git pull --ff-only origin main`을 실행합니다.

먼저 `schemas.py`, `providers.py`, `workflow/state.py`, `docs/interfaces.md`를 함께 읽습니다. 아래 파일 경로는 별도 표기가 없으면 `src/skala_agent/` 기준입니다.

공통 데이터 계약은 [interfaces.md](interfaces.md)에 정리했습니다. 서비스·구현 설정은 다음 담당자가 결정하고 PR에 남깁니다.

- 김동찬: VectorDB 제품, 확정 청킹 기준 적용, 원문 확보·보관 방법.
- 김강휘: LLM·웹검색 서비스, 확정 출력 schema 적용, 실행당 비용 한도.
- 윤소영: 실행 설정과 provider 주입 방식, 타임아웃·실패 처리.
- 이준형: 근거 검증 통과 조건, 인용 형식, 최종 보고서 형식.

공유 schema 변경은 관련 담당자 검토 후 작은 PR로 먼저 합칩니다. 기존 `src/agents/`, `src/rag/`는 원격에 있던 빈 자리 표시용 폴더입니다. 새 구현은 `src/skala_agent/` 안에 추가합니다. 기존 DOCX와 `docs/design.md`는 보존했으며 새 실행 방법은 README를 기준으로 합니다.

## 김동찬 — 검색 가능한 근거 만들기

**시작 파일:** `retrieval/interfaces.py`, `schemas.py`의 `Chunk`, `evaluation/retrieval.py`, 저장소 루트의 `configs/documents.json`.

1. 후보 논문의 제목·저자·원문 URL과 실제 페이지 수를 확인합니다. PDF는 `data/raw/`에 둡니다.
2. `PDFParser` 구현체를 추가해 페이지 번호와 텍스트를 추출합니다. 표·수식·OCR 한계를 기록합니다.
3. 토큰 단위 청킹과 metadata를 구현합니다. `paper_id`, `camp`, `role`, `section`, `page`, `source_url`을 유지하고 안정적인 chunk ID를 만듭니다.
4. `Embedder`에 BGE-M3를 연결하고 `VectorStore`의 적재·검색 구현을 추가합니다. 모델은 필요할 때 로딩합니다.
5. `Retriever.retrieve()`를 구현합니다. 기술 조사는 `role=primary`, 도메인은 전체 문서 검색을 지원합니다.
6. 한국어 질의와 정답 chunk ID 평가셋으로 Hit@1, Hit@3, MRR을 측정합니다. 설계서의 기존 점수를 새 실측 결과처럼 복사하지 않습니다.

**넘겨줄 결과:** Retriever 구현체, 인덱스 생성 명령, 청크 예제, 평가셋·측정 결과와 재현 절차. 김강휘에게 한국어 질의 → 원문 청크 검색 예제를 제공합니다.

**완료 기준:** 원문에서 인덱스를 다시 만들 수 있고, 검색 결과에 페이지·출처가 남으며, primary 필터가 reference를 제외하는 테스트가 통과합니다.

**권장 첫 PR:** `feat/retrieval-parser` — PDF 파싱·Chunk 생성·metadata 검사까지.

## 김강휘 — 네 관점 평가와 외부 서비스 연결

**시작 파일:** `agents/trl.py`, `market.py`, `stakeholder.py`, `domain.py`, `prompts/*.md`, `providers.py`, `schemas.py`의 `Assessment`, `Signal`, `Evidence`.

1. 확정된 Assessment.details의 TRLDetails, MarketDetails, StakeholderDetails, DomainDetails를 적용합니다. 세부 평가 질문과 집계 로직을 구현합니다.
2. 설계서 4장의 판정 규칙을 프롬프트와 검증 코드에 반영합니다. 현재 프롬프트 파일은 자동 로딩되지 않으므로 실제 provider에서 읽어 적용합니다.
3. 실제 Provider 구현을 별도 모듈에 추가합니다. `research()`는 김동찬의 primary 검색으로 개요·한계·실험 조건을 추출합니다.
4. `assess()`에서 TRL·시장성·이해관계자는 웹검색, 도메인은 전체 논문 RAG와 웹검색을 사용합니다. 각 호출은 선택된 기술별 Assessment 하나씩과 새 Evidence를 반환합니다.
5. `search_missing()`은 전달받은 부족 항목만 추가 검색합니다. 근거 ID와 원문 발췌를 유지합니다.
6. 구조화 출력 오류·자료 없음·상충 자료를 테스트합니다. `supports_claim`은 검색 성공만으로 True로 설정하지 않고 이준형과 합의한 검증을 거칩니다.

**넘겨줄 결과:** Provider 구현체, provider 설정 방법, 기술별 네 관점 출력 예제, 네트워크 없이 실행할 테스트 fixture.

**완료 기준:** 두 기술 모두 schema를 만족하고 출처와 질문별 signals가 연결됩니다. 자료 없음은 판단 보류, 두 출처 미만은 low이며, 서로 다른 평가 축을 단일 우열로 뭉개지 않습니다.

**권장 첫 PR:** `feat/agents-schema` — 확정 schema를 사용하는 TRL 한 관점의 fixture 기반 평가부터.

## 윤소영 — 결과를 하나의 실행으로 연결하기

**시작 파일:** `workflow/graph.py`, `workflow/state.py`, `cli.py`, `tests/test_workflow.py`.

1. 기존 demo와 테스트로 fan-out/fan-in, 선택적 retry, 재시도 상한 2회를 확인합니다.
2. 김강휘의 provider를 `build_graph(provider)`에 주입합니다. CLI에 demo/실제 실행 모드와 설정을 추가하되 기본 demo는 API 키 없이 동작하게 유지합니다.
3. 공유 schema 변경을 State에 반영합니다. 병렬 노드는 변경분만 반환하고, `analyses`는 관점별 교체, `evidence`는 같은 ID의 최신 값으로 갱신하는 규칙을 유지합니다.
4. 추가 검색 결과를 부족 관점에 전달하고 정상 관점의 결과가 보존되는지 확인합니다.
5. 외부 서비스 오류·타임아웃과 실행 로그를 구현합니다. 필요한 경우 checkpoint를 추가합니다.
6. 실제 provider를 통합한 실행 경로와 mock 실행 경로를 분리해 CI에서는 비용이 발생하지 않도록 합니다.

**넘겨줄 결과:** 설정 가능한 CLI, 실제 provider가 연결된 workflow, 정상/재시도/실패 실행 예제와 통합 테스트.

**완료 기준:** 모든 관점의 완료 후 한 번 종합하고, 부족 관점만 재실행하며, 최대 2회 후 종료합니다. 한 관점의 실패가 조용히 정상 결과로 바뀌지 않습니다.

**권장 첫 PR:** `feat/workflow-provider` — 실제 provider 주입과 실행 모드 분리.

## 이준형 — 검증 가능한 보고서 만들기

**시작 파일:** `agents/synthesis.py`, `validation.py`, `report.py`, `tests/test_validation.py`, `tests/test_workflow.py`, 저장소 루트의 README.

1. 현재 synthesis의 단순 취합을 설계서 4-7의 다섯 질문 기반 상충 분석으로 확장합니다. 종합 결과 schema는 윤소영·김강휘와 합의합니다.
2. 주장을 원문 발췌와 대조하는 검증을 구현합니다. URL 존재 확인만으로 통과시키지 말고 출처의 신뢰도·주장 지지·중립성을 확인합니다.
3. 부족 근거는 기술 ID·관점·구체적인 부족 내용을 `MissingEvidence`로 반환해 재검색에 전달합니다.
4. Evidence의 주장별 관계와 `supports_claim`을 누가 언제 설정하는지 통합합니다. 현재 validation은 해당 플래그를 읽기만 하므로 실제 검증 결과를 State에 반영하는 경로가 필요합니다.
5. 설계서 6장 순서로 본문과 인용을 생성합니다. 실제 사용한 출처만 REFERENCE에 포함하고 미검증 주장은 결론에서 제외합니다.
6. 정상 종료, 부족 근거 보완, 재시도 소진, 잘못된 인용을 E2E 테스트하고 README를 실제 실행 방법으로 갱신합니다.

**넘겨줄 결과:** 종합·검증·보고서 구현, 근거를 추적할 수 있는 예시 보고서, E2E 테스트와 실행 문서.

**완료 기준:** 보고서의 판정을 원문까지 추적할 수 있고, 근거 없는 판정이 확정 결론으로 나오지 않습니다. 검증 실패 이유와 평가 한계가 드러납니다.

**권장 첫 PR:** `feat/report-evidence` — fixture 기반 인용 검증과 보고서 출력부터.

## 병렬 작업과 통합 순서

| 단계 | 함께 진행할 작업 | 통합 조건 |
| --- | --- | --- |
| 1. 계약 확정 | 세부 schema, 검증 책임, provider 설정 합의 | 작은 공통 PR 먼저 병합 |
| 2. 독립 개발 | 김동찬은 검색, 김강휘는 fixture 기반 평가, 윤소영은 가짜 provider로 그래프, 이준형은 fixture 기반 보고서 | 모두 같은 계약을 사용 |
| 3. 첫 연결 | Retriever → research → 관점 하나 → 검증 → 보고서 | 원문 인용이 있는 최소 실행 성공 |
| 4. 전체 연결 | 네 관점 병렬 평가와 선택적 재검색 | 정상·부족 근거·실패 테스트 통과 |
| 5. 평가·제출 | 검색 지표 실측, 실제 보고서 검토, 실행 문서 정리 | 새 clone에서 재현 가능 |

기술 조사 연결은 김강휘가 구현하고 김동찬이 검색을 제공합니다. 추가 검색의 서비스 연동은 김강휘, 분기·루프 연결은 윤소영이 담당합니다. Evidence 검증 규칙은 이준형이 정의하고 다른 담당자와 연결합니다.

## PR 보내는 방법

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/담당기능
# 구현
make format
make lint
make test
make run
git add <변경한-파일>
git commit -m "기능: 담당 기능 구현 내용을 한글로 작성"
git push -u origin feat/담당기능
```

GitHub에서 `main` 대상으로 PR을 만듭니다. 변경 목적·입출력 예제·검증 명령을 적고, 공유 계약 변경 시 연관 담당자에게 검토를 요청합니다. API 키, PDF, 벡터 인덱스와 생성 보고서는 커밋하지 않습니다.

이슈 #1 반영: 청킹 기본값은 `retrieval/config.py`, 안정적 Evidence ID는 `evidence.py`, 점수 반환은 `RetrievalResult`, 실패는 `Assessment.status=failed`를 사용합니다.
