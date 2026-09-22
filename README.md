# SKALA Agent — KV Cache 기술 비교

TurboQuant(SW)와 ITME(HW)를 데이터센터·클라우드 서빙 관점에서 비교하는 Agentic RAG 프로젝트의 협업용 뼈대입니다.

**기본 실행은 API 키 없는 개발용 workflow입니다.** TRL·시장성·이해관계자·도메인은 Ollama Qwen3 + Tavily provider를 별도로 사용할 수 있습니다. PDF 검색, BGE-M3, VectorDB와 최종 의미 검증은 구현 대기입니다. 기본 provider는 결과를 만들어내지 않고 `판단 보류`를 반환합니다. 설계서의 기술 주장·논문 ID·성능 수치는 검증된 사실로 사용하지 않습니다.

## 빠른 시작

Python 3.12와 [uv](https://docs.astral.sh/uv/)를 사용합니다. `uv`가 관리할 Python 버전은 `.python-version`에 지정했습니다.

```bash
make setup
make run
make test
make lint
```

결과: `outputs/report.md` (Git 제외). 기본 실행은 외부 API를 호출하거나 임베딩 모델을 내려받지 않습니다. 최초 설치에는 인터넷 연결이 필요합니다.

```bash
uv run skala-agent --output outputs/my-report.md
```

## Ollama 4B / 8B 실행

기술 조사·추가 검색은 **qwen3:4b**, 평가·검증·종합·보고서는 **qwen3:8b**로 배정합니다. `.env`에 `USE_SINGLE_MODEL=true`를 설정하면 전 Agent를 **qwen3:4b 하나**로 배정합니다.

```bash
# macOS. Ollama 서버는 별도 터미널에서 실행해 둡니다.
brew install ollama
ollama serve
```

새 터미널에서:

```bash
ollama pull qwen3:4b
ollama pull qwen3:8b  # 저사양 단일 모델 모드는 생략

git clone https://github.com/kchanis1223/-SKALA-KVCache_Report_Agent.git
cd -- -SKALA-KVCache_Report_Agent
# PR 검토 중: main 병합 후에는 생략
git switch feat/issue-6-stakeholder-domain
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# .env.local에 TAVILY_API_KEY를 입력합니다 (Git 제외).
python app.py                        # 기본 demo: 모델 호출 없음
python app.py --mode real --timeout 600  # Ollama 연결
skala-evaluate --perspective all      # 네 관점 모두 평가
```

## 구성

```text
src/skala_agent/
  schemas.py              # Technology / Evidence / Assessment / Chunk 공통 계약
  providers.py            # 실제 검색·평가 연동 계약 + DemoProvider
  agents/                 # 기술 조사, 관점별 평가, 종합, 검증, 재검색, 보고서
  workflow/               # LangGraph 및 reducer가 있는 공통 State
  retrieval/              # PDFParser / Embedder / VectorStore / Retriever 계약
  evaluation/             # Hit@1, Hit@3, MRR 계산
  prompts/                # 관점별 프롬프트 초안
configs/                  # 문서 후보 목록
data/                     # raw PDF / processed 청크·벡터 (Git 제외)
docs/                     # 설계 참고 문서 및 인터페이스 설명
tests/                    # workflow, retry, 검증, 검색 지표 테스트
```

## Workflow

```mermaid
flowchart TD
    START --> research[기술 조사 · primary 논문 RAG]
    research --> dispatch{관점별 병렬 실행}
    dispatch --> trl[TRL · 웹]
    dispatch --> market[시장성 · 웹]
    dispatch --> stakeholder[이해관계자 · 웹]
    dispatch --> domain[도메인 · RAG + 웹]
    trl --> synthesis[종합]
    market --> synthesis
    stakeholder --> synthesis
    domain --> synthesis
    synthesis --> validate{근거 검증}
    validate -->|근거 충분 또는 재시도 2회| report[보고서]
    validate -->|근거 부족| search[부족 근거만 추가 검색]
    search -->|해당 관점만 재평가| dispatch
    report --> END
```

실제 구현은 LangGraph `Send`로 같은 평가 노드를 관점별로 병렬 실행하고 결과를 합칩니다. retry에서는 부족한 관점만 선택합니다. [공식 Send 문서](https://reference.langchain.com/python/langgraph/types/Send)를 참고했습니다.

- `analyses`는 관점 이름을 키로 병합하므로 병렬 쓰기가 충돌하지 않습니다.
- `evidence`는 `merge_evidence`로 ID별 갱신합니다. 같은 ID의 과거 버전은 교체합니다. 출처 수는 고유 URL 기준입니다.
- 같은 관점의 retry는 이전 판정을 교체합니다. 정상 관점의 결과는 유지합니다.
- 검증된 출처가 2개 미만이면 `confidence=low`입니다.
- 평가 서비스의 타임아웃·연결 실패는 `failed`로 남기고 정상 관점의 결과를 보존합니다.
- 근거가 없는 판정은 재시도 후에도 보고서에서 제외하고 판단 보류로 표시합니다.
- 실제 provider는 원문 발췌와 주장을 비교해 `supports_claim`을 갱신하고, 검증은 출처 연결·기술 ID·Signal 참조 관계를 함께 확인합니다. demo 모드는 외부 모델을 호출하지 않으므로 모든 근거를 미검증으로 유지합니다.

## Contributors

| 담당자 | 책임 | 시작할 파일/폴더 |
| --- | --- | --- |
| 김동찬 | PDF Parsing 및 문서 전처리, Chunking / Metadata 설계, BGE-M3 Embedding 적용, VectorDB 및 Retrieval Pipeline 구현, Retrieval 성능 평가 (Hit@1, Hit@3, MRR) | `retrieval/`, `evaluation/`, `schemas.py`의 Chunk |
| 김강휘 | TRL·시장성·이해관계자·도메인 평가 Agent, Agent별 Prompt 및 Structured Output 설계 | `agents/{trl,market,stakeholder,domain}.py`, `prompts/`, `providers.py` |
| 윤소영 | LangGraph Workflow, State Schema, Fan-out / Fan-in, Conditional Branch 및 Retry Loop, Agent 간 데이터 흐름 통합 | `workflow/`, `schemas.py`, `providers.py` |
| 이준형 | Synthesis Agent, Evidence Validation, Report Generation Agent, Citation / Reference, End-to-End 테스트 및 README | `agents/{synthesis,validation,report}.py`, `tests/`, `README.md` |

기술 조사 Agent는 김동찬의 Retrieval과 김강휘의 추출 프롬프트를 연결하고, 추가 검색 Agent는 김강휘의 검색 구현과 윤소영의 retry 흐름을 연결하는 공동 통합 지점입니다.

상세 작업 순서·인수인계 산출물·완료 기준은 [담당별 개발 안내](docs/development-guide.md)를 참고하세요.

## 다음 구현 순서

1. 김동찬: 문서 후보 원문 확인 → PDF 파싱 → 청크 metadata → BGE-M3 → VectorDB → 검색 평가셋.
2. 김강휘: 확정된 관점별 details schema 적용 → 프롬프트 구현 → 실제 provider의 `research`, `assess`, `search_missing` 구현.
3. 윤소영: provider를 `build_graph(provider)`에 주입하고 실제 데이터 통합. 필요하면 checkpoint·실행 로그·오류 처리 추가.
4. 이준형: 설계서 4-7 상충 탐지, 문장별 근거 검증·중립성 검사, 참고문헌 서식과 보고서 본문 구현.

`prompts/trl.md`와 `market.md`는 실제 평가 provider가 읽습니다. 나머지 프롬프트는 참고 초안이며 demo는 읽지 않습니다. 실제 provider에서 읽고 structured output schema와 함께 적용해야 합니다. 이슈 #5는 Ollama Qwen3·Tavily를 사용하며 model/search 객체는 교체할 수 있습니다. VectorDB 제품은 미정입니다.

## Git 협업

```bash
git clone https://github.com/kchanis1223/-SKALA-KVCache_Report_Agent.git
cd -- -SKALA-KVCache_Report_Agent
git switch main
make setup
git switch -c feat/담당기능
# 구현 및 검증 후
git add <변경한-파일>
git commit -m "기능: 변경 내용을 한글로 작성"
git push -u origin feat/담당기능
```

GitHub에서 `main` 대상으로 PR을 만듭니다. 기존 개인 브랜치가 있으면 최신 `origin/main`을 병합한 뒤 작업하세요.

`.env`, PDF, 벡터 인덱스, 출력 보고서는 제외됩니다. `uv.lock`은 커밋해서 동일한 의존성을 사용합니다. GitHub Actions에서 lint·테스트·demo 실행을 확인합니다.

개발 규칙: [CONTRIBUTING.md](CONTRIBUTING.md) · 계약: [docs/interfaces.md](docs/interfaces.md) · 제공 설계서: [docs/design-v1.1.md](docs/design-v1.1.md)

공통 계약 v1과 정상·미완료·실패 예제는 [인터페이스 문서](docs/interfaces.md)와 `tests/fixtures/contracts.json`을 기준으로 합니다.
