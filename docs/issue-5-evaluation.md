# 이슈 #5 — Ollama TRL / 시장성 평가

## 공통 모델 정책

재현성을 고려하여 로컬 실행 가능한 Qwen3 계열을 사용한다. 정보 추출·검색에는 Qwen3-4B, 평가·종합·검증에는 Qwen3-8B를 적용하며, 저사양 환경에서는 Qwen3-4B 단일 모델로 전체 Workflow를 실행할 수 있는 fallback 구성을 제공한다.

| Agent key | 담당 기능 | 기본 모델 |
| --- | --- | --- |
| research | 기술 조사 | qwen3:4b |
| additional_search | 추가 검색 | qwen3:4b |
| trl / market | TRL·시장성 평가 | qwen3:8b |
| stakeholder / domain | 이해관계자·도메인 평가 | qwen3:8b |
| validation / synthesis / report | 검증·종합·보고서 | qwen3:8b |

`ModelSettings`와 `ModelRouter`가 전 Agent의 배정을 관리합니다. `USE_SINGLE_MODEL=true`이면 모두 `SINGLE_MODEL=qwen3:4b`로 바뀌며 8B 객체나 요청을 만들지 않습니다. 14B·32B 모델은 설정 검증에서 거부합니다. `.env`와 `.env.local`을 자동 로딩하며 환경변수 > `.env.local` > `.env` 순서로 우선합니다.

**실제 LLM 연결은 #5의 TRL·시장성과 #6의 이해관계자·도메인에 적용됩니다.** 기술 조사 RAG는 pending, 종합·검증·보고서는 기존 코드 기반 뼈대입니다. 추가 검색은 현재 전달된 질의로 자료를 수집합니다. 나머지 담당자는 `models.for_agent("research")` 같은 공통 배정 API를 자신의 구현에 연결하면 됩니다. 모델 정책 제공과 9개 Agent의 구현 완료를 구분합니다.

```python
from skala_agent.model_config import ModelRouter, ModelSettings, read_environment
from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.tavily import TavilySearch

env = read_environment()
models = ModelRouter(ModelSettings.from_environment(env))
tech_model = models.for_agent("research")
eval_model = models.for_agent("trl")
provider = EvaluationProvider(models=models, search=TavilySearch(env["TAVILY_API_KEY"]))
```

직접 주입하는 `EvaluationProvider(model=custom_model, search=search)`도 유지합니다. `invoke(messages)`가 문자열 또는 문자열 content를 반환하면 됩니다. 기본 Ollama 모델은 native `/api/chat`의 `format`에 JSON Schema를 전달하며 `think=false`를 사용합니다. Pydantic 검증·원문 인용 검사도 별도로 적용합니다.

공식 API: [Ollama Chat](https://docs.ollama.com/api/chat), [구조화 출력](https://docs.ollama.com/capabilities/structured-outputs).

## 다른 PC에서 시작하기

Python 3.12를 권장합니다. macOS 예시는 다음과 같습니다. 다른 OS는 [Ollama 다운로드](https://ollama.com/download)를 사용합니다.

```bash
brew install ollama
# 별도 터미널에서 실행해 둡니다. Ollama 앱/서비스가 이미 실행 중이면 생략합니다.
ollama serve
```

서버가 실행되면 새 터미널에서:

```bash
ollama pull qwen3:4b
ollama pull qwen3:8b

git clone https://github.com/kchanis1223/-SKALA-KVCache_Report_Agent.git
cd -- -SKALA-KVCache_Report_Agent
# PR 검토 중에만 사용. main 병합 후에는 아래 전환을 생략합니다.
git switch feat/issue-6-stakeholder-domain
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# .env.local에 TAVILY_API_KEY를 입력합니다 (Git 제외).

python app.py                         # 기본 demo: Ollama/API 키 없이 전체 흐름 확인
python app.py --mode real --timeout 600  # Ollama + Tavily로 전체 그래프 실행
skala-evaluate --perspective all       # 네 관점을 JSON으로 평가
```

저사양 환경에서는 `qwen3:4b`만 pull하고 `.env`의 `USE_SINGLE_MODEL=true`만 바꾸면 됩니다. 이때 real 그래프의 실제 LLM 요청도 4B만 사용합니다. 모델 호출은 직렬화하고 `keep_alive=0`으로 사용 후 언로드해 두 모델을 동시에 상주시킬 필요를 줄였습니다. 대신 반복 로딩 시간이 발생합니다. 4B도 실행 가능한 메모리가 필요하며 CPU/GPU와 컨텍스트에 따라 속도가 다릅니다.

기존 uv 사용자는 `uv sync --locked` 후 `uv run skala-evaluate --perspective all`을 사용하세요. torch/transformers 설치나 모델 파일의 Python 직접 로딩은 더 이상 필요하지 않습니다.

`requirements.txt`는 `uv.lock`의 고정 버전을 export한 파일입니다. 의존성 변경 시 아래 명령으로 같이 갱신합니다.

```bash
uv export --no-dev --no-hashes --format requirements-txt --output-file requirements.txt
```

재현 실험에는 `ollama --version`, `ollama list`의 모델 ID, `.env`의 비밀값을 제외한 설정과 입력 자료 버전을 함께 기록하세요. 태그가 같아도 향후 모델 파일이 갱신될 수 있습니다.

## API 키가 필요한 범위

모델 추론은 로컬 Ollama를 사용하므로 LLM API 키가 필요 없습니다. 실시간 웹검색은 기존 Tavily를 사용하므로 `.env.local`에 `TAVILY_API_KEY`를 입력해야 합니다. 추가 데이터 파일이나 별도 검색 모드는 사용하지 않습니다. 기본 demo는 API 키와 모델 서버 없이 실행됩니다.

## 평가 규칙과 결과

1. 기술별 고정 질의 3개를 검색하고 URL 중복 제거 후 최대 6개·문서당 800자 발췌를 사용합니다. 추가 검색 후보 최대 2건을 우선 포함합니다.
2. 모델은 TRL 9개 / 시장성 11개 질문의 yes·no·unknown, 이유, source_id와 원문 quote를 추출합니다.
3. 질문 누락·중복·허구 출처·원문에 없는 quote를 거부합니다. 인용 없는 yes/no는 unknown으로 바꿉니다. JSON 형식 재요청은 최대 한 번입니다.
4. 등급·Evidence ID·URL은 코드로 결정합니다. TRL 7~9는 1차 근거를 요구합니다. 시장성은 수요·채택·생태계를 분리하고 종속성은 역방향 지표로 제외합니다.

signals의 grade는 답의 근거 강도(상=1차 자료, 중=간접 자료, 하=인용 없음)입니다. question에 yes/no/unknown을 함께 저장하고 축 집계에는 yes의 근거 강도만 반영합니다. 전체 근거 부재와 근거 있는 부정적 관측은 구분합니다.

수요·채택: 상 2개 이상이고 하가 없으면 최상위, 하 2개 이상이면 하위, 나머지는 중간입니다. 해당 축의 근거가 전혀 없으면 null입니다. 생태계: 상 3개 이상 + 종속성 부재 확인이면 확보, 하 2개 이상 또는 치명적 종속성이면 생태계 부재, 나머지는 형성 중입니다. 미확인 종속성만으로 확보 판정을 내리지 않습니다. 시장성 축이 하나라도 null이면 pending으로 남깁니다. 출처 2개 미만 또는 보류는 confidence=low, 그 외 잠정 평가는 최대 medium입니다.

Tavily 출처는 기본적으로 news, arxiv.org는 paper, `OFFICIAL_SOURCE_DOMAINS`의 정확한 호스트만 official로 분류합니다. 허용 목록은 작성 주체를 직접 확인하고 설정하세요.

`skala-evaluate`는 `outputs/evaluations.json`에 실제 모델 배정·Assessment·Evidence를 저장합니다. `--env-file`, `--perspective`, `--output`으로 설정합니다. 실패 결과를 저장한 경우 exit code 1로 종료합니다. `app.py --mode real`은 공통 `runtime.load_provider`와 `adapters.build_provider()`를 통해 같은 모델 정책을 사용하고 보고서를 생성합니다.

**출력은 의미적 근거 검증 전 잠정 평가입니다.** 새 Evidence의 supports_claim은 False이며 기존 검증된 ID·발췌가 같을 때만 상태를 보존합니다. #11 검증자가 최종 지지를 판정하기 전에는 전체 그래프에서 판단 보류로 처리됩니다. 연구 조사 및 근거가 부족한 관점의 pending도 보고서에 유지됩니다.

## 검증 범위

테스트는 9개 Agent 모델 배정, 단일 4B 객체 재사용, `.env` 우선순위·boolean 파싱, Ollama JSON Schema 요청, 4B/8B 실제 HTTP payload, 불완전 응답 거부, Tavily 키 검증, provider factory 및 전체 그래프 연결을 확인합니다. HTTP는 mock이며 실제 Ollama 4B/8B 생성 품질과 실행시간은 별도 실측 대상입니다. Tavily는 `.env.local`의 키로 실시간 검색을 확인했습니다. 키는 Git에 포함하지 않습니다.
