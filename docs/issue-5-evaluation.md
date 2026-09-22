# 이슈 #5 — TRL / 시장성 평가

## 모델 교체 방식

LangGraph는 workflow를 관리하고, 실제 추론은 주입된 model이 수행합니다. 현재 모델은 **로컬 Python에서 직접 실행하는 Qwen3**입니다. Ollama 서버와 OpenAI API/SDK는 사용하지 않습니다.

```python
from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.qwen import TransformersQwen
from skala_agent.integrations.tavily import TavilySearch

model = TransformersQwen(model_id="Qwen/Qwen3-4B")
# 더 작은 모델: model = TransformersQwen(model_id="Qwen/Qwen3-1.7B")
provider = EvaluationProvider(model=model, search=TavilySearch(api_key="..."))
```

`invoke(messages)`가 문자열 또는 문자열 content를 가진 메시지를 반환하면 다른 모델 객체도 그대로 주입할 수 있습니다. 평가 Agent의 입출력은 공통 Assessment/Evidence를 유지합니다. StructuredExtractor가 JSON Schema를 프롬프트에 포함하고 Pydantic으로 응답을 검사합니다. 서버의 강제 JSON decoding을 사용하지 않으므로 모델 출력 실패가 있을 수 있고, 형식 재요청은 최대 한 번입니다.

공식 사용 방식: [Qwen3-4B 모델 카드](https://huggingface.co/Qwen/Qwen3-4B), [Transformers chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating), [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search).

## 설치와 실행

```bash
uv sync --locked --extra local
export TAVILY_API_KEY='실제-검색-API-키'
uv run --extra local skala-evaluate --perspective trl
uv run --extra local skala-evaluate --perspective market
# 두 관점을 함께 평가
uv run --extra local skala-evaluate --perspective all
# 모델/장치 변경
uv run --extra local skala-evaluate --model Qwen/Qwen3-1.7B --device auto
```

최초 추론 때 Hugging Face에서 tokenizer와 모델 가중치를 다운로드합니다. 이후 캐시를 사용합니다. `auto`는 CUDA → MPS → CPU 순서로 선택합니다. GPU/MPS는 float16, CPU는 float32이므로 필요한 메모리와 속도가 다릅니다. 모델은 평가당 중복 로딩하지 않으며 병렬 Agent의 generate 호출은 한 번에 하나씩 실행합니다.

환경변수: `QWEN_MODEL`(기본 Qwen/Qwen3-4B), `QWEN_DEVICE`(auto), `QWEN_REVISION`(main), `TAVILY_API_KEY`, `OFFICIAL_SOURCE_DOMAINS`(쉼표로 구분한 정확한 호스트 목록). CLI는 `.env`를 자동으로 읽지 않습니다. `.env.example`을 참고하여 환경에 export하세요. 재현 실험에서는 `--revision`으로 모델 커밋을 고정하세요.

출력은 `outputs/evaluations.json`이며 모델·revision·기술별 Assessment·Evidence를 저장합니다. `--output`으로 경로를 지정할 수 있습니다. 실패 Assessment가 있으면 결과를 저장한 뒤 exit code 1로 종료합니다. **이 파일은 검증 전 잠정 평가이며 최종 보고서가 아닙니다.**

기본 demo `make run`은 그대로 API 키와 모델 다운로드 없이 실행됩니다. torch/transformers는 선택 의존성이며 실제 모델 호출 전에는 import하지 않습니다.

## 처리 흐름

1. 기술별 3개의 고정 조사 질의로 검색합니다. 각 질의는 최대 2개 결과를 요청합니다.
2. URL 중복을 제거해 최대 6개, 문서당 최대 800자의 검색 발췌를 사용합니다. 추가 검색 후보는 최대 2건을 우선 포함합니다.
3. Qwen3는 TRL 9개 / 시장성 11개 질문마다 yes·no·unknown, 이유, source_id와 원문 quote를 추출합니다. 긴 사고 출력은 `enable_thinking=False`로 끕니다.
4. 질문 누락·중복, 알 수 없는 출처, 발췌에 없는 quote를 거부합니다. 인용 없는 yes/no는 unknown으로 바꿉니다.
5. 코드가 근거 강도·축별 등급과 신뢰도를 계산하고 공통 schema로 반환합니다.

LLM에게 URL, Evidence ID, 최종 점수를 생성시키지 않습니다. 출처 ID와 Evidence ID는 코드에서 안정적으로 생성하고 동일 근거는 재실행해도 동일 ID를 유지합니다.

## 등급 계산

`signals`의 grade는 답을 뒷받침하는 근거 강도입니다. 상=paper/official/market_report, 중=news/community, 하=인용 없음입니다. 부정 답변도 강한 근거일 수 있으므로 question 필드에 응답 yes/no/unknown을 함께 기록합니다. 축 집계에는 yes의 근거 강도만 사용합니다.

- **TRL:** yes로 확인된 가장 높은 단계를 선택합니다. TRL 7~9는 1차 출처를 요구하며, 뉴스만으로 상용 단계를 확정하지 않습니다. 기준을 충족한 단계가 없으면 level=null 및 판단 보류입니다.
- **수요·채택:** 3개 질문 중 상 2개 이상이며 하가 없으면 최상위 등급, 하 2개 이상이면 하위 등급, 나머지는 중간 등급입니다. 축에 긍정·부정 근거가 모두 없으면 null로 보류합니다.
- **생태계:** 4개 질문 중 상 3개 이상이고 종속성 부재가 확인되면 확보. 하 2개 이상 또는 구현·확산을 막는 종속성이 확인되면 생태계 부재. 나머지는 형성 중입니다. 종속성이 미확인이면 확보로 올리지 않습니다. 종속성 질문은 4개 질문의 합산에서 제외합니다.
- 시장성 축이 하나라도 null이면 전체 status=pending으로 남기고 확인된 축은 details에 보존합니다.
- 고유 URL이 2개 미만이거나 판단 보류이면 confidence=low, 그 외 잠정 평가는 최대 medium입니다. 최종 의미 검증 전 high는 부여하지 않습니다.

Tavily가 출처 유형을 보증하지 않으므로 기본적으로 간접 자료(news)로 분류합니다. arxiv.org는 paper, 사용자가 1차 출처로 확인해 `OFFICIAL_SOURCE_DOMAINS`에 지정한 정확한 호스트만 official입니다. 검색 순위나 모델 추측으로 출처 유형을 격상하지 않습니다. 허용 목록은 실제 작성 주체를 확인하고 설정하세요.

## 기존 workflow와 연결

```python
from skala_agent.workflow.graph import build_graph, initial_state
result = build_graph(provider).invoke(initial_state())
```

TRL·시장성의 기존 node는 `provider.assess()`를 통해 새 구현을 실행합니다. 기술 조사 RAG, 이해관계자·도메인 평가는 아직 DemoProvider의 pending을 사용합니다. 따라서 이슈 #5만 검증할 때는 전체 그래프보다 `skala-evaluate` 독립 명령을 사용합니다. 공통 CLI는 윤소영 담당 변경과 충돌하지 않도록 수정하지 않았습니다.

새 Evidence의 `supports_claim`은 False입니다. 원문 quote 일치는 의미적 주장 지지와 다르므로 #11 검증자만 True로 확정합니다. 기존 검증된 ID와 발췌가 그대로인 경우에만 해당 검증 상태를 보존합니다. 전체 workflow는 미검증 결과를 판단 보류로 처리하며 재검색 상한 2회를 유지합니다.

네트워크 오류는 재시도 가능한 failed, 인증/설정 및 유효하지 않은 모델 출력은 재시도 불가능한 failed로 반환합니다. 한 기술의 실패가 다른 기술의 완료된 결과를 지우지 않습니다. 추가 검색 자체의 네트워크 backoff는 #10 범위입니다.

## 검증과 남은 한계

`make test`는 실제 API 없이 두 기술 평가, TRL 상한, 시장성 집계, 근거 부족/부정 근거 구분, 출처·quote 위조, 안정적 ID, 모델 교체, JSON 재요청, HTTP 오류 및 전체 그래프 연결을 검증합니다.

실제 Tavily 검색에는 별도 키가 필요합니다. 검색 발췌는 원문 전체가 아니므로 long-context 문서 검토나 원문 대조를 대신하지 못합니다. 소형 모델의 의미적 정확도와 실제 출처 수집 품질은 골든셋·실제 API 연결 후 평가해야 합니다. 실제 시장 상황이나 기술 성숙도를 테스트 fixture에서 추정하지 않습니다.

구현 시 확인: 자동 테스트 39개·lint·기본 demo 통과. Qwen3-0.6B를 실제 로컬 Transformers/MPS로 로딩하여 단일 질문의 unknown·빈 인용 JSON 출력도 확인했습니다. 이는 연결·출력 형식 smoke test이며 기본 4B 모델의 평가 품질 검증은 아닙니다. Tavily API 키가 없는 환경에서는 실제 웹검색 E2E를 수행하지 않았습니다.
