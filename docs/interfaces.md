# State / Agent I/O 계약 v2 (오케스트레이터 구조)

v1(고정 4관점 fan-out, Assessment·Signal·Details 스키마)을 오케스트레이터 구조로 단순화한 계약입니다. 코드 기준은 `src/skala_agent/workflow/state.py`, 규칙 기준은 `src/skala_agent/checks.py`입니다.

## 흐름

```
orchestrator ─Send→ worker × N → synthesize → validate ─┬→ report → END
     ▲                                                  │
     └──────── 부족 & retry_count < 1 ───────────────────┘
```

## State 읽기·쓰기 책임

| key | 타입 | 쓰는 노드 | 갱신 규칙 |
| --- | --- | --- | --- |
| run_id | str | CLI | 실행 중 고정. 체크포인트 thread_id, LangSmith metadata, 보고서 파일명 |
| question | str | CLI | 실행 중 고정 |
| tech_brief | str | orchestrator | 첫 계획 때 1회 |
| plan | list[SubTask] | orchestrator | 첫 계획은 전체, 재시도는 같은 id의 instruction만 교체 |
| worker_results | dict[task_id, WorkerResult] | worker | 병합 Reducer. 자기 task_id만 쓰고 재시도는 덮어씀 |
| sources | dict[source_id, Source] | worker | 병합 Reducer. finding이 인용한 출처만 |
| result | dict[agent, list[WorkerResult]] | synthesize | 매 회차 전체 교체 |
| verdict | Verdict | validate | 매 회차 전체 교체 |
| retry_count | int | orchestrator(재계획) | +1, 최대 1 |
| node_status | dict[str, str] | 모든 노드 | 병합 Reducer. pending / done / failed / fallback |
| last_error | str \| None | CLI | 실행이 예외로 멈췄을 때 기록 |
| report | str | report | 최종 Markdown |

## 데이터 모양

| 모델 | 필드 | 규칙 |
| --- | --- | --- |
| SubTask | id, agent, instruction | id는 영문·숫자·`-`·`_`만. agent는 domain / market / stakeholder / tech |
| Source | id, kind, title, url, text, page | id는 `{task_id}-{w\|p}{n}`. text는 도구가 모델에게 보여준 원문 그대로 |
| Finding | claim, source_id, quote | quote는 해당 Source.text의 연속된 부분(공백 차이만 허용). claim 400자·quote 500자를 넘으면 거부하지 않고 앞부분만 남김 |
| WorkerResult | task_id, from, success, findings, verdict, error | 유효한 finding이 1개 이상이면 success |
| Verdict | sufficient, feedback | feedback은 계획에 있는 task_id만 |

## 코드가 강제하는 규칙 (`checks.py`)

| 규칙 | 함수 | 위반 시 |
| --- | --- | --- |
| 과제 1~6개, 같은 agent 최대 2개, id 중복 금지 | `validate_plan` | 위반 내용을 알려 1회 재요청 → 기본 계획(4관점 × 1) |
| 재시도는 기존 id만, agent 변경 금지, 부족 판정 과제만 | `validate_plan(previous)` | 1회 재요청 → 기존 지시 + 보완점으로 대체 |
| 인용은 수집한 원문의 부분문자열 | `check_citations` | 해당 finding 제거 |
| 재시도 상한 1회, 보완 대상이 없으면 보고서로 | `route_after_validate` | 보고서로 이동 |
| 보고서 7개 목차 순서, 허용된 [S:id]만, URL 직접 표기 금지 | `check_report` | 1회 재작성 → 템플릿 보고서 |
| REFERENCE는 실제 인용 순서대로 코드가 생성 | `attach_references` | 해당 없음 |

## 실행 관리

- `recursion_limit=15`: 정상 경로는 최대 9 superstep(재시도 포함)입니다.
- 체크포인트: `outputs/checkpoints.sqlite`. `skala-agent --resume <run_id>`가 마지막 체크포인트부터 이어서 실행합니다. 역직렬화는 `workflow.graph.checkpoint_serde()`의 허용 목록 타입만 받습니다.
- worker 안의 예외(타임아웃, API 오류)는 그 과제만 `success=False`로 기록하고 진행합니다. 다른 노드의 예외는 실행을 멈추고 `last_error`를 남깁니다.
