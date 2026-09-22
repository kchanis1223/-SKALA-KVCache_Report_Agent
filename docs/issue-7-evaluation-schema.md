# 이슈 #7 — 평가 Agent 공통 출력 계약

네 평가 Agent는 `Assessment`와 중앙 `Evidence` 목록을 반환합니다. `EvaluationOutput`은 한 관점의 결과와 근거를 함께 검사하는 provider 경계 모델입니다. LangGraph의 State 및 `assess(...) -> (list[Assessment], list[Evidence])` 반환 형식은 유지합니다.

## 필드와 상태

| 항목 | 계약 |
| --- | --- |
| 판정 | technology_id, perspective, verdict, rationale, confidence, status, details |
| 관점별 details | trl: level / market: demand·adoption·ecosystem / stakeholder: 5개 positions·overall / domain: cost·sla_risk·operations·operational_risks |
| signals | question, grade(상/중/하), evidence_ids. 질문 중복 불가. 상·중에는 근거 참조 필수 |
| confidence | low / medium / high. 숫자 점수 불가 |
| Evidence | id, technology_id, claim, HTTP(S) url, title, source_type, excerpt, 선택 page·section_or_page·chunk_id, confidence, supports_claim |
| 페이지 | 1부터 시작하는 정수 또는 null. 웹자료에는 page/chunk_id가 없을 수 있음 |
| 근거 ID | 생성 시 evidence_id(owner, technology_id, claim, url, chunk_id)를 사용. validator는 기존 ID와 호환되도록 hash 모양을 강제하지 않음 |

평가 레코드는 알 수 없는 필드를 거부합니다. 식별자·판정·이유·질문·주장·제목·발췌는 공백만 있는 문자열을 허용하지 않으며, 발췌의 앞뒤 공백을 자동 제거하지 않습니다. 기존 `reason`/`evidence_text` 입력 alias와 canonical `rationale`/`excerpt` 출력은 유지합니다.

- `assessed`: 근거 참조 및 signals가 있고 평가 축이 완성되어야 합니다. TRL level, 시장성 3축, 이해관계자 overall, 도메인 3축이 필요합니다. SLA 판단 불가는 미완성입니다.
- `pending`: 일부 축 또는 근거 부족. details로 확인된 부분을 보존하며 confidence는 low로 정규화합니다.
- `failed`: error 필수. verdict는 판단 보류, confidence는 low로 정규화합니다. details는 없어도 됩니다.

개별 Assessment는 기존 DemoProvider 등과 호환되도록 details=null을 허용합니다. 실제 평가 경계인 EvaluationOutput에서는 failed를 제외하고 관점별 details를 요구합니다. 이 구분으로 미구현 데모와 구현된 평가의 계약을 분리합니다.

## 참조 및 신뢰도 검증

선택 기술마다 정확히 하나의 결과가 필요하며, 요청 관점과 달라서는 안 됩니다. 중복 결과·Evidence ID·근거 참조를 거부합니다. Assessment가 참조하는 모든 Evidence는 존재하고 같은 기술에 속해야 합니다. Signal/StakeholderPosition의 evidence_ids는 상위 Assessment의 evidence_ids에 포함되어야 합니다.

이해관계자는 5개 주체를 각각 한 번 출력합니다. 자료 없음 이외의 입장에는 근거가 필요하며, overall은 자료 없음을 제외한 엄격한 과반 규칙을 검사합니다.

confidence는 해당 Assessment가 직접 참조한 고유 URL 수로 제한합니다. 2개 미만이면 low입니다. 참조하지 않은 근거나 동일 논문의 여러 chunk는 출처 수를 늘리지 않습니다. 2개 이상이어도 신뢰도를 자동 승격하지 않으며, 실제 평가 provider는 의미 검증 전 최대 medium을 사용합니다. 정규화는 입력 객체를 수정하지 않고 복사본을 반환합니다.

이 검사는 구조·참조 검사입니다. URL이 두 개라는 사실은 서로 독립된 출처라는 뜻이 아닙니다. 최종 검증자가 주장 지지를 확인해야 supports_claim=True가 되며, 최종 confidence 제한은 기존 validation 단계가 검증된 출처를 기준으로 다시 적용합니다.

## 팀원이 사용하는 방법

```python
from skala_agent.evaluation_contracts import validate_evaluation_output

# provider의 assess() 마지막에 사용합니다.
return validate_evaluation_output(
    perspective, technologies, assessments, collected_evidence
)
```

네 관점의 EvaluationProvider에는 이미 연결했습니다. 다른 provider에서 기존 State의 Evidence를 직접 참조한다면 그 참조 대상도 검증에 제공해야 합니다. 구조 위반은 Pydantic ValidationError로 보고하며, 네트워크 실패처럼 숨기거나 재검색하지 않습니다.

독립 JSON을 검증할 때는 아래 모델을 사용합니다. technology_ids에는 호출자가 요청한 기술 ID를 넣습니다.

```python
from skala_agent.evaluation_contracts import EvaluationOutput

output = EvaluationOutput.model_validate(payload)
json_schema = EvaluationOutput.model_json_schema()
```

JSON Schema는 필드 타입·enum·관점별 discriminator를 표현합니다. 출처 수, 참조 관계, 과반 집계 등 교차 검사는 Python의 model_validate도 실행해야 합니다.

## 예제와 검증

`tests/fixtures/evaluation_outputs.json`에 네 관점 각각의 정상·근거 부족·실행 실패 예제 12개를 제공합니다. 모든 출처와 판정은 합성 데이터입니다. 잘못된 출력 예제는 `tests/test_evaluation_contracts.py`의 `invalid_payload()`에 있으며, 참조 누락·다른 기술·필드 오타·잘못된 페이지/신뢰도·중복·미완성 평가 등을 검증합니다.

```bash
uv run pytest tests/test_evaluation_contracts.py tests/test_contracts.py
make lint
make test
```

이슈 #5/#6의 실제 네 Agent 출력도 같은 검증기를 통과합니다. 기존 alias와 데모·재검색·실패 격리 테스트도 함께 유지합니다.
