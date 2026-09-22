# 설계서 ↔ 구현 변경 이력

> 제출한 설계서(`RAG-Design_...`)와 실제 구현이 달라진 지점을 기록합니다.
> 최종 제출 전에 **설계서를 이 문서 기준으로 맞춰야** 합니다.
> 채점 항목 "설계 구현 충실도 — 제출된 설계 문서 기준으로 Graph 흐름과 State 구조가
> 코드에 반영되어 있는가"(15점)에 직접 걸립니다.
>
> 작성: 윤소영 (Workflow 담당) · 최종 갱신 2026-09-22

---

## 요약

| # | 항목 | 성격 | 설계서 수정 필요 |
| --- | --- | --- | --- |
| 1 | 재검색 후 되돌아가는 지점 | **흐름 변경** | ✅ D절 Graph·Loop |
| 2 | 관점별 분석 결과를 `analyses` 하나로 통합 | 구조 변경 | ✅ D절 State 표 |
| 3 | `retry_count` 추가 | 누락 보완 | ✅ D절 State 표 |
| 4 | 이해관계자 Agent 입력에서 `market_analysis` 제거 | 흐름 정정 | ✅ D절 데이터 흐름 표 |
| 5 | `evidence` 병합 규칙 명시 | 누락 보완 | ✅ D절 State 표 |
| 6 | 평가 실패 상태(`failed`) 추가 | 기능 추가 | ⬜ 선택 |
| 7 | `MissingEvidence` 필드 확장 | 기능 추가 | ⬜ 선택 |
| 8 | `TechAnalysis` 모델 신설 | 구조 변경 | ⬜ 선택 |
| 9 | 노드별 실패 처리 방침 | 신규 | ⬜ 선택 (보고서 6장 한계점에 유용) |

---

## 1. 재검색 후 되돌아가는 지점 ⭐

**가장 중요한 변경입니다.**

| | |
| --- | --- |
| **설계서** | 근거 검증 → 부족 → 추가 검색 → State 보완 → **다시 종합** → 다시 근거 검증 |
| **구현** | 근거 검증 → 부족 → 추가 검색 → **부족한 관점만 다시 평가** → 종합 → 다시 근거 검증 |

**변경 이유**

설계서대로 종합부터 다시 하면 **근거만 쌓이고 판정은 그대로**입니다.

> 시장성 관점이 근거가 없어 `판단 보류`가 나옴 → 추가 검색으로 근거 3건 확보
> → 평가를 다시 안 하면 여전히 `판단 보류`. 근거만 State에 늘어남.

새 근거를 확보했으면 **그 관점의 판정을 다시 내려야** 재검색이 의미를 갖습니다.

**구현 위치**: `src/skala_agent/workflow/graph.py`
```python
graph.add_conditional_edges("additional_search", dispatch, ["evaluate"])
```

**설계서 수정안 (D절 Loop)**
```
근거 검증 → 부족 → 추가 검색 → 부족한 관점만 재평가 → 종합 → 다시 근거 검증
                                 (정상 관점 결과는 보존)
```

**발표 활용** — Lessons Learned에 그대로 쓸 수 있습니다.
> "설계 단계에서는 추가 검색 후 바로 종합으로 돌아가게 그렸는데, 구현하면서
> 그러면 근거만 쌓이고 판정은 안 바뀐다는 걸 발견해 해당 관점을 재평가하도록 바꿨습니다."

---

## 2. 관점별 분석 결과를 `analyses` 하나로 통합

| | |
| --- | --- |
| **설계서** | `trl_analysis`, `market_analysis`, `stakeholder_analysis`, `domain_analysis` (4개 키) |
| **구현** | `analyses: dict` — `analyses["trl"]`, `analyses["market"]`, … |

담는 내용은 같고 그릇 모양만 바뀌었습니다. 관점별로 칸이 분리돼 있어 **병렬 쓰기는 그대로 안전**합니다.

**변경 이유**
- `for key in PERSPECTIVES: analyses[key]` 로 반복 처리 가능 (키 이름을 일일이 부르지 않음)
- 관점이 늘어도 State 정의를 고칠 필요 없음
- 재평가 시 해당 관점 칸만 교체하면 됨

**구현 위치**: `src/skala_agent/workflow/state.py`
```python
analyses: Annotated[dict[str, list[Assessment]], merge_analyses]
```
이미 `docs/interfaces.md`에 기록돼 있습니다.

---

## 3. `retry_count` 추가

| | |
| --- | --- |
| **설계서** | State 표에 없음 |
| **구현** | `retry_count: int` (상한 `MAX_RETRIES = 2`) |

**변경 이유**: 없으면 **무한 루프**입니다. 근거를 끝내 못 찾는 항목이면 검증 → 추가 검색 →
재평가 → 검증 을 영원히 반복합니다. 몇 번 되돌아갔는지 세는 값이 있어야 멈춥니다.

상한 소진 시 보고서로 진행하고, 미해결 항목은 `판단 보류`로 남겨 보고서 6장 한계점에 표시합니다.

---

## 4. 이해관계자 Agent 입력에서 `market_analysis` 제거

| | |
| --- | --- |
| **설계서** | 이해관계자 Agent 입력 = `tech_analysis`, `market_analysis` |
| **구현** | 이해관계자 Agent 입력 = `tech_analysis`, `domain` |

**변경 이유**: 설계서 데이터 흐름 표에는 `market_analysis`가 입력인데, 같은 설계서의
Graph 그림에서는 시장성·이해관계자가 **형제 노드로 병렬 배치**돼 있습니다. 병렬이면
그 시점에 `market_analysis`가 아직 없어 모순입니다.

과제 가이드가 *"각 관점은 독립 축으로 조사하되, 겹치는 내용은 5번 종합 의견 단계에서
연결되도록 함"* 이라고 명시하고 있어, **완전 병렬을 유지**하고 관점 간 연결은 종합
단계에서 하도록 정리했습니다.

---

## 5. `evidence` 병합 규칙 명시

| | |
| --- | --- |
| **설계서** | `evidence: list` — 병합 규칙 없음 |
| **구현** | `merge_evidence` — 같은 ID는 최신으로 교체, 다른 주장에 ID 재사용 시 `ValueError` |

**변경 이유**: 네 관점이 `evidence` **한 칸에 동시에** 씁니다. 단순 누적(`operator.add`)이면
재평가 시 같은 ID의 옛 근거와 새 근거가 State에 공존합니다.

실제 재현 결과:
```
evidence 총건수 : 10
고유 ID 개수    : 8
중복된 ID       : market-itme, market-turboquant
```

폐기된 근거가 REFERENCE·인용 단계로 흘러가는 문제라 ID 기준 갱신 병합으로 바꿨습니다.
ID 발급은 `src/skala_agent/evidence.py`의 `evidence_id()`로 통일했습니다.

---

## 6. 평가 실패 상태 `failed` 추가

| | |
| --- | --- |
| **설계서** | 언급 없음 |
| **구현** | `Assessment.status: "pending" \| "assessed" \| "failed"` + `AgentError` |

**변경 이유**: 외부 서비스 타임아웃·연결 실패를 State에 표현할 수단이 없었습니다.
없으면 한 관점이 실패할 때 **전체 실행이 중단**돼 산출물이 0이 됩니다.

`failed`면 `verdict`가 자동으로 `판단 보류`, `confidence`가 `low`로 고정돼 실패가
정상 판정으로 둔갑하지 않습니다.

---

## 7. `MissingEvidence` 필드 확장

| | |
| --- | --- |
| **설계서** | `missing_evidence: list` — 부족 항목 저장 |
| **구현** | `technology_id`, `perspective`, `reason`, `kind`, `claim`, `queries`, `evidence_ids`, `retryable` |

**변경 이유**: `reason` 자유 문장만으로는 추가 검색 Agent가 **무엇을 다시 검색할지
판단할 수 없습니다.** `kind`(부족 유형)와 `queries`(검색 질의)를 구조화했습니다.

`retryable=False`인 항목은 재검색 대상에서 제외돼 헛된 재시도를 막습니다.

---

## 8. `TechAnalysis` 모델 신설

| | |
| --- | --- |
| **설계서** | `tech_analysis: dict` — 기술 원리, 장점, 한계, 실험 결과 |
| **구현** | `TechAnalysis` 모델 — `overview`, `scope`, `limitations`, `experiments`, `evidence_ids`, `status` |

**변경 이유**: 기술 ID → 문자열 하나로는 개요·범위·한계·실험 조건과 출처 연결을
담을 수 없었습니다.

---

## 9. 노드별 실패 처리 방침 (신규)

설계서에 없던 항목입니다. 보고서 6장 "한계점"에 쓸 수 있습니다.

| 단계 | 실패 시 | 이유 |
| --- | --- | --- |
| **논문 조사** | 3회 재시도 → 그래도 실패면 중단 | 모든 평가의 근거. 없으면 근거 없는 평가를 지어내게 됨 |
| **관점 평가** | 해당 관점만 `판단 보류`, 나머지 3관점은 진행 | 하나 느리다고 전체가 죽으면 산출물이 0 |
| **추가 검색** | 원인·회차·대상 관점을 밝히고 중단 | 네 관점 평가는 이미 끝난 상태임을 함께 알림 |

- 재시도는 **일시적 외부 오류**(`TimeoutError`, `ConnectionError`)만. 계약 위반이나
  프로그래밍 오류는 재시도하지 않고 그대로 전달합니다.
- 호출 1건당 상한 **120초** (`--timeout`으로 조정).

---

## 확인 방법

설계서를 고친 뒤, 아래로 구현과 일치하는지 확인할 수 있습니다.

```bash
uv run skala-agent --graph     # 컴파일된 실제 그래프를 mermaid로 출력
uv run skala-agent --dry-run   # 예상 호출 횟수 (최선 5회 / 최악 15회)
make test                      # tests/test_inspect.py 가 그래프 구조를 고정
```

`tests/test_inspect.py`가 실제 노드·엣지를 고정하므로, **코드가 바뀌어 문서와
어긋나면 CI에서 잡힙니다.**

---

## 아직 반영 전 (PR 대기 중)

| 항목 | 출처 | 상태 |
| --- | --- | --- |
| `synthesis_findings` State 키 + `SynthesisFinding` 모델 | 이준형 PR #21 | 리뷰 대기 |

병합되면 이 문서에 추가해야 합니다.
