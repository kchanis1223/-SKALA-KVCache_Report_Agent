# 모듈 간 계약

## State와 반환값

설계서의 `trl_analysis`, `market_analysis`, `stakeholder_analysis`, `domain_analysis`는 각각 `analyses["trl"]`, `analyses["market"]`, `analyses["stakeholder"]`, `analyses["domain"]`로 저장합니다. 나머지 주요 키는 설계서와 같습니다. 각 평가 호출은 선택된 기술마다 정확히 하나의 `Assessment`와 새 `Evidence` 목록을 반환합니다. 노드는 State를 직접 수정하지 않고 변경분만 반환합니다.

`Assessment`의 `verdict`는 현재 공통 문자열입니다. TRL 1~9, 시장성 3축, 이해관계자 5주체, 도메인 A/B/C의 상세 schema와 등급 집계는 김강휘 담당 구현 범위입니다. `signals`에는 질문별 상/중/하와 근거 ID를 저장합니다. 조사 완료 시에만 `status="assessed"`로 설정합니다.

`Evidence.id`는 실행 내 고유하고 안정적인 ID를 사용합니다. `supports_claim`은 기본 False이며 단순 URL 수집만으로 True를 주면 안 됩니다. 실제 검증자가 원문과 주장을 확인한 뒤 설정해야 합니다. 현재 skeleton은 이 플래그를 신뢰하므로 의미적 검증을 대신하지 않습니다.

## RAG

기술 조사: `role="primary"`. 도메인 평가: role 제한 없이 독립 검토 포함. TRL·시장성·이해관계자: 웹검색. 종합·검증·보고서는 누적 State만 사용합니다.

청크 필수 metadata: paper_id, camp, role, section, page, source_url. 기본 임베딩 선택은 `BAAI/bge-m3`; 모델 로딩은 실제 adapter에서 지연 실행하고 토큰 단위 chunking을 구현하세요. VectorDB 제품과 표·OCR 처리 도구는 미정입니다.

## 재검색과 한계

처음 네 관점을 병렬 실행한 뒤 fan-in합니다. 근거 부족 관점만 `Send`로 재실행하고 다른 관점의 결과는 유지합니다. 추가 검색은 최대 2회입니다. 같은 관점 안에서는 두 기술을 함께 재평가합니다. 예외는 숨기지 않고 호출자에게 전달합니다. 네트워크 재시도/backoff, checkpoint, 취소 및 타임아웃은 별도 구현 대상입니다.

출처 수는 고유 URL 기준이며 source 독립성까지 보장하지 않습니다. 인용 여부 검사는 의미적 지지·중립성 검증의 대체재가 아닙니다. 현재 종합은 관점 결과 취합이고, 보고서는 목차 템플릿입니다.
