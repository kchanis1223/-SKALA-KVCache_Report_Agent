# 진행 체크리스트

## 설계 단계 (Due: DAY3 10시)
- [x] 폴더 구조 생성
- [x] 기술 2건 선정 + 사유
- [x] RAG 적용 대상 결정
- [x] 임베딩 모델 선정 논리
- [x] 4관점 평가 기준 정의
- [x] State 스키마 설계
- [x] 그래프 흐름 설계 (mermaid)
- [x] 보고서 목차 초안
- [ ] 논문 PDF 확보 → 실제 페이지 수 확인 (200p 제한)
- [x] 조원 역할 분담 확정
- [ ] 설계 문서 PDF 변환 및 제출

## 개발 단계 (Due: DAY3 16시)

현재 경로와 담당별 순서는 [개발 안내](development-guide.md)를 따릅니다. 아래 체크는 실제 연동 완료 기준이며 demo 실행과 구분합니다.
- [ ] src/skala_agent/workflow/state.py — State 스키마 구현
- [ ] src/skala_agent/retrieval/ — 로딩·청킹·인덱싱 (bge-m3)
- [ ] 임베딩 후보 3종 Hit@3/MRR 실측
- [ ] 에이전트 6종 개별 구현 + 단위 테스트
- [ ] src/skala_agent/workflow/graph.py — fan-out/fan-in/중립성 루프 조립
- [ ] 전체 실행 → 보고서 생성 확인
- [ ] 재현성 점검 (clean 환경에서 재실행)
- [x] README Contributors 작성
- [ ] GitHub 푸시 + 보고서 PDF 제출

## 발표
- [ ] 차별점 정리 (중립성 검증 루프)
- [ ] 보고서 핵심 포인트
- [ ] Lessons Learned

## 실행 가능한 뼈대

- [x] 공통 schema 및 provider 계약
- [x] demo 병렬 평가·선택적 재시도·보고서 템플릿
- [x] 회귀 테스트와 CI 구성
- [x] 담당별 구현·인수인계 안내
- [ ] 실제 PDF·검색·LLM provider 연결
- [ ] 의미적 근거 검증·상충 분석·최종 보고서 구현
