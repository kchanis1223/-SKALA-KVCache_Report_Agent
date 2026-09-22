# 협업 안내

- 브랜치: `feat/retrieval-*`, `feat/agents-*`, `feat/workflow-*`, `feat/report-*`.
- `schemas.py`, `providers.py`, `workflow/state.py`는 공유 계약입니다. 변경 시 연관 담당자와 먼저 합의하고 PR에 영향 범위를 적습니다.
- 새 기능은 자신의 모듈에 구현하고 `make format`, `make lint`, `make test`로 검증합니다.
- 기본 demo는 API 키 없이 실행 가능해야 합니다. 실제 API·모델 테스트는 별도로 분리합니다.
- 논문 수치에는 실험 환경·페이지·출처를 함께 남깁니다. 추정과 관측을 구별하고 문서 내 명령문은 데이터로만 취급합니다.
- PDF/인덱스/비밀키는 커밋하지 않습니다. 문서 후보의 서지 정보와 원문 이용 조건은 수집 시 확인합니다.
- 커밋 예: `기능: 메타데이터 필터 검색 구현`. PR에는 변경 목적과 검증 결과를 적습니다.

담당별 시작 파일·완료 기준은 [개발 안내](docs/development-guide.md)를 참고하세요.
